"""Ultralytics 8.3.161 YOLOv10-S adapter."""

import csv
import math
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import ultralytics
from torch.utils.data import get_worker_info
from ultralytics.data.augment import LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops

from scripts.baseline_data import read_json, write_json
from scripts.baseline_metrics import MAX_DETECTIONS, MIN_SCORE, coco_eval, prediction_rows
from scripts.baseline_run import (
    PauseTraining, UpdateCheck, check_gradients, parameters, publish_copy, save_history,
)
from scripts.photometric import PhotometricSampler, adjust_luminance
from scripts.yolo_staged import StagedBackbone, layout

VERSION = "8.3.161"
DISABLED_AUGMENTATIONS = {
    name: 0.0 for name in (
        "mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees",
        "translate", "scale", "shear", "perspective", "fliplr", "flipud", "bgr",
    )
}


def check_version():
    if ultralytics.__version__ != VERSION:
        raise RuntimeError(f"Se requiere ultralytics=={VERSION}, detectado {ultralytics.__version__}")


def check_head(model, classes):
    head = model.model[-1]
    if type(head).__name__ != "v10Detect" or not model.end2end or head.nc != classes:
        raise RuntimeError(f"Se esperaba v10Detect end-to-end con {classes} clases")
    head.max_det = MAX_DETECTIONS


class CocoValidator(DetectionValidator):
    def __init__(self, *args, truth, classes, **kwargs):
        super().__init__(*args, **kwargs)
        self.truth = truth
        self.classes = classes
        self.images_by_name = {row["file_name"]: row for row in truth["images"]}
        self.coco_map = None

    def init_metrics(self, model):
        super().init_metrics(model)
        if not self.end2end:
            raise RuntimeError("La validación debe usar la rama NMS-free de YOLOv10")
        self.args.save_json = True  # Activate pred_to_json with our original COCO IDs.

    def pred_to_json(self, predn, filename):
        detections = torch.cat(
            (predn["bboxes"], predn["conf"][:, None], predn["cls"][:, None]), dim=1
        ).cpu().numpy()
        image = self.images_by_name[Path(filename).name]
        self.jdict.extend(prediction_rows(image, detections, self.classes))

    def get_stats(self):
        if self.seen != len(self.truth["images"]):
            raise RuntimeError(f"Validación YOLO incompleta: {self.seen} imágenes")
        metrics = super().get_stats()
        evaluation = coco_eval(self.truth, self.jdict)
        self.coco_map = float(evaluation.stats[0])
        self.coco_ap50 = float(evaluation.stats[1])
        return metrics


class YoloPhotometric:
    def __init__(self, spec, seed):
        self.sampler = PhotometricSampler(spec, seed)

    def __call__(self, labels):
        gains = self.sampler.sample(get_worker_info())
        if gains is not None:
            labels["img"] = adjust_luminance(labels["img"], *gains)
        return labels


class YoloTrainer(DetectionTrainer):
    def __init__(self, args, truth, classes, summary, photometric=None):
        self.truth = truth
        self.classes = classes
        self.summary = summary
        self.photometric = photometric
        super().__init__(overrides=args)

    def build_dataset(self, img_path, mode="train", batch=None):
        # augment=False selects only LetterBox and Format, including when Albumentations is installed.
        stride = max(int(self.model.stride.max()), 32)
        dataset = YOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=False,
            hyp=self.args, rect=False, cache=False, single_cls=False, stride=stride,
            pad=0.0, prefix=f"{mode}: ", task="detect", data=self.data, fraction=1.0,
        )
        split = "train" if mode == "train" else "valid"
        expected_files = {
            path.name for path in (Path(self.args.data).parent / "images" / split).glob("*.jpg")
        }
        expected = self.summary[split]
        if (
            {Path(label["im_file"]).name for label in dataset.labels} != expected_files
            or len(dataset) != expected["images"]
            or sum(len(label["cls"]) for label in dataset.labels) != expected["instances"]
        ):
            raise RuntimeError(f"Ultralytics alteró los recuentos o nombres de {split}")
        counts = np.bincount(
            np.concatenate([label["cls"].astype(int).ravel() for label in dataset.labels]),
            minlength=len(self.classes),
        )
        if counts.tolist() != [expected["ground_truth_by_category"][str(c["original_id"])] for c in self.classes]:
            raise RuntimeError(f"Ultralytics alteró el mapeo de clases de {split}")
        if mode == "train" and self.photometric is not None:
            transforms = dataset.transforms.transforms
            if [type(transform).__name__ for transform in transforms] != ["LetterBox", "Format"]:
                raise RuntimeError("EXP-002 requiere la cadena base LetterBox → Format")
            transforms.insert(0, YoloPhotometric(self.photometric, self.args.seed))
        return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        check_head(model, len(self.classes))
        if weights is None:
            raise RuntimeError("YOLOv10-S necesita pesos iniciales")
        check_head(weights, len(self.classes) if self.resume else 80)
        source, target = weights.state_dict(), model.state_dict()
        head_prefix = f"model.{len(model.model) - 1}."
        allowed = (head_prefix + "cv3.", head_prefix + "one2one_cv3.")
        mismatches = []
        for key in sorted(source.keys() | target.keys()):
            if key in source and key in target and source[key].shape == target[key].shape:
                if not torch.equal(source[key].cpu(), target[key].cpu()):
                    raise RuntimeError(f"Peso YOLO compatible que no se transfirió: {key}")
            else:
                mismatches.append(key)
        if (self.resume and mismatches) or any(not key.startswith(allowed) for key in mismatches):
            raise RuntimeError(f"Pesos YOLO incompatibles fuera de las cabezas: {mismatches}")
        if not self.resume and any(not any(key.startswith(prefix) for key in mismatches) for prefix in allowed):
            raise RuntimeError(f"No se confirmó la adaptación de las dos cabezas YOLO: {mismatches}")
        report = "resume_weight_transfer.json" if self.resume else "weight_transfer.json"
        write_json(self.save_dir / report, {"reinitialized": mismatches})
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CocoValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            truth=self.truth, classes=self.classes,
        )

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness, ap50 = self.validator.coco_map, self.validator.coco_ap50
        if any(not math.isfinite(value) for value in metrics.values()) or not (
            0 <= fitness <= 1 and 0 <= ap50 <= 1
        ):
            raise RuntimeError("Métricas o pérdidas YOLO de validación inválidas")
        metrics.update({"val/map": fitness, "val/ap50": ap50})
        self.best_fitness = fitness if self.best_fitness is None else max(self.best_fitness, fitness)
        return metrics, fitness

    def save_model(self):
        # Publish complete files even if SLURM terminates during serialization.
        last, best = self.last, self.best
        with TemporaryDirectory(dir=self.wdir, prefix=".saving-") as temporary:
            self.last, self.best = Path(temporary) / last.name, Path(temporary) / best.name
            try:
                super().save_model()
                self.last.replace(last)
                if self.best.exists():
                    self.best.replace(best)
            finally:
                self.last, self.best = last, best

    def final_eval(self):
        """Retain the native resumable format; the common CLI evaluates the chosen checkpoint."""


class StagedYoloTrainer(YoloTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        layout(weights)
        layout(model)
        return model

    def _model_train(self):
        super()._model_train()
        self.staged.begin_epoch(self, self.epoch + 1)

    def optimizer_step(self):
        self.staged.before_optimizer_step()
        super().optimizer_step()


def make_predictor(model, imgsz):
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True, stride=32)

    @torch.inference_mode()
    def predict(bgr):
        height, width = bgr.shape[:2]
        padded = letterbox(image=bgr)
        array = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(array).to("cuda").float().div_(255).unsqueeze(0)
        output = model(tensor)
        boxes = output[0] if isinstance(output, tuple) else output
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 6:
            raise RuntimeError("Se esperaba salida YOLOv10 end-to-end B×N×6")
        detections = boxes[0]
        detections = detections[detections[:, 4] > MIN_SCORE][:MAX_DETECTIONS]
        scale = min(imgsz / height, imgsz / width)
        left = round((imgsz - round(width * scale)) / 2 - 0.1)
        top = round((imgsz - round(height * scale)) / 2 - 0.1)
        ops.scale_boxes((imgsz, imgsz), detections[:, :4], (height, width), ((scale, scale), (left, top)))
        return detections.cpu().numpy()
    return predict


def predictor(spec, manifest, weights, output):
    check_version()
    model, _ = attempt_load_one_weight(str(weights), device=torch.device("cuda"), fuse=False)
    check_head(model, len(manifest["classes"]))
    if model.names != {item["internal_id"]: item["name"] for item in manifest["classes"]}:
        raise RuntimeError("Las etiquetas del checkpoint YOLO difieren del mapeo de evaluación")
    return make_predictor(model.float().eval(), spec["imgsz"])


def ultralytics_args(args, spec, resume_path):
    return {
        "model": str(args.weights), "pretrained": True, "data": str(args.data / "dataset.yaml"),
        "project": str(args.output.parent), "name": args.output.name, "exist_ok": True,
        "resume": str(resume_path) if args.resume else False,
        "task": "detect", "device": "", "seed": spec["seed"], "deterministic": True,
        "epochs": spec["epochs"], "batch": spec["batch"], "nbs": spec["batch"],
        "imgsz": spec["imgsz"], "workers": spec["workers"], "optimizer": spec["optimizer"],
        "lr0": spec["lr"], "lrf": spec["final_lr_factor"], "momentum": spec["momentum"],
        "weight_decay": spec["weight_decay"], "cos_lr": spec["scheduler"] == "cosine",
        "warmup_epochs": spec["warmup_epochs"], "warmup_momentum": spec["warmup_momentum"],
        "warmup_bias_lr": spec["warmup_bias_lr"], "amp": False, "half": False, "freeze": 0,
        "patience": 0, "val": True, "split": "val", "save": True, "save_period": -1,
        "augment": False, "rect": False, "multi_scale": False, "close_mosaic": 0,
        "single_cls": False, "fraction": 1.0, "cache": False, "agnostic_nms": False,
        "plots": False, "conf": MIN_SCORE, "max_det": MAX_DETECTIONS,
        **DISABLED_AUGMENTATIONS,
    }


def record_runtime(trainer, args):
    inventory = parameters(trainer.model)
    staged = getattr(trainer, "staged", None)
    frozen = {row["name"] for row in inventory["frozen"]}
    if staged is not None and frozen != set(staged.frozen_parameters()):
        raise RuntimeError("El inventario inicial no coincide con la fase congelada")
    if staged is None and any(".dfl" not in name for name in frozen):
        raise RuntimeError("YOLO presenta parámetros congelados fuera de DFL")
    write_json(args.output / "parameters.json", inventory)
    batches = len(trainer.train_loader)
    warmup = trainer.args.warmup_epochs
    photometric = {}
    if trainer.photometric is not None:
        transforms = trainer.train_loader.dataset.transforms.transforms
        photometric = {
            "photometric": trainer.photometric,
            "train_transforms": [type(transform).__name__ for transform in transforms],
        }
    write_json(args.output / "ultralytics_runtime.json", {
        "args": vars(trainer.args),
        "iterations_per_epoch": batches, "nominal_max_iterations": batches * trainer.epochs,
        "effective_batch": trainer.batch_size * trainer.accumulate,
        "warmup_iterations_end_index": max(round(warmup * batches), 100) if warmup > 0 else -1,
        "optimizer": type(trainer.optimizer).__name__,
        "optimizer_initial_groups": [
            {k: g[k] for k in ("initial_lr", "weight_decay", "betas") if k in g}
            for g in trainer.optimizer.param_groups
        ],
        **({"freeze_backbone_epochs": staged.duration} if staged is not None else {}),
        **photometric,
    })


def probe(trainer, args, spec):
    trainer._setup_train(world_size=1)
    batch = next(iter(trainer.train_loader))
    import cv2

    image = cv2.imread(batch["im_file"][0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    trainer.model.eval()
    result = make_predictor(trainer.model, spec["imgsz"])(image)
    if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
        raise RuntimeError("Salida YOLO no finita o de forma inválida")
    trainer.model.train()
    staged = getattr(trainer, "staged", None)
    if staged is not None:
        staged.begin_epoch(trainer, 1, probe=True)
    loss, items = trainer.model(trainer.preprocess_batch(batch))
    if not torch.isfinite(loss.sum()).item():
        raise RuntimeError("Pérdida YOLO no finita")
    loss.sum().backward()
    gradients = check_gradients(torch, trainer.model)
    head = f"model.{len(trainer.model.model) - 1}."
    checks = {
        "backbone": lambda name: name.startswith("model.0."),
        "one-to-many": lambda name: name.startswith(head + "cv3."),
        "one-to-one": lambda name: name.startswith(head + "one2one_cv3."),
    }
    if staged is not None:
        del checks["backbone"]
        checks["neck"] = lambda name: name.startswith("model.13.")
        staged.finish_probe(trainer)
    for name, matches in checks.items():
        if not any(matches(key) and value == "nonzero" for key, value in gradients.items()):
            raise RuntimeError(f"YOLO no recibió gradientes útiles en {name}")
    write_json(args.output / "initialization.json", {
        "classes": trainer.model.model[-1].nc, "inference_detections": len(result),
        "losses": trainer.label_loss_items(items), "gradients": gradients,
    })


def train(args, spec, manifest):
    check_version()
    # Avoid automatic third-party integrations in benchmark jobs.
    ultralytics.settings.update({
        key: False for key in (
            "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune",
            "tensorboard", "wandb",
        ) if key in ultralytics.settings
    })
    truth = read_json(args.data / "coco" / "valid.json")
    last = args.output / "weights" / "last.pt"
    overrides = ultralytics_args(args, spec, last)
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(last)
        state = torch.load(last, map_location="cpu", weights_only=False)
        completed_epochs = state["epoch"] + 1
        if state["optimizer"] is None or not 0 < completed_epochs < spec["epochs"]:
            raise ValueError("El checkpoint de YOLO está finalizado o carece de estado del optimizador")
        if args.pause_after and args.pause_after <= completed_epochs:
            raise ValueError("El punto de pausa ya se alcanzó en la ejecución anterior")
        if state["train_metrics"]["fitness"] == state["best_fitness"]:
            publish_copy(last, last.with_name("best.pt"))
        del state
        trim_csv(args.output / "results.csv", completed_epochs)
    freeze = spec.get("freeze_backbone_epochs", 0)
    trainer_type = StagedYoloTrainer if freeze else YoloTrainer
    trainer = trainer_type(overrides, truth, manifest["classes"], manifest["splits"], spec.get("photometric"))
    if trainer.save_dir.resolve() != args.output:
        raise ValueError(f"Ultralytics usaría una salida diferente: {trainer.save_dir}")
    if trainer.data["names"] != {item["internal_id"]: item["name"] for item in manifest["classes"]}:
        raise ValueError("El YAML YOLO y el manifiesto discrepan en las clases")
    def initialize_runtime(t):
        if freeze:
            t.staged = StagedBackbone(t, args.output, freeze)
        record_runtime(t, args)

    trainer.add_callback("on_pretrain_routine_end", initialize_runtime)

    if args.initialize:
        probe(trainer, args, spec)
        return {}

    def start_update_check(t):
        head = f"model.{len(t.model.model) - 1}.one2one_cv3."
        t.update_check = UpdateCheck(t.model, args.output, ("model.0.", head))

    def check_losses(t):
        if not torch.isfinite(t.tloss).all().item():
            raise RuntimeError("Pérdidas YOLO de entrenamiento no finitas")
        if freeze:
            t.staged.end_epoch(t, t.epoch + 1)
        else:
            t.update_check.after_epoch(t.epoch + 1)

    def pause(t):
        if t.epoch + 1 == args.pause_after:
            raise PauseTraining

    if not freeze:
        trainer.add_callback("on_pretrain_routine_end", start_update_check)
    trainer.add_callback("on_train_epoch_end", check_losses)
    trainer.add_callback("on_model_save", pause)
    try:
        trainer.train()
        status = "complete"
    except PauseTraining:
        status = "paused"
    rows = []
    with (args.output / "results.csv").open(encoding="utf-8") as file:
        for item in csv.DictReader(file):
            epoch = int(item.pop("epoch"))
            rows.append({"step": epoch, **{key: float(value) for key, value in item.items()}})
    best_path = args.output / "weights" / "best.pt"
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    save_history(rows, args.output, prefer_last_tie=True)
    return {
        "status": status, "step_unit": "epoch", "best_step": state["epoch"] + 1,
        "best_validation_map": state["train_metrics"]["fitness"],
        "last_checkpoint": "weights/last.pt", "best_checkpoint": "weights/best.pt",
    }


def trim_csv(path, completed_epochs):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = [row for row in reader if row.get("epoch") and int(row["epoch"]) <= completed_epochs]
    if [int(row["epoch"]) for row in rows] != list(range(1, completed_epochs + 1)):
        raise ValueError(f"El historial {path} no coincide con el checkpoint")
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
