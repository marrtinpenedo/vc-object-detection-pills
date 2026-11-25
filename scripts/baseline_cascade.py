"""Detectron2 Cascade R-CNN adapter."""

import json
import math
import re
from copy import copy
from pathlib import Path

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.evaluation import DatasetEvaluator
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from torch.utils.data import get_worker_info

from scripts.baseline_data import read_json, write_json
from scripts.baseline_metrics import MAX_DETECTIONS, MIN_SCORE, coco_eval, prediction_rows
from scripts.baseline_run import (
    PauseTraining, UpdateCheck, check_gradients, parameters, publish_copy, save_history,
)
from scripts.photometric import PhotometricSampler, adjust_luminance
from scripts.geometric import GeometrySampler, transform_coords, warp

PREFIX = "roi_heads.box_predictor."
EXPECTED_CLASSIFIERS = {
    f"{PREFIX}{stage}.cls_score.{kind}"
    for stage in range(3) for kind in ("weight", "bias")
}


class CheckedWeights(DetectionCheckpointer):
    def __init__(self, model, *, pretrained=False):
        super().__init__(model)
        self.pretrained = pretrained
        self.transfer = {}

    def _load_model(self, checkpoint):
        result = super()._load_model(checkpoint)
        missing = set(result.missing_keys) | {key for key, _, _ in result.incorrect_shapes}
        unexpected = set(result.unexpected_keys)
        if self.pretrained:
            if missing != EXPECTED_CLASSIFIERS or any(
                not key.startswith("roi_heads.mask_head.") for key in unexpected
            ):
                raise RuntimeError(f"Cascade COCO: incompatibilidades inesperadas: {result}")
        elif missing or unexpected:
            raise RuntimeError(f"Cascade: checkpoint ajustado incompatible: {result}")
        self.transfer = {"reinitialized": sorted(missing), "discarded": sorted(unexpected)}
        return result


def configure(spec, class_count, n_train, weights, out):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(spec["architecture"]))
    official_url = model_zoo.get_checkpoint_url(spec["architecture"])
    if Path(official_url).name != "model_final_480dd8.pkl":
        raise RuntimeError(f"El zoo devolvió un checkpoint inesperado: {official_url}")

    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_count
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MIN_SCORE
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = spec["nms_iou"]
    cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DETECTIONS
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.PRECISE_BN.ENABLED = False
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.MIN_SIZE_TRAIN = (spec["imgsz"],)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = spec["imgsz"]
    cfg.INPUT.MIN_SIZE_TEST = spec["imgsz"]
    cfg.INPUT.MAX_SIZE_TEST = spec["imgsz"]
    cfg.DATALOADER.NUM_WORKERS = spec["workers"]
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    iterations = math.ceil(n_train / spec["batch"])
    cfg.SOLVER.IMS_PER_BATCH = spec["batch"]
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.SOLVER.MAX_ITER = spec["epochs"] * iterations
    cfg.SOLVER.BASE_LR = spec["lr"]
    cfg.SOLVER.MOMENTUM = spec["momentum"]
    cfg.SOLVER.NESTEROV = False
    cfg.SOLVER.WEIGHT_DECAY = spec["weight_decay"]
    cfg.SOLVER.WEIGHT_DECAY_NORM = spec["weight_decay_norm"]
    cfg.SOLVER.LR_SCHEDULER_NAME = spec["scheduler"]
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_FACTOR = spec["warmup_factor"]
    cfg.SOLVER.WARMUP_ITERS = min(spec["warmup_iterations"], cfg.SOLVER.MAX_ITER - 1)
    cfg.SOLVER.AMP.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = spec["gradient_clip_norm"] > 0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = spec["gradient_clip_norm"]
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.TEST.EVAL_PERIOD = iterations * spec["eval_every_epochs"]
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations * spec["checkpoint_every_epochs"]
    cfg.SEED = spec["seed"]
    cfg.CUDNN_BENCHMARK = False
    cfg.OUTPUT_DIR = str(out)
    return cfg, iterations, official_url


def register_training_data(cfg, root, manifest):
    names = {}
    mapping = {item["original_id"]: item["internal_id"] for item in manifest["classes"]}
    for split in ("train", "valid"):
        name = f"pills_baseline_{split}"
        if name not in DatasetCatalog.list():
            register_coco_instances(
                name, {}, str(root / "coco" / f"{split}.json"), str(root / "images" / split)
            )
        records = DatasetCatalog.get(name)
        truth = read_json(root / "coco" / f"{split}.json")
        if (
            MetadataCatalog.get(name).thing_dataset_id_to_contiguous_id != mapping
            or [record["image_id"] for record in records]
            != [image["id"] for image in truth["images"]]
            or sum(len(record["annotations"]) for record in records) != manifest["splits"][split]["instances"]
        ):
            raise RuntimeError(f"Detectron2 cargó categorías o imágenes inesperadas en {split}")
        names[split] = name
    cfg.DATASETS.TRAIN = (names["train"],)
    cfg.DATASETS.TEST = (names["valid"],)
    cfg.freeze()


def as_detections(instances):
    return torch.cat(
        (instances.pred_boxes.tensor, instances.scores[:, None], instances.pred_classes[:, None]), dim=1
    ).cpu().numpy()


class CascadeEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        metadata = MetadataCatalog.get(dataset_name)
        self.truth = read_json(Path(metadata.json_file))
        self.images = {image["id"]: image for image in self.truth["images"]}
        self.classes = [
            {"original_id": original, "internal_id": internal}
            for original, internal in metadata.thing_dataset_id_to_contiguous_id.items()
        ]

    def reset(self):
        self.predictions = []
        self.seen = set()

    def process(self, inputs, outputs):
        for item, output in zip(inputs, outputs, strict=True):
            image_id = item["image_id"]
            if image_id in self.seen:
                raise RuntimeError(f"Imagen de validación repetida: {image_id}")
            self.seen.add(image_id)
            self.predictions.extend(prediction_rows(
                self.images[image_id], as_detections(output["instances"]), self.classes,
            ))

    def evaluate(self):
        if self.seen != self.images.keys():
            raise RuntimeError("Validación Cascade incompleta")
        evaluation = coco_eval(self.truth, self.predictions)
        return {"bbox": {"AP": float(evaluation.stats[0]), "AP50": float(evaluation.stats[1])}}


class EpochValidation(HookBase):
    def __init__(self, output, period):
        self.period = period
        self.output = output
        self.best_ap = -1.0
        self.best_step = 0

    def state_dict(self):
        return {"best_ap": self.best_ap, "best_step": self.best_step}

    def load_state_dict(self, state_dict):
        self.best_ap = state_dict["best_ap"]
        self.best_step = state_dict["best_step"]

    def before_train(self):
        prefixes = ("backbone.bottom_up.", "roi_heads.box_predictor.0.cls_score.")
        if isinstance(self.trainer, StagedCascadeTrainer):
            prefixes = ("backbone.fpn_lateral", "proposal_generator.", prefixes[1])
        self.updates = UpdateCheck(self.trainer.model, self.output, prefixes)

    def after_step(self):
        step = self.trainer.iter + 1
        if step % self.period:
            return
        self.updates.after_epoch(step)
        cfg, model = self.trainer.cfg, self.trainer.model
        metrics = self.trainer.test(cfg, model)["bbox"]
        ap, ap50 = metrics["AP"], metrics["AP50"]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (ap, ap50)):
            raise RuntimeError(f"Validación Cascade no válida: {metrics}")
        self.trainer.storage.put_scalar("val/map", ap, smoothing_hint=False)
        self.trainer.storage.put_scalar("val/ap50", ap50, smoothing_hint=False)
        if ap > self.best_ap:
            self.best_ap, self.best_step = ap, step


class EpochPublication(HookBase):
    def __init__(self, validation, stop_step):
        self.validation = validation
        self.stop_step = stop_step
        self.best_checkpoint = validation.output / "best" / "model_best.pth"

    def publish_best(self):
        publish_copy(Path(self.trainer.checkpointer.get_checkpoint_file()), self.best_checkpoint)

    def before_train(self):
        if self.trainer.start_iter and self.validation.best_step == self.trainer.start_iter:
            self.publish_best()

    def after_step(self):
        step = self.trainer.iter + 1
        if step == self.validation.best_step:
            self.publish_best()
        if step == self.stop_step:
            raise PauseTraining


class LuminanceTransform(T.Transform):
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def apply_image(self, image):
        return adjust_luminance(image, self.brightness, self.contrast)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation


class CascadePhotometric(T.Augmentation):
    def __init__(self, spec, seed):
        self.sampler = PhotometricSampler(spec, seed)

    def get_transform(self, image):
        gains = self.sampler.sample(get_worker_info())
        return LuminanceTransform(*gains) if gains is not None else T.NoOpTransform()


class CascadeAffine(T.Transform):
    def __init__(self, matrix):
        self.matrix = matrix

    def apply_image(self, image):
        return warp(image, self.matrix)

    def apply_coords(self, coords):
        return transform_coords(coords, self.matrix)


class CascadeGeometryMapper:
    def __init__(self, cfg):
        self.mapper = DatasetMapper(cfg, is_train=True)
        self.resize = self.mapper.augmentations.augs
        if [type(transform).__name__ for transform in self.resize] != ["ResizeShortestEdge"]:
            raise RuntimeError("La geometría requiere ResizeShortestEdge como preprocesado base")
        self.sampler = GeometrySampler(cfg.INPUT.GEOMETRIC, cfg.SEED)

    def __call__(self, record):
        boxes = [BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
                 for ann in record.get("annotations", []) if not ann.get("iscrowd", 0)]
        sample = self.sampler.sample(record["width"], record["height"], boxes, get_worker_info())
        if sample is None:
            return self.mapper(record)
        mapper = copy(self.mapper)
        mapper.augmentations = T.AugmentationList([CascadeAffine(sample[0]), *self.resize])
        return mapper(record)


class CascadeTrainer(DefaultTrainer):
    def __init__(self, cfg, stop_step):
        self.stop_step = stop_step
        super().__init__(cfg)

    def build_hooks(self):
        validation = EpochValidation(Path(self.cfg.OUTPUT_DIR), self.cfg.TEST.EVAL_PERIOD)
        return [
            hooks.IterationTimer(), hooks.LRScheduler(), validation,
            hooks.PeriodicWriter(self.build_writers(), math.gcd(20, self.cfg.TEST.EVAL_PERIOD)),
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=2),
            EpochPublication(validation, self.stop_step),
        ]

    @classmethod
    def build_train_loader(cls, cfg):
        if "GEOMETRIC" in cfg.INPUT:
            return build_detection_train_loader(cfg, mapper=CascadeGeometryMapper(cfg))
        if "PHOTOMETRIC" not in cfg.INPUT:
            return super().build_train_loader(cfg)
        mapper = DatasetMapper(cfg, is_train=True)
        transforms = mapper.augmentations.augs
        if [type(transform).__name__ for transform in transforms] != ["ResizeShortestEdge"]:
            raise RuntimeError("EXP-002 requiere ResizeShortestEdge como única transformación base")
        mapper.augmentations = T.AugmentationList([CascadePhotometric(cfg.INPUT.PHOTOMETRIC, cfg.SEED), *transforms])
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return CascadeEvaluator(dataset_name)


class CascadeFreezeHook(HookBase):
    def before_step(self):
        self.trainer.freeze_audit.before_step(self.trainer)

    def after_step(self):
        self.trainer.freeze_audit.after_step(self.trainer)


class StagedCascadeTrainer(CascadeTrainer):
    def build_hooks(self):
        ordered = super().build_hooks()
        ordered.insert(2, CascadeFreezeHook())
        return ordered


def make_predictor(cfg, model):
    resize = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, sample_style="choice",
    )

    @torch.inference_mode()
    def predict(bgr):
        height, width = bgr.shape[:2]
        resized = resize.get_transform(bgr).apply_image(bgr)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float()
        result = model([{"image": tensor, "height": height, "width": width}])[0]["instances"]
        return as_detections(result)
    return predict


def predictor(spec, manifest, weights, output):
    cfg, _, _ = configure(spec, len(manifest["classes"]), manifest["splits"]["train"]["images"], weights, output)
    model = build_model(cfg)
    CheckedWeights(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    return make_predictor(cfg, model)


def train(args, spec, manifest):
    setup_logger(name="detectron2")
    seed_all_rng(spec["seed"])
    torch.use_deterministic_algorithms(False)
    cfg, iterations, official_url = configure(
        spec, len(manifest["classes"]), manifest["splits"]["train"]["images"], args.weights, args.output,
    )
    if "photometric" in spec:
        cfg.INPUT.PHOTOMETRIC = CfgNode(spec["photometric"])
    if "geometric" in spec:
        cfg.INPUT.GEOMETRIC = CfgNode(spec["geometric"])
    register_training_data(cfg, args.data, manifest)
    (args.output / "best").mkdir(exist_ok=True)
    effective = args.output / "detectron2.yaml"
    if args.resume and (not effective.is_file() or effective.read_text(encoding="utf-8") != cfg.dump()):
        raise ValueError("La configuración efectiva de Detectron2 cambió al reanudar")
    effective.write_text(cfg.dump(), encoding="utf-8")
    freeze = spec.get("freeze_backbone_epochs", 0)
    trainer_type = StagedCascadeTrainer if freeze else CascadeTrainer
    trainer = trainer_type(cfg, (args.pause_after or 0) * iterations)
    if args.resume:
        if not trainer.checkpointer.has_checkpoint() or not Path(trainer.checkpointer.get_checkpoint_file()).is_file():
            raise FileNotFoundError("Falta el checkpoint local indicado por last_checkpoint")
        trainer.resume_or_load(resume=True)
        if trainer.start_iter >= cfg.SOLVER.MAX_ITER:
            raise ValueError("Cascade ya alcanzó el presupuesto de esta ejecución")
        if args.pause_after and args.pause_after * iterations <= trainer.start_iter:
            raise ValueError("El punto de pausa ya se alcanzó en la ejecución anterior")
        trim_metrics(args.output / "metrics.json", trainer.start_iter)
    else:
        weights = CheckedWeights(trainer.model, pretrained=True)
        weights.load(cfg.MODEL.WEIGHTS)
        write_json(args.output / "weight_transfer.json", weights.transfer)

    inventory = parameters(trainer.model)
    if any(p["name"].startswith("backbone.") for p in inventory["frozen"]):
        raise RuntimeError("Hay parámetros del backbone Cascade congelados antes de la intervención")
    if freeze:
        from scripts.cascade_staged import CascadeFreeze

        trainer.freeze_audit = CascadeFreeze(trainer, args.output, freeze, iterations)
        inventory = parameters(trainer.model)
    if trainer.model.roi_heads.num_classes != len(manifest["classes"]) or cfg.MODEL.MASK_ON:
        raise RuntimeError("Cabeza Cascade incorrecta")
    write_json(args.output / "parameters.json", inventory)
    write_json(args.output / "cascade_runtime.json", {
        "official_checkpoint": official_url, "iterations_per_epoch": iterations, "max_iter": cfg.SOLVER.MAX_ITER,
        "effective_batch": cfg.SOLVER.IMS_PER_BATCH, "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        **({"freeze_backbone_epochs": freeze} if freeze else {}),
    })
    if args.initialize:
        from detectron2.data.detection_utils import read_image

        batch = next(iter(trainer.data_loader))
        image = read_image(batch[0]["file_name"], format="BGR")
        model = trainer.model
        model.eval()
        predict = make_predictor(cfg, model)
        result = predict(image)
        if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
            raise RuntimeError("Salida de inferencia Cascade inválida")
        model.train()
        if freeze:
            model.backbone.bottom_up.eval()
        with EventStorage(0):
            losses = model(batch)
        loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise RuntimeError("Pérdida Cascade no finita")
        loss.backward()
        gradients = check_gradients(torch, model)
        backbone = "backbone.fpn_lateral" if freeze else "backbone."
        for prefix in (backbone, "roi_heads.box_predictor.", "proposal_generator."):
            if not any(name.startswith(prefix) and value == "nonzero" for name, value in gradients.items()):
                raise RuntimeError(f"Cascade: faltan gradientes útiles en {prefix}")
        write_json(args.output / "initialization.json", {
            "classes": trainer.model.roi_heads.num_classes,
            "inference_detections": len(result), "losses": {k: float(v.detach().cpu()) for k, v in losses.items()},
            "gradients": gradients,
        })
        return {}

    try:
        trainer.train()
        status = "complete"
    except PauseTraining:
        status = "paused"
    rows = {}
    with (args.output / "metrics.json").open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            step = item.pop("iteration") + 1
            rows.setdefault(step, {"step": step}).update(item)
    best = save_history([rows[key] for key in sorted(rows)], args.output, prefer_last_tie=False)
    if best is None:
        raise RuntimeError("Cascade no registró métricas de validación")
    last = Path(trainer.checkpointer.get_checkpoint_file()).relative_to(args.output)
    return {
        "status": status, "step_unit": "iteration", "best_step": best["step"], "best_validation_map": best["val/map"],
        "last_checkpoint": last.as_posix(), "best_checkpoint": "best/model_best.pth",
    }


def trim_metrics(path, completed_iterations):
    # An interruption can leave logged work after the last restorable checkpoint.
    with path.open("r+", encoding="utf-8") as file:
        valid = [
            line for line in file
            if line.endswith("\n") and json.loads(line)["iteration"] < completed_iterations
        ]
        file.seek(0)
        file.writelines(valid)
        file.truncate()
