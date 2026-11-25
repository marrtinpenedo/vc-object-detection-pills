"""CPU tests of the staged transfer policy and its audit."""

import csv
import json
import os
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.baseline_data import read_json, sha256, write_json
from scripts.baseline_run import REPO, load_config


class StagedConfigTest(unittest.TestCase):
    def test_variants_change_only_freeze_duration(self):
        baseline = load_config("yolo", None, {})
        self.assertNotIn("freeze_backbone_epochs", baseline)
        for freeze in (5, 10, 15, 20, 25, 30):
            config = REPO / f"configs/experiments/yolo-freeze-{freeze}.json"
            self.assertEqual(load_config("yolo", config, {}), {**baseline, "freeze_backbone_epochs": freeze})
        for name in ("yolo-lr-0005", "yolo-photometric", "yolo-warmup-3", "yolo-warmup-5"):
            config = REPO / f"configs/experiments/{name}.json"
            self.assertEqual(load_config("yolo", config, {}), read_json(config))

    def test_validates_type_range_backend_and_overrides(self):
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "cfg.json"
            yolo = load_config("yolo", None, {})
            for invalid in (True, 2.0, "2", None, -1, 31):
                with self.subTest(invalid=invalid):
                    config.write_text(json.dumps({**yolo, "freeze_backbone_epochs": invalid}))
                    with self.assertRaises(ValueError):
                        load_config("yolo", config, {})
            config.write_text(json.dumps({**yolo, "freeze_backbone_epochs": 0}))
            self.assertEqual(load_config("yolo", config, {}).get("freeze_backbone_epochs"), 0)
            config.write_text(json.dumps({**yolo, "freeze_backbone_epochs": 5}))
            self.assertEqual(load_config("yolo", config, {"epochs": 5})["freeze_backbone_epochs"], 5)
            with self.assertRaises(ValueError):
                load_config("yolo", config, {"epochs": 4})
            config.write_text(json.dumps({**load_config("cascade", None, {}), "freeze_backbone_epochs": 0}))
            with self.assertRaises(ValueError):
                load_config("cascade", config, {})

    def test_staged_cli_rejects_pause_and_resume_before_using_data(self):
        config = REPO / "configs/experiments/yolo-freeze-5.json"
        for options in (["--pause-after", "1"], ["--resume"]):
            result = subprocess.run(
                [sys.executable, "-m", "scripts.train_baseline", "yolo", "--config", str(config), *options],
                cwd=REPO, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("ejecución continua", result.stderr)

    def test_fixed_selection_margins_and_confirmation_requirements(self):
        analysis = import_module("experiments.exp-007-yolo-staged-transfer.analyze")
        control = {"map": 0.70, "map_without_ibuphil": 0.65, "recall": 0.67}
        results = {
            "control": control,
            "freeze-5": {"map": 0.70, "map_without_ibuphil": 0.67, "recall": 0.65},
            "freeze-10": {"map": 0.71, "map_without_ibuphil": 0.66, "recall": 0.64},
            "freeze-15": {"map": 0.70, "map_without_ibuphil": 0.67, "recall": 0.65},
        }
        self.assertEqual(analysis.select(results), "freeze-5")
        results["freeze-10"]["recall"] = 0.65
        self.assertEqual(analysis.select(results), "freeze-10")
        records = [analysis.deltas(control, results[arm]) for arm in ("freeze-10", "freeze-10", "freeze-5")]
        pairs = {seed: {"control": control, "freeze-10": results["freeze-10"], "delta": delta}
                 for seed, delta in zip((2026, 2027, 2028), records)}
        total = analysis.common.aggregate(pairs, "freeze-10")
        confirm = analysis.common.aggregate({seed: pairs[seed] for seed in (2027, 2028)}, "freeze-10")
        self.assertTrue(analysis.adoption(total, confirm))
        confirm["delta"]["map"]["mean"] = 0
        self.assertFalse(analysis.adoption(total, confirm))


@unittest.skipUnless(find_spec("torch") and find_spec("ultralytics"), "faltan PyTorch o Ultralytics")
class StagedCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import ultralytics

        if ultralytics.__version__ != "8.3.161":
            raise unittest.SkipTest("se requiere Ultralytics 8.3.161")
        cls.torch = torch
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def model(self):
        from ultralytics.nn.tasks import DetectionModel

        self.torch.manual_seed(2026)
        return DetectionModel("yolov10s.yaml", nc=8, verbose=False)

    def test_layout_and_native_branch_gradient_routes(self):
        from scripts.yolo_staged import layout

        model = self.model()
        modules = layout(model)
        self.assertEqual([row["index"] for row in modules if row["part"] == "backbone"], list(range(11)))
        self.assertEqual([row["index"] for row in modules if row["part"] == "neck"], list(range(11, 23)))
        self.assertEqual(modules[-1], {"index": 23, "type": "v10Detect", "part": "detector"})
        batch = self.torch.rand(2, 3, 64, 64)
        outputs = model(batch)
        sum(x.mean() for x in outputs["one2one"]).backward()
        grads = dict(model.named_parameters())
        self.assertIsNone(grads["model.0.conv.weight"].grad)
        self.assertTrue(any(p.grad is not None for name, p in grads.items() if name.startswith("model.23.one2one_cv3.")))
        model.zero_grad(set_to_none=True)
        sum(x.mean() for x in model(batch)["one2many"]).backward()
        self.assertIsNotNone(grads["model.0.conv.weight"].grad)
        model.model[-1].f = [15, 19, 22]
        with self.assertRaisesRegex(RuntimeError, "conexiones inesperadas"):
            layout(model)

    @unittest.skipUnless(os.environ.get("EXP007_PRETRAINED_WEIGHTS"), "checkpoint COCO local no indicado")
    def test_project_checkpoint_and_adapted_eight_class_model(self):
        from ultralytics.nn.tasks import attempt_load_one_weight

        from scripts.baseline_yolo import StagedYoloTrainer
        from scripts.yolo_staged import layout

        path = Path(os.environ["EXP007_PRETRAINED_WEIGHTS"])
        self.assertEqual(sha256(path), "65fa8332c38a7189597a268897bce6c1026f9499711b9761fb0c2c639b91d4d6")
        weights, _ = attempt_load_one_weight(str(path), device=self.torch.device("cpu"))
        with TemporaryDirectory() as temporary:
            trainer = StagedYoloTrainer.__new__(StagedYoloTrainer)
            trainer.classes, trainer.data = list(range(8)), {"channels": 3, "nc": 8}
            trainer.resume, trainer.save_dir = False, Path(temporary)
            model = trainer.get_model(weights.yaml, weights, verbose=False)
            self.assertEqual(layout(model), layout(weights))
            self.assertEqual(model.model[-1].nc, 8)
            self.assertEqual(sum(p.numel() for p in model.model[:11].parameters()), 3_888_800)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 8_072_544)
            self.assertEqual(len(read_json(trainer.save_dir / "weight_transfer.json")["reinitialized"]), 12)

    @unittest.skipUnless(os.environ.get("EXP007_PRETRAINED_WEIGHTS") and find_spec("pycocotools"),
                         "requiere checkpoint COCO local y pycocotools")
    def test_native_three_epoch_smoke_with_synthetic_images(self):
        from PIL import Image, ImageDraw

        from scripts.baseline_yolo import train

        weights = Path(os.environ["EXP007_PRETRAINED_WEIGHTS"])
        self.assertEqual(sha256(weights), "65fa8332c38a7189597a268897bce6c1026f9499711b9761fb0c2c639b91d4d6")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "coco").mkdir(parents=True)
            classes = [{"internal_id": i, "original_id": i + 1, "name": f"c{i + 1}"} for i in range(8)]
            categories = [{"id": row["original_id"], "name": row["name"]} for row in classes]
            manifest = {"classes": classes, "splits": {}}
            for split in ("train", "valid"):
                images, annotations = [], []
                for i in range(4):
                    image = Image.new("RGB", (64, 64), (i * 40 + 10, 80, 140))
                    ImageDraw.Draw(image).rectangle((12, 20, 44, 48), fill=(200, 30, i * 28))
                    file = data / f"images/{split}/{i}.jpg"
                    label = data / f"labels/{split}/{i}.txt"
                    file.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    image.save(file)
                    label.write_text(f"{i} 0.4375 0.53125 0.5 0.4375\n")
                    images.append({"id": i, "file_name": file.name, "width": 64, "height": 64})
                    annotations.append({"id": i + 1, "image_id": i, "category_id": i + 1,
                                        "bbox": [12, 20, 32, 28], "area": 896, "iscrowd": 0})
                write_json(data / f"coco/{split}.json",
                           {"images": images, "annotations": annotations, "categories": categories})
                manifest["splits"][split] = {
                    "images": 4, "instances": 4, "ground_truth_by_category": {str(i + 1): int(i < 4) for i in range(8)},
                }
            write_json(data / "dataset.yaml", {"path": str(data), "train": "images/train", "val": "images/valid",
                                              "names": [row["name"] for row in classes]})
            analysis = import_module("experiments.exp-007-yolo-staged-transfer.analyze")
            for duration, thaw, frozen in ((1, 2, [True, False, False]), (3, None, [True, True, True])):
                with self.subTest(duration=duration):
                    args = SimpleNamespace(weights=weights, data=data, output=root / f"smoke-{duration}",
                                           resume=False, initialize=False, pause_after=None)
                    spec = {**load_config("yolo", None, {"epochs": 3, "batch": 2, "imgsz": 64, "workers": 0}),
                            "freeze_backbone_epochs": duration}
                    status = train(args, spec, manifest)
                    audit = read_json(args.output / "staged_transfer.json")
                    self.assertEqual(status["status"], "complete")
                    self.assertTrue((args.output / status["best_checkpoint"]).is_file())
                    self.assertEqual(audit["thaw_epoch"], thaw)
                    self.assertEqual([item["backbone_frozen"] for item in audit["epochs"]], frozen)
                    self.assertEqual([item["ema_updates"] for item in audit["epochs"]], [2, 4, 6])
                    with (args.output / "history.csv").open(newline="") as stream:
                        history = [{key: float(row[key]) for key in ("step", *analysis.common.LR)}
                                   for row in csv.DictReader(stream)]
                    analysis.verify_stages(audit, read_json(args.output / "parameters.json"),
                                           read_json(args.output / "ultralytics_runtime.json"), history, duration)

    def test_native_optimizer_ema_and_training_modes_across_thaw(self):
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.torch_utils import ModelEMA

        from scripts.baseline_yolo import StagedYoloTrainer, YoloTrainer, record_runtime
        from scripts.yolo_staged import StagedBackbone, byte_digest

        torch = self.torch
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            trainer = StagedYoloTrainer.__new__(StagedYoloTrainer)
            trainer.model, trainer.freeze_layer_names = self.model(), [".dfl"]
            trainer.accumulate, trainer.train_loader = 1, [0, 1]
            trainer.epochs, trainer.batch_size, trainer.photometric = 4, 2, None
            trainer.args = get_cfg(overrides={"warmup_epochs": 1.0, "cos_lr": True, "lrf": 0.05})
            trainer.model.args = trainer.args
            trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=0.001, momentum=0.9, decay=0.0005)
            trainer._setup_scheduler()
            trainer.scheduler.last_epoch = -1
            trainer.ema = ModelEMA(trainer.model)
            trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
            args = SimpleNamespace(output=out)
            # A control follows the original inventory and Batch Normalization modes.
            YoloTrainer._model_train(trainer)
            record_runtime(trainer, args)
            self.assertEqual([row["name"] for row in read_json(out / "parameters.json")["frozen"]],
                             ["model.23.dfl.conv.weight"])
            self.assertNotIn("freeze_backbone_epochs", read_json(out / "ultralytics_runtime.json"))
            trainer.staged = stage = StagedBackbone(trainer, out, 2)
            record_runtime(trainer, args)
            self.assertEqual(sum(row["numel"] for row in read_json(out / "parameters.json")["frozen"]), 3_888_816)
            model_before = {name: value.detach().clone() for name, value in stage.tensors}
            optimizer, scheduler, ema = trainer.optimizer, trainer.scheduler, trainer.ema
            batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0, 1]),
                     "cls": torch.tensor([[0.0], [3.0]]),
                     "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.5, 0.5, 0.4, 0.3]])}
            # Native loop sets LR, then mode, then forward/backward and optimizer_step.
            import warnings

            for epoch in range(4):
                trainer.epoch = epoch
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    trainer.scheduler.step()
                trainer._model_train()
                self.assertEqual(all(not bn.training for bn in stage.bn.values()), epoch < 2)
                for _ in trainer.train_loader:
                    loss, _ = trainer.model(batch.copy())
                    loss.sum().backward()
                    self.assertEqual(all(p.grad is None for p in stage.backbone.values()), epoch < 2)
                    trainer.optimizer_step()
                stage.end_epoch(trainer, epoch + 1)
                if epoch < 2:
                    self.assertTrue(all(torch.equal(value, model_before[name]) for name, value in stage.tensors))
                    self.assertFalse(any(p in optimizer.state for p in stage.backbone.values()))
            self.assertIs(trainer.optimizer, optimizer)
            self.assertIs(trainer.scheduler, scheduler)
            self.assertIs(trainer.ema, ema)
            evidence = read_json(out / "staged_transfer.json")
            self.assertEqual(evidence["thaw_epoch"], 3)
            self.assertEqual([row["backbone_frozen"] for row in evidence["epochs"]], [True, True, False, False])
            self.assertEqual([row["ema_updates"] for row in evidence["epochs"]], [2, 4, 6, 8])
            self.assertEqual(evidence["epochs"][2]["adamw_steps"], {"backbone": 2, "neck": 6, "one2many": 6, "one2one": 6})
            self.assertNotEqual(byte_digest(stage.tensors), evidence["initial_backbone_sha256"])
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in stage.internal.values()))
            analysis = import_module("experiments.exp-007-yolo-staged-transfer.analyze")
            history = [{"step": item["epoch"], **dict(zip(analysis.common.LR, item["lr"]))}
                       for item in evidence["epochs"]]
            inventory, runtime = [read_json(out / name) for name in ("parameters.json", "ultralytics_runtime.json")]
            analysis.verify_stages(evidence, inventory, runtime, history, 2)
            for field, value in (
                ("backbone_sha256", "a" * 64), ("ema_updates", 0), ("backbone_bn_training", True),
            ):
                broken = deepcopy(evidence)
                broken["epochs"][0][field] = value
                with self.assertRaises(ValueError):
                    analysis.verify_stages(broken, inventory, runtime, history, 2)
