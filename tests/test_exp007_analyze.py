"""Regression tests for checking EXP-007 evidence from multiple SLURM jobs."""

import csv
from contextlib import redirect_stderr
from importlib import import_module
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.baseline_data import ANNOTATIONS, REPO_ID, REVISION, convert_split, read_json, sha256, write_json
from scripts.baseline_metrics import coco_eval, report
from scripts.baseline_run import REPO, load_config


analysis = import_module("experiments.exp-007-yolo-staged-transfer.analyze")


@unittest.skipUnless(find_spec("pycocotools"), "requiere pycocotools para verificar los artefactos COCO")
class AnalyzeCampaignTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "snapshot"
        self.exp, self.confirm = self.root / "exploration", self.root / "confirmation"
        self.dataset = self.root / "original"
        self.commit, self.weights_sha = "a" * 40, "b" * 64
        self.hardware = {"gpu": "NVIDIA A100-PCIE-40GB", "cuda_runtime": "12.1", "nvidia_driver": "580.178.04"}
        self.launch_hashes = {}
        for name in ("evaluate.sh", "evaluate.py"):
            path = self.repo / "slurm" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO / "slurm" / name).read_bytes())
            self.launch_hashes[name] = sha256(path)
        for arm in ("freeze-5", "freeze-10", "freeze-15"):
            relative = f"configs/experiments/yolo-{arm}.json"
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / relative).read_bytes())
        script = self.repo / "scripts/evaluate_baseline.py"
        script.parent.mkdir()
        script.write_bytes((REPO / "scripts/evaluate_baseline.py").read_bytes())
        self.code = {"scripts/evaluate_baseline.py": sha256(script)}

        classes = [{"original_id": cid, "internal_id": cid - 1,
                    "name": "Ibuphil 600 mg" if cid == 2 else f"class {cid}"} for cid in range(1, 9)]
        boxes = [{"image_id": 57, "category_id": c["original_id"], "bbox": [1, 1, 8, 8]} for c in classes]
        raw = {
            "images": [{"id": 57, "file_name": "57.jpg", "width": 12, "height": 12}],
            "categories": [{"id": c["original_id"], "name": c["name"]} for c in classes],
            "annotations": [{**box, "area": 64, "iscrowd": 0, "id": n} for n, box in enumerate(boxes)],
        }
        annotations = self.dataset / "valid" / ANNOTATIONS
        annotations.parent.mkdir(parents=True)
        write_json(annotations, raw)
        truth, _ = convert_split(raw, classes)
        coco = self.root / "valid.json"
        write_json(coco, truth)
        self.coco_sha = sha256(coco)
        manifest = {"dataset": REPO_ID, "revision": REVISION, "classes": classes,
                    "splits": {"valid": {"source_annotations_sha256": sha256(annotations)}},
                    "files": {"coco/valid.json": self.coco_sha}}
        for source in (self.exp, self.confirm):
            (source / "data").mkdir(parents=True)
            write_json(source / "data/manifest.json", manifest)
            self.manifest_sha = sha256(source / "data/manifest.json")
            (source / "source-commit").write_text(f"{self.commit}\n")
            (source / "inputs.sha256").write_text(
                f"{self.manifest_sha}  manifest.json\n{self.coco_sha}  valid.json\n{self.weights_sha}  yolov10s.pt\n"
            )
        baseline = self.repo / "experiments/baseline/results/yolo/training/run.json"
        baseline.parent.mkdir(parents=True)
        write_json(baseline, {"run": {"initial_weights_sha256": self.weights_sha,
                                      "dataset_manifest_sha256": self.manifest_sha}})
        self.coco_outputs = {}
        for arm, count in (("control", 6), ("freeze-5", 8), ("freeze-10", 6), ("freeze-15", 6)):
            predictions = [{**box, "score": 0.9} for box in boxes[:count]]
            metrics, curves, _ = report(coco_eval(truth, predictions))
            self.coco_outputs[arm] = metrics, curves, predictions
        self.create_cohort(self.exp, "a", 2026, tuple(analysis.ARMS))
        for seed in (2027, 2028):
            self.create_cohort(self.confirm, "b", seed, ("control", "freeze-5"))

    def create_cohort(self, source, phase, seed, arms):
        evaluation = source / f"eval-{phase}-s{seed}"
        evaluation.mkdir()
        job = str(seed * 10)
        (evaluation / "eval.job").write_text(f"{job}\n")
        (evaluation / "verified.ok").touch()
        entries = {}
        for index, arm in enumerate(arms):
            slot = source / f"s{seed}" / arm
            training = slot / "train"
            training.mkdir(parents=True)
            output = evaluation / arm
            (output / "examples").mkdir(parents=True)
            for ident in (57, 61, 80):
                (output / "examples" / f"image-{ident}.jpg").touch()
            remote_root = "/cesga//not-mounted/campaign"
            remote = f"{remote_root}/s{seed}/{arm}/train"
            normalized = str(Path(remote))
            train_job = str(seed * 10 + index + 1)
            (slot / "train.path").write_text(f"{remote}\n")
            (slot / "train.job").write_text(f"{train_job}\n")
            (evaluation / f"{arm}.train.sacct").write_text("COMPLETED|0:0\n")
            config_path = None if arm == "control" else self.repo / f"configs/experiments/yolo-{arm}.json"
            config = load_config("yolo", config_path, {"seed": seed})
            write_json(slot / "config.json", config)
            write_json(training / "run.json", {
                "run": {"backend": "yolo", "config": config, "dataset_manifest_sha256": self.manifest_sha,
                        "initial_weights_sha256": self.weights_sha},
                "mode": "train", "git_head": self.commit, "code_sha256": self.code,
                "packages": {"ultralytics": "8.3.161"},
                "hardware": {**self.hardware, "job_id": train_job, "host": f"training-{train_job}"},
            })
            metrics, curves, predictions = self.coco_outputs[arm]
            write_json(training / "status.json", {
                "status": "complete", "sessions": [{"job_id": train_job}], "best_checkpoint": "weights//best.pt",
                "best_step": 1, "best_validation_map": metrics["map"],
                "total_elapsed_seconds": 10, "cuda_peak_allocated_bytes": 10,
            })
            runtime_path = remote_root if index else str(Path(remote_root))
            write_json(training / "ultralytics_runtime.json", {
                "args": dict(seed=seed, optimizer="AdamW", epochs=30, imgsz=640, batch=8, amp=False,
                             cos_lr=True, lr0=0.001, lrf=0.05, augment=False, warmup_epochs=1,
                             warmup_momentum=0.8, warmup_bias_lr=0.001, momentum=0.9, weight_decay=0.0005,
                             data=f"{runtime_path}/data/dataset.yaml", model=f"{runtime_path}/yolov10s.pt"),
                "iterations_per_epoch": 40, "effective_batch": 8, "nominal_max_iterations": 1200,
                "warmup_iterations_end_index": 100, "optimizer_initial_groups": [],
            })
            changed = {"backbone": True, "head": True}
            write_json(training / "parameter_updates.json", [{"step": 1, "changed": changed}])
            write_json(training / "parameters.json", {"frozen": [{"name": analysis.DFL, "numel": 16}], "trainable": []})
            if arm != "control":
                write_json(training / "staged_transfer.json", {})
            write_json(training / "weight_transfer.json",
                       {"reinitialized": [f"model.23.{branch}.{i}" for branch in ("cv3", "one2one_cv3") for i in range(6)]})
            fields = ("step", "val/map", *analysis.common.LOSSES, *analysis.common.LR)
            with (training / "history.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for step in range(1, 31):
                    writer.writerow((step, metrics["map"], *([1.0] * 6), *([0.001] * 3)))
            checkpoint_sha = "c" * 64
            write_json(output / "metrics.json", {
                **metrics, "backend": "yolo", "model_config": config, "git_head": self.commit,
                "code_sha256": self.code, "packages": {"pycocotools": "2.0.8"},
                "hardware": {**self.hardware, "host": f"evaluation-{job}", "job_id": job},
                "split": "valid", "image_ids": [57], "dataset_manifest_sha256": self.manifest_sha,
                "coco_sha256": self.coco_sha, "checkpoint": f"{remote}/weights/best.pt",
                "checkpoint_sha256": checkpoint_sha, "checkpoint_kind": "best",
                "benchmark": dict(batch=1, nominal_shape=[640, 640], dtype="fp32", warmup_iterations=20,
                                  iterations=100, images=["57.jpg"], scope="common", samples_ms=[10] * 100,
                                  mean_ms_per_image=10, median_ms_per_image=10, p95_ms_per_image=10,
                                  fps_from_mean=100),
            })
            write_json(output / "predictions.json", predictions)
            write_json(output / "pr_curves.json", curves)
            Image.new("RGB", (1, 1)).save(output / "pr_curves.png")
            entries[arm] = {"run": normalized, "backend": "yolo", "training_job_id": train_job,
                            "training_slurm": "COMPLETED|0:0", "checkpoint": f"{normalized}/weights/best.pt",
                            "checkpoint_sha256": checkpoint_sha}
        write_json(evaluation / "launch.json", {
            "runs": entries, "job_id": job, "dataset_manifest_sha256": self.manifest_sha,
            "coco_sha256": self.coco_sha, "split": "valid", "launcher_sha256": self.launch_hashes,
            "benchmark": True, "warmup": 20, "iterations": 100, "checkpoint_kind": "best",
            "example_image_ids": [57, 61, 80],
        })

    def analyze(self, *, confirmation=False):
        output = self.root / "output"
        args = ["analyze", "--source", str(self.exp), "--data", str(self.dataset), "--output", str(output)]
        if confirmation:
            args.extend(("--confirmation", str(self.confirm)))
        # Keep the actual COCO, provenance and comparison checks in the CLI path. Freezing has separate tests.
        with (patch.object(analysis, "REPO", self.repo), patch.object(sys, "argv", args),
              patch.object(analysis, "verify_stages"), patch.object(analysis, "plot")):
            analysis.main()
        return read_json(output / "summary.json")

    def assert_rejected(self, message, *, confirmation=False):
        with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit) as failure:
            self.analyze(confirmation=confirmation)
        self.assertEqual(failure.exception.code, 1)
        self.assertIn(message, errors.getvalue())
        self.assertFalse((self.root / "output").exists())

    def test_equivalent_paths_and_different_training_nodes_pass_full_analysis(self):
        resolve = Path.resolve

        def forbid_remote_resolution(path, *args, **kwargs):
            if path.is_relative_to("/cesga"):
                raise AssertionError("se intentó resolver una ruta remota en local")
            return resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", forbid_remote_resolution):
            result = self.analyze(confirmation=True)
        self.assertTrue(result["coco_verified"])
        self.assertEqual(result["exploration"]["selected"], "freeze-5")
        self.assertEqual(set(result["confirmation"]["pairs"]), {"2027", "2028"})

    def test_distinct_remote_run_or_checkpoint_fails_full_analysis(self):
        evaluation = self.exp / "eval-a-s2026"
        for filename, arm, key in (
            ("launch.json", "control", "run"), ("launch.json", "control", "checkpoint"),
            ("control/metrics.json", None, "checkpoint"),
        ):
            with self.subTest(filename=filename, key=key):
                path = evaluation / filename
                saved = path.read_bytes()
                data = read_json(path)
                record = data["runs"][arm] if arm else data
                record[key] += "-different"
                write_json(path, data)
                try:
                    self.assert_rejected("procedencia de datos, checkpoint o código incompatible")
                finally:
                    path.write_bytes(saved)

    def test_training_hardware_mismatches_fail_full_analysis(self):
        path = self.exp / "s2026/freeze-5/train/run.json"
        for key in ("gpu", "cuda_runtime", "nvidia_driver"):
            with self.subTest(key=key):
                saved = path.read_bytes()
                data = read_json(path)
                data["hardware"][key] = "different"
                write_json(path, data)
                try:
                    self.assert_rejected("Semilla 2026, freeze-5: entorno")
                finally:
                    path.write_bytes(saved)

    def test_consistent_within_seed_but_different_between_phases_fails(self):
        for role, group, key in (
            ("train", "packages", "ultralytics"), ("train", "hardware", "gpu"),
            ("evaluation", "packages", "pycocotools"), ("evaluation", "hardware", "nvidia_driver"),
        ):
            with self.subTest(role=role, key=key):
                saved = {}
                for arm in ("control", "freeze-5"):
                    relative = (f"s2028/{arm}/train/run.json" if role == "train"
                                else f"eval-b-s2028/{arm}/metrics.json")
                    path = self.confirm / relative
                    saved[path] = path.read_bytes()
                    data = read_json(path)
                    data[group][key] = "different"
                    write_json(path, data)
                try:
                    self.assert_rejected("Semilla 2028: entorno de confirmación distinto al de exploración",
                                         confirmation=True)
                finally:
                    for path, payload in saved.items():
                        path.write_bytes(payload)
