"""Checks for EXP-006 configuration and analysis without model execution."""

import csv
import importlib
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.baseline_data import ANNOTATIONS, REPO_ID, REVISION, convert_split, read_json, sha256, write_json
from scripts.baseline_run import REPO, load_config


MODULE = "experiments.exp-006-yolo-warmup.analyze"
analysis = importlib.import_module(MODULE)


class YoloWarmupTest(unittest.TestCase):
    def test_only_warmup_changes_and_expected_end_indices(self):
        baseline = load_config("yolo", None, {})
        batches = (316 + baseline["batch"] - 1) // baseline["batch"]
        self.assertEqual(batches, 40)
        self.assertEqual(baseline["warmup_epochs"], 1.0)
        self.assertEqual(max(round(baseline["warmup_epochs"] * batches), 100), 100)
        for duration, end in ((3, 120), (5, 200)):
            with self.subTest(duration=duration):
                path = REPO / f"configs/experiments/yolo-warmup-{duration}.json"
                spec = load_config("yolo", path, {})
                self.assertEqual(spec, {**baseline, "warmup_epochs": float(duration)})
                self.assertEqual(max(round(spec["warmup_epochs"] * batches), 100), end)

    def prepare(self, root):
        (root / "data").mkdir()
        (root / "data/manifest.json").write_text("{}")
        digest = sha256(root / "data/manifest.json")
        commit, coco, weights = "a" * 40, "b" * 64, "c" * 64
        (root / "source-commit").write_text(f"{commit}\n")
        (root / "inputs.sha256").write_text(
            f"{digest}  data/manifest.json\n{coco}  data/coco/valid.json\n{weights}  yolov10s.pt\n")
        out = root / "eval-a-s2026"
        out.mkdir()
        (out / "eval.job").write_text("3\n")
        (out / "verified.ok").touch()
        hardware = {"gpu": "A100", "cuda_runtime": "12.1", "nvidia_driver": "driver", "host": "node", "job_id": "3"}
        for arm, warmup, job, score, peak in (
            ("control", 1.0, "1", 0.5, 50_317_300), ("warmup-5", 5.0, "2", 0.6, 5),
        ):
            run = root / "s2026" / arm / "train"
            evaluation = out / arm
            run.mkdir(parents=True)
            evaluation.mkdir()
            (run.parent / "train.job").write_text(job)
            (run.parent / "train.path").write_text(str(run))
            (out / f"{arm}.train.sacct").write_text("COMPLETED|0:0\n")
            config = {**load_config("yolo", None, {}), "warmup_epochs": warmup}
            runtime = {
                "args": {"warmup_epochs": warmup, "seed": 2026, "epochs": 30, "batch": 8,
                         "lr0": 0.001, "cos_lr": True, "lrf": 0.05, "amp": False,
                         "warmup_momentum": 0.8, "warmup_bias_lr": 0.001},
                "iterations_per_epoch": 40, "optimizer": "AdamW",
                "warmup_iterations_end_index": max(round(warmup * 40), 100),
            }
            status = {"status": "complete", "best_step": 20, "best_validation_map": score, "best_checkpoint": "weights/best.pt"}
            evidence = {"packages": {}, "git_head": None, "code_sha256": {"scripts/train_baseline.py": "digest"}}
            metadata = {
                **evidence,
                "run": {"backend": "yolo", "config": config, "dataset_manifest_sha256": digest,
                        "initial_weights_sha256": weights},
                "hardware": {**hardware, "job_id": job},
            }
            metrics = {
                **evidence, "backend": "yolo", "split": "valid", "checkpoint_kind": "best", "hardware": hardware,
                "dataset_manifest_sha256": digest, "coco_sha256": coco, "image_ids": list(range(90)),
                "model_config": config, "checkpoint": str(run / "weights/best.pt"),
                "map": score, "ap50": score, "ap75": score,
                "parameters": {"iou_type": "bbox"},
                "per_class": [{"category_id": cid, "name": "Ibuphil 600 mg" if cid == 2 else f"class {cid}",
                               "ground_truth": 1 if cid == 2 else 10, "ap": score} for cid in range(1, 9)],
                "operating_point": {"score": 0.25, "iou": 0.50, "precision": 0.5, "recall": 0.5,
                                    "tp": 10, "fp": 10, "fn": 10},
                "benchmark": {"warmup_iterations": 20, "iterations": 100, "batch": 1, "dtype": "fp32",
                              "nominal_shape": [640, 640], "samples_ms": [10.0] * 100,
                              "mean_ms_per_image": 10.0, "median_ms_per_image": 10.0, "p95_ms_per_image": 10.0,
                              "fps_from_mean": 100.0, "images": ["1.jpg"], "scope": "common"},
            }
            for file, value in (
                (run.parent / "config.json", config), (run / "run.json", metadata),
                (run / "ultralytics_runtime.json", runtime), (run / "status.json", status),
                (evaluation / "metrics.json", metrics),
            ):
                file.write_text(json.dumps(value))
            with (run / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("step", "val/map", *analysis.LOSSES, "lr/pg0"))
                writer.writeheader()
                for epoch in range(1, 31):
                    writer.writerow({
                        "step": epoch, "val/map": score if epoch == 20 else 0.0 if epoch == 2 else 0.25,
                        **dict.fromkeys(analysis.LOSSES, 1.0),
                        "val/cls_loss": peak if epoch == 2 else 1.0, "lr/pg0": 0.001,
                    })

    def test_summarizes_valid_pair_and_rejects_incompatible_benchmark(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            identity = analysis.campaign_identity(root)
            pair, histories = analysis.compare(root, 2026, "warmup-5", "a", identity)
            self.assertEqual(pair["control"]["peak_val_cls_loss_1_5"], 50_317_300)
            self.assertEqual(pair["control"]["zero_map_epochs_1_5"], 1)
            self.assertEqual(pair["control"]["mean_map_1_5"], 0.2)
            self.assertAlmostEqual(pair["delta"]["map"], 0.1)
            self.assertAlmostEqual(pair["delta"]["map_without_ibuphil"], 0.1)
            self.assertEqual(pair["warmup-5"]["best_epoch"], 20)
            self.assertEqual(histories["control"][0]["lr/pg0"], 0.001)
            self.assertNotIn("lr/pg1", histories["control"][0])
            summary = analysis.aggregate({2026: pair}, "warmup-5")
            self.assertEqual(summary["map_wins"], 1)
            self.assertIsNone(summary["delta"]["map"]["sample_sd"])
            second = {**pair, "delta": {**pair["delta"], "map": -0.1}}
            summary = analysis.aggregate({2026: pair, 2027: second}, "warmup-5")
            self.assertAlmostEqual(summary["delta"]["map"]["mean"], 0.0)
            self.assertAlmostEqual(summary["delta"]["map"]["sample_sd"], 2 ** 0.5 / 10)
            metrics = root / "eval-a-s2026/warmup-5/metrics.json"
            data = json.loads(metrics.read_text())
            data["hardware"]["host"] = "other-node"
            metrics.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "protocolo distintos"):
                analysis.compare(root, 2026, "warmup-5", "a", identity)

    def test_cli_and_invalid_history(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            output = root / "output"
            cmd = [sys.executable, "-m", MODULE, "--source", str(root), "--output", str(output),
                   "--variant", "warmup-5", "--pair", "2026", "a"]
            if find_spec("matplotlib"):
                cmd.append("--plots")
            result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "summary.json").read_text())
            self.assertIsNone(report["confirmation_only"])
            self.assertFalse(report["coco_verified"])
            if "--plots" in cmd:
                from PIL import Image

                with Image.open(output / "history-s2026.png") as figure:
                    figure.verify()
            result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            history = root / "s2026/control/train/history.csv"
            history.write_text(history.read_text().replace("50317300", "nan"))
            with self.assertRaisesRegex(ValueError, "valores inválidos"):
                analysis.read_history(history)

    @unittest.skipUnless(find_spec("pycocotools"), "pycocotools no disponible")
    def test_coco_replay_detects_changed_evidence(self):
        from PIL import Image

        from scripts.baseline_metrics import coco_eval, report

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            dataset = root / "original"
            original = dataset / "valid" / ANNOTATIONS
            original.parent.mkdir(parents=True)
            categories = [{"id": cid, "name": "Ibuphil 600 mg" if cid == 2 else f"class {cid}"} for cid in range(1, 9)]
            classes = [{"original_id": row["id"], "internal_id": index, "name": row["name"]}
                       for index, row in enumerate(categories)]
            boxes = [{"image_id": 0, "category_id": row["id"], "bbox": [2, 2, 6, 6]} for row in categories]
            raw = {"images": [{"id": 0, "file_name": "image.jpg", "width": 10, "height": 10}],
                   "categories": categories,
                   "annotations": [{**box, "id": idx, "iscrowd": 0, "area": 36} for idx, box in enumerate(boxes)]}
            write_json(original, raw)
            truth, _ = convert_split(raw, classes)
            prepared = root / "valid.json"
            write_json(prepared, truth)
            coco = sha256(prepared)
            manifest = root / "data/manifest.json"
            write_json(manifest, {
                "dataset": REPO_ID, "revision": REVISION, "classes": classes,
                "splits": {"valid": {"source_annotations_sha256": sha256(original)}},
                "files": {"coco/valid.json": coco},
            })
            digest = sha256(manifest)
            (root / "inputs.sha256").write_text(
                f"{digest}  data/manifest.json\n{coco}  data/coco/valid.json\n{'c' * 64}  yolov10s.pt\n")
            predictions = [{**box, "score": 0.9} for box in boxes]
            metrics, curves, _ = report(coco_eval(truth, predictions))
            for arm in ("control", "warmup-5"):
                run_path = root / f"s2026/{arm}/train/run.json"
                run = read_json(run_path)
                run["run"]["dataset_manifest_sha256"] = digest
                write_json(run_path, run)
                evaluation = root / f"eval-a-s2026/{arm}"
                write_json(evaluation / "metrics.json", {
                    **read_json(evaluation / "metrics.json"), **metrics, "image_ids": [0],
                    "dataset_manifest_sha256": digest, "coco_sha256": coco,
                })
                write_json(evaluation / "predictions.json", predictions)
                write_json(evaluation / "pr_curves.json", curves)
                Image.new("RGB", (1, 1)).save(evaluation / "pr_curves.png")
            output = root / "verified"
            cmd = [sys.executable, "-m", MODULE, "--source", str(root), "--data", str(dataset),
                   "--output", str(output), "--variant", "warmup-5", "--pair", "2026", "a"]
            verified = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertTrue(read_json(output / "summary.json")["coco_verified"])

            identity = analysis.campaign_identity(root)
            evaluation = root / "eval-a-s2026/control"
            for filename, change, message in (
                ("metrics.json", {"map": 0}, "map no coincide"),
                ("metrics.json", {"per_class": []}, "per_class no coincide"),
                ("pr_curves.json", {"recall": []}, "curvas PR incoherentes"),
            ):
                with self.subTest(message=message):
                    path = evaluation / filename
                    saved = read_json(path)
                    write_json(path, {**saved, **change})
                    with self.assertRaisesRegex(ValueError, message):
                        analysis.read_arm(root, 2026, "control", "a", identity, truth)
                    write_json(path, saved)
            with self.assertRaisesRegex(ValueError, "COCO de validación reconstruido"):
                analysis.validation_truth(dataset, root, {**identity[1], "valid.json": "0" * 64})
            original.write_text("{}")
            with self.assertRaisesRegex(ValueError, "anotaciones originales"):
                analysis.validation_truth(dataset, root, identity[1])
