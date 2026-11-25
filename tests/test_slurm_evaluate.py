"""CPU integration tests for the shared SLURM launcher."""

import importlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from importlib.util import find_spec

from PIL import Image

from scripts.baseline_data import ACTIVE_IDS, REPO_ID, REVISION, read_json, sha256, write_json
from scripts.baseline_metrics import coco_eval, report
from scripts.baseline_run import REPO, load_config


FAKE_EVALUATOR = """\
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

token = os.environ.get("EVAL_TEST_FAILURE")
flags = [arg for arg in sys.argv[1:] if arg != "--benchmark"]
opts = dict(zip(flags[::2], flags[1::2]))
run, output, data = (Path(opts[key]) for key in ("--run", "--output", "--data"))
with open(os.environ["EVAL_TEST_LOG"], "a") as log:
    log.write(json.dumps({
        "backend": os.environ["EVAL_TEST_BACKEND"], "output": output.name, "opts": opts,
        "settings": os.environ.get("YOLO_CONFIG_DIR"), "cwd": os.getcwd(),
    }) + "\\n")
if token == "evaluate" and output.name == "variant":
    sys.exit(23)

def read(path):
    return json.loads(path.read_text())

fixture = Path(os.environ["EVAL_TEST_FIXTURE"])
status, metadata = read(run / "status.json"), read(run / "run.json")
checkpoint = run / status[opts["--checkpoint"] + "_checkpoint"]
coco = data / "coco" / (opts["--split"] + ".json")
metrics = read(fixture / "summary.json")
metrics.update(
    backend=os.environ["EVAL_TEST_BACKEND"], checkpoint_kind=opts["--checkpoint"], checkpoint=str(checkpoint),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    split=opts["--split"], coco_sha256=hashlib.sha256(coco.read_bytes()).hexdigest(),
    dataset_manifest_sha256=metadata["run"]["dataset_manifest_sha256"],
    image_ids=[row["id"] for row in read(coco)["images"]], model_config=metadata["run"]["config"],
    git_head=metadata["git_head"], code_sha256=metadata["code_sha256"], packages={},
    hardware=dict(job_id=os.environ["SLURM_JOB_ID"], gpu="A100", host="node", compute_capability=[8, 0],
                  nvidia_driver="driver", cuda_runtime="12.1"),
    benchmark=None,
)
if "--benchmark" in sys.argv:
    count = int(opts["--iterations"])
    size = metadata["run"]["config"]["imgsz"]
    metrics["benchmark"] = dict(batch=1, nominal_shape=[size, size], dtype="fp32",
                                warmup_iterations=int(opts["--warmup"]), iterations=count,
                                images=[row["file_name"] for row in read(coco)["images"]],
                                samples_ms=[10] * count, mean_ms_per_image=10, fps_from_mean=100,
                                median_ms_per_image=10, p95_ms_per_image=10, scope="common")
if token == "checkpoint":
    metrics["checkpoint_sha256"] = "wrong"
if token == "benchmark" and output.name == "variant":
    metrics["hardware"]["host"] = "other"
output.mkdir()
(output / "metrics.json").write_text(json.dumps(metrics))
for name in ("predictions.json", "pr_curves.json", "pr_curves.png"):
    shutil.copyfile(fixture / name, output / name)
if token == "verify" and output.name == "variant":
    (output / "pr_curves.json").write_text("{}")
"""


@unittest.skipUnless(find_spec("pycocotools"), "pycocotools no disponible")
class EvaluationLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        (self.data / "coco").mkdir(parents=True)
        categories = [{"id": cid, "name": "Ibuphil 600 mg" if cid == 2 else f"class {cid}"} for cid in ACTIVE_IDS]
        imgs = [{"id": i, "file_name": f"{i}.jpg", "width": 12, "height": 12} for i in (57, 61, 80)]
        annotations = [dict(id=cid, image_id=57, category_id=cid, bbox=[2, 2, 8, 8], area=64, iscrowd=0)
                       for cid in ACTIVE_IDS]
        self.truth = {"images": imgs, "categories": categories, "annotations": annotations}
        for split in ("valid", "test"):
            write_json(self.data / f"coco/{split}.json", self.truth)
            for image in imgs:
                filename = self.data / "images" / split / image["file_name"]
                filename.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 12)).save(filename)
        write_json(self.data / "manifest.json", {
            "dataset": REPO_ID, "revision": REVISION,
            "classes": [{"original_id": row["id"], "internal_id": i, "name": row["name"]}
                        for i, row in enumerate(categories)],
            "files": {path.relative_to(self.data).as_posix(): sha256(path)
                      for path in self.data.rglob("*") if path.is_file()},
        })
        self.digest = sha256(self.data / "manifest.json")
        predictions = [{**row, "score": 0.9} for row in annotations]
        scores, curves, _ = report(coco_eval(self.truth, predictions))
        mock = self.root / "mock.py"
        mock.write_text(FAKE_EVALUATOR)
        for file, value in (("summary.json", scores), ("predictions.json", predictions), ("pr_curves.json", curves)):
            write_json(self.root / file, value)
        Image.new("RGB", (1, 1)).save(self.root / "pr_curves.png")
        self.log = self.root / "calls.log"
        commands = self.root / "commands"
        commands.mkdir()
        for name, code in (("module", "exit 0"), ("sacct", "printf '%s\\n' \"${TEST_SLURM_STATE:-COMPLETED|0:0}\"")):
            path = commands / name
            path.write_text(f"#!/bin/bash\n{code}\n")
            path.chmod(0o755)
        for backend in ("cascade", "yolo"):
            interpreter = self.root / f"pills-baseline/envs/{backend}/bin/python"
            interpreter.parent.mkdir(parents=True)
            exe = shlex.quote(sys.executable)
            interpreter.write_text(
                "#!/bin/bash\n"
                'if [[ $1 == -m && $2 == scripts.evaluate_baseline ]]; then\n'
                f'  shift 2; EVAL_TEST_BACKEND={backend} exec {exe} {shlex.quote(str(mock))} "$@"\n'
                f'fi\nexec {exe} "$@"\n'
            )
            interpreter.chmod(0o755)
        self.env = {
            **os.environ, "STORE": str(self.root), "SLURM_SUBMIT_DIR": str(REPO), "SLURM_JOB_ID": "99",
            "SLURM_JOB_NAME": "eval-fixture", "EVAL_TEST_LOG": str(self.log), "EVAL_TEST_FIXTURE": str(self.root),
            "PATH": f"{commands}:{os.environ['PATH']}",
        }
        self.commit = "a" * 40

    def training(self, path, backend, warmup=1.0, job="12"):
        path.mkdir(parents=True)
        config = load_config(backend, None, {})
        if backend == "yolo":
            config["warmup_epochs"] = warmup
            write_json(path / "ultralytics_runtime.json", {
                "args": dict(warmup_epochs=warmup, seed=2026, epochs=30, batch=8, lr0=0.001, cos_lr=True,
                             lrf=0.05, amp=False, warmup_bias_lr=0.001, warmup_momentum=0.8),
                "iterations_per_epoch": 40, "optimizer": "AdamW",
                "warmup_iterations_end_index": max(round(warmup * 40), 100),
            })
        write_json(path / "run.json", {
            "run": {"backend": backend, "dataset_manifest_sha256": self.digest, "config": config,
                    "initial_weights_sha256": "c" * 64},
            "hardware": {"job_id": job}, "git_head": self.commit, "code_sha256": {"scripts/evaluate_baseline.py": "old"},
        })
        for name, content in (("train.job", job), ("train.path", str(path))):
            (path.parent / name).write_text(content + "\n")
        write_json(path.parent / "config.json", config)
        write_json(path / "status.json", {
            "status": "complete", "best_checkpoint": "best.pt", "last_checkpoint": "last.pt",
            "sessions": [{"job_id": job}], "total_elapsed_seconds": 12, "cuda_peak_allocated_bytes": 1,
            "best_step": config["epochs"], "best_validation_map": 0.5,
        })
        (path / "best.pt").write_bytes(b"best")
        (path / "last.pt").write_bytes(b"last")
        loss_fields = "train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss"
        (path / "history.csv").write_text("step,val/map," + loss_fields + "\n"
                                          + "".join(f"{i},0.5,1,1,1,1,1,1\n" for i in range(1, config["epochs"] + 1)))
        shutil.copyfile(self.root / "pr_curves.png", path / "training.png")
        return path

    def launch(self, *args, **env):
        return subprocess.run(["bash", str(REPO / "slurm/evaluate.sh"), *map(str, args)], cwd=REPO,
                              env={**self.env, **env}, capture_output=True, text=True)

    def options(self, out, *runs):
        return ("--data", self.data, "--output", out, *(arg for name, path in runs for arg in ("--run", name, path)))

    def calls(self):
        import json

        return [json.loads(row) for row in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_legacy_and_named_mixed_backends(self):
        cascade = self.training(self.root / "r1/train", "cascade")
        yolo = self.training(self.root / "r2/train", "yolo")
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                out = self.root / f"out-{legacy}"
                args = ((self.data, cascade, yolo, out) if legacy
                        else self.options(out, ("r1", cascade), ("r2", yolo)))
                result = self.launch(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((out / "verified.ok").exists())
                names = ["cascade", "yolo"] if legacy else ["r1", "r2"]
                self.assertEqual([row["output"] for row in self.calls()[-2:]], names)
                self.assertEqual([row["backend"] for row in self.calls()[-2:]], ["cascade", "yolo"])
        wrong = self.launch(self.data, yolo, cascade, self.root / "wrong")
        self.assertIn("backend incorrecto", wrong.stderr)

    def test_one_model_test_last_and_benchmark_options(self):
        run = self.training(self.root / "arm/train", "yolo")
        (self.root / "pills-baseline/envs/cascade/bin/python").unlink()
        out = self.root / "result"
        result = self.launch(*self.options(out, ("solo", run)), "--split", "test", "--checkpoint", "last",
                             "--warmup", "3", "--iterations", "4", "--examples", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = read_json(out / "solo/metrics.json")
        self.assertEqual(metrics["checkpoint_sha256"], sha256(run / "last.pt"))
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["benchmark"]["iterations"], 4)
        self.assertEqual(self.calls()[0]["opts"]["--examples"], "0")
        self.assertEqual((out / "eval.job").read_text().strip(), "99")
        self.assertEqual(read_json(out / "launch.json")["job_name"], "eval-fixture")

    def test_one_cascade_without_benchmark_or_yolo_environment(self):
        run = self.training(self.root / "arm/train", "cascade")
        (self.root / "pills-baseline/envs/yolo/bin/python").unlink()
        out = self.root / "result"
        result = self.launch(*self.options(out, ("solo", run)), "--no-benchmark")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]["backend"], "cascade")
        self.assertIsNone(read_json(out / "solo/metrics.json")["benchmark"])
        self.assertFalse((out / ".ultralytics").exists())

    def test_staged_training_config_is_preserved_by_generic_evaluation(self):
        run = self.training(self.root / "freeze-5/train", "yolo")
        provenance = read_json(run / "run.json")
        provenance["run"]["config"]["freeze_backbone_epochs"] = 5
        write_json(run / "run.json", provenance)
        out = self.root / "staged-eval"
        result = self.launch(*self.options(out, ("freeze-5", run)), "--check-slurm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out / "verified.ok").is_file())
        metrics = read_json(out / "freeze-5/metrics.json")
        self.assertEqual(metrics["model_config"]["freeze_backbone_epochs"], 5)
        self.assertEqual(metrics["checkpoint_sha256"], sha256(run / "best.pt"))

    def test_two_yolo_runs_preserve_exp006_inventory_and_provenance(self):
        paths = [self.training(self.root / f"s2026/{arm}/train", "yolo", warmup, job)
                 for arm, warmup, job in (("control", 1.0, "12"), ("warmup-5", 5.0, "13"))]
        # Separate deployed launcher and checkout, as on CESGA for the historical training.
        layer = self.root / "deployed"
        checkout = self.root / "historical"
        layer.mkdir()
        checkout.mkdir()
        shutil.copyfile(REPO / "slurm/evaluate.py", layer / "evaluate.py")
        (checkout / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
        out = self.root / "eval-a-s2026"
        result = self.launch(*self.options(out, *zip(("control", "warmup-5"), paths)), "--check-slurm",
                             "--example-image-id", 57, "--example-image-id", 61, "--example-image-id", 80,
                             SLURM_SUBMIT_DIR=str(checkout), PILLS_EVAL_LAUNCHER_DIR=str(layer))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["backend"] for row in self.calls()], ["yolo", "yolo"])
        self.assertEqual({row["cwd"] for row in self.calls()}, {str(checkout)})
        self.assertEqual({row["settings"] for row in self.calls()}, {str(out / ".ultralytics")})
        launch = read_json(out / "launch.json")
        self.assertEqual(launch["launcher_sha256"]["evaluate.py"], sha256(layer / "evaluate.py"))
        self.assertEqual(launch["runs"]["warmup-5"]["training_job_id"], "13")
        (self.root / "source-commit").write_text(self.commit + "\n")
        (self.root / "inputs.sha256").write_text(
            f"{self.digest}  data/manifest.json\n{sha256(self.data / 'coco/valid.json')}  valid.json\n"
            f"{'c' * 64}  yolov10s.pt\n")
        analysis = importlib.import_module("experiments.exp-006-yolo-warmup.analyze")
        for arm in ("control", "warmup-5"):
            self.assertEqual({p.name for p in (out / arm / "examples").iterdir()},
                             {f"image-{i}.jpg" for i in (57, 61, 80)})
            _, metrics, _, _ = analysis.read_arm(self.root, 2026, arm, "a",
                                                 analysis.campaign_identity(self.root), self.truth)
            self.assertEqual(metrics["git_head"], self.commit)

    def test_rejects_invalid_args_paths_and_runs_before_inference(self):
        run = self.training(self.root / "r1/train", "yolo")
        out = self.root / "out"
        existing = self.root / "existing"
        existing.mkdir()
        linked = self.root / "link"
        linked.symlink_to(run, target_is_directory=True)
        for args in (
            self.options(out, ("a", run), ("a", run)),
            self.options(out, ("../a", run)),
            self.options(out, ("a", "")),
            self.options(out, ("a", run)) + ("--split", "train"),
            self.options(out, ("a", run)) + ("--iterations", "0"),
            self.options(out, ("a", run)) + ("--example-image-id", "999"),
            self.options(out, ("a", self.root / "missing")),
            self.options(existing, ("a", run)),
            self.options(run / "out", ("a", run)),
            self.options(linked / "out", ("a", run)),
            self.options(self.data / "out", ("a", run)),
        ):
            with self.subTest(args=args):
                self.assertNotEqual(self.launch(*args).returncode, 0)
                self.assertEqual(self.calls(), [])
        self.assertFalse(out.exists())
        for path, changes in (
            (run / "status.json", {"status": "paused"}),
            (run / "status.json", {"best_checkpoint": "../missing"}),
            (run / "run.json", {"run": {"backend": "unknown"}}),
            (run / "run.json", {"run": {"backend": "yolo", "dataset_manifest_sha256": "wrong"}}),
        ):
            saved = read_json(path)
            write_json(path, {**saved, **changes})
            self.assertNotEqual(self.launch(*self.options(out, ("a", run))).returncode, 0)
            self.assertFalse(out.exists())
            write_json(path, saved)
        self.assertNotEqual(self.launch(*self.options(out, ("a", run)), "--check-slurm",
                                       TEST_SLURM_STATE="FAILED|1:0").returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_failure_never_creates_success_marker(self):
        control = self.training(self.root / "r1/train", "yolo")
        variant = self.training(self.root / "r2/train", "yolo")
        for fault, code in (("evaluate", 23), ("verify", 1), ("checkpoint", 1), ("benchmark", 1)):
            with self.subTest(fault=fault):
                out = self.root / fault
                result = self.launch(*self.options(out, ("control", control), ("variant", variant)),
                                     EVAL_TEST_FAILURE=fault)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertTrue((out / "launch.json").exists())
                self.assertFalse((out / "verified.ok").exists())

    def test_help_without_slurm_or_store(self):
        result = subprocess.run(["bash", str(REPO / "slurm/evaluate.sh"), "--help"], cwd=REPO,
                                env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run NOMBRE RUTA", result.stdout)
