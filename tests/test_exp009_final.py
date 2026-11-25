"""Exercise the final protocol gates and export with synthetic files."""
import importlib
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.baseline_data import read_json, sha256, write_json
from scripts.campaign_analysis.import_results import verify_inventory
from scripts.exp009_followup.final_export import FILES, export
from scripts.exp009_followup.final_import import RECEIPT, ROOT, normalize

driver = importlib.import_module("slurm.evaluate")


class FinalProtocolTest(unittest.TestCase):
    def test_test_identity_and_protocol(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "coco").mkdir()
            manifest = {"classes": list(range(8)), "splits": {"test": {"images": 45}}}
            truth = {"images": list(range(45)), "categories": list(range(8))}
            write_json(root / "manifest.json", manifest)
            write_json(root / "coco/test.json", truth)
            selection = {
                "status": "pendiente_sha256_fisico", "c_slurm_accounting_verified": True,
                "operating_point": {"score": .25, "iou": .5},
                "dataset_manifest_sha256": sha256(root / "manifest.json"),
                "test_coco_expected_sha256_from_manifest": sha256(root / "coco/test.json"),
                "yolo": {"code_snapshot_sha256": "historic-code"}, "evaluation_entrypoint_sha256": "evaluator",
            }
            path = root / "selection.json"
            write_json(path, selection)
            args = SimpleNamespace(selection=path, split="test", checkpoint="best", benchmark=True,
                                   warmup=20, iterations=100, check_slurm=True)
            runs = [("yolo", root / "a"), ("cascade", root / "b")]
            with patch("scripts.campaign.verify_source", return_value=(
                    "historic-code", {"sha256": {"scripts/evaluate_baseline.py": "evaluator"}})) as code:
                self.assertEqual(driver.final_contract(args, root, manifest, truth, runs), selection)
                code.assert_called_once_with("historic-code")
                with self.assertRaisesRegex(ValueError, "45 ejemplos"):
                    driver.final_contract(args, root, manifest, {**truth, "images": truth["images"][:-1]}, runs)
                args.checkpoint = "last"
                with self.assertRaisesRegex(ValueError, "Protocolo final"):
                    driver.final_contract(args, root, manifest, truth, runs)

    def test_checkpoint_identity_includes_recipe_and_best_epoch(self):
        folder = Path("/example/run")
        checkpoint = folder / "best/model_best.pth"
        recipe = {"epochs": 100, "seed": 2027}
        record = {"cascade": {"checkpoint_path": str(checkpoint), "checkpoint_sha256": "expected",
                              "training_job": "150", "recipe": recipe, "validation": {"best_epoch": 34}}}
        status = {"best_step": 2686}
        run = {"backend": "cascade", "config": recipe}
        driver.check_final_weights(record, "cascade", folder, run, status, checkpoint, "expected", "150")
        for changed, digest in (({"best_step": 2687}, "expected"), (status, "different")):
            with self.assertRaisesRegex(ValueError, "checkpoint físico"):
                driver.check_final_weights(record, "cascade", folder, run, changed, checkpoint, digest, "150")

    def test_directory_copy_keeps_prediction_bytes_and_marks_the_local_inventory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / ROOT
            run.mkdir()
            for file in FILES:
                path = run / file
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"fixture\n")
            (run / "extra.txt").write_bytes(b"excluded")
            source = root / "original.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                archive.add(run, arcname=ROOT)
            destination = normalize(source, root / "imported")
            verify_inventory(destination)
            self.assertEqual((destination / "yolo/predictions.json").read_bytes(), b"fixture\n")
            self.assertFalse((destination / "extra.txt").exists())
            self.assertFalse((destination / "final-eval-sacct.txt").exists())
            receipt = read_json(destination / RECEIPT)
            self.assertEqual(receipt["source_archive_sha256"], sha256(source))
            self.assertIs(receipt["local_inventory_only"], True)

    def test_export_requires_exit_zero_and_exactly_the_light_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, target = root / "run", root / "verified"
            run.mkdir()
            for file in FILES:
                path = run / file
                path.parent.mkdir(exist_ok=True)
                path.write_text("fixture")
            (run / "eval.job").write_text("150\n")
            write_json(run / "launch.json", {"job_id": "150", "split": "test", "selection_sha256": "expected"})
            (run / "ignored.pth").write_text("checkpoint")
            output = root / "final.tar.gz"
            with patch("subprocess.check_output", return_value="150|COMPLETED|1:0\n"):
                with self.assertRaises(ValueError):
                    export(run, output)
                self.assertFalse(output.exists())
            with patch("subprocess.check_output", return_value="150|COMPLETED|0:0\n"):
                export(run, output)
            with tarfile.open(output) as archive:
                archive.extractall(target, filter="data")
            self.assertEqual({p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()},
                             {*FILES, "final-eval-sacct.txt", "export.sha256.json"})
            verify_inventory(target)
