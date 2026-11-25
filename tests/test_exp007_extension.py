"""Local tests for EXP-007 evidence, provenance and the SLURM submission plan."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.baseline_data import read_json, write_json
from scripts.baseline_run import REPO


MODULE = "experiments.exp-007-yolo-staged-transfer"
provenance = import_module(f"{MODULE}.provenance")
analysis = import_module(f"{MODULE}.analyze_c")
planner = import_module(f"{MODULE}.campaign")
evidence = import_module(f"{MODULE}.evidence")
A = REPO / "experiments/exp-007-yolo-staged-transfer/results/phase-a"
C = A.with_name("phase-c")


def historical_checkouts(root):
    old, published = (root / name for name in ("original", "published"))
    revision = (C / "source-commit").read_text().strip()
    for folder, commit in ((old, provenance.ORIGINAL), (published, revision)):
        folder.mkdir()
        archive = subprocess.check_output(["git", "archive", commit], cwd=REPO)
        subprocess.run(["tar", "-x", "-C", str(folder)], input=archive, check=True)
    return old, published, revision


class EvidenceTest(unittest.TestCase):
    def test_publication_retains_figures_without_job_sidecars_or_duplicate_training(self):
        for phase, (seeds, arms) in evidence.COHORTS.items():
            with self.subTest(phase=phase):
                archived = set(evidence.inventory(phase))
                raw = set(evidence.inventory(phase, raw=True))
                self.assertLess(archived, raw)
                self.assertFalse(any(name.endswith((".train.sacct", "/train.job", "/train.path",
                                                   "/eval.job", "/verified.ok")) for name in archived))
                self.assertEqual(sum(name.endswith("/pr_curves.png") for name in archived), len(seeds) * len(arms))
                self.assertEqual(sum(name.endswith("/training.png") for name in archived),
                                 {"a": 4, "b": 4, "c": 13}[phase])
        hashes = provenance.verify_evidence(A)
        examples = {f"eval-a-s2026/{arm}/examples/failure-{ident}.jpg"
                    for arm in ("control", "freeze-15") for ident in evidence.EXAMPLE_IDS}
        examples.update({"eval-a-s2026/control/examples/failure-10.jpg",
                         "eval-a-s2026/freeze-15/examples/correct-10.jpg"})
        self.assertEqual(set(hashes), set(evidence.inventory("a")) | examples)

    def test_example_labels_distinguish_errors_and_empty_scenes(self):
        cases = [
            ({"tp": [1], "fp": [], "fn": []}, "correct"),
            ({"tp": [1], "fp": [2], "fn": []}, "failure"),
            ({"tp": [], "fp": [], "fn": [3]}, "failure"),
            ({"tp": [], "fp": [], "fn": []}, "empty"),
        ]
        for groups, label in cases:
            with self.subTest(label=label, groups=groups):
                self.assertEqual(evidence.example_name(10, groups), f"{label}-10.jpg")

    def test_authorizes_only_the_pinned_scientific_delta(self):
        with TemporaryDirectory() as tmp:
            old, published, revision = historical_checkouts(Path(tmp))
            historical, extension = provenance.verify_revisions(old, published, A)
            provenance.verify_commit(REPO, provenance.ORIGINAL, historical)
            provenance.verify_commit(REPO, revision, extension)
            with self.assertRaisesRegex(ValueError, "Huella Git incompatible"):
                provenance.verify_commit(REPO, provenance.ORIGINAL, extension)
            with (published / "scripts/evaluate_baseline.py").open("a") as stream:
                stream.write("# unexpected change\n")
            with self.assertRaisesRegex(ValueError, "no autorizado"):
                provenance.verify_revisions(old, published, A)

    def test_reuses_historical_predictions_with_new_benchmark_only(self):
        analysis.verify_evidence(A)
        historical = A / "eval-a-s2026/freeze-15"
        with TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            for name in ("metrics.json", "pr_curves.json"):
                shutil.copyfile(historical / name, fresh / name)
            metrics = read_json(fresh / "metrics.json")
            metrics["benchmark"]["fps_from_mean"] /= 2
            write_json(fresh / "metrics.json", metrics)
            analysis.verify_reused(historical, fresh)
            for key, value in (("checkpoint_sha256", "0" * 64), ("map", 0.5)):
                with self.subTest(key=key):
                    write_json(fresh / "metrics.json", {**metrics, key: value})
                    with self.assertRaisesRegex(ValueError, key):
                        analysis.verify_reused(historical, fresh)
            write_json(fresh / "metrics.json", metrics)
            write_json(fresh / "pr_curves.json", [])
            with self.assertRaisesRegex(ValueError, "Curvas PR"):
                analysis.verify_reused(historical, fresh)

    def test_duplicate_training_cannot_fill_a_second_slot(self):
        training = {seed: {arm: {"job": f"{seed}-{arm}", "path": f"/train/{seed}/{arm}"}
                           for arm in provenance.C_ARMS} for seed in provenance.SEEDS}
        analysis.verify_unique(training)
        for key in ("job", "path"):
            with self.subTest(key=key):
                record = training[2028]["freeze-30"]
                previous = record[key]
                record[key] = training[2026]["freeze-30"][key]
                with self.assertRaisesRegex(ValueError, rf"duplicados \({key}\)"):
                    analysis.verify_unique(training)
                record[key] = previous

    @unittest.skipUnless(find_spec("pycocotools") and (REPO / "data/pills-sxdht/valid").is_dir(),
                         "requiere anotaciones locales de validación y pycocotools")
    def test_cohort_accepts_authorized_training_sources_without_relabeling_evaluation(self):
        with TemporaryDirectory() as tmp:
            old, published, _ = historical_checkouts(Path(tmp))
            mixed = Path(tmp) / "mixed"
            original, extension = provenance.verify_revisions(old, published, A)
            arm = "freeze-15"
            slot = mixed / "s2026" / arm
            shutil.copytree(A / "s2026" / arm, slot)
            run_path = slot / "train/run.json"
            run = read_json(run_path)
            run["code_sha256"] = provenance.script_hashes(extension)
            write_json(run_path, run)
            (mixed / "evidence.sha256").write_text("".join(
                f"{provenance.digest(file)}  {file.relative_to(mixed)}\n"
                for file in sorted(slot.rglob("*")) if file.is_file()
            ))
            provenance.verify_evidence(mixed)
            common = analysis.paired.common
            identity = common.campaign_identity(A)
            truth = common.validation_truth(REPO / "data/pills-sxdht", A, identity[1])
            sources = {name: (A, provenance.ORIGINAL, provenance.script_hashes(original))
                       for name in provenance.A_ARMS}
            sources[arm] = mixed, "f" * 40, run["code_sha256"]
            values, _, _ = analysis.paired.read_cohort(
                A, "a", 2026, provenance.A_ARMS, identity, truth, training_sources=sources, evaluation_checkout=old,
            )
            self.assertEqual(analysis.paired.select(values), arm)
            self.assertEqual(read_json(A / "eval-a-s2026" / arm / "metrics.json")["code_sha256"],
                             provenance.script_hashes(original))
            sources[arm] = mixed, provenance.ORIGINAL, provenance.script_hashes(original)
            with self.assertRaisesRegex(ValueError, "procedencia"):
                analysis.paired.read_cohort(
                    A, "a", 2026, provenance.A_ARMS, identity, truth, training_sources=sources, evaluation_checkout=old,
                )


class CampaignTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        old, new = self.root / "historical", self.root / "extension"
        campaign = planner.Campaign.__new__(planner.Campaign)
        campaign.original = self.root / "original-code"
        campaign.roots = {"a": old, "b": old, "c": new, "smoke": new}
        campaign.extension, campaign.queue, campaign.historical = new, new / "queue", old
        campaign.data, campaign.weights = old / "data", self.root / "yolov10s.pt"
        campaign.python = self.root / "python"
        (campaign.original / "configs/experiments").mkdir(parents=True)
        shutil.copyfile(REPO / "configs/experiments/yolo-freeze-15.json",
                        campaign.original / "configs/experiments/yolo-freeze-15.json")
        campaign.retry = False
        historical_eval = campaign.slot(planner.Task("a", 2026))
        historical_eval.mkdir(parents=True)
        (historical_eval / "eval.job").write_text("12005631\n")
        campaign.queue.mkdir(parents=True)
        self.campaign = campaign

    def test_submits_once_with_historical_reuse_and_smoke_dependency(self):
        commands = []

        def submit(command, **options):
            commands.append((command, options))
            return f"{100 + len(commands)};cluster\n"

        with (patch.object(self.campaign, "prepare_campaign"),
              patch.object(self.campaign, "verify_evaluation") as verify,
              patch.object(planner, "slurm", return_value=""),
              patch.object(planner, "state", return_value="active"),
              patch.object(planner.subprocess, "check_output", side_effect=submit),
              patch("builtins.print")):
            self.campaign.submit()
            self.campaign.submit()
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(commands), 23)
        smoke, _ = commands[0]
        self.assertIn("--gres=gpu:a100:1", smoke)
        self.assertIn("&&", smoke[-1])
        self.assertIn("verify-smoke", smoke[-1])
        jobs = dict(zip(planner.tasks(), commands))
        for task, (argv, options) in jobs.items():
            expected = REPO if task.arm and task.phase in {"c", "smoke"} else self.campaign.original
            self.assertEqual(options["cwd"], expected)
            self.assertEqual(options["env"]["PILLS_YOLO_CONFIG_DIR"], str(self.campaign.slot(task) / "ultralytics"))
            if task.phase != "smoke":
                dependencies = next(arg for arg in argv if arg.startswith("--dependency="))
                self.assertIn("101", dependencies.split(":")[1:])
        c_evaluation = jobs[planner.Task("c", 2027)][0]
        self.assertEqual(c_evaluation.count("--run"), 7)
        for arm in ("control", "freeze-15"):
            self.assertIn(str(self.campaign.slot(planner.Task("b", 2027, arm)) / "train"), c_evaluation)
        self.assertNotIn(planner.Task("c", 2026, "freeze-15"), jobs)

    def test_unconfirmed_submission_never_gets_resent(self):
        task = planner.Task("smoke", 2026, "freeze-30")
        with (patch.object(planner, "slurm", return_value=""),
              patch.object(planner.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "sbatch"))):
            with self.assertRaises(subprocess.CalledProcessError):
                self.campaign.schedule(task, set(), set())
        with patch.object(planner.subprocess, "check_output") as submit:
            with self.assertRaisesRegex(ValueError, "Envío sin confirmar"):
                self.campaign.schedule(task, set(), set())
            submit.assert_not_called()

    def test_retry_preserves_failed_output_and_reuses_completed_job(self):
        task = planner.Task("smoke", 2026, "freeze-30")
        slot = self.campaign.slot(task)
        self.campaign.command(task)
        self.campaign.legacy_job_file(task).write_text("100\n")
        original = self.campaign.records(task)
        write_json(original, {"job": "100", "afterok": []})
        self.campaign.retry = True
        with (patch.object(planner, "state", side_effect=["failed", "complete"]),
              patch.object(planner, "slurm", return_value=""),
              patch.object(planner.subprocess, "check_output", return_value="101\n") as submit,
              patch.object(self.campaign, "verify_training") as verify, patch("builtins.print")):
            self.assertEqual(self.campaign.schedule(task, set(), set()), "101")
            self.assertIsNone(self.campaign.schedule(task, set(), set()))
        submit.assert_called_once()
        verify.assert_called_once_with(task, "101")
        archived = self.campaign.extension / "attempts" / f"{task.name}-100"
        self.assertEqual(read_json(archived / "submission.json")["job"], "100")
        self.assertEqual((archived / "output/train.job").read_text().strip(), "100")
        self.assertEqual((slot / "train.job").read_text().strip(), "101")

    def test_smoke_requires_all_three_epochs_of_permanent_freeze(self):
        slot = self.root / "smoke"
        training = slot / "train"
        archived = A / "s2026/freeze-15/train"
        training.mkdir(parents=True)
        for name in ("parameters.json", "status.json"):
            shutil.copyfile(archived / name, training / name)
        run = read_json(archived / "run.json")
        run["run"]["config"].update(epochs=3, freeze_backbone_epochs=3)
        write_json(training / "run.json", run)
        runtime = read_json(archived / "ultralytics_runtime.json")
        write_json(training / "ultralytics_runtime.json", {**runtime, "freeze_backbone_epochs": 3})
        rows = (archived / "history.csv").read_text().splitlines()
        (training / "history.csv").write_text("\n".join(rows[:4]) + "\n")
        audit = read_json(archived / "staged_transfer.json")
        audit.update(freeze_backbone_epochs=3, thaw_epoch=None, optimizer_groups_verified_at_thaw=False,
                     epochs=audit["epochs"][:3])
        write_json(training / "staged_transfer.json", audit)
        with patch("sys.argv", ["campaign", "verify-smoke", str(slot)]):
            planner.main()
        self.assertTrue((slot / "verified.ok").is_file())
        (slot / "verified.ok").unlink()
        audit["epochs"][-1]["adamw_steps"]["backbone"] = 1
        write_json(training / "staged_transfer.json", audit)
        with (patch("sys.argv", ["campaign", "verify-smoke", str(slot)]), patch("sys.stderr"),
              self.assertRaises(SystemExit) as failed):
            planner.main()
        self.assertEqual(failed.exception.code, 1)
        self.assertFalse((slot / "verified.ok").exists())


if __name__ == "__main__":
    unittest.main()
