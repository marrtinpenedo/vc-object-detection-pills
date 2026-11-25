"""CPU checks for the prospective factorial, scheduler graph and selection rule."""

import csv
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts import campaign
from scripts.baseline_data import read_json, write_json
from scripts.baseline_run import load_config
from scripts.geometric import GeometrySampler
from scripts.photometric import PhotometricSampler

analysis = import_module("experiments.exp-009-final-optimization.analyze")
PHOTOMETRIC = {"brightness": .1, "contrast": .1, "probability": .5}


class Exp009Test(unittest.TestCase):
    def test_exact_factorial_and_pinned_augmentation(self):
        plan = read_json(campaign.REPO / campaign.EXP009_A)
        old = read_json(campaign.REPO / campaign.PLAN)
        self.assertEqual({k: plan[k] for k in ("geometric", "mosaic_probability", "evaluation", "seed")},
                         {k: old[k] for k in ("geometric", "mosaic_probability", "evaluation", "seed")})
        self.assertEqual(plan["evaluation"]["split"], "valid")
        self.assertEqual(len(plan["conditions"]), 14)
        baseline = load_config("yolo", None, {"epochs": 100})
        self.assertEqual((baseline["lr"], baseline["warmup_epochs"], baseline["scheduler"]), (.001, 1, "cosine"))
        for duration in analysis.DURATIONS:
            for tag in analysis.CELLS:
                arm, = [d for d in plan["conditions"] if d["name"] == analysis.name(duration, tag)]
                expected = {**baseline}
                if duration:
                    expected["freeze_backbone_epochs"] = duration
                if tag == "geom":
                    expected.update(geometric=old["geometric"], mosaic=.3)
                self.assertEqual(campaign.condition_config(plan, arm), expected)
        graph = campaign.graph(plan)
        self.assertEqual(len(graph), 29)
        self.assertEqual(graph["preflight-yolo"]["probes"], ["f000-control", "smoke-temporal", "smoke-permanent"])

    def test_photometry_config_and_independent_worker_streams(self):
        plan = read_json(campaign.REPO / campaign.EXP009_A)
        geom = plan["geometric"]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            yolo = load_config("yolo", None, {})
            write_json(path, {**yolo, "photometric": PHOTOMETRIC, "geometric": geom, "mosaic": .3,
                              "freeze_backbone_epochs": 20})
            config = load_config("yolo", path, {})
            self.assertEqual(config["photometric"], PHOTOMETRIC)
            self.assertEqual(config["freeze_backbone_epochs"], 20)
            cascade = load_config("cascade", None, {})
            write_json(path, {**cascade, "photometric": PHOTOMETRIC, "geometric": geom})
            with self.assertRaises(ValueError):
                load_config("cascade", path, {})
        worker = SimpleNamespace(seed=2026)
        norm = GeometrySampler(geom, 2026)
        photo = PhotometricSampler(PHOTOMETRIC, 2026, stream=17)
        repeat = PhotometricSampler(PHOTOMETRIC, 2026, stream=17)
        self.assertNotEqual(norm.worker_rng(worker).random(), photo.worker_rng(worker).random())
        self.assertEqual(photo.worker_rng(worker).random(), [repeat.worker_rng(worker).random() for _ in range(2)][-1])

    def test_frozen_permanent_can_span_the_full_budget(self):
        published = campaign.REPO / "experiments/exp-007-yolo-staged-transfer/results/phase-c/s2026/freeze-30/train"
        audit, inventory, runtime = (read_json(published / file) for file in
                                     ("staged_transfer.json", "parameters.json", "ultralytics_runtime.json"))
        with (published / "history.csv").open() as stream:
            prototype = next(csv.DictReader(stream))
        history = []
        initial = audit["epochs"][0]
        steps = runtime["iterations_per_epoch"]
        for epoch in range(1, 101):
            history.append({"step": epoch, **{key: float(prototype[key]) for key in ("lr/pg0", "lr/pg1", "lr/pg2")}})
        audit["epochs"] = [
            {**initial, "epoch": e, "adamw_steps": {k: 0 if k == "backbone" else e * steps
                                                     for k in initial["adamw_steps"]},
             "scheduler_epoch": e - 1, "ema_updates": e * steps}
            for e in range(1, 101)
        ]
        audit["freeze_backbone_epochs"] = runtime["freeze_backbone_epochs"] = 100
        analysis.staged.verify_stages(audit, inventory, runtime, history, 100)
        audit["thaw_epoch"] = 100
        with self.assertRaises(ValueError):
            analysis.staged.verify_stages(audit, inventory, runtime, history, 100)

    def test_interaction_and_rare_class_sensitive_selection(self):
        values = {analysis.name(f, a): {"map": .7, analysis.SENSITIVITY: .75}
                  for f in analysis.DURATIONS for a in analysis.CELLS}
        values["f000-geom"]["map"] = .75
        values["f020-geom"].update({"map": .8, analysis.SENSITIVITY: .74})
        values["f040-control"].update({"map": .76, analysis.SENSITIVITY: .76})
        self.assertEqual(analysis.select_duration(values)["duration"], 40)
        row, = [r for r in analysis.factorial(values, ["map"]) if r["freeze_epochs"] == 20]
        self.assertAlmostEqual(row["interaction"], .05)
        values["f040-control"][analysis.SENSITIVITY] = .74
        self.assertIsNone(analysis.select_duration(values))

    def test_launch_dag_and_retry_do_not_duplicate_successful_jobs(self):
        plan = read_json(campaign.REPO / campaign.EXP009_A)
        nodes = campaign.graph(plan)
        recipes = {c["name"]: campaign.condition_config(plan, c) for c in plan["conditions"]}
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, work, data = (base / n for n in ("results", "work", "data"))
            data.mkdir()
            write_json(data / "manifest.json", {})
            write_json(base / "snapshot.json", {})
            snapshot = campaign.sha256(base / "snapshot.json")
            plan["dataset_manifest_sha256"] = campaign.sha256(data / "manifest.json")
            manifest = {"plan": plan, "nodes": nodes, "recipes": recipes, "work": str(work),
                        "data": str(data), "snapshot_sha256": snapshot}
            calls, states = [], {}

            def submit(args, **_):
                calls.append(args)
                ident = str(len(calls))
                states[ident] = "PENDING"
                return SimpleNamespace(returncode=0, stdout=ident, stderr="")

            with (patch.object(campaign, "REPO", base),
                  patch.object(campaign, "deployment", return_value=manifest),
                  patch.object(campaign, "job_states", return_value=states),
                  patch.object(campaign.subprocess, "run", side_effect=submit),
                  redirect_stdout(StringIO())):
                campaign.launch(work, root, plan_path=campaign.EXP009_A)
                self.assertEqual(len(calls), 29)
                self.assertEqual(sum("--kill-on-invalid-dep=yes" in cmd for cmd in calls), 28)
                self.assertTrue(all("exp009-a" in cmd for cmd in calls))
                record = read_json(root / "jobs.json")
                evaluation = record["eval-f010-geom"][0]
                train = record["train-f010-geom"][0]
                self.assertEqual(evaluation["parent_id"], train["id"])
                states[train["id"]] = "COMPLETED"
                write_json(campaign.attempt_dir(root, "train-f010-geom", 1) / "done.json",
                           {"job_id": train["id"], "snapshot_sha256": snapshot})
                states[evaluation["id"]] = "FAILED"
                campaign.launch(work, root, retry=True, plan_path=campaign.EXP009_A)
                self.assertEqual(len(calls), 30)
                self.assertEqual(calls[-1][-2:], ["eval-f010-geom", "2"])
