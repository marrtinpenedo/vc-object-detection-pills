"""Local campaign tests with simulated scheduler receipts and archives."""
import io
import tarfile
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts import campaign
from scripts.baseline_data import read_json, sha256, write_json

importer = import_module("experiments.campaign-003-005.import_results")
analysis = import_module("experiments.campaign-003-005.analyze")
ensemble = import_module("experiments.exp-003-ensemble.analyze")


class CampaignTest(unittest.TestCase):
    def test_graph_and_retries_keep_independent_work(self):
        plan = read_json(campaign.REPO / campaign.PLAN)
        nodes = campaign.graph(plan)
        self.assertEqual(len(nodes), 20)
        self.assertEqual(nodes["train-exp004-e"]["requires"], "preflight-yolo")
        self.assertEqual(nodes["eval-exp005-d"]["requires"], "train-exp005-d")
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            work, root, data = base / "work", base / "run", base / "data"
            data.mkdir()
            (data / "manifest.json").write_text("{}\n")
            plan["dataset_manifest_sha256"] = sha256(data / "manifest.json")
            snapshot = {"base_commit": "fixture", "sha256": {}}
            write_json(base / campaign.SNAPSHOT, snapshot)
            manifest = {"snapshot_sha256": sha256(base / campaign.SNAPSHOT), "nodes": nodes,
                        "data": str(data), "work": str(work), "plan": plan,
                        "recipes": {c["name"]: campaign.condition_config(plan, c) for c in plan["conditions"]}}
            commands = []
            states = {}

            def submit(command, **kwargs):
                commands.append(command)
                job_id = str(1000 + len(commands))
                states[job_id] = "PENDING"
                return SimpleNamespace(returncode=0, stdout=f"{job_id};cluster\n", stderr="")

            with (patch.object(campaign, "REPO", base),
                  patch.object(campaign, "deployment", return_value=manifest),
                  patch.object(campaign, "job_states", return_value=states),
                  patch.object(campaign.subprocess, "run", side_effect=submit),
                  patch("builtins.print")):
                campaign.launch(work, root)
                self.assertEqual(len(commands), 20)
                self.assertEqual(sum(any(s.startswith("--dependency=afterok:") for s in cmd) for cmd in commands), 18)
                campaign.launch(work, root)
                self.assertEqual(len(commands), 20)
                ledger = read_json(root / "jobs.json")
                for name, attempts in ledger.items():
                    attempt = attempts[-1]
                    if name in {"train-exp004-a", "eval-exp004-a"}:
                        states[attempt["id"]] = "CANCELLED"
                        continue
                    states[attempt["id"]] = "COMPLETED"
                    directory = campaign.attempt_dir(root, name, attempt["attempt"])
                    write_json(directory / "done.json", {"job_id": attempt["id"],
                                                          "snapshot_sha256": manifest["snapshot_sha256"]})
                campaign.launch(work, root, retry=True)
                ledger = read_json(root / "jobs.json")
                self.assertEqual(len(commands), 22)
                self.assertEqual(ledger["eval-exp004-a"][-1]["parent_id"], ledger["train-exp004-a"][-1]["id"])
                self.assertFalse(ledger["train-exp004-a"][-1]["afterok"])
                self.assertTrue(ledger["eval-exp004-a"][-1]["afterok"])
                bundle = base / "results.tar.gz"
                campaign.collect(root, bundle)
                imported = importer.import_results(bundle, base / "imported")
            self.assertEqual(read_json(imported / "manifest.json"), manifest)
            self.assertEqual(read_json(imported / "slurm-states.json"), states)

    def test_tar_traversal_is_rejected(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            tar = base / "results.tar.gz"
            entry = tarfile.TarInfo("../escape")
            entry.size = 2
            with tarfile.open(tar, "w:gz") as stream:
                stream.addfile(entry, io.BytesIO(b"{}"))
            with self.assertRaises(ValueError):
                importer.import_results(tar, base / "imported")
            self.assertFalse((base / "escape").exists())

    def test_yolo_recovery_keeps_original_cascade_and_versions_attempts(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            work, root, deployed = base / "work", base / "run", base / "fixed"
            deployed.mkdir()
            (root / "configs").mkdir(parents=True)
            plan = read_json(campaign.REPO / campaign.PLAN)
            (root / "prepared-manifest.json").write_text("{}\n")
            plan["dataset_manifest_sha256"] = sha256(root / "prepared-manifest.json")
            nodes = campaign.graph(plan)
            sources = {name: "1" * 64 for name in campaign.YOLO_RECOVERY_FILES}
            sources["scripts/baseline_yolo.py"] = "a" * 64
            original = {"base_commit": "fixture", "sha256": sources}
            fixed = {**original, "sha256": {**sources, **dict.fromkeys(campaign.YOLO_RECOVERY_FILES, "2" * 64)}}
            write_json(root / campaign.SNAPSHOT, original)
            write_json(deployed / campaign.SNAPSHOT, fixed)
            before, after = (sha256(folder / campaign.SNAPSHOT) for folder in (root, deployed))
            recipes = {c["name"]: campaign.condition_config(plan, c) for c in plan["conditions"]}
            manifest = {"snapshot_sha256": before, "work": str(work), "data": "fixture",
                        "nodes": nodes, "plan": plan, "recipes": recipes}
            write_json(root / "manifest.json", manifest)
            for name, recipe in recipes.items():
                write_json(root / "configs" / f"{name}.json", recipe)
            ledger, states, cascade_files = {}, {}, {}
            failed = "11994101"
            for index, (name, node) in enumerate(nodes.items()):
                ident = failed if name == "preflight-yolo" else str(12000000 + index)
                parent = node["requires"]
                ledger[name] = [{"attempt": 1, "id": ident, "job_name": f"original-{name}",
                                 "parent_id": ledger[parent][0]["id"] if parent else None, "afterok": bool(parent)}]
                folder = campaign.attempt_dir(root, name, 1)
                folder.mkdir(parents=True)
                if node["backend"] == "cascade":
                    states[ident] = "COMPLETED"
                    write_json(folder / "done.json", {"job_id": ident, "snapshot_sha256": before})
                    cascade_files[folder / "done.json"] = (folder / "done.json").read_bytes()
                else:
                    states[ident] = "FAILED" if ident == failed else "CANCELLED"
            write_json(root / campaign.LEDGER, ledger)

            commands = []

            def submit(command, **kwargs):
                commands.append(command)
                ident = str(13000000 + len(commands))
                states[ident] = "PENDING"
                return SimpleNamespace(returncode=0, stdout=f"{ident}\n", stderr="")

            with (patch.object(campaign, "REPO", deployed),
                  patch.object(campaign, "deployment", return_value={**manifest, "snapshot_sha256": after}) as deployment,
                  patch.object(campaign, "job_states", return_value=states),
                  patch.object(campaign.subprocess, "run", side_effect=submit),
                  patch("builtins.print")):
                with self.assertRaisesRegex(ValueError, "ligada a otro snapshot"):
                    campaign.launch(work, root, retry=True)
                deployment.return_value = {**manifest, "snapshot_sha256": after, "data": "other"}
                with self.assertRaisesRegex(ValueError, "rutas"):
                    campaign.launch(work, root, retry=True, failed_yolo_preflight=failed)
                deployment.return_value = {**manifest, "snapshot_sha256": after}
                with self.assertRaisesRegex(ValueError, "no confirma"):
                    campaign.launch(work, root, retry=True, failed_yolo_preflight="123")
                self.assertFalse(commands)
                campaign.launch(work, root, retry=True, failed_yolo_preflight=failed)
                campaign.launch(work, root, retry=True, failed_yolo_preflight=failed)
                self.assertEqual(len(commands), 11)
                self.assertEqual(sum(any(s.startswith("--dependency=afterok:") for s in cmd) for cmd in commands), 10)
                new_ledger = read_json(root / campaign.LEDGER)
                for name, node in nodes.items():
                    self.assertEqual(new_ledger[name][0], ledger[name][0])
                    if node["backend"] == "cascade":
                        self.assertEqual(new_ledger[name], ledger[name])
                    else:
                        record = new_ledger[name][-1]
                        self.assertEqual(record["snapshot_sha256"], after)
                        if node["requires"]:
                            self.assertEqual(record["parent_id"], new_ledger[node["requires"]][-1]["id"])
                        states[record["id"]] = "COMPLETED"
                        write_json(campaign.attempt_dir(root, name, 2) / "done.json",
                                   {"job_id": record["id"], "snapshot_sha256": after})
                campaign.launch(work, root, retry=True, failed_yolo_preflight=failed)
                self.assertEqual(len(commands), 11)
                bundle = base / "recovered.tar.gz"
                campaign.collect(root, bundle)
            imported = importer.import_results(bundle, base / "imported")
            self.assertEqual(analysis.attempt_snapshots(imported, manifest, new_ledger), {before: original, after: fixed})
            self.assertEqual(sha256(root / campaign.SNAPSHOT), before)
            self.assertEqual(read_json(root / "manifest.json"), manifest)
            self.assertTrue(all(path.read_bytes() == original_bytes for path, original_bytes in cascade_files.items()))

    def test_factorial_contrasts_and_pending_arms(self):
        arms = {arm: {"map": score} for arm, score in zip("abcde", [0.6, 0.7, 0.65, 0.8, 0.75])}
        contrasts = {r["contrast"]: r["delta"] for r in analysis.contrast_rows(arms, "yolo", ("map",))}
        self.assertAlmostEqual(contrasts["epochs_no_aug"], 0.1)
        self.assertAlmostEqual(contrasts["interaction"], 0.05)
        self.assertAlmostEqual(contrasts["mosaic_100"], 0.05)
        remaining = list(analysis.contrast_rows({"a": arms["a"]}, "cascade", ("map",)))
        self.assertTrue(all(r["delta"] is None and r["pending_arms"] for r in remaining))

    def test_class_contributions_explain_reversal_in_official_map(self):
        reporting = import_module("experiments.campaign-003-005.reporting")
        rare = reporting.RARE_CLASS
        names = [rare, *(f"class {i}" for i in range(7))]
        a = dict.fromkeys(names, 0.9)
        c = dict.fromkeys(names, 0.8)
        a[rare], c[rare] = 0.1, 0.9
        report, = reporting.effect_contributions({"a": a, "c": c}, names, analysis.CONTRASTS)
        self.assertEqual(report["contrast"], "augmentation_30")
        self.assertAlmostEqual(report["aporte_ibuphil_al_map"], 0.1)
        self.assertAlmostEqual(report["aporte_otras_clases_al_map"], -0.0875)
        self.assertAlmostEqual(report["delta_sin_ibuphil"], -0.1)
        self.assertAlmostEqual(report["delta_map"], 0.0125)
        self.assertEqual((report["otras_suben"], report["otras_bajan"]), (0, 7))

    def test_yolo_effective_recipe_ignores_only_output_paths(self):
        training = (campaign.REPO / "experiments/exp-002-photometric/results/s2026/yolo/control/training")
        record = read_json(training / "ultralytics_runtime.json")
        spec = read_json(training / "config.json")
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "ultralytics_runtime.json"
            write_json(path, record)
            _, first = analysis.verify_runtime(folder, "yolo", spec, 316)
            record["args"].update(project="/different/attempt-02", name="training",
                                  save_dir="/different/attempt-02/training")
            write_json(path, record)
            _, second = analysis.verify_runtime(folder, "yolo", spec, 316)
            self.assertEqual(first, second)
            record["args"]["lr0"] = spec["lr"] / 2
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "Receta efectiva"):
                analysis.verify_runtime(folder, "yolo", spec, 316)

    def test_ensemble_pairs_only_same_class_and_keeps_primary_score(self):
        def box(cls, score, x):
            return {"image_id": 1, "category_id": cls, "score": score, "bbox": [x, 0, 10, 10]}

        rows = ensemble.fuse([box(1, 0.8, 0), box(2, 0.7, 0)],
                             [box(1, 0.6, 1), box(1, 0.5, 1), box(3, 0.4, 0)])
        self.assertEqual(len(rows), 3)
        by_class = {r["category_id"]: r for r in rows}
        self.assertAlmostEqual(by_class[1]["score"], 0.7)
        self.assertAlmostEqual(by_class[1]["bbox"][0], 0.6 / 1.4)
        self.assertEqual(by_class[2]["score"], 0.7)
        self.assertEqual(by_class[3]["score"], 0.2)
