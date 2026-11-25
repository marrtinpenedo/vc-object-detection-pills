"""Exercise the C controller from its transferred layout with a fake SLURM."""
import os
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from scripts.campaign import REPO


class ConfirmationBundleSimulation(unittest.TestCase):
    def test_deployed_graph_and_selective_retry(self):
        bundle = REPO / "experiments/exp-009-final-optimization/results/evidence/exp-009-c.tar.gz"
        if not bundle.is_file():
            bundle = REPO / "results/exp-009-c.tar.gz"
        if not bundle.is_file():
            self.skipTest("Paquete C local ausente")
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with tarfile.open(bundle) as archive:
                archive.extractall(folder, filter="data")
            check = dedent("""
                from pathlib import Path
                from types import SimpleNamespace
                from unittest.mock import patch

                import controller as c

                plan, fingerprint = c.activate()
                engine = c.engine
                work = c.PACKAGE / "work"
                data, root = work / "data", work / "results"
                data.mkdir(parents=True)
                c.write_json(data / "manifest.json", {})
                nodes = c.graph(plan)
                manifest = {"plan": {**plan, "dataset_manifest_sha256": c.sha256(data / "manifest.json")},
                            "nodes": nodes, "recipes": {row["name"]: c.recipe(plan, row) for row in plan["conditions"]},
                            "work": str(work), "data": str(data), "snapshot_sha256": plan["reference"]["snapshot_sha256"],
                            "controller_sha256": fingerprint}
                calls, states = [], {}

                def submit(command, **kwargs):
                    assert kwargs["cwd"] == c.PACKAGE / "runtime"
                    assert command[-7:-5] == [str(c.PACKAGE / "campaign.sh"), "job"]
                    calls.append(command)
                    return SimpleNamespace(stdout=str(len(calls)), stderr="", returncode=0)

                with (patch.object(engine, "deployment", return_value=manifest),
                      patch.object(engine, "job_states", return_value=states),
                      patch.object(engine.subprocess, "run", side_effect=submit)):
                    engine.launch(work, root, plan_path=c.SPEC)
                    assert len(calls) == 25
                    assert sum("--kill-on-invalid-dep=yes" in call for call in calls) == 24
                    engine.launch(work, root, plan_path=c.SPEC)
                    assert len(calls) == 25
                    ledger = c.read_json(root / "jobs.json")
                    done, = ledger["train-a-s2027"]
                    states[done["id"]] = "COMPLETED"
                    c.write_json(engine.attempt_dir(root, "train-a-s2027", 1) / "done.json",
                                 {"job_id": done["id"], "snapshot_sha256": manifest["snapshot_sha256"]})
                    failures = {"eval-a-s2027": "FAILED", "train-f-s2028": "FAILED", "eval-f-s2028": "CANCELLED"}
                    for name, status in failures.items():
                        states[ledger[name][0]["id"]] = status
                    engine.launch(work, root, retry=True, plan_path=c.SPEC)
                    assert len(calls) == 28
                    assert {call[-2] for call in calls[25:]} == set(failures)
                    assert all(call[-1] == "2" for call in calls[25:])
                    assert not any(arg.startswith("--dependency=afterok") for arg in calls[25])
                    assert "--dependency=afterok:27" in calls[-1]
            """)
            result = subprocess.run([sys.executable, "-c", check], cwd=folder, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(folder)})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
