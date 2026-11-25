"""CPU regression tests for the Cascade phase boundary and campaign contract."""

import io
import tarfile
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import campaign
from scripts.baseline_data import read_json, sha256, write_json
from scripts.baseline_run import load_config
from scripts.campaign_analysis.import_results import import_results
from scripts.cascade_staged import BOTTOM_UP, REST, CascadeFreeze, thaw_iteration

analysis = import_module("experiments.exp-008-cascade-freezing.analyze")


class Parameter:
    def __init__(self):
        self.requires_grad, self.grad = True, None
        self.value = np.array([1.0], dtype=np.float32)

    def detach(self):
        return self

    cpu = contiguous = detach

    def numpy(self):
        return self.value

    def numel(self):
        return self.value.size


class FrozenBatchNorm2d:
    pass


class ResNet:
    def __init__(self):
        self.res2, self.res3, self.res4, self.res5 = (range(n) for n in (3, 4, 6, 3))
        self.training = True
        self.weight, self.buffer = Parameter(), Parameter()

    def modules(self):
        return (self, FrozenBatchNorm2d())

    def named_parameters(self):
        return (("stem.conv1.weight", self.weight),)

    def named_buffers(self):
        return (("stem.conv1.norm.running_mean", self.buffer),)

    def requires_grad_(self, mode):
        self.weight.requires_grad = mode

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)


class FPN:
    def __init__(self):
        self.bottom_up = ResNet()


class SGD:
    def __init__(self, parameters):
        self.param_groups = [{"lr": 0.1, "params": list(parameters)}]
        self.state = {}


def trainer_fixture():
    model = SimpleNamespace(training=True, backbone=FPN())
    params = {f"{BOTTOM_UP}stem.conv1.weight": model.backbone.bottom_up.weight,
              **{f"{prefix}weight": Parameter() for prefix in REST}}
    model.named_parameters = params.items
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(MAX_ITER=4),
                          MODEL=SimpleNamespace(RESNETS=SimpleNamespace(NORM="FrozenBN")))
    return SimpleNamespace(model=model, optimizer=SGD(params.values()), cfg=cfg,
                           scheduler=SimpleNamespace(last_epoch=0))


class CascadeFreezeTest(unittest.TestCase):
    def setUp(self):
        torch, optim = ModuleType("torch"), ModuleType("torch.optim")
        optim.SGD = SGD
        patcher = patch.dict("sys.modules", {"torch": torch, "torch.optim": optim})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_phase_boundary_buffer_and_momentum_invariants(self):
        for frozen_epochs in (1, 2):
            with self.subTest(epochs=frozen_epochs), TemporaryDirectory() as tmp:
                trainer = trainer_fixture()
                opt = trainer.optimizer
                audit = CascadeFreeze(trainer, Path(tmp), frozen_epochs, 2)
                backbone = trainer.model.backbone.bottom_up
                for iteration in range(4):
                    trainer.iter = iteration
                    # Simulate Detectron2 restoring train() after a validation.
                    backbone.train()
                    audit.before_step(trainer)
                    frozen = iteration < 2 * frozen_epochs
                    self.assertEqual(backbone.training, not frozen)
                    self.assertEqual(backbone.weight.requires_grad, not frozen)
                    self.assertIn(backbone.weight, opt.param_groups[0]["params"])
                    gradients = {}
                    for name, parameter in trainer.model.named_parameters():
                        if parameter.requires_grad:
                            gradients[name] = "nonzero"
                            parameter.grad = 1
                            opt.state.setdefault(parameter, {"momentum_buffer": 0})["momentum_buffer"] += 1
                            parameter.value += 1
                    trainer.scheduler.last_epoch += 1
                    with patch("scripts.cascade_staged.check_gradients", return_value=gradients):
                        audit.after_step(trainer)
                analysis.verify_freezing(read_json(audit.path), frozen_epochs, 2, 2)
                self.assertEqual(audit.report["first_unfrozen_step"], 3 if frozen_epochs == 1 else None)
                self.assertEqual(opt.state[next(p for p in audit.others)]["momentum_buffer"], 4)

    def test_missing_optimizer_parameter_or_frozen_mutation_fails(self):
        with TemporaryDirectory() as tmp:
            trainer = trainer_fixture()
            trainer.optimizer.param_groups[0]["params"].pop(0)
            with self.assertRaisesRegex(RuntimeError, "entrar en SGD"):
                CascadeFreeze(trainer, Path(tmp), 1, 2)
            trainer = trainer_fixture()
            audit = CascadeFreeze(trainer, Path(tmp), 1, 2)
            trainer.iter = 0
            audit.before_step(trainer)
            trainer.scheduler.last_epoch += 1
            trainer.model.backbone.bottom_up.weight.grad = 1
            with self.assertRaisesRegex(RuntimeError, "gradiente o estado"):
                audit.after_step(trainer)

    def test_real_dataset_iteration_boundaries(self):
        self.assertEqual(thaw_iteration(10, 79, 2370), 790)
        self.assertEqual(thaw_iteration(50, 79, 7900), 3950)
        self.assertIsNone(thaw_iteration(100, 79, 7900))


class Exp008CampaignTest(unittest.TestCase):
    def test_compact_xz_export_uses_the_same_inventory_verification(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            evidence = base / "jobs.json"
            write_json(evidence, {"preflight-cascade": []})
            inventory = base / "export.sha256.json"
            write_json(inventory, {evidence.name: sha256(evidence)})
            archive = base / "evidence.tar.xz"
            with tarfile.open(archive, "w:xz") as stream:
                for file in (evidence, inventory):
                    stream.add(file, arcname=file.name)
            imported = import_results(archive, base / "import")
            self.assertEqual((imported / evidence.name).read_bytes(), evidence.read_bytes())

    def test_recipes_change_only_freeze_for_each_budget(self):
        plan = read_json(campaign.REPO / campaign.EXP008)
        definitions = {row["name"]: row for row in plan["conditions"]}
        self.assertEqual(len(definitions), 8)
        for name, definition in definitions.items():
            epochs = definition["epochs"]
            recipe = campaign.condition_config(plan, definition)
            expected = load_config("cascade", None, {"epochs": epochs})
            if not name.endswith("f000"):
                expected["freeze_backbone_epochs"] = definition["freeze_backbone_epochs"]
            self.assertEqual(recipe, expected)
        self.assertEqual(plan["evaluation"]["split"], "valid")
        self.assertEqual(plan["evaluation"]["checkpoint"], "best")
        self.assertEqual(len(campaign.graph(plan)), 17)

    def test_new_launch_uses_cascade_batch_mode_and_continues_idempotently(self):
        plan = read_json(campaign.REPO / campaign.EXP008)
        nodes = campaign.graph(plan)
        recipes = {d["name"]: campaign.condition_config(plan, d) for d in plan["conditions"]}
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            work, root, data = (base / name for name in ("work", "run", "data"))
            data.mkdir()
            (data / "manifest.json").write_text("{}")
            plan["dataset_manifest_sha256"] = campaign.sha256(data / "manifest.json")
            (base / "snapshot.json").write_text("{}")
            manifest = {"plan": plan, "nodes": nodes, "recipes": recipes, "work": str(work), "data": str(data),
                        "snapshot_sha256": campaign.sha256(base / "snapshot.json")}
            commands, states = [], {}

            def register(command, **_):
                commands.append(command)
                ident = str(len(commands))
                states[ident] = "PENDING"
                return SimpleNamespace(returncode=0, stdout=ident, stderr="")

            with (patch.object(campaign, "REPO", base),
                  patch.object(campaign, "deployment", return_value=manifest),
                  patch.object(campaign, "job_states", return_value=states),
                  patch.object(campaign.subprocess, "run", side_effect=register),
                  redirect_stdout(io.StringIO())):
                campaign.launch(work, root, plan_path=campaign.EXP008)
                campaign.launch(work, root, plan_path=campaign.EXP008)
            self.assertEqual(len(commands), 17)
            self.assertTrue(all("exp008" in command for command in commands))
            self.assertEqual(sum("--kill-on-invalid-dep=yes" in command for command in commands), 16)
            self.assertEqual(nodes["preflight-cascade"]["probes"], ["e030-f000", "smoke"])
