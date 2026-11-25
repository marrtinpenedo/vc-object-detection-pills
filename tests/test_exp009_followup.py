"""CPU checks of EXP-009 intervention isolation and paired factorial contrasts."""
import copy
import unittest
from importlib import import_module
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import campaign
from scripts.baseline_data import read_json
from scripts.exp009_followup.bundle import plan_path
from scripts.exp009_followup.recipes import SMOKE, RUNTIME_PLAN, graph, recipe, smoke_recipe, verify_plan

followup = import_module("experiments.exp-009-final-optimization.followup")
confirmation = import_module("experiments.exp-009-final-optimization.confirmation")


class Exp009FollowupTest(unittest.TestCase):
    def test_reference_and_new_factors(self):
        for phase, count in (("B", 9), ("C", 25), ("D", 5)):
            with self.subTest(phase=phase):
                plan = read_json(plan_path(phase))
                historical = read_json(campaign.REPO / RUNTIME_PLAN[phase])
                verify_plan(plan, historical, campaign.condition_config)
                nodes = graph(plan)
                backend, = plan["packages"]
                self.assertEqual(len(nodes), count)
                for condition in plan["conditions"]:
                    name = condition["name"]
                    self.assertEqual(nodes[f"train-{name}"]["requires"], f"preflight-{backend}")
                    self.assertEqual(nodes[f"eval-{name}"]["requires"], f"train-{name}")
                    base = plan["reference"]["arms"][condition["reference"]]["config"]
                    changed = recipe(plan, condition)
                    if phase == "B":
                        self.assertEqual(changed.pop("photometric"), {"probability": .5, "brightness": .1, "contrast": .1})
                    else:
                        self.assertIn(changed["seed"], (2027, 2028))
                        changed["seed"] = 2026
                    self.assertEqual(changed, base)
                changed = copy.deepcopy(plan)
                changed["evaluation"]["split"] = "test"
                with self.assertRaisesRegex(ValueError, "protocolo"):
                    verify_plan(changed, historical, campaign.condition_config)
                changed = copy.deepcopy(plan)
                next(iter(changed["reference"]["arms"].values()))["config"]["lr"] = 0
                with self.assertRaisesRegex(ValueError, "referencia"):
                    verify_plan(changed, historical, campaign.condition_config)

    def test_composed_native_smoke(self):
        for phase in ("B", "C"):
            plan = read_json(plan_path(phase))
            smoke = smoke_recipe(plan)
            self.assertIn(SMOKE, graph(plan)["preflight-yolo"]["probes"])
            self.assertEqual((smoke["epochs"], smoke["freeze_backbone_epochs"]), (3, 1))
            self.assertEqual((smoke["photometric"], smoke["geometric"], smoke["mosaic"]),
                             (plan["photometric"], plan["geometric"], .3))

    def test_confirmation_pairing_and_candidate_gate(self):
        values = {(arm, seed): {"map": merit, confirmation.SENSITIVITY: merit}
                  for seed in confirmation.SEEDS
                  for arm, merit in zip("ABCDEF", (.71, .78, .77, .82, .80, .88))}
        self.assertEqual(confirmation.choose(values), ("F", 2027, list("ABCDEF")))
        for seed in (2027, 2028):
            values["F", seed]["map"] = .819
        values["F", 2026]["map"] = .96
        self.assertEqual(confirmation.choose(values)[0], "D")
        first, *_ = confirmation.paired_rows(values, ["map"])
        self.assertEqual((first["contrast"], first["positive_new_seeds"], first["positive_seeds"]), ("F-D", 0, 1))
        self.assertAlmostEqual(first["delta_2026"], .14)
        self.assertEqual(len(confirmation.COMPARISONS), 15)

    def test_accounting_requires_literal_success_for_every_allocation(self):
        completed = {"preflight": {"id": "10"}, "train": {"id": "11"}}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "sacct.txt"
            path.write_text("11|COMPLETED|0:0|\n10|COMPLETED|0:0\n")
            self.assertEqual(set(confirmation.verify_slurm_accounting(path, completed)["allocations"]), {"10", "11"})
            for rows in ("10|COMPLETED|0:0", "10|COMPLETED|0:0\n11|COMPLETED|1:0",
                         "10|COMPLETED|0:0\n10|COMPLETED|0:0", "10|COMPLETED"):
                path.write_text(rows)
                with self.assertRaises(ValueError):
                    confirmation.verify_slurm_accounting(path, completed)

    def test_factorial_distinguishes_simple_effect_from_interaction(self):
        values = {f"f{80*f:03d}-{'geom' if g else 'control'}" + ("-photo" if p else ""):
                  1 + f + 2*g + 3*p + 5*f*g + 7*f*g*p for f, g, p in product((0, 1), repeat=3)}
        effects = {name: sum(weights[arm] * values[arm] for arm in weights)
                   for name, weights in followup.contrasts().items()}
        self.assertEqual(effects["fotometria_f000-control"], 3)
        self.assertEqual(effects["fotometria_f080-geom"], 10)
        self.assertEqual(effects["freezing×geometria"], 8.5)
        self.assertEqual(effects["freezing×geometria×fotometria"], 7)


if __name__ == "__main__":
    unittest.main()
