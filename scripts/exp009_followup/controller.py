"""Run the existing campaign engine against immutable training sources and new recipes."""
import argparse
import os
import sys
from pathlib import Path

from recipes import SMOKE, RUNTIME_PLAN, graph, recipe, smoke_recipe, verify_plan

PACKAGE = Path(__file__).resolve().parent
SPEC = PACKAGE / "plan.json"
sys.path.insert(0, str(PACKAGE / "runtime"))
from scripts import campaign as engine  # noqa: E402
from scripts.baseline_data import read_json, sha256, write_json  # noqa: E402


def activate():
    descriptor = PACKAGE / "controller.json"
    identity = read_json(descriptor)
    files = ("controller.py", "recipes.py", "campaign.sh", "plan.json")
    engine.require(set(identity) == set(files), "Inventario del controlador incompleto")
    for name in files:
        engine.require(sha256(PACKAGE / name) == identity[name], f"Controlador alterado: {name}")
    plan = read_json(SPEC)
    engine.verify_source(plan["reference"]["snapshot_sha256"])
    verify_plan(plan, read_json(engine.REPO / RUNTIME_PLAN[plan["phase"]]), engine.condition_config)
    native_deployment, native_train = engine.deployment, engine.train
    fingerprint = sha256(descriptor)

    def deployment(work, plan_path):
        return {**native_deployment(work, plan_path), "controller_sha256": fingerprint}

    def train(root, manifest, arm, backend, data, weights, output, initialize=False):
        if arm != SMOKE:
            return native_train(root, manifest, arm, backend, data, weights, output, initialize)
        engine.require(backend == "yolo" and initialize, "El smoke de composición solo pertenece al preflight YOLO")
        engine.require(sha256(weights) == plan["weights"][backend]["sha256"], "Pesos del smoke alterados")
        config = output.with_suffix(".json")
        write_json(config, smoke_recipe(plan))
        engine.run_module("scripts.train_baseline", backend, "--data", data, "--weights", weights,
                          "--config", config, "--output", output)
        runtime = read_json(output / "ultralytics_runtime.json")
        audit = read_json(output / "staged_transfer.json")
        engine.require(
            read_json(output / "status.json")["status"] == "complete"
            and runtime["train_transforms"] == ["YoloMosaic", "YoloGeometry", "YoloPhotometric", "LetterBox", "Format"]
            and runtime["args"]["close_mosaic"] == 0 and runtime["args"]["mosaic"] == plan["mosaic_probability"]
            and [row["epoch"] for row in audit["epochs"]] == [1, 2, 3]
            and audit["thaw_epoch"] == 2
            and [row["adamw_steps"]["backbone"] for row in audit["epochs"]] == [0, 40, 80],
            "Falló el smoke nativo de composición y descongelación")

    # The historical engine owns the ledger, jobs, SLURM dependencies and export.
    engine.graph = graph
    engine.condition_config = recipe
    engine.deployment = deployment
    engine.train = train
    engine.BATCH = PACKAGE / "campaign.sh"
    return plan, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("plan")
    for mode in ("launch", "status", "collect", "reconcile"):
        command = commands.add_parser(mode)
        command.add_argument("--work", type=Path, required=True)
        command.add_argument("--root", type=Path)
        if mode == "launch":
            command.add_argument("--retry-failed", action="store_true")
        elif mode == "collect":
            command.add_argument("output", type=Path)
        elif mode == "reconcile":
            command.add_argument("name")
            choice = command.add_mutually_exclusive_group(required=True)
            choice.add_argument("--job-id")
            choice.add_argument("--confirmed-not-submitted", action="store_true")
    job = commands.add_parser("job")
    job.add_argument("work", type=Path)
    job.add_argument("backend")
    job.add_argument("root", type=Path)
    job.add_argument("name")
    job.add_argument("attempt", type=int)
    args = parser.parse_args()
    plan, fingerprint = activate()
    if args.mode == "plan":
        for name, node in graph(plan).items():
            print(f"{name:40} {node['requires'] or '-'}")
        return
    work = args.work.resolve()
    root = args.root.resolve() if args.root else work / "results" / f"exp-009-{plan['phase'].lower()}"
    if args.mode == "launch":
        engine.launch(work, root, retry=args.retry_failed, plan_path=SPEC)
        return
    receipt = read_json(root / "manifest.json")
    engine.require(receipt["plan"] == plan and receipt["controller_sha256"] == fingerprint
                   and receipt["snapshot_sha256"] == plan["reference"]["snapshot_sha256"],
                   "La raíz está ligada a otro controlador o snapshot")
    if args.mode == "job":
        engine.require(bool(os.environ.get("SLURM_JOB_ID")), "Usa sbatch para las tareas GPU")
        engine.job(work, args.backend, root, args.name, args.attempt, plan_path=SPEC)
    elif args.mode == "status":
        engine.status(root)
    elif args.mode == "collect":
        engine.collect(root, args.output.resolve())
    else:
        engine.reconcile(root, args.name, args.job_id, args.confirmed_not_submitted)


if __name__ == "__main__":
    main()
