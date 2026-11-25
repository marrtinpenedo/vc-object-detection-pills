"""Audit and reproduce phase B or D, including the pinned controls."""
import argparse
from importlib import import_module
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from scripts.baseline_data import read_json, sha256, write_json
from scripts.baseline_metrics import plot_pr
from scripts.campaign import attempt_dir
from scripts.campaign_analysis.analyze import (
    archive, attempt_snapshots, evaluate_condition, job_record, load_dataset, verify_provenance, verify_runtime,
)
from scripts.campaign_analysis.import_results import require, verify_inventory
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, effect_contributions, save, write_csv
from scripts.exp009_followup.bundle import check_reference, plan_path
from scripts.exp009_followup.recipes import SMOKE, graph, recipe, smoke_recipe

phase_a = import_module("experiments.exp-009-final-optimization.analyze")
reports = import_module("experiments.exp-009-final-optimization.followup_reports")


def contrasts():
    def cell(freeze, geometry, photo):
        return f"f{freeze:03d}-{'geom' if geometry else 'control'}" + ("-photo" if photo else "")

    deltas = {}
    for freeze in (0, 80):
        for geometry in (0, 1):
            off, on = (cell(freeze, geometry, photo) for photo in (0, 1))
            deltas[f"fotometria_{off}"] = {off: -1, on: 1}
    factors = ("freezing", "geometria", "fotometria")
    for mask in range(1, 8):
        indices = [i for i in range(3) if mask & (1 << i)]
        effects = {}
        for bits in product((0, 1), repeat=3):
            freeze, geometry, photo = bits
            sign = (-1) ** sum(not bits[i] for i in indices)
            effects[cell(80 * freeze, geometry, photo)] = sign / 2 ** (3 - len(indices))
        deltas["×".join(factors[i] for i in indices)] = effects
    return deltas


def read_results(root, plan, snapshots, truth, splits, conditions, followup=False):
    manifest = read_json(root / "manifest.json")
    ledger, states = (read_json(root / file) for file in ("jobs.json", "slurm-states.json"))
    options = {"recipe_resolver": recipe, "extra_smokes": (SMOKE,)} if followup else {}
    records = {c["name"]: evaluate_condition(root, manifest, snapshots, ledger, states, c, truth, splits, **options)
               for c in conditions}
    require(all(row is not None for row in records.values()), "Faltan evaluaciones válidas")
    for name, row in records.items():
        training = phase_a.by_job(root, ledger, f"train-{name}", row["audit"]["training_job"]) / "training"
        status = read_json(training / "status.json")
        row["resources"] = {"training_seconds": status["total_elapsed_seconds"],
                            "peak_gpu_allocated_mib": status["cuda_peak_allocated_bytes"] / 2**20,
                            "peak_gpu_reserved_mib": max(s["cuda_peak_reserved_bytes"] for s in status["sessions"]) / 2**20}
        if "yolo" in plan["packages"]:
            row["staging"] = phase_a.verify_freezing(training, row["history"], manifest["recipes"][name])
    return records


def verify_completion(root, nodes, ledger, states, snapshot):
    completed = {}
    for name, node in nodes.items():
        successes = [entry for entry in ledger[name] if states.get(entry["id"]) == "COMPLETED"]
        require(len(successes) == 1, f"Se espera un único trabajo completo: {name}")
        entry, = successes
        require(entry.get("snapshot_sha256", snapshot) == snapshot
                and job_record(root, snapshot, name, entry), f"Recibo o snapshot incompatible: {name}")
        parent = node["requires"]
        require(entry["parent_id"] == (completed[parent]["id"] if parent else None),
                f"Dependencia realizada incompatible: {name}")
        completed[name] = entry
    return completed


def load_import(imported, plan, controller_receipt):
    verify_inventory(imported)
    manifest = read_json(imported / "manifest.json")
    ledger = read_json(imported / "jobs.json")
    snapshot = plan["reference"]["snapshot_sha256"]
    snapshots = attempt_snapshots(imported, manifest, ledger)
    receipt = read_json(controller_receipt)
    require(manifest["plan"] == plan and manifest["nodes"] == graph(plan)
            and manifest["controller_sha256"] == sha256(controller_receipt)
            and receipt["plan.json"] == sha256(plan_path(plan["phase"]))
            and set(snapshots) == {snapshot} and set(ledger) == manifest["nodes"].keys(),
            "Controlador, matriz o snapshot inesperado")
    states = read_json(imported / "slurm-states.json")
    completed = verify_completion(imported, manifest["nodes"], ledger, states, snapshot)
    return manifest, ledger, states, snapshots, completed


def verify_composed_smoke(root, ledger, snapshot, plan, job, n_train):
    smoke = phase_a.by_job(root, ledger, "preflight-yolo", job) / SMOKE
    spec = smoke_recipe(plan)
    run, status = (read_json(smoke / filename) for filename in ("run.json", "status.json"))
    verify_provenance(run, "yolo", job, snapshot, plan)
    require(run["mode"] == "train" and run["run"]["config"] == spec and status["status"] == "complete"
            and run["run"]["initial_weights_sha256"] == plan["weights"]["yolo"]["sha256"]
            and run["run"]["dataset_manifest_sha256"] == plan["dataset_manifest_sha256"], "Smoke compuesto incompatible")
    stride, _ = verify_runtime(smoke, "yolo", spec, n_train)
    history, _ = archive.read_history(smoke / "history.csv", stride, spec["epochs"], status)
    phase_a.verify_freezing(smoke, history, spec)


def analyze(phase, imported, reference, data, output, controller_receipt):
    plan = read_json(plan_path(phase))
    reference_manifest = check_reference(plan, reference)
    manifest, ledger, states, snapshots, completed = load_import(imported, plan, controller_receipt)
    snapshot = plan["reference"]["snapshot_sha256"]
    truth, splits = load_dataset(data, imported, plan)
    old_truth, old_splits = load_dataset(data, reference, reference_manifest["plan"])
    require(truth == old_truth and splits == old_splits, "Datos incompatibles entre fases")
    old = read_results(reference, plan, snapshots, truth, splits,
                       [item["condition"] for item in plan["reference"]["arms"].values()])
    new = read_results(imported, plan, snapshots, truth, splits, plan["conditions"], followup=True)
    for condition in plan["conditions"]:
        arm = condition["name"]
        original, current = (row["effective"].copy() for row in (old[condition["reference"]], new[arm]))
        if phase == "D":
            del original["SEED"], current["SEED"]
        require(original == current, f"Receta efectiva no equivalente: {arm}")
    if phase == "B":
        verify_composed_smoke(imported, ledger, snapshots[snapshot], plan,
                              completed["preflight-yolo"]["id"], splits["train"]["images"])
    output.mkdir(parents=True, exist_ok=False)
    rows = {**old, **new}
    recipes = {**reference_manifest["recipes"], **manifest["recipes"]}
    metrics = [dict(arm=name, seed=recipes[name]["seed"], source="reference" if name in old else phase, **r["values"],
                    **r["benchmark"], **r["resources"]) for name, r in rows.items()]
    write_csv(output / "resumen.csv", metrics)
    write_csv(output / "por_clase.csv", [{"arm": name, **c} for name, r in rows.items() for c in r["classes"]])
    histories = [{"arm": name, **entry} for name, r in rows.items() for entry in r["history"]]
    write_csv(output / "historial.csv", histories, tuple(dict.fromkeys(key for r in histories for key in r)))
    write_json(output / "auditoria.json", {"controller_sha256": manifest["controller_sha256"],
                              "verified_inventory_entries": len(read_json(imported / "export.sha256.json")),
               "reference": plan["reference"], "attempts": ledger, "slurm_states": states,
               "completed_jobs": {name: entry["id"] for name, entry in completed.items()},
               "conditions": {name: {**r["audit"], "benchmark_protocol": r["protocol"]} for name, r in rows.items()}})
    fields = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall")
    write_csv(output / "ibuphil.csv", reports.ibuphil_rows(rows))
    if phase == "B":
        names = [c["name"] for c in truth["categories"]]
        aps = {name: {c["name"]: c["ap"] for c in r["classes"]} for name, r in rows.items()}
        effects = list(effect_contributions(aps, names, contrasts()))
        write_csv(output / "factorial.csv", effects)
        write_csv(output / "efectos_metricas.csv", [
            {"contrast": contrast, "metric": field,
             "delta": sum(weight * rows[arm]["values"][field] for arm, weight in weights.items())}
            for contrast, weights in contrasts().items() for field in fields
        ])
        reports.write_b(output, rows, recipes, plan, effects)
    else:
        write_csv(output / "semillas.csv", [{"metric": field, "mean": mean(r[field] for r in metrics),
                                             "sample_sd": stdev(r[field] for r in metrics)} for field in fields])
        reports.write_d(output, rows)
    for name, r in rows.items():
        plot_pr(r["pr_curves"], output / f"pr-{name}.png")
    plot(output, rows, phase)


def plot(output, rows, phase):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    x = range(len(rows))
    for offset, metric, label in ((-.18, "map", "8 clases"), (.18, SENSITIVITY, f"Sin {RARE_CLASS}")):
        bars = axes[0].bar([i + offset for i in x], [100 * row["values"][metric] for row in rows.values()],
                           width=.34, label=label)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set(xticks=list(x), xticklabels=list(rows), ylim=(0, 108), ylabel="mAP COCO (%)", title=f"EXP-009 {phase}")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].legend()
    for name, row in rows.items():
        history = [entry for entry in row["history"] if "val/map" in entry]
        axes[1].plot([p["step"] / row["stride"] for p in history], [100 * p["val/map"] for p in history],
                     label=name, lw=1)
        axes[1].scatter(row["values"]["best_epoch"], 100 * row["values"]["best_validation_map_history"], s=15)
    axes[1].set(xlabel="Época", ylabel="mAP interno (%)", title="Validación durante el entrenamiento")
    axes[1].legend(fontsize=8, ncols=2)
    save(fig, output / "rendimiento.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("B", "D"))
    for required in ("imported", "reference", "data", "output", "controller_receipt"):
        parser.add_argument(required, type=Path)
    args = parser.parse_args()
    analyze(args.phase, args.imported, args.reference, args.data, args.output, args.controller_receipt)
