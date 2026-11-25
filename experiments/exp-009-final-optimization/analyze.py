"""Reproduce the 14 phase-A evaluations and calculate fixed factorial contrasts."""

import argparse
from importlib import import_module
from pathlib import Path

from scripts.baseline_data import read_json, write_json
from scripts.baseline_metrics import plot_pr
from scripts.campaign import EXP009_A, REPO, attempt_dir, condition_config, graph, yolo_smoke_recipe
from scripts.campaign_analysis.analyze import (
    archive, attempt_snapshots, evaluate_condition, load_dataset, verify_provenance,
    verify_runtime, write_error_outputs,
)
from scripts.campaign_analysis.import_results import require, sha256, verify_inventory
from scripts.campaign_analysis.reporting import SENSITIVITY, save, write_csv

staged = import_module("experiments.exp-007-yolo-staged-transfer.analyze")
DURATIONS = (0, 10, 20, 40, 60, 80, 100)
CELLS = ("control", "geom")
METRICS = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall")


def name(duration, geometry):
    return f"f{duration:03d}-{geometry}"


def by_job(root, ledger, node, job):
    entry, = [item for item in ledger[node] if item["id"] == job]
    return attempt_dir(root, node, entry["attempt"])


def verify_freezing(folder, history, spec):
    duration = spec.get("freeze_backbone_epochs", 0)
    path = folder / "staged_transfer.json"
    require(path.is_file() == bool(duration), f"Auditoría por etapas inesperada: {folder}")
    audit = read_json(path) if duration else None
    runtime, inventory = (read_json(folder / file) for file in ("ultralytics_runtime.json", "parameters.json"))
    staged.verify_stages(audit, inventory, runtime, history, duration)
    return audit


def verify_smoke(folder, manifest, snapshot, plan, job_id, n_train):
    for arm in ("smoke-temporal", "smoke-permanent"):
        training = folder / arm
        run, status = (read_json(training / file) for file in ("run.json", "status.json"))
        verify_provenance(run, "yolo", job_id, snapshot, plan)
        spec = yolo_smoke_recipe(manifest, arm)
        require(run["mode"] == "train" and run["run"]["config"] == spec and status["status"] == "complete"
                and run["run"]["dataset_manifest_sha256"] == plan["dataset_manifest_sha256"]
                and run["run"]["initial_weights_sha256"] == plan["weights"]["yolo"]["sha256"],
                f"Receta o entradas del smoke: {arm}")
        stride, _ = verify_runtime(training, "yolo", spec, n_train)
        history, _ = archive.read_history(training / "history.csv", stride, spec["epochs"], status)
        verify_freezing(training, history, spec)


def factorial(values, fields):
    for duration in DURATIONS:
        first, geometry = (values[name(duration, cell)] for cell in CELLS)
        baseline, baseline_geometry = (values[name(0, cell)] for cell in CELLS)
        for metric in fields:
            delta_geom = geometry[metric] - first[metric]
            yield {"freeze_epochs": duration, "metric": metric, "geometry_delta": delta_geom,
                   "freeze_delta_control": first[metric] - baseline[metric],
                   "freeze_delta_geometry": geometry[metric] - baseline_geometry[metric],
                   "interaction": delta_geom - (baseline_geometry[metric] - baseline[metric])}


def select_duration(values):
    """Nominate a positive freezing level vs. the contemporaneous zero in the same cell."""
    candidates = []
    for duration in DURATIONS[1:]:
        for cell in CELLS:
            variant, control = values[name(duration, cell)], values[name(0, cell)]
            gain = variant["map"] - control["map"]
            sensitivity = variant[SENSITIVITY] - control[SENSITIVITY]
            if gain >= .01 and sensitivity >= 0:
                candidates.append({"duration": duration, "cell": cell, "delta_map": gain, "delta_map7": sensitivity,
                                   "map": variant["map"], "map7": variant[SENSITIVITY]})
    return max(candidates, key=lambda row: (row["map"], row["map7"], -row["duration"], row["cell"] == "control"),
               default=None)


def analyze(root, dataset, output, expected_snapshot):
    verify_inventory(root)
    manifest = read_json(root / "manifest.json")
    ledger, states = (read_json(root / file) for file in ("jobs.json", "slurm-states.json"))
    plan = read_json(REPO / EXP009_A)
    snapshots = attempt_snapshots(root, manifest, ledger)
    require(manifest["plan"] == plan and manifest["nodes"] == graph(plan)
            and set(snapshots) == {expected_snapshot}
            and snapshots[expected_snapshot]["sha256"][EXP009_A.as_posix()] == sha256(REPO / EXP009_A),
            "Manifiesto o snapshot incorrecto para EXP-009 A")
    require([(r["name"], r["epochs"]) for r in plan["conditions"]]
            == [(name(d, c), 100) for d in DURATIONS for c in CELLS], "La matriz no contiene los 14 brazos fijados")
    truth, splits = load_dataset(dataset, root, plan)
    results, checked_smokes = {}, set()
    for condition in plan["conditions"]:
        arm = condition["name"]
        result = evaluate_condition(root, manifest, snapshots, ledger, states, condition, truth, splits)
        require(result is not None, f"Evaluación pendiente: {arm}")
        audit = result["audit"]
        smoke = audit["preflight_job"]
        if smoke not in checked_smokes:
            verify_smoke(by_job(root, ledger, "preflight-yolo", smoke), manifest, snapshots[expected_snapshot],
                         plan, smoke, splits["train"]["images"])
            checked_smokes.add(smoke)
        train = by_job(root, ledger, f"train-{arm}", audit["training_job"]) / "training"
        runtime = read_json(train / "ultralytics_runtime.json")
        spec = condition_config(plan, condition)
        require(runtime["iterations_per_epoch"] == 40 and runtime["warmup_iterations_end_index"] == 100,
                f"Lotes o warm-up efectivo incorrecto: {arm}")
        result["staging"] = verify_freezing(train, result["history"], spec)
        status = read_json(train / "status.json")
        result["resources"] = {"training_seconds": status["total_elapsed_seconds"],
                               "peak_gpu_allocated_mib": status["cuda_peak_allocated_bytes"] / 2**20,
                               "peak_gpu_reserved_mib": max(s["cuda_peak_reserved_bytes"] for s in status["sessions"]) / 2**20}
        results[arm] = result
    reference = results[name(0, CELLS[0])]["effective"]
    require(all(r["effective"] == reference for r in results.values()), "Recetas efectivas no comparables")
    write_outputs(output, results, truth, dataset / "valid", manifest, expected_snapshot, checked_smokes)


def write_outputs(output, results, truth, images, manifest, snapshot, checked_smokes):
    output.mkdir(parents=True, exist_ok=False)
    fields = (*METRICS, *(category["name"] for category in truth["categories"]))
    values, summary, all_history, windows, classes = {}, [], [], [], []
    for arm, result in results.items():
        duration = manifest["recipes"][arm].get("freeze_backbone_epochs", 0)
        values[arm] = {**result["values"], **{c["name"]: c["ap"] for c in result["classes"]}}
        summary.append({"arm": arm, "freeze_epochs": duration, "geometry": arm.endswith("-geom"),
                        **result["values"], **result["benchmark"], **result["resources"]})
        classes.extend({"arm": arm, **row} for row in result["classes"])
        all_history.extend({"arm": arm, **row} for row in result["history"])
        if result["staging"] and result["staging"]["thaw_epoch"]:
            selected = range(max(1, duration - 1), min(100, duration + 3) + 1)
            windows.extend({"arm": arm, "relative_epoch": int(row["step"]) - duration,
                            **row} for row in result["history"] if int(row["step"]) in selected)
        plot_pr(result["pr_curves"], output / f"pr-{arm}.png")
    write_csv(output / "resumen.csv", summary)
    write_csv(output / "por_clase.csv", classes)
    write_csv(output / "factorial.csv", factorial(values, fields))
    for filename, rows in (("historial", all_history), ("transiciones", windows)):
        write_csv(output / f"{filename}.csv", rows, tuple(dict.fromkeys(key for r in rows for key in r)))
    write_error_outputs(results, truth, images, output)
    nomination = select_duration(values)
    reused = {name(d, c): {"config": manifest["recipes"][name(d, c)], **results[name(d, c)]["audit"]}
              for d in (0, nomination["duration"]) for c in CELLS} if nomination else {}
    write_json(output / "auditoria.json", {"snapshot_sha256": snapshot, "preflights": sorted(checked_smokes),
                                         "conditions": {arm: {**r["audit"], "benchmark_protocol": r["protocol"],
                                             "thaw_epoch": r["staging"]["thaw_epoch"] if r["staging"] else None,
                                             "freezing_verified": r["staging"] is not None}
                                                        for arm, r in results.items()}})
    write_json(output / "fase-b-propuesta.json", {"snapshot_sha256": snapshot, "nomination": nomination,
                                                 "reuse_candidates": reused})
    plot_outputs(output, results)
    table = ["# EXP-009 A · Validación reproducida", "",
             "| Brazo | mAP 8 clases | mAP 7 clases | Época best |", "| --- | ---: | ---: | ---: |"]
    for r in summary:
        table.append(f"| {r['arm']} | {r['map']:.6f} | {r[SENSITIVITY]:.6f} | {r['best_epoch']} |")
    table += ["", f"Propuesta para B · freeze-{nomination['duration']}." if nomination
              else "Ninguna duración supera el criterio de activación de B.",
              "", "Exploración con semilla 2026. Las latencias describen asignaciones independientes.", ""]
    (output / "findings.md").write_text("\n".join(table), encoding="utf-8")


def plot_outputs(output, results):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, metric, title in zip(axes, ("map", SENSITIVITY), ("Ocho clases", "Sin Ibuphil 600 mg"), strict=True):
        for cell, label in zip(CELLS, ("Sin augmentación", "Afín + Mosaic"), strict=True):
            ax.plot(DURATIONS, [100 * results[name(d, cell)]["values"][metric] for d in DURATIONS], "o-", label=label)
        ax.set(title=title, xlabel="Épocas congeladas", xticks=DURATIONS)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("mAP COCO (%)")
    axes[0].legend()
    save(fig, output / "duracion.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True)
    for column, cell in enumerate(CELLS):
        for d in DURATIONS:
            history = results[name(d, cell)]["history"]
            thaw = d + 1 if 0 < d < 100 else None
            map_line, = axes[0, column].plot([r["step"] for r in history], [100*r["val/map"] for r in history],
                                             label=f"Freeze {d}")
            if thaw:
                axes[0, column].scatter(thaw, 100 * history[thaw - 1]["val/map"], color=map_line.get_color(), s=12)
            axes[1, column].plot([r["step"] for r in history], [r["train/cls_loss"] for r in history],
                                 color=map_line.get_color())
        axes[0, column].set(title=cell, ylabel="mAP COCO (%)")
        axes[1, column].set(ylabel="Pérdida de clasificación", xlabel="Época")
        axes[0, column].legend(ncol=2, fontsize=8)
        for ax in axes[:, column]:
            ax.grid(alpha=.25)
    fig.suptitle("Historial de validación y entrenamiento · puntos en la primera época descongelada")
    save(fig, output / "evolucion.png")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("imported", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht")
    cli.add_argument("--snapshot-sha256", required=True)
    args = cli.parse_args()
    analyze(args.imported, args.data, args.output, args.snapshot_sha256)
