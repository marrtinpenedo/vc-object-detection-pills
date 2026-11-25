"""Reproduce EXP-008 validation from the selective export, including the freezing audit."""

import argparse
import re
from pathlib import Path

from scripts.baseline_data import read_json, write_json
from scripts.baseline_metrics import plot_pr
from scripts.campaign import EXP008, REPO, attempt_dir, graph
from scripts.campaign_analysis.analyze import attempt_snapshots, evaluate_condition, load_dataset, verify_provenance
from scripts.campaign_analysis.import_results import require, sha256, verify_inventory
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, plot_history, write_csv


def verify_freezing(record, freeze, epochs, stride):
    expected = freeze * stride + 1 if freeze < epochs else None
    require((record["freeze_epochs"], record["steps_per_epoch"], record["first_unfrozen_step"],
             record["frozen_steps"], record["normalization"]) ==
            (freeze, stride, expected, freeze * stride, "FrozenBatchNorm2d"), "Presupuesto de congelación")
    initial = record["initial_weights_sha256"]
    buffers = record["initial_buffers_sha256"]
    require(record["backbone_parameters"] and record["backbone_scalars"] > 0
            and all(re.fullmatch("[0-9a-f]{64}", digest) for digest in (initial, buffers)), "Huella inicial de ResNet")
    first = record["first_step"]
    transition = record["transition"]
    require(first["step"] == 1 and first["epoch"] == 1 and first["weights_sha256"] == initial
            and first["backbone_momentum_buffers"] == 0 < first["other_momentum_after"]
            and first["scheduler_after"] == first["scheduler_before"] + 1, "Primer paso congelado")
    if expected is None:
        require(transition is None, "Una variante permanente se descongeló")
    else:
        require(transition["step"] == expected and transition["epoch"] == freeze + 1
                and transition["weights_sha256"] != initial and transition["backbone_momentum_buffers"] > 0
                and transition["scheduler_before"] - first["scheduler_before"] == freeze * stride
                and transition["scheduler_after"] == transition["scheduler_before"] + 1
                and transition["other_momentum_after"] >= transition["other_momentum_before"] > 0,
                "ResNet no se actualizó en el primer paso descongelado")
    require(len(record["epochs"]) == epochs, "Auditoría de épocas incompleta")
    for epoch, row in enumerate(record["epochs"], 1):
        require((row["epoch"], row["step"], row["frozen"], row["buffers_sha256"]) ==
                (epoch, epoch * stride, epoch <= freeze, buffers), "Fase o buffers de una época")
        require(re.fullmatch("[0-9a-f]{64}", row["weights_sha256"])
                and (epoch > freeze or row["weights_sha256"] == initial), "ResNet congelada cambió de pesos")


def by_job(root, ledger, name, job):
    entry = next(entry for entry in ledger[name] if entry["id"] == job)
    return attempt_dir(root, name, entry["attempt"])


def analyze(root, dataset, output, expected_snapshot):
    verify_inventory(root)
    manifest = read_json(root / "manifest.json")
    ledger, states = (read_json(root / name) for name in ("jobs.json", "slurm-states.json"))
    snapshots = attempt_snapshots(root, manifest, ledger)
    plan = read_json(REPO / EXP008)
    require(manifest["plan"] == plan and manifest["nodes"] == graph(plan)
            and set(snapshots) == {expected_snapshot}
            and snapshots[expected_snapshot]["sha256"][EXP008.as_posix()] == sha256(REPO / EXP008),
            "Manifiesto o snapshot EXP-008 incompatible")
    truth, splits = load_dataset(dataset, root, plan)
    conditions = plan["conditions"]
    results, freezing = {}, {}
    checked_smoke = set()
    for condition in conditions:
        name = condition["name"]
        result = evaluate_condition(root, manifest, snapshots, ledger, states, condition, truth, splits)
        require(result is not None, f"Evaluación pendiente: {name}")
        audit = result["audit"]
        if audit["preflight_job"] not in checked_smoke:
            smoke = by_job(root, ledger, "preflight-cascade", audit["preflight_job"]) / "smoke"
            require(read_json(smoke / "smoke.json") ==
                    {"steps": 3, "transition_step": 2, "dataset_split": "train", "status": "complete"},
                    "Preflight nativo de transición incompleto")
            native = read_json(smoke / "run.json")
            verify_provenance(native, "cascade", audit["preflight_job"], snapshots[expected_snapshot], plan)
            require(native["weights_sha256"] == plan["weights"]["cascade"]["sha256"], "Pesos del smoke no coinciden")
            verify_freezing(read_json(smoke / "cascade_freezing.json"), 1, 3, 1)
            checked_smoke.add(audit["preflight_job"])
        training = by_job(root, ledger, f"train-{name}", audit["training_job"]) / "training"
        status = read_json(training / "status.json")
        result["resources"] = {"training_seconds": status["total_elapsed_seconds"],
                               "peak_gpu_allocated_mib": status["cuda_peak_allocated_bytes"] / 2**20,
                               "peak_gpu_reserved_mib": max(s["cuda_peak_reserved_bytes"] for s in status["sessions"]) / 2**20}
        freeze = condition.get("freeze_backbone_epochs", 0)
        if freeze:
            record = read_json(training / "cascade_freezing.json")
            verify_freezing(record, freeze, condition["epochs"], result["stride"])
            freezing[name] = record
        else:
            require(not (training / "cascade_freezing.json").exists(), "El control recibió freezing")
        results[name] = result
    write_outputs(output, conditions, results, freezing, checked_smoke, expected_snapshot)


def write_outputs(output, conditions, results, freezing, checked_smoke, snapshot):
    contrast_metrics = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall", "tp", "fp", "fn")
    summary, classes, deltas, around, history, candidates = [], [], [], [], [], []
    output.mkdir(parents=True, exist_ok=False)
    for budget in (30, 100):
        rows = [c for c in conditions if c["epochs"] == budget]
        control = results[f"e{budget:03d}-f000"]
        plot_rows = {}
        ranked = []
        for letter, condition in zip("abcd", rows, strict=True):
            name = condition["name"]
            freeze = condition.get("freeze_backbone_epochs", 0)
            result = results[name]
            require(result["effective"] == control["effective"], f"Configuración comparable: {name}")
            plot_rows[letter] = result
            summary.append({"arm": name, "budget": budget, "freeze_epochs": freeze,
                            **result["values"], **result["benchmark"], **result["resources"]})
            history.extend({"arm": name, "epoch": r["step"] / result["stride"], **r}
                           for r in result["history"])
            by_class = {c["name"]: c["ap"] for c in control["classes"]}
            classes.extend({"arm": name, **c, "ap_minus_control": c["ap"] - by_class[c["name"]]}
                           for c in result["classes"])
            plot_pr(result["pr_curves"], output / f"pr-{name}.png")
            if not freeze:
                continue
            delta = {metric: result["values"][metric] - control["values"][metric] for metric in contrast_metrics}
            rare = next(c["ap"] for c in result["classes"] if c["name"] == RARE_CLASS)
            if delta["map"] >= 0.01 and delta[SENSITIVITY] >= 0:
                ranked.append((delta["map"], delta[SENSITIVITY], name))
            deltas.append({"arm": name, "budget": budget, **delta,
                           "rare_contribution_to_map_delta": (rare - by_class[RARE_CLASS]) / len(by_class)})
            transition = freezing[name]["transition"]
            if transition:
                scored = {int(row["step"]) // result["stride"]: row for row in result["history"] if "val/map" in row}
                for epoch in range(max(1, freeze - 1), min(budget, freeze + 2) + 1):
                    selected = scored[epoch]
                    around.append({"arm": name, "epoch": epoch, "phase": "frozen" if epoch <= freeze else "unfrozen",
                                   "first_unfrozen_step": transition["step"], "lr_at_thaw": transition["lr"],
                                   **{key: selected.get(key) for key in ("val/map", "val/ap50", "total_loss", "lr")}})
        candidates.append({"budget": budget, "candidate": max(ranked)[2] if ranked else None})
        labels = {letter: condition["name"] for letter, condition in zip(plot_rows, rows, strict=True)}
        plot_history(plot_rows, "cascade", output / f"entrenamiento-{budget}.png", labels=labels)
    for name, rows in (("resumen", summary), ("por_clase", classes), ("contrastes", deltas), ("transicion", around)):
        write_csv(output / f"{name}.csv", rows)
    write_csv(output / "historial.csv", history, ("arm", "epoch", *sorted({k for r in history for k in r} - {"arm", "epoch"})))
    write_json(output / "auditoria.json", {"snapshot_sha256": snapshot, "smoke_jobs": sorted(checked_smoke),
                                         "conditions": {k: v["audit"] for k, v in results.items()},
                                         "freezing": freezing, "candidates": candidates})
    lines = ["# EXP-008 · Exploración en validación", "",
             "| Brazo | mAP oficial | mAP sin Ibuphil | Mejor época |", "| --- | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['arm']} | {row['map']:.6f} | {row[SENSITIVITY]:.6f} | {row['best_epoch']} |")
    lines.extend(["", "Candidatos para confirmar con 2027 y 2028"])
    for choice in candidates:
        lines.append(f"- {choice['budget']} épocas · {choice['candidate'] or 'Ningún brazo cumple el criterio'}.")
    lines.extend(["", "Una sola semilla en 90 imágenes de validación. El mAP de siete clases es una sensibilidad.", ""])
    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht")
    parser.add_argument("--snapshot-sha256", required=True)
    args = parser.parse_args()
    analyze(args.imported, args.data, args.output, args.snapshot_sha256)
