"""Derived phase-A figures and exact comparisons of shared training prefixes."""

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scripts.baseline_data import read_json, sha256
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, save, write_csv


def table(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def diagnose(imported, analysis):
    import numpy as np
    from matplotlib import pyplot as plt

    audit = read_json(analysis / "auditoria.json")
    require_snapshot = sha256(imported / "snapshot.json")
    if audit["snapshot_sha256"] != require_snapshot:
        raise ValueError("La importación y el análisis pertenecen a snapshots distintos")
    arms = {r["arm"]: r for r in table(analysis / "resumen.csv")}
    ledger = read_json(imported / "jobs.json")
    target = analysis / "diagnosticos"
    target.mkdir(exist_ok=False)
    histories = defaultdict(dict)
    for r in table(analysis / "historial.csv"):
        histories[r["arm"]][int(float(r["step"]))] = {k: float(v) for k, v in r.items()
                                                    if k not in {"arm", "step", "time"} and v != ""}
    prefixes, windows, predictions = [], [], []
    for arm, row in arms.items():
        node = f"eval-{arm}"
        job = audit["conditions"][arm]["evaluation_job"]
        entry, = (e for e in ledger[node] if e["id"] == job)
        path = imported / "jobs" / node / f"attempt-{entry['attempt']:02d}" / "evaluation"
        predictions.append({"arm": arm, "best_epoch": row["best_epoch"], "evaluation_job": job,
                            "checkpoint_sha256": audit["conditions"][arm]["checkpoint_sha256"],
                            "predictions_sha256": sha256(path / "predictions.json"),
                            "pr_curves_sha256": sha256(path / "pr_curves.json")})
    cells = ("control", "geom")
    for cell in cells:
        names = [arm for arm, row in arms.items() if arm.endswith(f"-{cell}") and int(row["freeze_epochs"])]
        for a, b in combinations(names, 2):
            prefix = min(int(arms[a]["freeze_epochs"]), int(arms[b]["freeze_epochs"]))
            equal = sum(histories[a][e] == histories[b][e] for e in range(1, prefix + 1))
            prefixes.append({"recipe": cell, "arm_a": a, "arm_b": b, "frozen_common_epochs": prefix,
                             "epochs_equal_excluding_walltime": equal})
        for arm in names:
            n = int(arms[arm]["freeze_epochs"])
            if n == 100:
                continue
            values = histories[arm]
            frozen = histories[f"f100-{cell}"]
            best_pre = max(range(1, n + 1), key=lambda e: values[e]["val/map"])
            best_post = max(range(n + 1, 101), key=lambda e: values[e]["val/map"])
            windows.append({"arm": arm, "first_unfrozen_epoch": n + 1,
                            "map_last_frozen": values[n]["val/map"], "map_first_unfrozen": values[n+1]["val/map"],
                            "map_second_unfrozen": values[n+2]["val/map"], "map_permanent_same_epoch": frozen[n+1]["val/map"],
                            "lr_first_unfrozen": values[n+1]["lr/pg0"],
                            "val_cls_loss_last_frozen": values[n].get("val/cls_loss"),
                            "val_cls_loss_first_unfrozen": values[n+1].get("val/cls_loss"),
                            "best_pre_epoch": best_pre, "best_pre_map": values[best_pre]["val/map"],
                            "best_post_epoch": best_post, "best_post_map": values[best_post]["val/map"]})
    for filename, data in (("prefijos", prefixes), ("predicciones_best", predictions), ("descongelacion", windows)):
        write_csv(target / f"{filename}.csv", data)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained", sharey=True)
    rows = table(analysis / "factorial.csv")
    for ax, column, title in zip(axes, ("geometry_delta", "interaction"),
                                 ("Efecto afín + Mosaic", "Interacción frente a freeze-0"), strict=True):
        for field, color, label in (("map", "#174c80", "Ocho clases"), (SENSITIVITY, "#b05c12", "Sin Ibuphil 600 mg")):
            selected = [r for r in rows if r["metric"] == field]
            x = [int(r["freeze_epochs"]) for r in selected]
            ax.plot(x, [100 * float(r[column]) for r in selected], "o-", color=color, label=label)
        ax.axhline(0, color="#686868", linewidth=.7)
        ax.set(xticks=x, title=title, xlabel="Épocas congeladas")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Diferencia de mAP (pp)")
    axes[0].legend(loc="best")
    save(fig, target / "interaccion.png")

    by_class = table(analysis / "por_clase.csv")
    names = list(dict.fromkeys(r["name"] for r in by_class))
    ap = {(r["arm"], r["name"]): float(r["ap"]) for r in by_class}
    matrices = []
    for cell in cells:
        selected = [arm for arm in arms if arm.endswith(f"-{cell}") and arm != f"f000-{cell}"]
        matrices.append(np.array([[100 * (ap[arm, cls] - ap[f"f000-{cell}", cls]) for arm in selected] for cls in names]))
    limit = max(1, max(abs(data).max() for data in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, values, label in zip(axes, matrices, ("Sin augmentación", "Afín + Mosaic"), strict=True):
        heat = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xticks=range(6), xticklabels=[10, 20, 40, 60, 80, 100],
               yticks=range(len(names)), yticklabels=names, title=label, xlabel="Épocas congeladas")
        ax.axhline(names.index(RARE_CLASS) + .5, color="white", linewidth=1)
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > .65 * limit else "#222222")
    fig.colorbar(heat, ax=axes, shrink=.85, label="Δ AP frente a freeze-0 (pp)")
    save(fig, target / "ap_por_clase_delta.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharey=True)
    for ax, cell, title in zip(axes, cells, ("Sin augmentación", "Afín + Mosaic"), strict=True):
        for row in windows:
            if not row["arm"].endswith(f"-{cell}"):
                continue
            n = row["first_unfrozen_epoch"] - 1
            ax.plot(range(-1, 4), [100 * histories[row["arm"]][e]["val/map"] for e in range(n-1, n+4)],
                    "o-", label=f"Freeze {n}")
        ax.axvline(.5, color="#787878", linestyle="--", linewidth=.9)
        ax.set(xticks=range(-1, 4), xlabel="Épocas respecto al final de la congelación", title=title)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("mAP de validación (%)")
    axes[1].legend(fontsize=8, ncols=2)
    save(fig, target / "ventana_descongelacion.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for cell, label in zip(cells, ("Sin augmentación", "Afín + Mosaic"), strict=True):
        selected = [row for row in arms.values() if row["arm"].endswith(f"-{cell}")]
        for row, x, y in zip(selected, [float(r["training_seconds"]) / 60 for r in selected],
                             [100 * float(r["map"]) for r in selected], strict=True):
            ax.annotate(row["freeze_epochs"], (x, y), xytext=(5, 2), textcoords="offset points", fontsize=8)
        ax.scatter([float(r["training_seconds"]) / 60 for r in selected],
                   [100 * float(r["map"]) for r in selected], label=label)
    ax.set(xlabel="Tiempo interno de entrenamiento (min)", ylabel="mAP COCO oficial (%)",
           title="Coste y rendimiento por duración de la congelación")
    ax.legend(loc="lower left")
    ax.grid(alpha=.25)
    save(fig, target / "coste_gpu.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    diagnose(args.imported, args.analysis)
