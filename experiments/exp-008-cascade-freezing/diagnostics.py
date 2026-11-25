"""Compare verified EXP-008 arms, transition windows and archived Cascade references."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from tarfile import open as open_tar

from scripts.baseline_data import read_json
from scripts.campaign import EXP008, REPO
from scripts.campaign_analysis.analyze import load_dataset
from scripts.campaign_analysis.import_results import require, sha256
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, save, write_csv


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def historical_references(plan):
    baseline = REPO / "experiments/baseline/results/cascade/evaluation/metrics.json"
    exp002 = REPO / "experiments/exp-002-photometric/results/s2026/cascade/control/evaluation/metrics.json"
    exp005 = REPO / "experiments/exp-004-yolo-factorial/results/evidence/campaign-003-005-results-recovered.tar.gz"
    references = [("Baseline histórica", 30, read_json(baseline)), ("EXP-002 control", 30, read_json(exp002))]
    with open_tar(exp005) as archive:
        for arm, budget in zip("abcd", (30, 100, 30, 100), strict=True):
            member = f"jobs/eval-exp005-{arm}/attempt-01/evaluation/metrics.json"
            references.append((f"EXP-005 {arm.upper()}", budget, json.load(archive.extractfile(member))))
    for name, budget, metrics in references:
        require((metrics["backend"], metrics["split"], metrics["coco_sha256"], metrics["dataset_manifest_sha256"])
                == ("cascade", "valid", plan["valid_coco_sha256"], plan["dataset_manifest_sha256"]),
                f"Protocolo de referencia distinto: {name}")
        by_class = {r["name"]: r["ap"] for r in metrics["per_class"]}
        yield {"reference": name, "budget": budget, "map": metrics["map"], "ap50": metrics["ap50"],
               "ap75": metrics["ap75"], SENSITIVITY: mean(ap for k, ap in by_class.items() if k != RARE_CLASS),
               "ap_ibuphil_600_mg": by_class[RARE_CLASS], **metrics["operating_point"]}


def rare_rankings(imported, audit, truth):
    from pycocotools.mask import iou

    category, = [c["id"] for c in truth["categories"] if c["name"] == RARE_CLASS]
    rare, = [ann for ann in truth["annotations"] if ann["category_id"] == category]
    ledger = read_json(imported / "jobs.json")
    for arm, info in audit["conditions"].items():
        name = f"eval-{arm}"
        record, = [row for row in ledger[name] if row["id"] == info["evaluation_job"]]
        path = imported / "jobs" / name / f"attempt-{record['attempt']:02d}" / "evaluation/predictions.json"
        predictions = [row for row in read_json(path) if row["category_id"] == category]
        ranked = sorted(enumerate(predictions), key=lambda pair: (-pair[1]["score"], pair[1]["image_id"], pair[0]))
        for rank, (_, detection) in enumerate(ranked, start=1):
            overlap = (float(iou([detection["bbox"]], [rare["bbox"]], [0])[0, 0])
                       if detection["image_id"] == rare["image_id"] else 0)
            if rank == 1 or overlap >= .5:
                yield {"arm": arm, "rank": rank, "image_id": detection["image_id"], "score": detection["score"],
                       "iou_to_ibuphil": overlap}


def diagnostics(imported, analysis, dataset):
    from matplotlib import pyplot as plt
    import numpy as np

    plan = read_json(REPO / EXP008)
    audit = read_json(analysis / "auditoria.json")
    require(audit["snapshot_sha256"] == sha256(imported / "snapshot.json"),
            "El análisis y la importación corresponden a snapshots distintos")
    truth, _ = load_dataset(dataset, imported, plan)
    target = analysis / "diagnosticos"
    target.mkdir(exist_ok=False)
    summary = rows(analysis / "resumen.csv")
    by_class = rows(analysis / "por_clase.csv")
    histories = {row["arm"]: {} for row in summary}
    for row in rows(analysis / "historial.csv"):
        if row["val/map"]:
            histories[row["arm"]][int(float(row["epoch"]))] = float(row["val/map"])
    write_csv(target / "referencias_cascade.csv", historical_references(plan))
    write_csv(target / "ranking_ibuphil.csv", rare_rankings(imported, audit, truth))

    contrast = []
    for name, info in audit["freezing"].items():
        transition = info["transition"]
        if transition is None:
            continue
        values = histories[name]
        n = info["freeze_epochs"]
        budget = len(values)
        permanent = histories[f"e{budget:03d}-f{budget:03d}"]
        best_pre = max(range(1, n + 1), key=values.get)
        best_post = max(range(n + 1, budget + 1), key=values.get)
        contrast.append({"arm": name, "budget": budget, "freeze": n, "lr_at_thaw": transition["lr"],
                         "before": values[n], "after_1": values[n + 1], "after_2": values[n + 2],
                         "change_first_epoch": values[n + 1] - values[n],
                         "change_first_epoch_permanent": permanent[n + 1] - permanent[n],
                         "best_pre_epoch": best_pre, "best_pre_map": values[best_pre],
                         "best_post_epoch": best_post, "best_post_map": values[best_post],
                         "num_identical_pre_epochs_vs_permanent": sum(values[e] == permanent[e] for e in range(1, n + 1))})
    write_csv(target / "diagnostico_descongelacion.csv", contrast)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharey=True)
    for ax, budget in zip(axes, (30, 100), strict=True):
        selected = [r for r in summary if int(r["budget"]) == budget]
        x = [int(r["freeze_epochs"]) for r in selected]
        for metric, color, name in (("map", "#174c80", "mAP oficial"), (SENSITIVITY, "#b05c12", "Sin Ibuphil 600 mg")):
            ax.plot(x, [100 * float(r[metric]) for r in selected], "o", color=color, label=name)
        ax.set(title=f"{budget} épocas totales", xlabel="Épocas congeladas", xticks=x)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("mAP COCO (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Freezing de Cascade · Rendimiento del checkpoint best")
    save(fig, target / "map_por_presupuesto.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained", sharey=True)
    for ax, row in zip(axes.flat, contrast, strict=True):
        n = row["freeze"]
        perm = f"e{row['budget']:03d}-f{row['budget']:03d}"
        for name, color, label in ((row["arm"], "#174c80", "Temporal"), (perm, "#b05c12", "Permanente")):
            ax.plot([-1, 0, 1, 2], [100 * histories[name][e] for e in range(n - 1, n + 3)], "o-", color=color, label=label)
        ax.axvline(.5, linestyle="--", linewidth=1, color="#666666")
        ax.set(xticks=[-1, 0, 1, 2], xlabel="Épocas respecto al final de la congelación temporal",
               ylabel="mAP valid (%)", title=f"{row['budget']} épocas, freeze-{n}")
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Validación antes y después de descongelar ResNet")
    save(fig, target / "ventana_descongelacion.png")

    arms = [r["arm"] for r in summary if int(r["freeze_epochs"])]
    classes = list(dict.fromkeys(r["name"] for r in by_class))
    index = {(r["arm"], r["name"]): float(r["ap_minus_control"]) for r in by_class}
    changes = np.array([[100 * index[arm, cls] for arm in arms] for cls in classes])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    heat = ax.imshow(changes, cmap="RdBu", vmin=-8, vmax=8, aspect="auto")
    ax.set(xticks=range(len(arms)), xticklabels=arms, yticks=range(len(classes)), yticklabels=classes,
           title="Diferencia de AP frente al control emparejado (pp)")
    ax.axvline(2.5, color="white", linewidth=2)
    for (y, x), value in np.ndenumerate(changes):
        ax.text(x, y, f"{value:+.1f}", ha="center", va="center", color="white" if abs(value) >= 5.5 else "#222222")
    fig.colorbar(heat, ax=ax, extend="both", label="Δ AP (pp, color acotado a ±8)")
    save(fig, target / "ap_delta_por_clase.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht")
    args = parser.parse_args()
    diagnostics(args.imported, args.analysis, args.data)
