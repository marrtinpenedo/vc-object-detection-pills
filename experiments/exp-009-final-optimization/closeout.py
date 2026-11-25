"""Compare validation cohorts and record checkpoints proposed for the final evaluation."""
import argparse
import csv
from importlib import import_module
from pathlib import Path

from scripts.baseline_data import read_json, sha256, write_json
from scripts.campaign_analysis.import_results import require, verify_inventory
from scripts.campaign_analysis.reporting import SENSITIVITY, save, write_csv

confirmation = import_module("experiments.exp-009-final-optimization.confirmation")
METRICS = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall", "mean_ms_per_image", "p95_ms_per_image")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def figures(output, contrasts, comparison):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained", sharey=True)
    labels = ["F−D · Fotometría, con geometría", "E−C · Fotometría, sin geometría", "D−B · Freezing, con geometría"]
    for ax, metric, title in zip(axes, ("map", SENSITIVITY), ("mAP COCO oficial", "mAP sin Ibuphil 600 mg")):
        ax.axvline(0, lw=.7, color="#252525")
        rows = [r for r in contrasts if r["metric"] == metric and r["contrast"] in ("F-D", "E-C", "D-B")]
        for seed, offset, style, color in ((2026, -.18, "o", "#797979"), (2027, 0, "s", "#174c80"),
                                           (2028, .18, "^", "#b05c12")):
            ax.scatter([100 * float(r[f"delta_{seed}"]) for r in rows], [i + offset for i in range(3)],
                       marker=style, color=color, s=48, label=f"{seed} · exploración" if seed == 2026 else str(seed))
        ax.set(yticks=range(3), yticklabels=labels, title=title, xlabel="Diferencia emparejada (pp)", xlim=(-9, 9))
        ax.grid(axis="x", alpha=.16)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="best", fontsize=8)
    save(fig, output / "contrastes_prioritarios.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained", sharey=True)
    for ax, metric, title in zip(axes, ("map", SENSITIVITY), ("mAP COCO oficial", "mAP sin Ibuphil 600 mg")):
        for y, backend, color in ((0, "yolo", "#174c80"), (1, "cascade", "#b05c12")):
            cohort = [r for r in comparison if r["backend"] == backend]
            ax.scatter([100 * r[metric] for r in cohort], [y]*3, color=color, s=35)
            chosen, = [r for r in cohort if r["checkpoint_propuesto"]]
            ax.scatter(100 * chosen[metric], y, facecolors="none", edgecolors="#161616", s=140, linewidths=1.3)
        ax.set(yticks=(0, 1), yticklabels=("YOLOv10-S", "Cascade R-CNN"), xlim=(79, 92), title=title, xlabel="Validación (%)")
        ax.grid(axis="x", alpha=.16)
    fig.suptitle("Tres semillas por receta · círculo exterior marca el checkpoint propuesto")
    save(fig, output / "comparacion_validacion.png")


def closeout(c_report, c_import, d_report, d_import, output):
    c_audit = read_json(c_report / "auditoria.json")
    yolo = read_json(c_report / "candidato-yolo-valid.json")
    yolo_report = read_csv(c_report / "por_semilla.csv")
    d_rows = read_csv(d_report / "resumen.csv")
    as_values = lambda row: {key: float(row[key]) for key in METRICS}
    values = {(row["arm"], int(row["seed"])): as_values(row) for row in yolo_report}
    arm, seed, eligible = confirmation.choose(values)
    certified = c_audit["slurm_accounting"] is not None
    require((arm, seed, eligible) == (yolo["arm"], yolo["seed"], yolo["eligible_arms"])
            and (yolo["status"] == "propuesto_para_congelar") == certified,
            "La selección YOLO no concuerda con sus contrastes o su auditoría")
    require({int(row["seed"]) for row in d_rows} == set(confirmation.SEEDS) and len(d_rows) == 3,
            "Faltan las tres semillas Cascade")
    cascade_row = sorted(d_rows, key=lambda r: (float(r["map"]), float(r[SENSITIVITY]), int(r["seed"])))[1]
    verify_inventory(c_import)
    verify_inventory(d_import)
    dc = read_json(d_import / "manifest.json")
    cascade_arm = cascade_row["arm"]
    cascade_job = read_json(d_report / "auditoria.json")["conditions"][cascade_arm]
    ledger = read_json(d_import / "jobs.json")
    evaluation = confirmation.followup.phase_a.by_job(d_import, ledger, f"eval-{cascade_arm}", cascade_job["evaluation_job"])
    cascade = read_json(evaluation / "evaluation/metrics.json")
    require(dc["recipes"][cascade_arm] == cascade["model_config"] and cascade["map"] == float(cascade_row["map"])
            and cascade["checkpoint_sha256"] == cascade_job["checkpoint_sha256"] and cascade["split"] == "valid",
            "La propuesta Cascade difiere de la evaluación auditada")
    cascade_candidate = {"arm": cascade_arm, "seed": int(cascade_row["seed"]), "recipe": cascade["model_config"],
                         "snapshot_sha256": dc["snapshot_sha256"], "initial_weights": dc["plan"]["weights"]["cascade"],
                         "training_job": cascade_job["training_job"], "evaluation_job": cascade_job["evaluation_job"],
                         "checkpoint_path": cascade["checkpoint"], "checkpoint_sha256": cascade["checkpoint_sha256"],
                         "validation": {**as_values(cascade_row), "best_epoch": int(cascade_row["best_epoch"])}}
    ds = c_import / "prepared-manifest.json"
    require(sha256(ds) == sha256(d_import / ds.name) == yolo["dataset_manifest_sha256"],
            "Los modelos no comparten el dataset validado")
    c_identity, d_identity = (read_json(root / "snapshot.json")["sha256"] for root in (c_import, d_import))
    evaluator = c_identity["scripts/evaluate_baseline.py"]
    require(evaluator == d_identity["scripts/evaluate_baseline.py"], "Evaluadores COCO distintos")
    contract = {"status": "pendiente_sha256_fisico" if certified else "pendiente_sacct_y_sha256_fisico",
                "c_slurm_accounting_verified": certified, "approval_required": True,
                "dataset_manifest_sha256": sha256(ds), "valid_coco_sha256": yolo["valid_coco_sha256"],
                "test_coco_expected_sha256_from_manifest": read_json(ds)["files"]["coco/test.json"],
                "evaluation_entrypoint_sha256": evaluator, "operating_point": yolo["operating_point"],
                "benchmark": {"warmup": 20, "iterations": 100, "batch": 1, "precision": "fp32", "image_size": 640},
                "yolo": yolo, "cascade": cascade_candidate}
    comparison = [{"backend": "yolo", "arm": arm, "seed": s, "checkpoint_propuesto": s == seed, **values[arm, s]}
                  for s in confirmation.SEEDS]
    comparison.extend({"backend": "cascade", "arm": "e100-f000", "seed": int(row["seed"]),
                       "checkpoint_propuesto": row == cascade_row, **as_values(row)} for row in d_rows)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "seleccion-valid.json", contract)
    write_csv(output / "comparacion_por_semilla.csv", comparison)
    write_csv(output / "comparacion_resumen.csv", [
        {"backend": backend, "metric": key, **confirmation.descriptive(
            [r[key] for r in comparison if r["backend"] == backend])}
        for backend in ("yolo", "cascade") for key in METRICS])
    figures(output, read_csv(c_report / "contrastes.csv"), comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("c_report", "c_import", "d_report", "d_import", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    closeout(args.c_report, args.c_import, args.d_report, args.d_import, args.output)
