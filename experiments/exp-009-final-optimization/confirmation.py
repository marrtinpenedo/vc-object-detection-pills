"""Reproduce validation of EXP-009 C and propose one final YOLO recipe."""
import argparse
from importlib import import_module
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from scripts.baseline_data import read_json, sha256, write_json
from scripts.baseline_metrics import plot_pr
from scripts.campaign_analysis.analyze import load_dataset
from scripts.campaign_analysis.import_results import require
from scripts.campaign_analysis.reporting import SENSITIVITY, save, write_csv
from scripts.exp009_followup.bundle import confirmation_references, plan_path
from scripts.exp009_followup.recipes import CONFIRMATION_ARMS, CONFIRMATION_SEEDS

followup = import_module("experiments.exp-009-final-optimization.followup")

SEEDS = (2026, *CONFIRMATION_SEEDS)
METRICS = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall", "mean_ms_per_image", "p95_ms_per_image")
NUMERICAL_PACKAGES = ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pillow", "pycocotools")
PRIORITY = (("F", "D"), ("E", "C"), ("D", "B"))
COMPARISONS = PRIORITY + tuple((hi, lo) for lo, hi in combinations(CONFIRMATION_ARMS, 2) if (hi, lo) not in PRIORITY)
INTERACTIONS = {
    "freezing_x_geometria_sin_fotometria": {"A": 1, "B": -1, "C": -1, "D": 1},
    "geometria_x_fotometria_f080": {"C": 1, "D": -1, "E": -1, "F": 1},
}


def descriptive(values):
    return {"mean": mean(values), "sample_sd": stdev(values), "n": len(values)}


def deltas(rows, variant, control, metric):
    return [rows[variant, seed][metric] - rows[control, seed][metric] for seed in SEEDS]


def adoption(rows, variant, control):
    official = deltas(rows, variant, control, "map")
    sensitivity = deltas(rows, variant, control, SENSITIVITY)
    return mean(official) >= .01 and mean(sensitivity) >= 0 and sum(d > 0 for d in official) >= 2


def choose(rows):
    """Apply the prespecified paired rule and rank eligible recipes by mean mAP."""
    matched = {"B": "A", "C": "A", "D": "B", "E": "C", "F": "D"}
    eligible = ["A"]
    for arm, control in matched.items():
        if adoption(rows, arm, control) and adoption(rows, arm, "A"):
            eligible.append(arm)
    winner = max(eligible, key=lambda arm: (mean(rows[arm, s]["map"] for s in SEEDS),
                                            mean(rows[arm, s][SENSITIVITY] for s in SEEDS), -ord(arm)))
    middle = sorted(SEEDS, key=lambda s: (rows[winner, s]["map"], rows[winner, s][SENSITIVITY], s))[1]
    return winner, middle, eligible


def paired_rows(values, metrics, interactions=True):
    definitions = {f"{variant}-{control}": {variant: 1, control: -1} for variant, control in COMPARISONS}
    if interactions:
        definitions.update(INTERACTIONS)
    for contrast, weights in definitions.items():
        for metric in metrics:
            ds = [sum(weight * values[arm, seed][metric] for arm, weight in weights.items()) for seed in SEEDS]
            yield {"contrast": contrast, "metric": metric, **dict(zip((f"delta_{seed}" for seed in SEEDS), ds)),
                   "mean_new_seeds": mean(ds[1:]), "positive_new_seeds": sum(d > 0 for d in ds[1:]),
                   "positive_seeds": sum(d > 0 for d in ds), **descriptive(ds)}


def transition_summaries(runs, configs):
    for (arm, seed), run in runs.items():
        hist = run["history"]
        peak = max(r["val/map"] for r in hist)
        duration = configs[arm, seed].get("freeze_backbone_epochs", 0)
        thaw = run["staging"]["thaw_epoch"] if duration else None
        prev, first, second = hist[thaw-2:thaw+1] if thaw else ({}, {}, {})
        best = run["values"]["best_epoch"]
        yield {"arm": arm, "seed": seed, "thaw_epoch": thaw, "best_epoch": best,
               "best_phase": ("descongelado" if best >= thaw else "congelado") if thaw else "sin_freezing",
               "best_internal_map": peak, "last_internal_map": hist[-1]["val/map"],
               "epochs_within_one_pp": sum(peak - row["val/map"] <= .01 for row in hist),
               "last_ten_mean_map": mean(row["val/map"] for row in hist[-10:]),
               "map_before_thaw": prev.get("val/map"), "map_first_active": first.get("val/map"),
               "map_second_active": second.get("val/map"),
               "map_delta_thaw": first["val/map"] - prev["val/map"] if thaw else None,
               "lr_first_active": first.get("lr/pg0"), "val_class_loss_before": prev.get("val/cls_loss"),
               "val_class_loss_after": first.get("val/cls_loss")}


def plot(output, runs, summaries):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained", sharex=True, sharey=True)
    for arm, ax in zip(CONFIRMATION_ARMS, axes.flat):
        for seed in SEEDS:
            run = runs[arm, seed]
            hist = run["history"]
            line, = ax.plot([r["step"] for r in hist], [100*r["val/map"] for r in hist],
                            label=str(seed), ls="--" if seed == 2026 else "-")
            epoch = run["values"]["best_epoch"]
            ax.scatter(epoch, 100*hist[epoch-1]["val/map"], s=18, color=line.get_color())
        if arm in "CDEF":
            ax.axvline(81, lw=.8, ls=":", color="#303030")
        ax.set_title(arm)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Época")
    fig.supylabel("mAP interno de validación (%)")
    save(fig, output / "curvas_por_receta.png")

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for offset, metric, label in ((-.12, "map", "COCO oficial"), (.12, SENSITIVITY, "Sin Ibuphil 600 mg")):
        positions = [idx + offset for idx in range(6)]
        values = [r for r in summaries if r["metric"] == metric]
        ax.errorbar(positions, [100*r["mean"] for r in values], yerr=[100*r["sample_sd"] for r in values],
                    fmt="o", capsize=4, label=label)
    ax.set(xticks=range(6), xticklabels=list(CONFIRMATION_ARMS), xlabel="Receta", ylabel="mAP (%)",
           title="Media y desviación estándar muestral · tres semillas")
    ax.legend()
    save(fig, output / "map_por_receta.png")


def verify_slurm_accounting(path, completed):
    """Accept only explicit success for each allocation used by the analysis."""
    expected = {entry["id"] for entry in completed.values()}
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.rstrip("|").split("|")]
        require(len(fields) == 3, f"Fila sacct inválida: {line}")
        job, state, exit_code = fields
        require(job in expected and job not in records, f"ID sacct inesperado o repetido: {job}")
        require((state, exit_code) == ("COMPLETED", "0:0"), f"SLURM {job}: {state} {exit_code}")
        records[job] = {"state": state, "exit_code": exit_code}
    require(set(records) == expected, f"Faltan IDs en sacct: {sorted(expected - records.keys())}")
    return {"sha256": sha256(path), "allocations": records}


def write_reports(output, plan, manifest, ledger, states, completed, runs, configs, integrity):
    output.mkdir(parents=True, exist_ok=False)
    values = {key: {**r["values"], **r["benchmark"], **r["resources"]} for key, r in runs.items()}
    aps = {key: {item["name"]: item["ap"] for item in run["classes"]} for key, run in runs.items()}
    summaries = [{"arm": arm, "metric": m, **descriptive([values[arm, s][m] for s in SEEDS])}
                 for arm in CONFIRMATION_ARMS for m in METRICS]
    write_csv(output / "por_semilla.csv", [{"arm": a, "seed": s, "cohort": "exploratoria" if s == 2026 else "confirmacion",
                                            **row} for (a, s), row in values.items()])
    write_csv(output / "resumen.csv", summaries)
    write_csv(output / "contrastes.csv", paired_rows(values, METRICS))
    names = tuple(next(iter(aps.values())))
    write_csv(output / "contrastes_ap_por_clase.csv", paired_rows(aps, names, interactions=False))
    write_csv(output / "media_ap_por_clase.csv", [
        {"arm": arm, "class": name, **descriptive([aps[arm, seed][name] for seed in SEEDS])}
        for arm in CONFIRMATION_ARMS for name in names])
    write_csv(output / "por_clase.csv", [{"arm": a, "seed": s, **item}
                                        for (a, s), run in runs.items() for item in run["classes"]])
    rows = [{"arm": a, "seed": s, **point} for (a, s), run in runs.items() for point in run["history"]]
    write_csv(output / "historial.csv", rows, tuple(dict.fromkeys(key for row in rows for key in row)))
    write_csv(output / "descongelacion.csv", transition_summaries(runs, configs))
    audit = {f"{a}-s{s}": {**r["audit"], "latency_protocol": r["protocol"],
                           "thaw_epoch": r["staging"]["thaw_epoch"] if r["staging"] else None}
             for (a, s), r in runs.items()}
    write_json(output / "auditoria.json", {**integrity, "controller_sha256": manifest["controller_sha256"],
               "references": plan["reference"], "completed_jobs": {k: v["id"] for k, v in completed.items()},
               "attempts": ledger, "slurm_states": states, "conditions": audit})
    winner, seed, eligible = choose(values)
    write_json(output / "candidato-yolo-valid.json", {
        "status": "propuesto_para_congelar" if integrity["slurm_accounting"] else "provisional_pendiente_sacct",
        "arm": winner, "eligible_arms": eligible, "seed": seed,
        "operating_point": {"score": .25, "iou": .50}, "evaluation": plan["evaluation"],
        "code_snapshot_sha256": manifest["snapshot_sha256"], "weights": plan["weights"]["yolo"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"], "valid_coco_sha256": plan["valid_coco_sha256"],
        "recipe": configs[winner, seed], "validation": values[winner, seed], **audit[f"{winner}-s{seed}"]})
    for (arm, seed), run in runs.items():
        plot_pr(run["pr_curves"], output / f"pr-{arm.lower()}-s{seed}.png")
    plot(output, runs, summaries)


def analyze(imported, reference_a, reference_b, data, output, receipt, *, accounting=None, diagnostic=False):
    require(diagnostic != (accounting is not None), "Indica sacct para aceptar C o ejecuta un diagnóstico explícito")
    plan = read_json(plan_path("C"))
    historical = confirmation_references(plan, reference_a, reference_b)
    manifest, ledger, states, snapshots, completed = followup.load_import(imported, plan, receipt)
    inventory = imported / "export.sha256.json"
    integrity = {"verified_inventory_entries": len(read_json(inventory)), "inventory_sha256": sha256(inventory),
                 "slurm_accounting": None if diagnostic else verify_slurm_accounting(accounting, completed)}
    truth, splits = load_dataset(data, imported, plan)
    snapshot = plan["reference"]["snapshot_sha256"]
    followup.verify_composed_smoke(imported, ledger, snapshots[snapshot], plan,
                                   completed["preflight-yolo"]["id"], splits["train"]["images"])
    cohorts = {"A": reference_a, "B": reference_b}
    for phase, root in cohorts.items():
        require(load_dataset(data, root, historical[phase]["plan"]) == (truth, splits),
                f"Datos de referencia incompatibles: {phase}")
    # COCOeval annotates the ground truth with internal matching fields.
    refs = {}
    for phase, root in cohorts.items():
        conditions = [row["condition"] for row in plan["reference"]["arms"].values() if row["phase"] == phase]
        refs.update(followup.read_results(root, historical[phase]["plan"], snapshots, truth, splits, conditions,
                                         followup=phase == "B"))
    new = followup.read_results(imported, plan, snapshots, truth, splits, plan["conditions"], followup=True)
    runs = {(arm, 2026): refs[name] for arm, name in CONFIRMATION_ARMS.items()}
    configs = {(arm, 2026): plan["reference"]["arms"][name]["config"] for arm, name in CONFIRMATION_ARMS.items()}
    for c in plan["conditions"]:
        name, arm, seed = c["name"], c["id"], c["seed"]
        old, current = refs[c["reference"]], new[name]
        old_args = old["effective"]["args"]
        require({**old["effective"], "args": {**old_args, "seed": seed}} == current["effective"]
                and old_args["seed"] == 2026,
                f"Receta efectiva no comparable: {name}")
        require(all(old["packages"][stage][package] == current["packages"][stage][package]
                    for stage in ("training", "evaluation") for package in NUMERICAL_PACKAGES),
                f"Versiones de entrenamiento o evaluación no comparables: {name}")
        runs[arm, seed], configs[arm, seed] = current, manifest["recipes"][name]
    write_reports(output, plan, manifest, ledger, states, completed, runs, configs, integrity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("imported", "reference_a", "reference_b", "data", "output", "receipt"):
        parser.add_argument(argument, type=Path)
    verification = parser.add_mutually_exclusive_group(required=True)
    verification.add_argument("--slurm-accounting", type=Path, help="Salida sacct con JobIDRaw,State,ExitCode")
    verification.add_argument("--diagnostic", action="store_true", help="Reproduce COCO y marca la selección provisional")
    args = parser.parse_args()
    analyze(args.imported, args.reference_a, args.reference_b, args.data, args.output, args.receipt,
            accounting=args.slurm_accounting, diagnostic=args.diagnostic)
