"""Analyze the 21 runs of the post-exploration EXP-007 extension."""

import argparse
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
import subprocess

from scripts.baseline_data import DESTINATION, read_json, write_json

from . import analyze as paired
from .provenance import (
    A_ARMS, B_ARMS, C_ARMS, ORIGINAL, SEEDS, receipt, script_hashes, training_phase,
    verify_commit, verify_evidence, verify_revisions,
)


def verify_reused(reference, fresh):
    old, new = (read_json(folder / "metrics.json") for folder in (reference, fresh))
    for key in ("checkpoint_sha256", "dataset_manifest_sha256", "coco_sha256", "map", "ap50", "ap75",
                "ap_small", "ap_medium", "ap_large", "per_class", "operating_point", "parameters"):
        paired.require(old[key] == new[key], f"Reevaluación distinta ({key}): {fresh}. Investigar antes de combinar.")
    paired.require(read_json(reference / "pr_curves.json") == read_json(fresh / "pr_curves.json"),
                   f"Curvas PR distintas en la reevaluación: {fresh}")


def verify_unique(training):
    for key in ("job", "path"):
        seen = [row[key] for runs in training.values() for row in runs.values()]
        paired.require(len(seen) == len(set(seen)) == len(SEEDS) * len(C_ARMS), f"Entrenamientos duplicados ({key})")


def paired_summary(results, reference):
    return {
        arm: paired.common.aggregate(
            {seed: {"control": arms[reference], "variant": arms[arm], "delta": paired.deltas(arms[reference], arms[arm])}
             for seed, arms in results.items()}, "variant",
        ) for arm in C_ARMS if arm != reference
    }


def frozen_prefixes(histories):
    variables = ("val/map", *paired.common.LOSSES, *paired.common.LR)
    result = {}
    for left, right in combinations(C_ARMS[1:], 2):
        epochs = min(paired.ALL_ARMS[left], paired.ALL_ARMS[right])
        result[f"{left}/{right}"] = {
            "epochs": epochs,
            "max_abs_delta": {key: max(abs(a[key] - b[key])
                                       for a, b in zip(histories[left][:epochs], histories[right][:epochs]))
                              for key in variables},
        }
    return result


def plot_performance(results, histories, output):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True, sharey="row")
    durations = list(paired.ALL_ARMS.values())
    for ax, (key, label) in zip(axes.flat, (
        ("map", "mAP COCO"), ("map_without_ibuphil", "mAP sin Ibuphil 600 mg"),
        ("recall", "Recall a score 0,25"), ("precision", "Precisión a score 0,25"),
    )):
        means, deviations = [], []
        for arm in C_ARMS:
            group = [results[seed][arm][key] for seed in SEEDS]
            defined = [value for value in group if value is not None]
            means.append(mean(defined) if defined else float("nan"))
            deviations.append(stdev(defined) if len(defined) > 1 else float("nan"))
        for seed in SEEDS:
            ax.plot(durations, [results[seed][arm][key] for arm in C_ARMS], alpha=0.45, label=str(seed))
        ax.errorbar(durations, means, yerr=deviations, color="black", marker="o", capsize=3, label="Media ± DE")
        ax.set(title=label, xticks=durations, xlabel="Épocas congeladas")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "duracion.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True, sharey="row")
    for index, seed in enumerate(SEEDS):
        for arm in C_ARMS[1:]:
            early = histories[seed][arm][:5]
            axes[0, index].plot([r["step"] for r in early], [r["val/map"] for r in early], label=arm)
            axes[1, index].plot([r["step"] for r in early], [r["val/cls_loss"] for r in early])
        axes[0, index].set_title(str(seed))
        axes[1, index].set(xlabel="Época", xticks=range(1, 6), yscale="symlog")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("mAP COCO")
    axes[1, 0].set_ylabel("Pérdida de clasificación en valid")
    fig.savefig(output / "prefijo-congelado.png", dpi=140)
    plt.close(fig)


def analyze(args):
    sources = {"a": args.source, "b": args.confirmation, "c": args.extension}
    for source in sources.values():
        verify_evidence(source)
    old, new = verify_revisions(args.original_checkout, paired.REPO, args.source)
    identity = paired.common.campaign_identity(args.source)
    extension_revision, inputs = paired.common.campaign_identity(args.extension)
    paired.require(extension_revision != ORIGINAL and inputs == identity[1]
                   and paired.common.campaign_identity(args.confirmation) == identity
                   and receipt(args.extension / "code-config.sha256") == new
                   and receipt(args.confirmation / "code-config.sha256") == old,
                   "Revisión o entradas incompatibles en la ampliación")
    verify_commit(args.git_repo, ORIGINAL, old)
    verify_commit(args.git_repo, extension_revision, new)
    smoke = args.extension / "smoke"
    paired.verify_permanent_smoke(smoke)
    smoke_run = read_json(smoke / "train/run.json")
    smoke_job = smoke_run["hardware"]["job_id"]
    if not (args.extension / "evidence.sha256").is_file():
        submission = read_json(args.extension / "queue/exp007-yolo-smoke-permanent-s2026.json")
        paired.require((smoke / "verified.ok").is_file() and submission["job"] == smoke_job,
                       "Falta la verificación del smoke de congelación permanente")
    status = read_json(smoke / "train/status.json")
    paired.require(isinstance(smoke_job, str) and smoke_job.isdecimal()
                   and status["sessions"][-1]["job_id"] == smoke_job and smoke_run["mode"] == "train"
                   and smoke_run["code_sha256"] == script_hashes(new)
                   and smoke_run["git_head"] in (None, extension_revision)
                   and smoke_run["run"]["dataset_manifest_sha256"] == inputs["manifest.json"]
                   and smoke_run["run"]["initial_weights_sha256"] == inputs["yolov10s.pt"],
                   "Procedencia del smoke distinta de la ampliación")
    truth = paired.common.validation_truth(args.data, args.source, inputs)
    evaluation = {"evaluation_checkout": args.original_checkout}
    a, _, environment = paired.read_cohort(args.source, "a", 2026, A_ARMS, identity, truth, **evaluation)
    paired.require(paired.select(a) == "freeze-15", "La selección histórica de fase A no se reprodujo")
    for seed in SEEDS[1:]:
        _, _, b_env = paired.read_cohort(args.confirmation, "b", seed, B_ARMS, identity, truth, **evaluation)
        paired.require(b_env == environment, f"Entorno distinto en B, semilla {seed}")

    results, histories, training, initial_backbones = {}, {}, {}, set()
    for seed in SEEDS:
        origins = {arm: training_phase(seed, arm) for arm in C_ARMS}
        versions = {arm: (sources[phase], extension_revision if phase == "c" else ORIGINAL,
                          script_hashes(new if phase == "c" else old)) for arm, phase in origins.items()}
        values, histories[seed], c_env = paired.read_cohort(
            args.extension, "c", seed, C_ARMS, identity, truth, training_sources=versions, **evaluation,
        )
        paired.require(c_env == environment, f"Entorno distinto en C, semilla {seed}")
        launch = read_json(args.extension / f"eval-c-s{seed}/launch.json")
        training[seed] = {}
        for arm, phase in origins.items():
            run_root = sources[phase] / f"s{seed}" / arm / "train"
            current = args.extension / f"eval-c-s{seed}" / arm
            if phase != "c":
                verify_reused(sources[phase] / f"eval-{phase}-s{seed}" / arm, current)
            duration = paired.ALL_ARMS[arm]
            rows = histories[seed][arm]
            values[arm].update(
                map_max_frozen=max((r["val/map"] for r in rows[:duration]), default=None),
                map_max_unfrozen=max((r["val/map"] for r in rows[duration:]), default=None),
            )
            if duration:
                initial_backbones.add(read_json(run_root / "staged_transfer.json")["initial_backbone_sha256"])
            entry = launch["runs"][arm]
            training[seed][arm] = {
                "phase": phase, "training_revision": versions[arm][1], "evaluation_revision": ORIGINAL,
                "job": entry["training_job_id"], "evaluation_job": launch["job_id"], "path": str(Path(entry["run"])),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "best_phase": "frozen" if values[arm]["best_epoch"] <= duration else "fine-tune",
            }
        results[seed] = values
    verify_unique(training)
    paired.require(len(initial_backbones) == 1, "Las huellas iniciales del backbone congelado no son iguales")
    control, freeze15 = (paired_summary(results, arm) for arm in ("control", "freeze-15"))
    candidates = []
    for arm, summary in control.items():
        delta = {key: summary["delta"][key]["mean"] for key in ("map", "map_without_ibuphil", "recall")}
        if (paired.meets(delta["map"], 0.01) and paired.meets(delta["map_without_ibuphil"], 0)
                and summary["map_wins"] >= 2 and paired.meets(delta["recall"], -0.02)):
            candidates.append(arm)
    candidates.sort(key=lambda arm: (
        *(-control[arm]["variant"][key]["mean"] for key in ("map", "map_without_ibuphil", "recall")),
        paired.ALL_ARMS[arm],
    ))
    weights = {row["id"]: sum(a["category_id"] == row["id"] for a in truth["annotations"])
               for row in truth["categories"]}
    return {
        "phase": "C", "coco_verified": True, "permanent_smoke_job": smoke_job, "training": training, "runs": results,
        "per_class": [{**row, "support": weights[row["id"]]} for row in truth["categories"]],
        "by_arm": {"control": control["freeze-5"]["control"],
                   **{arm: summary["variant"] for arm, summary in control.items()}},
        "against_control": control, "against_freeze_15": freeze15,
        "deltas_by_seed": {seed: {
            reference: {arm: paired.deltas(values[reference], values[arm]) for arm in C_ARMS if arm != reference}
            for reference in ("control", "freeze-15")
        } for seed, values in results.items()},
        "ranking": candidates, "candidate": candidates[0] if candidates else None,
        "transition_windows": {seed: paired.transition_windows(h) for seed, h in histories.items()},
        "frozen_prefixes": {seed: frozen_prefixes(h) for seed, h in histories.items()},
    }, histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "confirmation", "extension", "original-checkout", "git-repo", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DESTINATION)
    args = parser.parse_args()
    try:
        result, histories = analyze(args)
        args.output.mkdir(parents=True)
        write_json(args.output / "summary.json", result)
        for seed, rows in histories.items():
            paired.plot(rows, args.output / f"history-s{seed}.png")
        plot_performance(result["runs"], histories, args.output)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
