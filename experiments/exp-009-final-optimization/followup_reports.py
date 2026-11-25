"""Generate the phase-C proposal and interpretation figures from audited runs."""
from statistics import mean, stdev

import numpy as np

from scripts.baseline_data import write_json
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, save, write_csv

C_SEEDS = (2027, 2028)


def confirmation_plan(rows, recipes, phase_plan, photo_pairs):
    controls = list(phase_plan["reference"]["arms"])
    candidates = []
    for control, variant in photo_pairs:
        left, right = rows[control]["values"], rows[variant]["values"]
        gain = right["map"] - left["map"]
        sensitivity = right[SENSITIVITY] - left[SENSITIVITY]
        if gain >= .01 and sensitivity >= 0:
            candidates.append((control, variant, right["map"], right[SENSITIVITY], gain, sensitivity))
    selected = sorted(candidates, key=lambda row: (-row[2], -row[3], row[1]))[:2]
    f000c, f000g, f080c, f080g = controls
    definitions = {
        "freezing_sin_geometria": {f000c: -1, f080c: 1},
        "freezing_con_geometria": {f000g: -1, f080g: 1},
        "geometria_f000": {f000c: -1, f000g: 1},
        "geometria_f080": {f080c: -1, f080g: 1},
        "freezing_x_geometria_sin_fotometria": {f000c: 1, f000g: -1, f080c: -1, f080g: 1},
    }
    photo = {}
    for control, variant, _, _, gain, sensitivity in selected:
        definitions[f"fotometria_{control}"] = {control: -1, variant: 1}
        photo[variant] = {"paired_control": control, "map_gain": gain, "map7_gain": sensitivity}
    if all(name in photo for name in (f080c + "-photo", f080g + "-photo")):
        definitions["geometria_x_fotometria_f080"] = {f080c: 1, f080g: -1, f080c + "-photo": -1, f080g + "-photo": 1}
    arms = controls + list(photo)
    return {
        "scope": "proposal",
        "exploratory_seed": 2026,
        "confirmation_seeds": C_SEEDS,
        "new_trainings": len(arms) * len(C_SEEDS),
        "dataset_manifest_sha256": phase_plan["dataset_manifest_sha256"],
        "valid_coco_sha256": phase_plan["valid_coco_sha256"],
        "weights": phase_plan["weights"],
        "evaluation": phase_plan["evaluation"],
        "photo_selection": photo,
        "recipes": {name: recipes[name] for name in arms},
        "validation_2026": {name: rows[name]["audit"] for name in arms},
        "contrasts": definitions,
    }


def ibuphil_rows(rows):
    for name, run in rows.items():
        focus, = [row for row in run["focused"] if row["gt_class"] == RARE_CLASS]
        ranked = sorted((row for row in run["rare"] if row["pred_class"] == RARE_CLASS),
                        key=lambda row: row["rank_ibuphil_600_mg"])
        first = next(iter(ranked), None)
        correct = next((row for row in ranked if row["max_iou_ibuphil_600_mg"] >= .5), None)
        ap = next(row["ap"] for row in run["classes"] if row["name"] == RARE_CLASS)
        yield {"arm": name, "image_id": focus["image_id"], "gt_id": focus["gt_id"], "ap": ap,
               "best_same_iou": focus["best_same_iou"], "best_same_score": focus["best_same_score"],
               "correct_rank_at_iou50": correct["rank_ibuphil_600_mg"] if correct else None,
               "first_score": first["score"] if first else None,
               "first_iou": first["max_iou_ibuphil_600_mg"] if first else None,
               "first_overlapping_gt_class": first["best_gt_class"] if first else None,
               "operating_tp": focus["tp_score025_iou50"]}


def cascade_class_deltas(rows):
    by_seed = [{c["name"]: c["ap"] for c in run["classes"]} for run in rows.values()]
    for name in by_seed[0]:
        old, second, third = (r[name] for r in by_seed)
        yield {"class": name, "s2026_ap": old, "s2027_ap": second, "s2028_ap": third,
               "delta_2027_vs_2026": second - old, "delta_2028_vs_2026": third - old,
               "delta_2028_vs_2027": third - second, "mean_ap": mean((old, second, third)),
               "sample_sd_ap": stdev((old, second, third))}


def write_b(output, rows, recipes, plan, effects):
    pairs = [(arm["reference"], arm["name"]) for arm in plan["conditions"]]
    write_json(output / "fase-c-propuesta.json", confirmation_plan(rows, recipes, plan, pairs))
    from matplotlib import pyplot as plt

    photo = effects[:len(pairs)]
    plt_labels = ["Freeze 0 · sin geom.", "Freeze 0 · con geom.", "Freeze 80 · sin geom.", "Freeze 80 · con geom."]
    fig, (axis, classes) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained",
                                        gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(pairs))
    for offset, key, label, color in ((-.18, "delta_map", "mAP 8 clases", "#174c80"),
                                       (.18, "delta_sin_ibuphil", "mAP 7 clases", "#b05c12")):
        bars = axis.bar(x + offset, [100 * row[key] for row in photo], width=.34, label=label, color=color)
        axis.bar_label(bars, fmt="%+.1f", fontsize=9, padding=2)
    axis.set(xticks=x, xticklabels=plt_labels, ylabel="Cambio al añadir fotometría (pp)", ylim=(-6.6, 2.5))
    axis.tick_params(axis="x", rotation=20)
    axis.axhline(0, color="black", lw=.7)
    axis.legend()
    names = [c["name"] for c in rows[pairs[0][0]]["classes"]]
    ds = np.array([[100 * row[f"delta_ap_{n}"] for row in photo] for n in names])
    magnitude = abs(ds).max()
    panel = classes.imshow(ds, cmap="RdBu", vmin=-magnitude, vmax=magnitude, aspect="auto")
    for i, line in enumerate(ds):
        for j, delta in enumerate(line):
            classes.text(j, i, f"{delta:+.1f}", fontsize=9, ha="center", va="center",
                         color="white" if abs(delta) > .6 * magnitude else "black")
    classes.set(xticks=x, xticklabels=plt_labels, yticks=range(len(names)), yticklabels=names,
                title="Cambios de AP por clase")
    classes.tick_params(axis="x", rotation=20)
    fig.colorbar(panel, ax=classes, label="Δ AP (pp)")
    fig.suptitle("YOLOv10-S · Efecto de la fotometría en validación")
    save(fig, output / "fotometria_por_receta.png")


def write_d(output, rows):
    table = list(cascade_class_deltas(rows))
    write_csv(output / "diferencias_por_clase.csv", table)
    values = [run["values"] for run in rows.values()]
    rare = next(r for r in table if r["class"] == RARE_CLASS)
    pairs = [(0, 1), (0, 2), (1, 2)]
    differences = []
    for before, after in pairs:
        seed, compared = 2026 + before, 2026 + after
        delta_rare = rare[f"s{compared}_ap"] - rare[f"s{seed}_ap"]
        first, second = values[before], values[after]
        differences.append({"reference_seed": seed, "comparison_seed": compared,
                            "delta_map": second["map"] - first["map"],
                            "delta_map7": second[SENSITIVITY] - first[SENSITIVITY],
                            "rare_contribution": delta_rare / 8,
                            "other_classes_contribution": (second[SENSITIVITY] - first[SENSITIVITY]) * 7 / 8})
    write_csv(output / "diferencias_semillas.csv", differences)
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 5), layout="constrained")
    x = np.arange(len(table))
    for seed, color, symbol in ((2026, "#174c80", "o"), (2027, "#b05c12", "s"), (2028, "#28734f", "^")):
        axis.scatter(x + .18 * (seed - 2027), [100 * row[f"s{seed}_ap"] for row in table],
                     color=color, marker=symbol, label=str(seed), s=55)
    axis.set(xticks=x, xticklabels=[r["class"] for r in table], ylim=(38, 101),
             ylabel="AP COCO (%)", title="Cascade a 100 épocas · AP por clase y semilla")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=.25)
    axis.legend(loc="lower right")
    save(fig, output / "ap_por_semilla.png")
