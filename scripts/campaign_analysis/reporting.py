"""Tables and figures derived exclusively from verified campaign conditions."""

from importlib import import_module
from statistics import mean

import numpy as np

write_csv = import_module("experiments.exp-002-photometric.analyze").write_csv

RARE_CLASS = "Ibuphil 600 mg"
SENSITIVITY = "map_without_ibuphil_600_mg"
DETECTORS = {"yolo": ("YOLOv10-S", "exp-004"), "cascade": ("Cascade R-CNN", "exp-005")}
EFFECTS = {
    "epochs_no_aug": "B−A · Épocas, control",
    "epochs_aug": "D−C · Épocas, con geometría",
    "augmentation_30": "C−A · Geometría a 30",
    "augmentation_100": "D−B · Geometría a 100",
    "interaction": "(D−B)−(C−A) · Interacción",
    "mosaic_100": "D−E · Habilitar Mosaic",
    "affine_only_100": "E−B · Afín a 100",
}
MAPS = {"map": "mAP COCO (8 clases)", SENSITIVITY: "mAP sin Ibuphil 600 mg (7 clases)"}
ARM_COLORS = dict(zip("abcde", ("#174c80", "#5098c5", "#c26a0b", "#983e27", "#4c794c")))
MAP_COLORS = ("#174c80", "#b05c12")


def effect_contributions(arms, names, contrasts):
    for effect, coefficients in contrasts.items():
        if not all(arm in arms for arm in coefficients):
            continue
        deltas = {name: sum(factor * arms[arm][name] for arm, factor in coefficients.items()) for name in names}
        rare = deltas[RARE_CLASS]
        common = [delta for name, delta in deltas.items() if name != RARE_CLASS]
        yield {
            "contrast": effect, "delta_map": mean(deltas.values()), "delta_sin_ibuphil": mean(common),
            "delta_ap_ibuphil": rare, "aporte_ibuphil_al_map": rare / len(names),
            "aporte_otras_clases_al_map": sum(common) / len(names),
            "otras_suben": sum(delta > 1e-12 for delta in common),
            "otras_bajan": sum(delta < -1e-12 for delta in common),
            **{f"delta_ap_{name}": delta for name, delta in deltas.items()},
        }


def save(fig, path):
    from matplotlib import pyplot as plt

    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_overview(results, output):
    from matplotlib import pyplot as plt

    arms = [(DETECTORS[backend][0], arm, row["values"])
            for backend, rows in results.items() for arm, row in rows.items()]
    fig, ax = plt.subplots(figsize=(12, 5.3), layout="constrained")
    x = np.arange(len(arms))
    for j, (metric, label) in enumerate(MAPS.items()):
        bars = ax.bar(x + (j - .5) * .36, [100 * v[metric] for _, _, v in arms],
                      width=.36, label=label, color=MAP_COLORS[j])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=[f"{name}\n{arm.upper()}" for name, arm, _ in arms],
           ylim=(0, 107), ylabel="mAP (%)", title="Rendimiento en validación por condición")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    save(fig, output)


def plot_effects(rows, detector, output):
    from matplotlib import pyplot as plt

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, max(3.2, len(rows) * .60)), layout="constrained")
    for j, (field, label) in enumerate(zip(("delta_map", "delta_sin_ibuphil"), MAPS.values())):
        bars = ax.barh(y + (j - .5) * .36, [r[field] * 100 for r in rows], height=.34,
                       color=MAP_COLORS[j], label=label)
        ax.bar_label(bars, fmt="%+.1f", fontsize=8, padding=2)
    ax.set(yticks=y, yticklabels=[EFFECTS[r["contrast"]] for r in rows],
           xlabel="Diferencia (puntos porcentuales)", title=f"{detector} · Efectos en validación")
    ax.axvline(0, color="#303030", linewidth=.7)
    ax.invert_yaxis()
    ax.margins(x=.14)
    ax.legend(loc="best")
    save(fig, output)


def plot_classes(arms, names, detector, output):
    from matplotlib import pyplot as plt

    arm_ids = list(arms)
    ap = np.array([[arms[arm][name] * 100 for arm in arms] for name in names])
    control = {arm: "a" if arm in ("a", "c") else "b" for arm in arms}
    has_control = [arm in ("c", "d", "e") and control[arm] in arms for arm in arms]
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5.6), layout="constrained",
                             gridspec_kw={"width_ratios": (len(arms), max(2, sum(has_control)))})
    mat = axes[0].imshow(ap, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axes[0].set(xticks=range(len(arm_ids)), xticklabels=[a.upper() for a in arm_ids], yticks=range(len(names)),
                yticklabels=names, title="AP COCO por clase (%)")
    for (i, j), value in np.ndenumerate(ap):
        axes[0].text(j, i, f"{value:.1f}", va="center", ha="center", color="white" if value > 60 else "black")
    fig.colorbar(mat, ax=axes[0], shrink=.72)
    labels = [f"{a.upper()}−{control[a].upper()}" for a, shown in zip(arm_ids, has_control) if shown]
    if labels:
        diff = np.array([[100 * (arms[a][name] - arms[control[a]][name])
                          for a, shown in zip(arm_ids, has_control) if shown] for name in names])
        limit = max(1, np.max(np.abs(diff)))
        mat = axes[1].imshow(diff, vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
        axes[1].set(xticks=range(len(labels)), xticklabels=labels, yticks=[], title="Cambio frente al control (pp)")
        for (i, j), value in np.ndenumerate(diff):
            axes[1].text(j, i, f"{value:+.1f}", va="center", ha="center",
                         color="white" if abs(value) > .65 * limit else "black")
        fig.colorbar(mat, ax=axes[1], shrink=.72)
    else:
        axes[1].set_visible(False)
    fig.suptitle(f"{detector} · Rendimiento por clase")
    save(fig, output)


def history_diagnostics(rows, backend):
    loss = "train/cls_loss" if backend == "yolo" else "total_loss"
    lr = "lr/pg0" if backend == "yolo" else "lr"
    for arm, result in rows.items():
        stride, history = result["stride"], result["history"]
        scores = [row["val/map"] for row in history if "val/map" in row]
        best_step = stride * result["values"]["best_epoch"]
        best = next(row for row in history if row["step"] == best_step)
        tail = [row for row in history if row["step"] > best_step]
        yield {"arm": arm.upper(), "best_epoch": best_step // stride, "last_epoch": len(scores),
               **{k: result["values"][k] for k in ("best_validation_map_history", "last_validation_map_history",
                                                   "best_minus_last_history")},
               "score_epoch_30": scores[29] if len(scores) >= 30 else None,
               "min_map_last_5_epochs": min(scores[-5:]), "max_map_last_5_epochs": max(scores[-5:]),
               "loss_field": loss, "loss_best": best.get(loss), "loss_last": history[-1].get(loss),
               "lr_field": lr, "lr_best": best.get(lr), "lr_last": history[-1].get(lr),
               "post_best_epochs": sum("val/map" in row for row in tail)}


def plot_history(rows, backend, output):
    from matplotlib import pyplot as plt

    if backend == "yolo":
        fields = [("val/map", "mAP (%)"), ("lr/pg0", "LR (grupo 0)")]
        fields += [(f"{split}/{loss}_loss", f"{label} · {loss}")
                   for split, label in (("train", "Entrenamiento"), ("val", "Validación"))
                   for loss in ("box", "cls", "dfl")]
    else:
        fields = [("val/map", "mAP (%)"), ("lr", "LR"), ("total_loss", "Pérdida de entrenamiento")]
    fields = [(key, label) for key, label in fields
              if any(key in row for r in rows.values() for row in r["history"])]
    fig, axes = plt.subplots((len(fields) + 1) // 2, 2, figsize=(12, 2.9 * ((len(fields) + 1) // 2)),
                             squeeze=False, sharex=True, layout="constrained")
    for ax, (key, label) in zip(axes.flat, fields):
        factor = 100 if key == "val/map" else 1
        for arm, result in rows.items():
            points = [(row["step"] / result["stride"], factor * row[key])
                      for row in result["history"] if key in row]
            if points:
                ax.plot(*zip(*points), label=arm.upper(), color=ARM_COLORS[arm], linewidth=1.3)
            if key == "val/map":
                values = result["values"]
                ax.scatter(values["best_epoch"], 100 * values["best_validation_map_history"],
                           marker="D", s=25, color=ARM_COLORS[arm], zorder=3)
        if key == "val/cls_loss":
            ax.set_yscale("symlog", linthresh=1)
            label += " (escala simétrica log)"
        ax.set(ylabel=label, xlabel="Época lógica")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    axes[0, 0].legend(title="Condición", ncols=len(rows))
    fig.suptitle(f"{DETECTORS[backend][0]} · Historial de entrenamiento y validación (◇ checkpoint best)")
    save(fig, output)


def percent(value, signed=False):
    return "—" if value is None else format(value * 100, "+.2f" if signed else ".2f").replace(".", ",")


def table(headers, rows):
    def row(cells):
        return "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join((row(headers), row(["---"] * len(headers)), *(row(cells) for cells in rows)))


def arm_table(records):
    return table(("Modelo", "Condición", "Épocas", "mAP 8 (%)", "mAP 7 (%)", "AP50 (%)", "AP75 (%)",
                  "Precisión (%)", "Recall (%)", "TP", "FP", "FN", "ms/img", "p95 ms"),
                 ((r["modelo"], r["arm"], r["epochs"], *(percent(r[k]) for k in ("map", SENSITIVITY, "ap50",
                   "ap75", "precision", "recall")), *(r[k] for k in ("tp", "fp", "fn")),
                   f"{r['mean_ms_per_image']:.2f}", f"{r['p95_ms_per_image']:.2f}") for r in records))


def experiment_report(detector, records, effects, diagnostics, arms, names):
    by_effect = {r["contrast"]: r for r in effects}
    focus = ("augmentation_100", "mosaic_100") if detector == DETECTORS["yolo"][0] else ("augmentation_30", "epochs_aug")
    interpretation = []
    for key in focus:
        if key not in by_effect:
            continue
        effect = by_effect[key]
        ranked = sorted(((name, effect[f"delta_ap_{name}"]) for name in names if name != RARE_CLASS),
                        key=lambda pair: pair[1], reverse=True)
        moved = ", ".join(f"{name} {percent(delta, signed=True)} pp" for name, delta in
                          (ranked[:2] + ranked[-2:]) if abs(delta) > 1e-12)
        interpretation.append(f"- **{EFFECTS[key]}** · Cambio oficial {percent(effect['delta_map'], True)} pp, "
                              f"con un aporte de Ibuphil de {percent(effect['aporte_ibuphil_al_map'], True)} pp. "
                              f"El cambio del promedio de siete clases es {percent(effect['delta_sin_ibuphil'], True)} pp "
                              f"({effect['otras_suben']} suben y {effect['otras_bajan']} bajan)."
                              + (f" Entre las siete, los extremos son {moved}." if moved else ""))
    summary = [
        f"# {detector} · Hallazgos en validación",
        arm_table(records),
        "Precisión, recall, TP, FP y FN usan confianza ≥ 0,25 e IoU ≥ 0,50. Las latencias son descriptivas por job. "
        "El mAP principal incluye ocho clases. El promedio de siete es un análisis de sensibilidad.",
        "## Contrastes (puntos porcentuales)",
        table(("Contraste", "Δ mAP 8", "Δ mAP 7", "Δ AP Ibuphil 600 mg", "Aporte Ibuphil al Δ mAP 8",
               "Aporte de las otras siete", "Otras suben / bajan"),
              ((EFFECTS[r["contrast"]], *(percent(r[k], signed=True) for k in (
                    "delta_map", "delta_sin_ibuphil", "delta_ap_ibuphil", "aporte_ibuphil_al_map",
                    "aporte_otras_clases_al_map")), f"{r['otras_suben']} / {r['otras_bajan']}") for r in effects)),
        "Cada clase pesa 1/8 en el mAP oficial. El aporte de las otras siete es 7/8 del cambio del promedio "
        "de siete clases. Consultar `aporte_por_clase.csv` para los cambios de cada clase por contraste.",
        *interpretation,
        "## AP por clase (%)",
        table(("Clase", *(arm.upper() for arm in arms)),
              ((name, *(percent(v[name]) for v in arms.values())) for name in names)),
        "## Selección del checkpoint",
        table(("Condición", "Época best", "mAP best (%)", "mAP última (%)", "Diferencia (pp)", "mAP época 30 (%)"),
              ((r["arm"], r["best_epoch"], *(percent(r[k]) for k in ("best_validation_map_history",
                 "last_validation_map_history", "best_minus_last_history", "score_epoch_30"))) for r in diagnostics)),
        "El mAP de la última época procede del historial de validación. La evaluación externa reproducida corresponde "
        "al archivo `best`. Los presupuestos de 30 y 100 épocas recorren horizontes coseno distintos. "
        "Una caída después de `best` es compatible con fluctuación o degradación tardía. Interpretarla junto a "
        "la pérdida y el LR de `diagnostico_historial.csv` y `entrenamiento.png`.",
        "## Errores y ejemplos",
        "Consultar `por_clase.csv` para TP, FP y FN por clase, `confusiones.csv` para los solapamientos "
        "con una clase distinta y `ibuphil.csv` para el ranking y el IoU de las detecciones de interés. "
        "Las imágenes disponibles se anotan a confianza ≥ 0,25 e IoU ≥ 0,50 en `ejemplos/`. "
        "Las curvas PR por clase a IoU 0,50 y 0,75 se guardan en `curvas_pr/`.",
        "## Alcance",
        "La semilla 2026 y las 90 imágenes de validación permiten una comparación emparejada exploratoria. "
        "La comparación entre arquitecturas también cambia la receta y el preprocesado. "
        "Ibuphil 600 mg tiene una sola instancia en validación. Los resultados del conjunto de test quedan reservados.",
    ]
    if detector == DETECTORS["yolo"][0]:
        summary.insert(3, "D−E mide habilitar Mosaic en la receta de 100 épocas. Además del montaje y "
                          "reescalado de imágenes, cambia la frecuencia efectiva de la afín.")
    return "\n\n".join(summary) + "\n"


def write_reports(output, results, plan, names, values, contrasts, pending):
    combined = []
    budgets = {c["name"]: c["epochs"] for c in plan["conditions"]}
    for backend, rows in results.items():
        if not rows:
            continue
        detector, exp = DETECTORS[backend]
        target = output / exp
        records = [{"modelo": detector, "arm": arm.upper(), "epochs": budgets[f"{exp.replace('-', '')}-{arm}"],
                    **r["values"], **r["benchmark"]} for arm, r in rows.items()]
        combined.extend(records)
        effects = list(effect_contributions(values[backend], names, contrasts))
        if effects:
            write_csv(target / "aporte_por_clase.csv", effects)
            plot_effects(effects, detector, target / "efectos_factoriales.png")
        diagnostics = list(history_diagnostics(rows, backend))
        write_csv(target / "diagnostico_historial.csv", diagnostics)
        plot_history(rows, backend, target / "entrenamiento.png")
        plot_classes(values[backend], names, detector, target / "ap_por_clase.png")
        (target / "findings.md").write_text(
            experiment_report(detector, records, effects, diagnostics, values[backend], names), encoding="utf-8")
    if combined:
        write_csv(output / "resumen.csv", combined)
        plot_overview(results, output / "map_condiciones.png")
    report = [
        "# EXP-004 y EXP-005 · Resumen ejecutivo",
        f"Se reprodujeron {len(combined)} de {len(plan['conditions'])} evaluaciones de validación. "
        "La auditoría vincula cada preflight, entrenamiento y evaluación con su descriptor y la huella de `best`.",
        "Diseño factorial 30/100 épocas × control/geometría, más la ablación sin Mosaic en YOLO. "
        "Pesos COCO, semilla 2026 y 90 imágenes de validación. Los checkpoints se seleccionaron en la misma "
        "partición. Los ocho AP tienen el mismo peso en el mAP COCO@0,50:0,95. La latencia describe cada asignación.",
    ]
    if combined:
        report.append(arm_table(combined))
    if pending:
        report.append(f"Evaluaciones pendientes de verificación: {', '.join(pending)}.")
    else:
        report += ["## Resultados principales"]
        for backend, (detector, _) in DETECTORS.items():
            arms = values[backend]
            rows = {r["contrast"]: r for r in effect_contributions(arms, names, contrasts)}
            key = "augmentation_100" if backend == "yolo" else "augmentation_30"
            focus = rows[key]
            interaction = rows["interaction"]
            report.append(f"- **{detector}** · {EFFECTS[key]} da {percent(focus['delta_map'], True)} pp oficiales "
                          f"y {percent(focus['delta_sin_ibuphil'], True)} pp sin Ibuphil. La interacción es "
                          f"{percent(interaction['delta_map'], True)} pp oficiales y "
                          f"{percent(interaction['delta_sin_ibuphil'], True)} pp a siete clases.")
        report.append("## Candidatos de validación")
        for detector, _ in DETECTORS.values():
            subset = [r for r in combined if r["modelo"] == detector]
            leaders = [" / ".join(r["arm"] for r in subset if r[key] == max(s[key] for s in subset))
                       for key in MAPS]
            report.append(f"- **{detector}** · {leaders[0]} lidera por mAP oficial y {leaders[1]} por el promedio "
                          "sin Ibuphil 600 mg.")
        report.append("El criterio primario de candidatura es el mAP oficial. La sensibilidad, los errores y la "
                      "latencia permiten explicar los compromisos. Reservar test para los candidatos congelados.")
    report.append("Una semilla y un único split limitan la replicabilidad. "
                  "Los efectos de época incluyen el horizonte del scheduler y la selección entre más checkpoints.")
    (output / "resumen-ejecutivo.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
