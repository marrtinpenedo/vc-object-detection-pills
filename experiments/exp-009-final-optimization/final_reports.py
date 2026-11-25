"""Presentation tables and diagnostics from reproduced final predictions."""
from collections import defaultdict
from statistics import mean

from scripts.baseline_data import read_json, sha256
from scripts.baseline_metrics import OPERATING_IOU, coco_eval, operating_point
from scripts.campaign_analysis.reporting import SENSITIVITY, save, write_csv

NAMES = {"yolo": "YOLOv10-S D", "cascade": "Cascade R-CNN"}
COLORS = {"yolo": "#196191", "cascade": "#b45a15"}
COMPARE = ("map", SENSITIVITY, "ap50", "ap75", "precision", "recall",
           "mean_ms_per_image", "p95_ms_per_image", "fps_from_mean")


def diagnostics(root, truth, name):
    from pycocotools.mask import iou

    ev = coco_eval(truth, read_json(root / name / "predictions.json"))
    _, matches = operating_point(ev)
    errors = []
    for image_id, groups in sorted(matches.items()):
        anns = ev.cocoGt.imgToAnns[image_id]
        image = ev.cocoGt.imgs[image_id]
        for event in ("fp", "fn"):
            for row in groups[event]:
                candidates = iou([row["bbox"]], [ann["bbox"] for ann in anns], [0] * len(anns))[0] if anns else []
                best = max(zip(anns, candidates), key=lambda pair: pair[1], default=(None, 0.0))
                gt, overlap = best
                pred_cls = ev.cocoGt.cats[row["category_id"]]["name"]
                gt_cls = ev.cocoGt.cats[gt["category_id"]]["name"] if gt and overlap > 0 else None
                if event == "fn":
                    reason = "omision"
                elif overlap < OPERATING_IOU:
                    reason = "sin_solape_iou50"
                else:
                    reason = "duplicado" if pred_cls == gt_cls else "confusion_de_clase"
                errors.append({
                    "backend": name, "image_id": image_id, "file_name": image["file_name"], "event": event,
                    "reason": reason, "detection_class": pred_cls, "score": row.get("score"),
                    "iou_with_gt": float(overlap), "nearest_gt_class": gt_cls,
                    "bbox_xywh": row["bbox"], "gt_bbox_xywh": gt["bbox"] if gt and overlap > 0 else None,
                })
    return errors, ev


def comparison_rows(selection, result):
    for name, (test, *_) in result.items():
        valid = selection[name]["validation"].copy()
        valid.setdefault("fps_from_mean", 1000 / valid["mean_ms_per_image"])
        for metric in COMPARE:
            a, b = valid[metric], test[metric]
            yield {"backend": name, "metric": metric, "valid": a, "test": b,
                   "test_minus_valid": b - a if a is not None and b is not None else None}


def overview(output, comparison, result):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    keys = ((SENSITIVITY, "mAP COCO · siete clases"), ("precision", "Precisión operativa"), ("recall", "Recall operativo"))
    for ax, (metric, title) in zip(axes, keys):
        for index, name in enumerate(NAMES):
            row, = [r for r in comparison if r["backend"] == name and r["metric"] == metric]
            for offset, split, alpha in ((-.18, "valid", .48), (.18, "test", 1)):
                bars = ax.bar(index + offset, 100 * row[split], width=.34, color=COLORS[name], alpha=alpha)
                ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
        ax.set(title=title, ylabel="%", ylim=(0, 111), xticks=(0, 1), xticklabels=list(NAMES.values()))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Checkpoints de la semilla 2027 · validación (barra clara) y test (barra oscura)")
    save(fig, output / "validacion_test.png")

    fig, (ax, fps) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for panel, key, ylabel in ((ax, "mean_ms_per_image", "Latencia media (ms/imagen)"),
                               (fps, "fps_from_mean", "Rendimiento (FPS)")):
        bars = panel.bar(list(NAMES.values()), [result[name][0][key] for name in NAMES], color=list(COLORS.values()))
        panel.bar_label(bars, fmt="%.2f", padding=3)
        panel.margins(y=.17)
        panel.set_ylabel(ylabel)
        panel.grid(axis="y", alpha=.18)
        panel.set_axisbelow(True)
    fig.suptitle("Test · misma A100, batch 1, FP32")
    save(fig, output / "velocidad_test.png")


def by_iou(output, root):
    from matplotlib import pyplot as plt

    rows = []
    for name in NAMES:
        classes = read_json(root / name / "pr_curves.json")["classes"]
        thresholds = [key for key in next(iter(classes.values())) if key != "name"]
        for threshold in thresholds:
            supported = [mean(row[threshold]) for row in classes.values() if row[threshold][0] is not None]
            rows.append({"backend": name, "iou": float(threshold), "ap": mean(supported) if supported else None,
                         "supported_classes": len(supported)})
    write_csv(output / "ap_por_iou.csv", rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    for name in NAMES:
        group = [row for row in rows if row["backend"] == name]
        ax.plot([row["iou"] for row in group], [100 * row["ap"] if row["ap"] is not None else float("nan")
                                              for row in group],
                label=NAMES[name], color=COLORS[name], marker="o")
    ax.set(title="Test · AP por umbral IoU", xlabel="Umbral IoU", ylabel="AP medio (%)",
           xticks=[row["iou"] for row in group], ylim=(0, 107))
    ax.grid(alpha=.18)
    ax.legend()
    save(fig, output / "ap_por_iou.png")


def error_panel(path, errors, evals, dataset, manifest):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    cases = defaultdict(list)
    for row in errors:
        cases[row["backend"], row["image_id"]].append(row)
    available = []
    for (name, ident), rows in cases.items():
        file_name = rows[0]["file_name"]
        image = dataset / "test" / file_name
        if image.is_file() and sha256(image) == manifest["files"][f"images/test/{file_name}"]:
            available.append((name, ident, rows, image))
    if not available:
        return
    cols = min(4, len(available))
    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.4 * rows), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, available):
        name, ident, failures, image = case
        with Image.open(image) as source:
            ax.imshow(source.convert("RGB"))
        for annotation in evals[name].cocoGt.imgToAnns[ident]:
            x, y, w, h = annotation["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#42df78", lw=1.8))
        for failure in failures:
            x, y, w, h = failure["bbox_xywh"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#fc5d5d", lw=1.6, ls="--"))
        trans = {"duplicado": "Caja duplicada", "confusion_de_clase": "Confusión de clase",
                 "sin_solape_iou50": "Sin solape a IoU 0,50", "omision": "Omisión"}
        ax.set_title(f"{NAMES[name]} · imagen {ident}\n{trans[failures[0]['reason']]}", fontsize=11)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.legend(handles=[Patch(edgecolor="#2a8f4d", facecolor="none", label="Anotación"),
                        Patch(edgecolor="#cd3232", facecolor="none", label="Error en el punto operativo")],
               loc="outside lower center", ncols=2)
    save(fig, path)


def write(output, root, truth, selection, result, dataset, manifest):
    comparison = list(comparison_rows(selection, result))
    write_csv(output / "validacion_test.csv", comparison)
    by_model = {name: diagnostics(root, truth, name) for name in NAMES}
    errors = [row for records, _ in by_model.values() for row in records]
    fields = ("backend", "image_id", "file_name", "event", "reason", "detection_class", "score",
              "iou_with_gt", "nearest_gt_class", "bbox_xywh", "gt_bbox_xywh")
    write_csv(output / "errores_operativos.csv", errors, fields)
    overview(output, comparison, result)
    by_iou(output, root)
    error_panel(output / "errores_operativos.png", errors,
                {name: value[1] for name, value in by_model.items()}, dataset, manifest)
