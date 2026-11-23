"""Shared COCO scoring and bounded qualitative outputs."""

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

MIN_SCORE = 0.001
MAX_DETECTIONS = 100
OPERATING_SCORE = 0.25
OPERATING_IOU = 0.50


def prediction_rows(image, detections, classes):
    """Convert native-image xyxy, confidence and contiguous class to COCO."""
    original_ids = {c["internal_id"]: c["original_id"] for c in classes}
    rows = []
    for x1, y1, x2, y2, score, cls in np.asarray(detections).reshape(-1, 6):
        if not np.isfinite([x1, y1, x2, y2, score, cls]).all():
            raise ValueError(f"Predicción no finita en {image['file_name']}")
        if int(cls) != cls or int(cls) not in original_ids or not 0 <= score <= 1:
            raise ValueError(f"Clase o confianza inválida en {image['file_name']}")
        # Subtract in double precision so the COCO extents preserve the clipped edges.
        x1, x2 = map(float, np.clip([x1, x2], 0, image["width"]))
        y1, y2 = map(float, np.clip([y1, y2], 0, image["height"]))
        if x2 <= x1 or y2 <= y1:
            continue
        if score >= MIN_SCORE:
            rows.append({
                "image_id": image["id"],
                "category_id": original_ids[int(cls)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return sorted(rows, key=lambda row: -row["score"])[:MAX_DETECTIONS]


def validate_predictions(truth, predictions):
    images = {image["id"]: image for image in truth["images"]}
    categories = {category["id"] for category in truth["categories"]}
    counts = Counter()
    for row in predictions:
        image_id, cls, score = row["image_id"], row["category_id"], row["score"]
        if type(image_id) is not int or image_id not in images or type(cls) is not int or cls not in categories:
            raise ValueError(f"ID de imagen o categoría ajeno al COCO de evaluación: {row}")
        x, y, w, h = row["bbox"]
        image = images[image_id]
        if (
            not np.isfinite([x, y, w, h, score]).all() or min(x, y) < 0 or min(w, h) <= 0
            or x + w > image["width"] + 1e-6 or y + h > image["height"] + 1e-6
            or not MIN_SCORE <= score <= 1
        ):
            raise ValueError(f"Predicción fuera de los límites del protocolo: {row}")
        counts[image_id] += 1
    if any(count > MAX_DETECTIONS for count in counts.values()):
        raise ValueError("Se superó el máximo común de detecciones por imagen")


def coco_eval(truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    validate_predictions(truth, predictions)
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = truth
        gt.createIndex()
        if predictions:
            # loadRes muta los diccionarios y añade sus propios IDs.
            dt = gt.loadRes([dict(row) for row in predictions])
        else:
            dt = COCO()
            dt.dataset = {**truth, "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(gt.getImgIds())
        ev.params.catIds = sorted(gt.getCatIds())
        ev.params.maxDets = [1, 10, MAX_DETECTIONS]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def defined(value):
    return None if value < 0 else float(value)


def operating_point(ev):
    """Use COCO's score-ordered, class-aware matches at the fixed operating point."""
    idx = int(np.flatnonzero(np.isclose(ev.params.iouThrs, OPERATING_IOU))[0])
    by_image = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})
    counts = Counter()
    for item in ev.evalImgs:
        if item is None or item["aRng"] != ev.params.areaRng[0]:
            continue
        groups = by_image[item["image_id"]]
        tps = set()
        for j, (ident, score) in enumerate(zip(item["dtIds"], item["dtScores"])):
            if score < OPERATING_SCORE or item["dtIgnore"][idx, j]:
                continue
            gt_id = int(item["dtMatches"][idx, j])
            key = "tp" if gt_id else "fp"
            tps.add(gt_id)
            groups[key].append(ev.cocoDt.anns[ident])
            counts[key] += 1
        for j, ident in enumerate(item["gtIds"]):
            if ident not in tps and not item["gtIgnore"][j]:
                groups["fn"].append(ev.cocoGt.anns[ident])
                counts["fn"] += 1
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    metrics = {
        "score": OPERATING_SCORE, "iou": OPERATING_IOU, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
    return metrics, by_image


def report(ev):
    params = ev.params
    counts = Counter(ann["category_id"] for ann in ev.cocoGt.dataset["annotations"])
    precision = ev.eval["precision"][:, :, :, 0, -1]
    per_class = []
    curves = {}
    for index, cid in enumerate(params.catIds):
        values = precision[:, :, index]
        valid = values[values >= 0]
        name = ev.cocoGt.cats[cid]["name"]
        per_class.append({
            "category_id": int(cid), "name": name, "ground_truth": counts[cid],
            "ap": float(valid.mean()) if len(valid) else None,
        })
        curves[str(cid)] = {
            "name": name,
            **{f"{iou:.2f}": [defined(v) for v in values[i]] for i, iou in enumerate(params.iouThrs)},
        }
    metrics = {
        **dict(zip(
            ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"),
            map(defined, ev.stats[:6]),
        )),
        "per_class": per_class,
        "images": len(params.imgIds),
        "instances": sum(counts.values()),
        "parameters": {
            "iou_type": "bbox", "iou_thresholds": params.iouThrs.tolist(),
            "recall_thresholds": params.recThrs.tolist(), "max_detections": params.maxDets,
            "areas_px_squared": dict(zip(params.areaRngLbl, params.areaRng)),
            "minimum_export_score": MIN_SCORE, "use_categories": bool(params.useCats),
        },
    }
    metrics["operating_point"], examples = operating_point(ev)
    return metrics, {"recall": params.recThrs.tolist(), "classes": curves}, examples


def plot_pr(curves, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, iou in zip(axes, ("0.50", "0.75")):
        for item in curves["classes"].values():
            if any(value is not None for value in item[iou]):
                ax.plot(curves["recall"], item[iou], label=item["name"])
        ax.set(xlim=(0, 1), ylim=(0, 1.03), title=f"IoU {iou}", xlabel="Recall")
        if ax.lines:
            ax.legend(fontsize="small")
    axes[0].set_ylabel("Precisión interpolada")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_examples(ev, examples, images_root, output, limit):
    from PIL import Image, ImageDraw

    output.mkdir(exist_ok=True)
    ids = sorted(examples)
    failures = [i for i in ids if examples[i]["fp"] or examples[i]["fn"]]
    correct = [i for i in ids if i not in failures and examples[i]["tp"]]
    failures.sort(key=lambda i: (-(len(examples[i]["fp"]) + len(examples[i]["fn"])), i))
    for kind, selected in (("correct", correct[:limit]), ("failure", failures[:limit])):
        for image_id in selected:
            with Image.open(images_root / ev.cocoGt.imgs[image_id]["file_name"]) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for group, color in (("tp", "#00ff78"), ("fp", "#ff574f"), ("fn", "#ffc000")):
                for row in examples[image_id][group]:
                    x, y, w, h = row["bbox"]
                    draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                    label = ev.cocoGt.cats[row["category_id"]]["name"]
                    draw.text((x, max(0, y - 12)), f"{group.upper()} {label}", fill=color, stroke_width=1)
            image.save(output / f"{kind}-{image_id}.jpg")
