"""Reproduce one fixed ensemble of the archived EXP-002 controls."""
import argparse
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from statistics import mean

from scripts.baseline_data import read_json, write_json
from scripts.baseline_metrics import MAX_DETECTIONS, MIN_SCORE, coco_eval, plot_pr, report, validate_predictions

archive = import_module("experiments.exp-002-photometric.analyze")
HERE = Path(__file__).resolve().parent
IOU = 0.55
YOLO_SINGLE_WEIGHT = 0.5


def overlap(a, b):
    x, y, w, h = a
    u, v, p, q = b
    area = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return area / (w * h + p * q - area)


def fuse(cascade, yolo):
    def group(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["image_id"], row["category_id"]].append(row)
        return groups

    c_groups, y_groups = group(cascade), group(yolo)
    images = defaultdict(list)
    for image, cls in sorted(c_groups.keys() | y_groups.keys()):
        primary = sorted(c_groups[image, cls], key=lambda row: -row["score"])
        secondary = y_groups[image, cls]
        matched = set()
        for c in primary:
            options = ((overlap(c["bbox"], y["bbox"]), -i) for i, y in enumerate(secondary) if i not in matched)
            iou, neg_index = max(options, default=(0, 0))
            if iou < IOU:
                images[image].append(c)
                continue
            index = -neg_index
            matched.add(index)
            y = secondary[index]
            confidence = c["score"] + y["score"]
            bbox = [(u * c["score"] + v * y["score"]) / confidence for u, v in zip(c["bbox"], y["bbox"])]
            images[image].append({**c, "bbox": bbox, "score": confidence / 2})
        for index, y in enumerate(secondary):
            score = y["score"] * YOLO_SINGLE_WEIGHT
            if index not in matched and score >= MIN_SCORE and all(overlap(c["bbox"], y["bbox"]) < IOU for c in primary):
                images[image].append({**y, "score": score})
    return [row for _, rows in sorted(images.items())
            for row in sorted(rows, key=lambda row: -row["score"])[:MAX_DETECTIONS]]


def run(data, output, source=archive.RESULTS / "s2026"):
    output.mkdir(parents=True, exist_ok=True)
    archive.verify_inventory(source)
    provenance = source / "provenance"
    code = archive.source_identity(provenance)
    truth, _, digests, splits = archive.load_inputs(provenance, data)
    image_ids = [image["id"] for image in truth["images"]]
    inputs, predictions, reports = {}, {}, {}
    for detector in ("cascade", "yolo"):
        folder = source / detector / "control"
        evaluation = folder / "evaluation"
        metrics = read_json(evaluation / "metrics.json")
        archive.verify_run(folder / "training", metrics, detector, "control", code, digests,
                           splits["train"]["images"], image_ids)
        predictions[detector] = read_json(evaluation / "predictions.json")
        reports[detector], curves, _ = report(coco_eval(truth, predictions[detector]))
        archive.require(all(metrics[key] == value for key, value in reports[detector].items())
                        and curves == read_json(evaluation / "pr_curves.json"), f"COCO exportado: {detector}")
        inputs[detector] = {
            "predictions_sha256": archive.sha256(evaluation / "predictions.json"),
            "checkpoint_sha256": metrics["checkpoint_sha256"], "job_id": metrics["hardware"]["job_id"],
        }
    fused = fuse(predictions["cascade"], predictions["yolo"])
    validate_predictions(truth, fused)
    reports["ensemble"], curves, _ = report(coco_eval(truth, fused))
    values = {}
    for name, metrics in reports.items():
        per_class = {row["name"]: row["ap"] for row in metrics["per_class"]}
        values[name] = {
            **{key: metrics[key] for key in ("map", "ap50", "ap75")},
            "map_without_ibuphil_600_mg": mean(ap for cls, ap in per_class.items() if cls != "Ibuphil 600 mg"),
            **{key: metrics["operating_point"][key] for key in ("precision", "recall", "tp", "fp", "fn")},
            **per_class,
        }
    rows = []
    for metric in values["cascade"]:
        cols = {key: value[metric] for key, value in values.items()}
        delta = cols["ensemble"] - cols["cascade"] if None not in (cols["ensemble"], cols["cascade"]) else None
        rows.append({"metric": metric, **cols, "delta_ensemble_cascade": delta})
    archive.write_csv(output / "resumen.csv", rows)
    write_json(output / "predictions.json", fused)
    write_json(output / "pr_curves.json", curves)
    write_json(output / "metrics.json", reports["ensemble"])
    write_json(output / "provenance.json", {"split": "valid", "coco_sha256": digests["valid.json"],
                                            "source_commit": archive.REVISION, "inputs": inputs,
                                            "fusion_code_sha256": archive.sha256(Path(__file__)),
                                            "iou": IOU, "yolo_single_weight": YOLO_SINGLE_WEIGHT,
                                            "prediction_counts": {**{k: len(v) for k, v in predictions.items()},
                                                                  "ensemble": len(fused)}})
    plot_pr(curves, output / "pr_curves.png")
    print(output / "resumen.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=archive.REPO / "data/pills-sxdht")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.data, args.output)
