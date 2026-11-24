"""Recalcula EXP-001 con predicciones exportadas y COCO local, sin inferencia."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from scripts.baseline_data import ACTIVE_IDS, class_mapping, convert_split, read_json, sha256
from scripts.baseline_metrics import OPERATING_IOU, OPERATING_SCORE, coco_eval, report

EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
SEEDS = (2026, 2027, 2028)
ARMS = {"control": "configs/baseline/yolo.json", "variante": "configs/experiments/yolo-lr-0005.json"}
RARE, COLD = 2, 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


@cache
def git_blob(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=REPO)


def verify_downloads(root):
    with (root / "provenance/downloaded.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            local = root / row["local_path"]
            # Los PNG originales son copias de consulta. Todos los datos tabulares son obligatorios.
            if local.suffix == ".png" and not local.exists():
                continue
            require(sha256(local) == row["sha256"], f"Artefacto alterado: {local}")


def verify_run(root, seed, arm, metrics):
    batch = "s2026" if seed == 2026 else "repeticiones"
    revision = (root / "provenance" / batch / "source-commit").read_text().strip()
    training = root / f"s{seed}" / arm / "training"
    run = read_json(training / "run.json")
    status = read_json(training / "status.json")
    runtime = read_json(training / "ultralytics_runtime.json")
    saved = read_json(training / "config.json")
    name = f"{arm}-s{seed}"
    expected = {**json.loads(git_blob(revision, ARMS[arm])), "seed": seed}
    require(saved == expected == run["run"]["config"] == metrics["model_config"], f"Configuración: {name}")
    args = runtime["args"]
    require(
        (args["seed"], args["lr0"], args["epochs"], args["batch"], args["imgsz"], args["optimizer"])
        == (seed, saved["lr"], saved["epochs"], saved["batch"], saved["imgsz"], saved["optimizer"]),
        f"Configuración efectiva: {name}",
    )
    invocation = run["invocation"]
    original_run = Path(invocation[invocation.index("--output") + 1])
    original_config = Path(ARMS[arm]) if seed == 2026 else original_run.with_suffix(".config.json")
    require(Path(invocation[invocation.index("--config") + 1]) == original_config
            and int(invocation[invocation.index("--seed") + 1]) == seed, f"Invocación: {name}")
    require(original_run.name == name and Path(metrics["checkpoint"]) == original_run / status["best_checkpoint"],
            f"Origen del checkpoint: {name}")
    require((metrics["backend"], metrics["split"], metrics["checkpoint_kind"]) == ("yolo", "valid", "best"),
            f"Evaluación: {name}")
    job = (training / "slurm.job").read_text().strip()
    require(
        status["status"] == "complete" and run["hardware"]["job_id"] == job
        and [(s["status"], s["job_id"]) for s in status["sessions"]] == [("complete", job)],
        f"Estado e ID SLURM: {name}",
    )
    require(run["code_sha256"] == metrics["code_sha256"], f"Archivos de entrenamiento/evaluación: {name}")
    for path, digest in run["code_sha256"].items():
        require(hashlib.sha256(git_blob(revision, path)).hexdigest() == digest, f"Revisión: {name}, {path}")
    require(run["run"]["dataset_manifest_sha256"] == metrics["dataset_manifest_sha256"],
            f"Manifiesto del dataset: {name}")
    with (training / "history.csv").open(newline="") as stream:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    steps = [row["step"] for row in history]
    require(steps == list(range(1, saved["epochs"] + 1)), f"Épocas: {name}")
    best = max(history, key=lambda row: (row["val/map"], row["step"]))
    require(best["step"] == status["best_step"] and math.isclose(
        best["val/map"], status["best_validation_map"], abs_tol=5e-7), f"Mejor época: {name}")
    return run, runtime, history


def iou(box_a, box_b):
    x, y, w, h = box_a
    u, v, width, height = box_b
    intersection = max(0, min(x + w, u + width) - max(x, u)) * max(0, min(y + h, v + height) - max(y, v))
    return intersection / (w * h + width * height - intersection)


def match_rows(ev):
    """Emparejamientos COCO en el rango de área completo, para IoU 0,50 y 0,75."""
    threshold_75 = int(np.flatnonzero(np.isclose(ev.params.iouThrs, 0.75))[0])
    for item in ev.evalImgs:
        if item is None or item["aRng"] != ev.params.areaRng[0]:
            continue
        for index, score in enumerate(item["dtScores"]):
            if not item["dtIgnore"][0, index]:
                yield (item["category_id"], score, int(item["dtMatches"][0, index]),
                       int(item["dtMatches"][threshold_75, index]))


def recall_at_precision(matches, instances):
    grouped = defaultdict(list)
    for _, score, gt_50, _ in matches:
        grouped[score].append(bool(gt_50))
    tp = predicted = best_tp = 0
    best_score = None
    for score, hits in sorted(grouped.items(), reverse=True):
        tp += sum(hits)
        predicted += len(hits)
        if tp > best_tp and tp / predicted >= 0.82:
            best_tp, best_score = tp, score
    return best_tp / instances, best_score


def diagnosis(truth, predictions, matches):
    by_image = defaultdict(list)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    at_point = [row for row in matches if row[1] >= OPERATING_SCORE]
    correct_at_point = {gt_50 for _, _, gt_50, _ in at_point if gt_50}
    errors = Counter()
    for ann in truth["annotations"]:
        if ann["id"] in correct_at_point:
            continue
        cid = ann["category_id"]
        eligible = [row for row in by_image[ann["image_id"]]
                    if row["score"] >= OPERATING_SCORE and row["category_id"] != cid
                    and iou(row["bbox"], ann["bbox"]) >= OPERATING_IOU]
        if eligible:
            errors[cid, max(eligible, key=lambda row: row["score"])["category_id"]] += 1

    values = {}
    for cid in ACTIVE_IDS:
        rows = [row for row in matches if row[0] == cid]
        point = [row for row in at_point if row[0] == cid]
        values[cid] = {
            "tp_export_iou50": sum(bool(row[2]) for row in rows),
            "tp_export_iou75": sum(bool(row[3]) for row in rows),
            "tp_score025_iou50": sum(bool(row[2]) for row in point),
            "fp_score025_iou50": sum(not row[2] for row in point),
            "fn_with_other_class_score025_iou50": sum(errors[cid, other] for other in ACTIVE_IDS),
        }
    return values, errors


def ibuphil_rows(truth, predictions, seed, arm):
    images = {row["id"]: row for row in truth["images"]}
    names = {row["id"]: row["name"] for row in truth["categories"]}
    anns = defaultdict(list)
    for row in truth["annotations"]:
        anns[row["image_id"]].append(row)
    rare_images = {ann["image_id"] for ann in truth["annotations"] if ann["category_id"] == RARE}
    for index, pred in enumerate(predictions):
        image = pred["image_id"]
        cid = pred["category_id"]
        overlaps = [(iou(pred["bbox"], ann["bbox"]), ann) for ann in anns[image]]
        on_cold = any(overlap >= 0.5 and ann["category_id"] == COLD for overlap, ann in overlaps)
        if image not in rare_images and cid != RARE and not on_cold:
            continue
        overlap, nearest = max(overlaps, key=lambda pair: pair[0], default=(0, None))
        yield {
            "seed": seed, "arm": arm, "prediction_index": index, "image_id": image,
            "file_name": images[image]["file_name"], "pred_class": names[cid], "score": pred["score"],
            "pred_bbox_xywh": json.dumps(pred["bbox"]),
            "nearest_gt_class": names[nearest["category_id"]] if nearest else "",
            "nearest_gt_id_prepared": nearest["id"] if nearest else "",
            "nearest_gt_bbox_xywh": json.dumps(nearest["bbox"]) if nearest else "",
            "iou_nearest_gt": overlap, "iou_to_cold_max": max(
                (overlap for overlap, ann in overlaps if ann["category_id"] == COLD), default=0
            ),
        }


def run_metrics(metrics, recalculated, curves, matches, history):
    benchmark = metrics["benchmark"]
    samples = benchmark["samples_ms"]
    require(len(samples) == benchmark["iterations"] == 100 and min(samples) > 0, "Muestras del benchmark")
    latencies = {
        "mean_ms_per_image": float(np.mean(samples)), "median_ms_per_image": float(np.median(samples)),
        "p95_ms_per_image": float(np.percentile(samples, 95)), "fps_from_mean": 1000 / float(np.mean(samples)),
    }
    for key, value in latencies.items():
        require(math.isclose(value, benchmark[key], abs_tol=1e-12), f"Latencia exportada: {key}")
    early = [row for row in history if row["step"] <= 5]
    peak = max(early, key=lambda row: row["val/cls_loss"])
    best = max(history, key=lambda row: (row["val/map"], row["step"]))
    recall, threshold = recall_at_precision(matches, metrics["instances"])
    return {
        **{key: recalculated[key] for key in ("map", "ap50", "ap75")}, **latencies,
        "map7": mean(row["ap"] for row in recalculated["per_class"] if row["category_id"] != RARE),
        "ap50_7": mean(mean(curves["classes"][str(cid)]["0.50"]) for cid in ACTIVE_IDS if cid != RARE),
        "ap75_7": mean(mean(curves["classes"][str(cid)]["0.75"]) for cid in ACTIVE_IDS if cid != RARE),
        "precision_score025_iou50": recalculated["operating_point"]["precision"],
        "recall_score025_iou50": recalculated["operating_point"]["recall"],
        "best_micro_recall_precision_ge082": recall, "score_best_micro_recall_precision_ge082": threshold,
        "epoch_best": best["step"], "map_best_history": best["val/map"], "map_epoch30": history[-1]["val/map"],
        "map_history_gap": best["val/map"] - history[-1]["val/map"],
        "peak_val_cls_loss_e1_e5": peak["val/cls_loss"], "epoch_peak_val_cls_loss_e1_e5": peak["step"],
        "zero_map_epochs_e1_e5": sum(row["val/map"] == 0 for row in early),
        "mean_map_e1_e5": mean(row["val/map"] for row in early),
        **{f"map_epoch{int(row['step'])}": row["val/map"] for row in early},
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(values):
    for metric, class_name, support in values[SEEDS[0], "control"]:
        key = (metric, class_name, support)
        row = {"metric": metric, "class": class_name, "n_valid": support}
        by_arm = {arm: [values[seed, arm][key] for seed in SEEDS] for arm in ARMS}
        deltas = [variant - control for control, variant in zip(by_arm["control"], by_arm["variante"])]
        for index, seed in enumerate(SEEDS):
            for arm in ARMS:
                row[f"s{seed}_{arm}"] = by_arm[arm][index]
            row[f"s{seed}_delta"] = deltas[index]
        for arm, series in [*by_arm.items(), ("delta", deltas)]:
            row[f"mean_{arm}"] = mean(series)
            row[f"sd_{arm}"] = stdev(series)
        yield row


def plot_history(histories, output):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, len(SEEDS), figsize=(10, 5.5), sharex=True, sharey="row", layout="constrained")
    for col, seed in enumerate(SEEDS):
        map_axis, loss_axis = axes[:, col]
        for arm, color, label in (("control", "#204d78", "Control · 0,001"), ("variante", "#bb5417", "Variante · 0,0005")):
            history = histories[seed, arm]
            epochs = [row["step"] for row in history]
            map_axis.plot(epochs, [row["val/map"] for row in history], color=color, label=label)
            best = max(history, key=lambda row: (row["val/map"], row["step"]))
            map_axis.scatter(best["step"], best["val/map"], color=color, s=23, zorder=3)
            loss_axis.plot(epochs, [row["val/cls_loss"] for row in history], color=color)
        for ax in (map_axis, loss_axis):
            ax.axvspan(1, 5, color="#ebebeb", zorder=0)
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        map_axis.set(title=f"Semilla {seed}", ylim=(-0.02, 0.85))
        loss_axis.set(yscale="log", xlabel="Época")
    axes[0, 0].set_ylabel("mAP COCO en validación")
    axes[1, 0].set_ylabel("Pérdida de clasificación en validación")
    axes[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(output, dpi=125)
    plt.close(fig)


def analyze(dataset):
    root = EXPERIMENT / "results"
    verify_downloads(root)
    raw = read_json(dataset / "valid/_annotations.coco.json")
    train = read_json(dataset / "train/_annotations.coco.json")
    classes = class_mapping({row["id"]: row for row in train["categories"]}, train["annotations"])
    truth, _ = convert_split(raw, classes)
    coco_bytes = (json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    input_hashes = {Path(path).name: digest for digest, path in (
        line.split(maxsplit=1) for line in (root / "provenance/s2026/inputs.sha256").read_text().splitlines()
    )}
    require(hashlib.sha256(coco_bytes).hexdigest() == input_hashes["valid.json"], "COCO local de validación")
    for split, data in (("train", train), ("valid", raw)):
        images = {row["id"]: row for row in data["images"]}
        targets = [images[ann["image_id"]] for ann in data["annotations"] if ann["category_id"] == RARE]
        require(len(targets) == 1, f"Soporte de Ibuphil en {split}")
        for image in targets:
            require((dataset / split / image["file_name"]).is_file(), f"JPEG de Ibuphil en {split}")
    values, histories, ibuphil = {}, {}, []
    benchmark_protocol = None
    all_errors = {}
    for seed in SEEDS:
        paired_args = []
        paired_benchmarks = []
        for arm in ARMS:
            evaluation = root / f"s{seed}" / arm / "evaluation"
            metrics = read_json(evaluation / "metrics.json")
            run, runtime, history = verify_run(root, seed, arm, metrics)
            paired_args.append({key: value for key, value in runtime["args"].items()
                                if key not in {"lr0", "name", "save_dir"}})
            require(run["run"]["initial_weights_sha256"] == input_hashes["yolov10s.pt"]
                    and metrics["dataset_manifest_sha256"] == input_hashes["manifest.json"]
                    and metrics["coco_sha256"] == input_hashes["valid.json"]
                    and metrics["image_ids"] == [image["id"] for image in truth["images"]], "Datos y pesos comunes")
            benchmark = metrics["benchmark"]
            protocol = {key: benchmark[key] for key in (
                "batch", "nominal_shape", "dtype", "warmup_iterations", "iterations", "images", "scope"
            )}
            require(protocol["images"] == [image["file_name"] for image in truth["images"][:16]],
                    "Orden del benchmark")
            if benchmark_protocol is None:
                benchmark_protocol = protocol
            require(protocol == benchmark_protocol, "Protocolos de benchmark diferentes")
            paired_benchmarks.append(metrics["hardware"])

            predictions = read_json(evaluation / "predictions.json")
            ev = coco_eval(truth, predictions)
            recalculated, curves, _ = report(ev)
            require(all(metrics[key] == value for key, value in recalculated.items()), f"COCO: {evaluation}")
            require(curves == read_json(evaluation / "pr_curves.json"), f"Curvas PR: {evaluation}")
            matches = list(match_rows(ev))
            diagnostic, errors = diagnosis(truth, predictions, matches)
            all_errors[seed, arm] = errors
            values[seed, arm] = {(key, "", ""): value for key, value in
                                 run_metrics(metrics, recalculated, curves, matches, history).items()}
            for row in recalculated["per_class"]:
                cid, name, support = row["category_id"], row["name"], row["ground_truth"]
                curve = curves["classes"][str(cid)]
                data = {"ap": row["ap"], "ap50": mean(curve["0.50"]), "ap75": mean(curve["0.75"]), **diagnostic[cid]}
                values[seed, arm].update({(key, name, support): value for key, value in data.items()})
            histories[seed, arm] = history
            ibuphil.extend(ibuphil_rows(truth, predictions, seed, arm))
        require(paired_args[0] == paired_args[1], f"Configuración efectiva pareada en {seed}")
        require(paired_benchmarks[0] == paired_benchmarks[1], f"Entorno de benchmark pareado en {seed}")

    names = {row["id"]: row["name"] for row in truth["categories"]}
    for cid, other in sorted(set().union(*(errors.keys() for errors in all_errors.values()))):
        for pair, errors in all_errors.items():
            values[pair][(f"fn_predicted_as_{names[other]}", names[cid], "")] = errors[cid, other]

    output = root / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "resumen.csv", list(summary_rows(values)))
    write_csv(output / "ibuphil.csv", ibuphil)
    plot_history(histories, output / "entrenamiento.png")
    print(f"Verificadas {len(histories)} evaluaciones de validación. Resultados en {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht",
                        help="directorio del dataset COCO original")
    analyze(parser.parse_args().data)
