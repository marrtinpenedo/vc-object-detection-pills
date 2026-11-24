"""Audita EXP-002 y recalcula COCO y las latencias a partir de archivos locales."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from scripts.baseline_data import ACTIVE_IDS, INSTANCE_COUNTS, SPLITS, convert_split, prepared_manifest, read_json
from scripts.baseline_metrics import OPERATING_IOU, OPERATING_SCORE, coco_eval, report

from .import_results import (
    ARMS, DETECTORS, EXPERIMENT, INVENTORY, RESULTS, RUNS, SEED,
    checksum_entries, require, required_paths, run_name, sha256,
)

REPO = EXPERIMENT.parents[1]
REVISION = "66ac3471594aa95411a92244b5cf03517ec7672d"
RARE, COLD = 2, 3
FOCUS = {1, RARE, COLD, 4, 6}
COLORS = (5, 6, 7, 8)
DETECTOR_NAMES = {"yolo": "YOLOv10-S", "cascade": "Cascade R-CNN"}
ARM_COLORS = {"control": "#235681", "variante": "#ad4b18"}
INITIAL_WEIGHTS = {"yolo": "yolov10s.pt", "cascade": "model_final_480dd8.pkl"}
BACKEND_PACKAGES = {"yolo": ("ultralytics", "8.3.161"), "cascade": ("detectron2", "0.6")}
BEST_CHECKPOINTS = {"yolo": Path("weights/best.pt"), "cascade": Path("best/model_best.pth")}
LATENCIES = ("mean_ms_per_image", "median_ms_per_image", "p95_ms_per_image", "fps_from_mean")


def verify_inventory(root):
    seen = set()
    required = set(required_paths())
    with (root / INVENTORY).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            local = Path(row["local_path"])
            path = root / local
            require(not local.is_absolute() and ".." not in local.parts and local not in seen,
                    f"Ruta de inventario inválida o repetida: {local}")
            seen.add(local)
            if local not in required and not path.exists() and not path.is_symlink():
                continue
            require(path.is_file() and path.resolve() == path and sha256(path) == row["sha256"],
                    f"Archivo ausente o huella distinta a la importación: {path}")
    missing = required - seen
    require(not missing, f"Archivos obligatorios sin inventariar: {', '.join(map(str, sorted(missing)))}")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})


def source_identity(provenance):
    source = (provenance / "source-commit").read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{7,40}", source) and REVISION.startswith(source),
            f"Se esperaba 66ac347 en source-commit, se obtuvo {source!r}")
    paths = git("ls-tree", "-r", "--name-only", REVISION, "scripts").decode().splitlines()
    return {path: hashlib.sha256(git("show", f"{REVISION}:{path}")).hexdigest()
            for path in paths if Path(path).parent == Path("scripts") and path.endswith(".py")}


def verify_preflight_hashes(provenance):
    path = provenance / "code-config.sha256"
    if not path.is_file():
        return None
    entries = list(checksum_entries(path))
    require(entries and len({name for _, name in entries}) == len(entries),
            "Huella de código o configuración vacía o duplicada")
    for digest, name in entries:
        require(not name.is_absolute() and ".." not in name.parts and
                hashlib.sha256(git("show", f"{REVISION}:{name}")).hexdigest() == digest,
                f"Huella de la revisión distinta del preflight: {name}")
    return len(entries)


def verify_split_isolation(manifest):
    images = {split: {} for split in SPLITS}
    for path, digest in manifest["files"].items():
        parts = Path(path).parts
        if len(parts) == 3 and parts[0] == "images":
            images[parts[1]][parts[2]] = digest
    for split, entries in images.items():
        summary = manifest["splits"][split]
        require(len(entries) == summary["images"] == SPLITS[split] and
                summary["instances"] == INSTANCE_COUNTS[split], f"Recuentos del manifiesto en {split}")
    for a, b in combinations(images, 2):
        require(not (images[a].keys() & images[b].keys()) and
                not (set(images[a].values()) & set(images[b].values())),
                f"Coinciden nombres o huellas de imágenes entre {a} y {b}")


def load_inputs(provenance, dataset):
    digests = {}
    for digest, path in checksum_entries(provenance / "inputs.sha256"):
        name = path.name
        require(name not in digests, f"Huella de entrada duplicada: {path}")
        digests[name] = digest
    require({"manifest.json", "valid.json", *INITIAL_WEIGHTS.values()} <= set(digests),
            "inputs.sha256 debe registrar el manifiesto, el COCO valid y ambos pesos iniciales")
    require(sha256(provenance / "manifest.json") == digests["manifest.json"], "Huella del manifiesto de datos")
    manifest = prepared_manifest(provenance)
    verify_split_isolation(manifest)
    counts = {}
    for split in ("train", "valid"):
        path = dataset / split / "_annotations.coco.json"
        expected = manifest["splits"][split]
        require(sha256(path) == expected["source_annotations_sha256"], f"Anotaciones originales de {split}")
        raw = read_json(path)
        counts[split] = Counter(ann["category_id"] for ann in raw["annotations"])
        require(len(raw["images"]) == expected["images"] and
                {str(cid): counts[split][cid] for cid in ACTIVE_IDS} == expected["ground_truth_by_category"],
                f"Recuentos de {split}")
    truth, _ = convert_split(raw, manifest["classes"])
    encoded = (json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(encoded).hexdigest() == digests["valid.json"] == manifest["files"]["coco/valid.json"],
            "El COCO reconstruido de validación difiere del evaluado")
    return truth, counts, digests, manifest["splits"]


def expected_config(detector, arm):
    path = f"configs/baseline/{detector}.json" if arm == "control" else f"configs/experiments/{detector}-photometric.json"
    return json.loads(git("show", f"{REVISION}:{path}"))


def verify_runtime(training, detector, spec, train_images):
    iterations = math.ceil(train_images / spec["batch"])
    photo = spec.get("photometric")
    if detector == "yolo":
        runtime = read_json(training / "ultralytics_runtime.json")
        args = runtime["args"]
        require((args["lr0"], args["seed"], args["epochs"], args["batch"], args["imgsz"]) ==
                (spec["lr"], SEED, spec["epochs"], spec["batch"], spec["imgsz"]) and not args["augment"],
                f"Argumentos efectivos de YOLO: {training}")
        chain = ["YoloPhotometric", "LetterBox", "Format"] if photo else None
        require(runtime.get("photometric") == photo and runtime.get("train_transforms") == chain,
                f"Transformaciones efectivas de YOLO: {training}")
        require(runtime["nominal_max_iterations"] == iterations * spec["epochs"],
                f"Presupuesto efectivo de YOLO: {training}")
        effective = runtime
        for key in ("name", "save_dir"):
            del args[key]
        for key in ("photometric", "train_transforms"):
            effective.pop(key, None)
    else:
        runtime = read_json(training / "cascade_runtime.json")
        cfg = yaml.safe_load((training / "detectron2.yaml").read_text())
        solver, input_cfg = cfg["SOLVER"], cfg["INPUT"]
        require((solver["BASE_LR"], cfg["SEED"], solver["MAX_ITER"], solver["IMS_PER_BATCH"]) ==
                (spec["lr"], SEED, iterations * spec["epochs"], spec["batch"]), f"Receta efectiva de Cascade: {training}")
        require(input_cfg.get("PHOTOMETRIC") == photo and input_cfg["RANDOM_FLIP"] == "none"
                and not input_cfg["CROP"]["ENABLED"], f"Transformaciones efectivas de Cascade: {training}")
        require(cfg["DATASETS"]["TRAIN"] == ["pills_baseline_train"] and
                cfg["DATASETS"]["TEST"] == ["pills_baseline_valid"] and
                cfg["MODEL"]["ROI_HEADS"]["NUM_CLASSES"] == len(ACTIVE_IDS) and not cfg["MODEL"]["MASK_ON"],
                f"Particiones de entrenamiento/validación o cabeza de Cascade: {training}")
        require(runtime["max_iter"] == iterations * spec["epochs"], f"Presupuesto efectivo de Cascade: {training}")
        input_cfg.pop("PHOTOMETRIC", None)
        del cfg["OUTPUT_DIR"]
        effective = {"runtime": runtime, "detectron2": cfg}
    require(runtime["iterations_per_epoch"] == iterations and runtime["effective_batch"] == spec["batch"],
            f"Iteraciones por época o lote efectivo: {training}")
    return (1 if detector == "yolo" else iterations), effective


def read_history(path, step_per_epoch, epochs, status):
    with path.open(encoding="utf-8", newline="") as stream:
        history = [{key: float(value) for key, value in row.items() if value != ""}
                   for row in csv.DictReader(stream)]
    steps = [row["step"] for row in history]
    require(steps and steps == sorted(set(steps)) and
            all(step.is_integer() and 1 <= step <= epochs * step_per_epoch for step in steps),
            f"Pasos del historial: {path}")
    scored = {row["step"]: row["val/map"] for row in history if "val/map" in row}
    require(list(scored) == list(range(step_per_epoch, (epochs + 1) * step_per_epoch, step_per_epoch))
            and all(0 <= value <= 1 for value in scored.values()), f"Validaciones por época: {path}")
    best_score = status["best_validation_map"]
    best_step = status["best_step"]
    require(best_step in scored and math.isclose(best_score, scored[best_step], abs_tol=5e-7)
            and math.isclose(best_score, max(scored.values()), abs_tol=5e-7), f"Selección de best: {path}")
    return history, {
        "best_epoch": best_step // step_per_epoch,
        "best_validation_map_history": best_score,
        "last_validation_map_history": scored[epochs * step_per_epoch],
        "best_minus_last_history": best_score - scored[epochs * step_per_epoch],
    }


def verify_run(training, metrics, detector, arm, code, digests, train_images, image_ids):
    run = read_json(training / "run.json")
    status = read_json(training / "status.json")
    spec = expected_config(detector, arm)
    require(read_json(training / "config.json") == spec == run["run"]["config"] == metrics["model_config"]
            and (spec["seed"], spec["epochs"]) == (SEED, 30), f"Configuración y semilla: {training}")
    require(run["run"]["backend"] == detector and run["mode"] == "train"
            and (metrics["backend"], metrics["split"], metrics["checkpoint_kind"]) == (detector, "valid", "best"),
            f"Tipo de entrenamiento y evaluación: {training}")
    job = (training / "slurm.job").read_text().strip().split(";")[0]
    require(job.isdecimal() and run["hardware"]["job_id"] == job and status["status"] == "complete"
            and status["sessions"] and status["sessions"][-1]["status"] == "complete",
            f"Estado o ID SLURM de entrenamiento: {training}")
    final_job = status["sessions"][-1]["job_id"]
    metadata_sources = [run, metrics]
    if final_job != job:
        resumed = read_json(training / "resume_runtime.json")
        require(resumed["mode"] == "resume" and resumed["run"] == run["run"]
                and resumed["hardware"]["job_id"] == final_job, f"Procedencia de la reanudación: {training}")
        metadata_sources.append(resumed)
    for metadata in metadata_sources:
        require(metadata["git_head"] in (None, REVISION) and metadata["code_sha256"] == code,
                f"Revisión del código registrada en {training}")
        package, version = BACKEND_PACKAGES[detector]
        require(metadata["packages"].get(package) == version and metadata["packages"].get("pycocotools") == "2.0.8",
                f"Versiones del detector o de COCOeval en {training}")
    invocation = run["invocation"]
    output = Path(invocation[invocation.index("--output") + 1])
    checkpoint = BEST_CHECKPOINTS[detector]
    require(output.name == run_name(detector, arm) and Path(status["best_checkpoint"]) == checkpoint
            and Path(metrics["checkpoint"]) == output / checkpoint
            and re.fullmatch(r"[0-9a-f]{64}", metrics["checkpoint_sha256"]),
            f"Correspondencia del checkpoint best: {training}")
    require(run["run"]["initial_weights_sha256"] == digests[INITIAL_WEIGHTS[detector]]
            and run["run"]["dataset_manifest_sha256"] == metrics["dataset_manifest_sha256"] == digests["manifest.json"]
            and metrics["coco_sha256"] == digests["valid.json"] and metrics["image_ids"] == image_ids,
            f"Datos, pesos e imágenes: {training}")
    step_unit = "epoch" if detector == "yolo" else "iteration"
    require(status["step_unit"] == step_unit, f"Unidad del historial: {training}")
    stride, effective = verify_runtime(training, detector, spec, train_images)
    history, summary = read_history(training / "history.csv", stride, spec["epochs"], status)
    return run, output.parent, job, final_job, stride, effective, history, summary


def benchmark_metrics(metrics, image_names):
    benchmark = metrics["benchmark"]
    require(isinstance(benchmark, dict), "Falta el benchmark de la evaluación")
    samples = benchmark["samples_ms"]
    require(len(samples) == benchmark["iterations"] and samples and
            all(math.isfinite(ms) and ms > 0 for ms in samples), "Muestras de latencia inválidas")
    values = dict(zip(LATENCIES, (float(np.mean(samples)), float(np.median(samples)),
                                float(np.percentile(samples, 95)), 1000 / float(np.mean(samples)))))
    require(all(math.isclose(value, benchmark[key], rel_tol=1e-12) for key, value in values.items()),
            "La latencia recalculada difiere del informe exportado")
    standard = (benchmark["batch"], benchmark["nominal_shape"], benchmark["dtype"],
                benchmark["warmup_iterations"], benchmark["iterations"], benchmark["images"])
    prescribed = (1, [640, 640], "fp32", 20, 100, image_names[:16])
    keys = ("job_id", "host", "gpu", "compute_capability", "cuda_runtime", "nvidia_driver", "cpu")
    hardware = metrics["hardware"]
    conditions = {key: hardware.get(key) for key in keys}
    return values, {
        "protocol": standard == prescribed,
        "scope": benchmark["scope"],
        "hardware": conditions,
    }


def benchmark_comparable(a, b):
    return (a["protocol"] and b["protocol"] and a["hardware"] == b["hardware"]
            and all(a["hardware"].values()) and bool(a["scope"]) and a["scope"] == b["scope"])


def match_scores(ev):
    matches = {threshold: {} for threshold in (OPERATING_IOU, 0.75)}
    for threshold, by_annotation in matches.items():
        index = int(np.flatnonzero(np.isclose(ev.params.iouThrs, threshold))[0])
        for item in ev.evalImgs:
            if item is None or item["aRng"] != ev.params.areaRng[0]:
                continue
            for gt, score, ignored in zip(item["dtMatches"][index], item["dtScores"], item["dtIgnore"][index]):
                if gt and not ignored:
                    by_annotation[int(gt)] = score
    return matches


def describe_candidate(candidate, prefix, names):
    detection, overlap = candidate if candidate else (None, None)
    return {
        f"{prefix}_class": names[detection["category_id"]] if detection else None,
        f"{prefix}_score": detection["score"] if detection else None,
        f"{prefix}_iou": overlap,
        f"{prefix}_bbox_xywh": json.dumps(detection["bbox"]) if detection else None,
    }


IBUPHIL_FIELDS = (
    "detector", "arm", "image_id", "file_name", "prediction_index", "pred_class", "score", "bbox_xywh",
    "rank_ibuphil_600_mg", "max_iou_ibuphil_600_mg", "max_iou_ibuphil_cold", "best_gt_id", "best_gt_class",
    "best_gt_iou", "best_gt_bbox_xywh",
)


def ibuphil_row(detector, arm, image, names, detection, index, rank, overlaps):
    best = max(overlaps, key=lambda pair: pair[1], default=None)
    annotation, overlap = best if best and best[1] > 0 else (None, 0)
    max_iou = {
        cid: max((value for ann, value in overlaps if ann["category_id"] == cid), default=0)
        for cid in (RARE, COLD)
    }
    if not (any(ann["category_id"] == RARE for ann, _ in overlaps)
            or detection["category_id"] == RARE or max_iou[COLD] >= OPERATING_IOU):
        return None
    return dict(zip(IBUPHIL_FIELDS, (
        detector, arm, image["id"], image["file_name"], index, names[detection["category_id"]],
        detection["score"], json.dumps(detection["bbox"]), rank.get(index), max_iou[RARE], max_iou[COLD],
        annotation["id"] if annotation else None, names[annotation["category_id"]] if annotation else None,
        overlap, json.dumps(annotation["bbox"]) if annotation else None,
    )))


def diagnose(detector, arm, truth, predictions, ev, examples):
    from pycocotools.mask import iou

    names = {c["id"]: c["name"] for c in truth["categories"]}
    matches = match_scores(ev)
    matched_at_operating_point = {ann for ann, score in matches[OPERATING_IOU].items() if score >= OPERATING_SCORE}
    ap_counts = {
        f"tp_export_iou{code}": Counter(ev.cocoGt.anns[ann]["category_id"] for ann in matches[threshold])
        for threshold, code in ((OPERATING_IOU, 50), (0.75, 75))
    }
    counts = {
        **ap_counts,
        **{f"{event}_score025_iou50": Counter(row["category_id"] for image in examples.values() for row in image[event])
           for event in ("tp", "fp", "fn")},
    }
    anns_by_image = defaultdict(list)
    preds_by_image = defaultdict(list)
    for ann in truth["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    for index, detection in enumerate(predictions):
        preds_by_image[detection["image_id"]].append((index, detection))
    rare_predictions = sorted((i for i, p in enumerate(predictions) if p["category_id"] == RARE),
                              key=lambda i: -predictions[i]["score"])
    ranks = {i: rank for rank, i in enumerate(rare_predictions, start=1)}
    confusions = Counter()
    focused, rare = [], []
    for image in truth["images"]:
        anns = anns_by_image[image["id"]]
        indexed = preds_by_image[image["id"]]
        preds = [det for _, det in indexed]
        overlaps = (iou([det["bbox"] for det in preds], [ann["bbox"] for ann in anns], [0] * len(anns))
                    if preds and anns else np.zeros((len(preds), len(anns))))
        for column, ann in enumerate(anns):
            cid = ann["category_id"]
            match_score = matches[OPERATING_IOU].get(ann["id"])
            tp = match_score is not None and match_score >= OPERATING_SCORE
            same, other = [], []
            for det, overlap in zip(preds, overlaps[:, column]):
                if det["category_id"] == cid:
                    same.append((det, overlap))
                elif overlap >= OPERATING_IOU:
                    other.append((det, overlap))
            best_other = max(other, key=lambda pair: pair[0]["score"], default=None)
            if not tp and best_other and best_other[0]["score"] >= OPERATING_SCORE:
                confusions[cid, best_other[0]["category_id"]] += 1
            if cid in FOCUS:
                best_same = max(same, key=lambda pair: (pair[1], pair[0]["score"]), default=None)
                focused.append({
                    "detector": detector, "arm": arm, "image_id": image["id"], "file_name": image["file_name"],
                    "gt_id": ann["id"], "gt_class": names[cid], "gt_bbox_xywh": json.dumps(ann["bbox"]),
                    "tp_score025_iou50": int(tp), "coco_match_score_iou50": match_score,
                    **describe_candidate(best_same, "best_same", names),
                    **describe_candidate(best_other, "best_other", names),
                })
        for row, (index, detection) in enumerate(indexed):
            record = ibuphil_row(detector, arm, image, names, detection, index, ranks, list(zip(anns, overlaps[row])))
            if record:
                rare.append(record)
    return counts, confusions, focused, rare, matched_at_operating_point


def comparison_rows(values, comparable, labels=("metric",)):
    for keys in values[RUNS[0]]:
        row = dict(zip(labels, keys if isinstance(keys, tuple) else (keys,)))
        for detector, arm in RUNS:
            row[f"{detector}_{arm}"] = values[detector, arm][keys]
        contrasts = {
            "yolo_delta": (("yolo", "variante"), ("yolo", "control")),
            "cascade_delta": (("cascade", "variante"), ("cascade", "control")),
            "cascade_minus_yolo_control": (("cascade", "control"), ("yolo", "control")),
            "cascade_minus_yolo_variante": (("cascade", "variante"), ("yolo", "variante")),
        }
        for name, (a, b) in contrasts.items():
            left, right = values[a][keys], values[b][keys]
            valid = left is not None and right is not None and (keys not in LATENCIES or comparable[a, b])
            row[name] = left - right if valid else None
        yolo, cascade = row["yolo_delta"], row["cascade_delta"]
        valid = yolo is not None and cascade is not None
        if keys in LATENCIES:
            valid = valid and all(comparable.values())
        row["delta_cascade_minus_delta_yolo"] = cascade - yolo if valid else None
        yield row


def write_csv(path, rows, fieldnames=()):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(dataset, results):
    root = results / f"s{SEED}"
    verify_inventory(root)
    provenance = root / "provenance"
    code = source_identity(provenance)
    preflight_hashes = verify_preflight_hashes(provenance)
    truth, support, digests, splits = load_inputs(provenance, dataset)
    image_ids = [image["id"] for image in truth["images"]]
    image_names = [image["file_name"] for image in truth["images"]]
    names = {c["id"]: c["name"] for c in truth["categories"]}
    global_values, class_values, confusion_values = {}, {}, {}
    benchmarks, histories, receipts, effective_configs = {}, {}, {}, {}
    operating_matches = {}
    focused, rare, training_roots = [], [], set()
    for detector, arm in RUNS:
        pair = (detector, arm)
        folder = root / detector / arm
        metrics = read_json(folder / "evaluation/metrics.json")
        run, training_root, job, final_job, stride, effective, history, history_values = verify_run(
            folder / "training", metrics, detector, arm, code, digests, splits["train"]["images"], image_ids
        )
        effective_configs[pair] = effective
        training_roots.add(training_root)
        predictions = read_json(folder / "evaluation/predictions.json")
        evaluation = coco_eval(truth, predictions)
        recalculated, curves, examples = report(evaluation)
        require(all(metrics[key] == value for key, value in recalculated.items()), f"Desacuerdo con COCO exportado: {folder}")
        require(curves == read_json(folder / "evaluation/pr_curves.json"), f"Desacuerdo con curvas PR: {folder}")
        latency, benchmarks[pair] = benchmark_metrics(metrics, image_names)
        counts, confusions, inst, ibuphil, operating_matches[pair] = diagnose(
            detector, arm, truth, predictions, evaluation, examples
        )
        focused.extend(inst)
        rare.extend(ibuphil)
        ap = {row["category_id"]: row["ap"] for row in recalculated["per_class"]}
        operating = recalculated["operating_point"]
        global_values[pair] = {
            **{key: recalculated[key] for key in ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")},
            "map_without_ibuphil_600_mg": mean(ap[cid] for cid in ACTIVE_IDS if cid != RARE),
            "map_color_classes": mean(ap[cid] for cid in COLORS),
            **{f"{key}_score025_iou50": operating[key] for key in ("precision", "recall", "tp", "fp", "fn")},
            **latency, **history_values,
        }
        class_values[pair] = {}
        for cid in ACTIVE_IDS:
            keys = (names[cid], support["train"][cid], support["valid"][cid])
            curve = curves["classes"][str(cid)]
            values = {
                "ap": ap[cid], "ap50": mean(curve["0.50"]), "ap75": mean(curve["0.75"]),
                **{key: by_class[cid] for key, by_class in counts.items()},
            }
            class_values[pair].update({(key, *keys): value for key, value in values.items()})
        confusion_values[pair] = {(names[cid], names[other]): confusions[cid, other]
                                  for cid in ACTIVE_IDS for other in ACTIVE_IDS if cid != other}
        histories[pair] = (stride, history, history_values["best_epoch"])
        receipts[run_name(detector, arm)] = {
            "train_job": job, "final_train_job": final_job, "train_hardware": run["hardware"],
            "evaluation": benchmarks[pair],
            "backend_version": BACKEND_PACKAGES[detector][1],
            "checkpoint": metrics["checkpoint"], "checkpoint_sha256": metrics["checkpoint_sha256"],
        }

    require(len(training_roots) == 1 and len({receipt["train_job"] for receipt in receipts.values()}) == len(RUNS),
            "Se esperaban cuatro trabajos distintos en una sola tanda de entrenamiento")
    evaluation_jobs = {entry["hardware"]["job_id"] for entry in benchmarks.values()}
    require(len(evaluation_jobs) == 1 and all(evaluation_jobs), "Se esperaba un único trabajo SLURM de evaluación")
    for detector in DETECTORS:
        require(effective_configs[detector, "control"] == effective_configs[detector, "variante"],
                f"Diferencia adicional en la configuración efectiva de {detector}")
    contrasts = [
        (("yolo", "variante"), ("yolo", "control")),
        (("cascade", "variante"), ("cascade", "control")),
        (("cascade", "control"), ("yolo", "control")),
        (("cascade", "variante"), ("yolo", "variante")),
    ]
    comparable = {(a, b): benchmark_comparable(benchmarks[a], benchmarks[b]) for a, b in contrasts}
    overlap = {}
    for arm in ARMS:
        yolo, cascade = operating_matches["yolo", arm], operating_matches["cascade", arm]
        overlap[arm] = {"both": len(yolo & cascade), "yolo_only": len(yolo - cascade),
                        "cascade_only": len(cascade - yolo), "neither": len(truth["annotations"]) - len(yolo | cascade)}
    output = results / "analysis"
    require(not output.is_symlink(), f"Se requiere una carpeta local para el análisis: {output}")
    with TemporaryDirectory(prefix=".analysis-exp002-", dir=results) as temporary:
        staged = Path(temporary)
        write_csv(staged / "resumen.csv", comparison_rows(global_values, comparable))
        write_csv(staged / "por_clase.csv",
                  comparison_rows(class_values, comparable, ("metric", "class", "n_train", "n_valid")))
        write_csv(staged / "confusiones.csv", comparison_rows(confusion_values, comparable, ("gt_class", "other_class")))
        write_csv(staged / "errores_instancias.csv", focused)
        write_csv(staged / "ibuphil.csv", rare, IBUPHIL_FIELDS)
        audit = {
            "source_commit": REVISION, "original_training_root": str(next(iter(training_roots))),
            "inputs_sha256": digests, "dataset_splits": splits, "runs": receipts,
            "evaluation_job": next(iter(evaluation_jobs)), "preflight_hashes_verified": preflight_hashes,
            "operating_match_overlap": overlap,
            "comparable_latency": {f"{run_name(*a)} - {run_name(*b)}": flag for (a, b), flag in comparable.items()},
        }
        (staged / "verificacion.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        plot_history(histories, staged / "entrenamiento.png")
        plot_comparison(global_values, staged / "rendimiento.png")
        plot_class_deltas(class_values, names, support["valid"], staged / "ap_por_clase.png")
        output.mkdir(exist_ok=True)
        for path in staged.iterdir():
            path.replace(output / path.name)
    return output, comparable


def plot_history(histories, output):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", layout="constrained")
    for col, detector in enumerate(DETECTORS):
        ap_axis, loss_axis = axes[:, col]
        # Training losses have different definitions and ranges in the two backends.
        loss = "train/cls_loss" if detector == "yolo" else "total_loss"
        for arm in ARMS:
            stride, history, best = histories[detector, arm]
            color = ARM_COLORS[arm]
            epochs = [row["step"] / stride for row in history if "val/map" in row]
            scores = [100 * row["val/map"] for row in history if "val/map" in row]
            ap_axis.plot(epochs, scores, color=color, label=arm.capitalize())
            ap_axis.scatter(best, scores[epochs.index(best)], color=color, s=34, zorder=3)
            loss_axis.plot([row["step"] / stride for row in history if loss in row],
                           [row[loss] for row in history if loss in row], color=color, linewidth=1.15)
        ap_axis.set(title=DETECTOR_NAMES[detector], ylim=(0, 103))
        loss_axis.set(xlabel="Época", ylabel="Pérdida de clasificación" if detector == "yolo" else "Pérdida total")
        for axis in (ap_axis, loss_axis):
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("mAP COCO en validación (%)")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_comparison(values, output):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True, layout="constrained")
    metrics = ("map", "ap50", "ap75")
    for axis, detector in zip(axes, DETECTORS):
        centers = np.arange(len(metrics))
        for index, arm in enumerate(ARMS):
            heights = [100 * values[detector, arm][key] for key in metrics]
            bars = axis.bar(centers + (index - 0.5) * 0.34, heights, 0.31,
                            label=arm.capitalize(), color=ARM_COLORS[arm])
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set(title=DETECTOR_NAMES[detector], ylim=(0, 110), xticks=centers,
                 xticklabels=("mAP COCO", "AP50", "AP75"), yticks=range(0, 101, 20))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validación (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_class_deltas(values, names, support, output):
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    positions = np.arange(len(ACTIVE_IDS))
    for detector, offset, marker, color in (("yolo", -0.15, "o", "#235681"), ("cascade", 0.15, "D", "#ad4b18")):
        controls = {key[1]: val for key, val in values[detector, "control"].items() if key[0] == "ap"}
        variants = {key[1]: val for key, val in values[detector, "variante"].items() if key[0] == "ap"}
        deltas = [100 * (variants[names[cid]] - controls[names[cid]]) for cid in ACTIVE_IDS]
        axis.scatter(deltas, positions + offset, marker=marker, color=color, s=40, label=DETECTOR_NAMES[detector], zorder=3)
    axis.axvline(0, color="#5b6269", linewidth=1)
    axis.grid(axis="x", alpha=0.25)
    axis.set(yticks=positions, yticklabels=[f"{names[cid]} ({support[cid]})" for cid in ACTIVE_IDS],
             xlabel="Δ AP COCO (puntos porcentuales)")
    axis.invert_yaxis()
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht", help="dataset COCO original local")
    parser.add_argument("--results", type=Path, default=RESULTS, help="directorio de resultados importados")
    options = parser.parse_args()
    try:
        output, comparable = analyze(options.data.resolve(), options.results.resolve())
    except (OSError, ValueError, IndexError, KeyError, TypeError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Verificadas cuatro evaluaciones de validación. Análisis en {output}")
    if not all(comparable.values()):
        print("Hay deltas de latencia sin comparación válida. Consulta verificacion.json.")


if __name__ == "__main__":
    main()
