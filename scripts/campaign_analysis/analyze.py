"""Reproduce COCO validation, training histories and factorial contrasts locally."""
import argparse
import hashlib
import json
import math
import re
from importlib import import_module
from pathlib import Path
from statistics import mean

import yaml

from scripts.baseline_data import ACTIVE_IDS, convert_split, read_json, write_json
from scripts.baseline_metrics import coco_eval, plot_pr, report, save_examples
from scripts.campaign import (
    PLAN, REPO, attempt_dir, attempt_snapshot, condition_config, graph, snapshot_file, verify_yolo_recovery,
)
from .import_results import require, sha256, verify_inventory
from .reporting import RARE_CLASS, SENSITIVITY, write_reports

archive = import_module("experiments.exp-002-photometric.analyze")

CONTRASTS = {
    "epochs_no_aug": {"b": 1, "a": -1},
    "epochs_aug": {"d": 1, "c": -1},
    "augmentation_30": {"c": 1, "a": -1},
    "augmentation_100": {"d": 1, "b": -1},
    "interaction": {"d": 1, "b": -1, "c": -1, "a": 1},
    "mosaic_100": {"d": 1, "e": -1},
    "affine_only_100": {"e": 1, "b": -1},
}
COCO_METRICS = ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
BEST = {"yolo": "weights/best.pt", "cascade": "best/model_best.pth"}


def load_dataset(dataset, root, plan):
    path = root / "prepared-manifest.json"
    require(sha256(path) == plan["dataset_manifest_sha256"], "Huella del manifiesto preparado")
    prepared = read_json(path)
    archive.verify_split_isolation(prepared)
    require([c["original_id"] for c in prepared["classes"]] == list(ACTIVE_IDS), "Mapeo de clases")
    for split in ("train", "valid"):
        source = dataset / split / "_annotations.coco.json"
        require(sha256(source) == prepared["splits"][split]["source_annotations_sha256"], f"Anotaciones de {split}")
    truth, _ = convert_split(read_json(source), prepared["classes"])
    encoded = (json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(encoded).hexdigest() == plan["valid_coco_sha256"], "COCO de validación reconstruido")
    return truth, prepared["splits"]


def job_record(root, snapshot, name, record):
    path = attempt_dir(root, name, record["attempt"])
    if not (path / "done.json").is_file():
        return None
    require(read_json(path / "done.json") == {"job_id": record["id"], "snapshot_sha256": snapshot},
            f"Recibo incoherente en {path}")
    return path


def verify_runtime(folder, backend, spec, n_train):
    iterations = math.ceil(n_train / spec["batch"])
    geom, mosaic = spec.get("geometric"), spec.get("mosaic", 0.0)
    if backend == "yolo":
        summary = read_json(folder / "ultralytics_runtime.json")
        args = summary["args"]
        fields = {"lr0": spec["lr"], "seed": spec["seed"], "epochs": spec["epochs"], "batch": spec["batch"],
                  "imgsz": spec["imgsz"], "workers": spec["workers"], "optimizer": spec["optimizer"],
                  "lrf": spec["final_lr_factor"], "warmup_epochs": spec["warmup_epochs"], "cos_lr": True,
                  "val": True, "split": "val", "close_mosaic": 0, "mosaic": mosaic, "augment": False,
                  "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "amp": False}
        chain = (["YoloMosaic"] if mosaic else []) + ["YoloGeometry", "LetterBox", "Format"] if geom else None
        require(all(args[key] == value for key, value in fields.items())
                and summary.get("geometric") == geom and summary.get("train_transforms") == chain
                and summary["nominal_max_iterations"] == iterations * spec["epochs"], f"Receta efectiva: {folder}")
        # Each attempt has its own output parent, including retries from another snapshot.
        effective = {"args": {k: v for k, v in args.items() if k not in
                              ("project", "name", "save_dir", "epochs", "mosaic")},
                     "groups": summary["optimizer_initial_groups"], "warmup": summary["warmup_iterations_end_index"]}
    else:
        summary = read_json(folder / "cascade_runtime.json")
        cfg = yaml.safe_load((folder / "detectron2.yaml").read_text())
        solver, image = cfg["SOLVER"], cfg["INPUT"]
        fields = {"BASE_LR": spec["lr"], "MAX_ITER": spec["epochs"] * iterations, "IMS_PER_BATCH": spec["batch"],
                  "WARMUP_ITERS": spec["warmup_iterations"], "CHECKPOINT_PERIOD": iterations,
                  "LR_SCHEDULER_NAME": spec["scheduler"]}
        require(all(solver[key] == value for key, value in fields.items())
                and image.get("GEOMETRIC") == geom and image["RANDOM_FLIP"] == "none" and not image["CROP"]["ENABLED"]
                and not cfg["TEST"]["AUG"]["ENABLED"] and cfg["TEST"]["EVAL_PERIOD"] == iterations
                and cfg["SEED"] == spec["seed"] and cfg["DATASETS"]["TRAIN"] == ["pills_baseline_train"]
                and cfg["DATASETS"]["TEST"] == ["pills_baseline_valid"] and cfg["MODEL"]["ROI_HEADS"]["NUM_CLASSES"] == 8
                and not cfg["MODEL"]["MASK_ON"] and summary["max_iter"] == solver["MAX_ITER"], f"Receta efectiva: {folder}")
        del solver["MAX_ITER"], cfg["OUTPUT_DIR"]
        image.pop("GEOMETRIC", None)
        effective = cfg
    require((summary["iterations_per_epoch"], summary["effective_batch"]) == (iterations, spec["batch"]),
            f"Presupuesto efectivo: {folder}")
    return (1 if backend == "yolo" else iterations), effective


def verify_provenance(record, backend, job_id, snapshot, plan):
    scripts = {name: digest for name, digest in snapshot["sha256"].items() if name.startswith("scripts/")}
    package, version = plan["packages"][backend]
    require(record["code_sha256"] == scripts and record["packages"][package] == version
            and record["hardware"]["job_id"] == job_id, "Código, versión o trabajo distintos")


def evaluate_condition(root, manifest, snapshots, ledger, states, condition, truth, splits):
    name, backend = condition["name"], condition["backend"]
    plan = manifest["plan"]
    training_name, evaluation_name, probe_name = f"train-{name}", f"eval-{name}", f"preflight-{backend}"
    candidates = [entry for entry in ledger.get(evaluation_name, [])
                  if states.get(entry["id"]) == "COMPLETED"]
    if not candidates:
        return None
    require(len(candidates) == 1, f"Dos evaluaciones completas para {name}")
    ev_job = candidates[0]
    digest = attempt_snapshot(manifest, ev_job)
    snapshot = snapshots[digest]
    train_job = next(entry for entry in ledger[training_name] if entry["id"] == ev_job["parent_id"])
    probe_job = next(entry for entry in ledger[probe_name] if entry["id"] == train_job["parent_id"])
    require(all(states.get(job["id"]) == "COMPLETED" and attempt_snapshot(manifest, job) == digest
                for job in (train_job, probe_job)), f"Cadena incompleta o con distinto snapshot: {name}")
    probe = job_record(root, digest, probe_name, probe_job)
    training = job_record(root, digest, training_name, train_job)
    evaluated = job_record(root, digest, evaluation_name, ev_job)
    require(probe and training and evaluated, f"Faltan recibos de la cadena: {name}")
    for initialized in manifest["nodes"][probe_name]["probes"]:
        folder = probe / initialized
        status = read_json(folder / "status.json")
        run = read_json(folder / "run.json")
        verify_provenance(run, backend, probe_job["id"], snapshot, plan)
        require(status["status"] == "initialized" and run["mode"] == "initialize"
                and run["run"]["config"] == manifest["recipes"][initialized], f"Preflight incompleto: {initialized}")
        verify_runtime(folder, backend, manifest["recipes"][initialized], splits["train"]["images"])
        init = read_json(folder / "initialization.json")
        require(init["classes"] == 8 and "nonzero" in init["gradients"].values()
                and init["losses"] and all(math.isfinite(x) for x in init["losses"].values()),
                f"Pérdidas o gradientes del preflight inválidos: {initialized}")

    train = training / "training"
    evaluation = evaluated / "evaluation"
    run, status = (read_json(train / file) for file in ("run.json", "status.json"))
    original = read_json(evaluation / "metrics.json")
    spec = condition_config(plan, condition)
    require(spec == read_json(root / "configs" / f"{name}.json") == manifest["recipes"][name]
            == run["run"]["config"] == original["model_config"], f"Receta declarada: {name}")
    verify_provenance(run, backend, train_job["id"], snapshot, plan)
    verify_provenance(original, backend, ev_job["id"], snapshot, plan)
    expected_path = (Path(manifest["work"]) / plan["weights"][backend]["path"])
    args = run["invocation"]
    train_output = Path(args[args.index("--output") + 1])
    require(train_output.parts[-4:] == ("jobs", training_name, f"attempt-{train_job['attempt']:02d}", "training")
            and Path(original["checkpoint"]) == train_output / BEST[backend]
            and status["best_checkpoint"] == BEST[backend], f"Ruta del checkpoint: {name}")
    require(re.fullmatch("[0-9a-f]{64}", original["checkpoint_sha256"])
            and original["checkpoint_sha256"] == (train / "best.sha256").read_text().strip(),
            f"SHA-256 del checkpoint: {name}")
    dataset_hash = plan["dataset_manifest_sha256"]
    require(run["run"]["initial_weights_sha256"] == plan["weights"][backend]["sha256"]
            and Path(run["run"]["initial_weights"]) == expected_path
            and run["run"]["dataset_manifest_sha256"] == original["dataset_manifest_sha256"] == dataset_hash
            and original["coco_sha256"] == plan["valid_coco_sha256"]
            and original["image_ids"] == [image["id"] for image in truth["images"]]
            and (original["split"], original["checkpoint_kind"], original["backend"]) == ("valid", "best", backend)
            and run["mode"] == "train" and run["run"]["backend"] == backend
            and status["status"] == "complete"
            and [(item["job_id"], item["status"]) for item in status["sessions"]] == [(train_job["id"], "complete")],
            f"Procedencia de los datos o estado del entrenamiento: {name}")
    stride, effective = verify_runtime(train, backend, spec, splits["train"]["images"])
    history, details = archive.read_history(train / "history.csv", stride, spec["epochs"], status)
    predictions = read_json(evaluation / "predictions.json")
    evaluator = coco_eval(truth, predictions)
    metrics, curves, examples = report(evaluator)
    require(all(original[key] == value for key, value in metrics.items())
            and read_json(evaluation / "pr_curves.json") == curves, f"COCO exportado no reproducible: {name}")
    benchmark, protocol = archive.benchmark_metrics(original, [img["file_name"] for img in truth["images"]])
    require(protocol["protocol"], f"Protocolo del benchmark: {name}")
    counts, confusions, focused, rare, _ = archive.diagnose(backend, name[-1].upper(), truth, predictions,
                                                         evaluator, examples)
    classes = [{**row, "ap50": mean(curves["classes"][str(row["category_id"])]["0.50"]),
                "ap75": mean(curves["classes"][str(row["category_id"])]["0.75"]),
                **{event: values[row["category_id"]] for event, values in counts.items()}}
               for row in metrics["per_class"]]
    per_class = {row["name"]: row["ap"] for row in classes}
    require(math.isclose(metrics["map"], mean(per_class.values()), abs_tol=1e-12),
            f"El mAP no coincide con la media de las ocho clases: {name}")
    values = {
        **{key: metrics[key] for key in COCO_METRICS},
        SENSITIVITY: mean(ap for cls, ap in per_class.items() if cls != RARE_CLASS),
        **metrics["operating_point"], **details,
    }
    return {"values": values, "classes": classes, "benchmark": benchmark, "protocol": protocol,
            "effective": effective, "history": history, "stride": stride,
            "confusions": confusions, "focused": focused, "rare": rare, "examples": examples, "evaluator": evaluator,
            "pr_curves": curves,
            "audit": {"training_job": train_job["id"], "evaluation_job": ev_job["id"], "preflight_job": probe_job["id"],
                      "snapshot_sha256": digest,
                      "checkpoint_sha256": original["checkpoint_sha256"], "training_hardware": run["hardware"],
                      "evaluation_hardware": original["hardware"]}}


def contrast_rows(arms, kind, fields):
    for name, coefficients in CONTRASTS.items():
        if kind == "cascade" and "e" in coefficients:
            continue
        for metric in fields:
            inputs = {arm: arms.get(arm, {}).get(metric) for arm in coefficients}
            value = (sum(factor * inputs[arm] for arm, factor in coefficients.items())
                     if all(value is not None for value in inputs.values()) else None)
            yield {"contrast": name, "metric": metric, "delta": value,
                   "pending_arms": ",".join(arm for arm in coefficients if arm not in arms)}


def attempt_snapshots(root, manifest, ledger):
    original = manifest["snapshot_sha256"]
    snapshots = {original: read_json(snapshot_file(root, manifest, original))}
    for name, entries in ledger.items():
        for entry in entries:
            digest = attempt_snapshot(manifest, entry)
            if digest == original:
                continue
            require(manifest["nodes"][name]["backend"] == "yolo", f"Snapshot de recuperación fuera de YOLO: {name}")
            if digest not in snapshots:
                patched = read_json(snapshot_file(root, manifest, digest))
                verify_yolo_recovery(snapshots[original], patched)
                snapshots[digest] = patched
    require(len(snapshots) <= 2, "La campaña registra más de un snapshot de recuperación")
    return snapshots


def analyze(root, dataset, output, expected_snapshot=None, expected_recovery=None, require_complete=False):
    verify_inventory(root)
    manifest = read_json(root / "manifest.json")
    ledger, states = (read_json(root / file) for file in ("jobs.json", "slurm-states.json"))
    snapshots = attempt_snapshots(root, manifest, ledger)
    snapshot = snapshots[manifest["snapshot_sha256"]]
    plan = manifest["plan"]
    require(plan == read_json(REPO / PLAN)
            and snapshot["sha256"][PLAN.as_posix()] == sha256(REPO / PLAN) and manifest["nodes"] == graph(plan),
            "Manifiesto o snapshot incompatibles con la campaña local")
    if expected_snapshot is not None:
        require(expected_snapshot == manifest["snapshot_sha256"], "SHA-256 del snapshot distinto del esperado")
    if expected_recovery is not None:
        require(expected_recovery != manifest["snapshot_sha256"] and expected_recovery in snapshots,
                "SHA-256 de la recuperación YOLO distinto del esperado")
    # The inputs must be archived OpenCV/COCO data, never test detections.
    truth, splits = load_dataset(dataset, root, plan)
    results = {"yolo": {}, "cascade": {}}
    for condition in plan["conditions"]:
        result = evaluate_condition(root, manifest, snapshots, ledger, states, condition, truth, splits)
        if result:
            expected = expected_recovery if condition["backend"] == "yolo" else manifest["snapshot_sha256"]
            if expected is not None:
                require(result["audit"]["snapshot_sha256"] == expected,
                        f"El intento exitoso usa otro snapshot: {condition['name']}")
            results[condition["backend"]][condition["name"].split("-")[-1]] = result
    pending = [c["name"] for c in plan["conditions"]
               if c["name"][-1] not in results[c["backend"]]]
    require(not require_complete or not pending, f"Faltan evaluaciones completas: {', '.join(pending)}")
    for backend, rows in results.items():
        effective = [r["effective"] for r in rows.values()]
        require(not effective or all(item == effective[0] for item in effective),
                f"Recetas efectivas no comparables: {backend}")
    output.mkdir(parents=True, exist_ok=False)
    values = {}
    names = [c["name"] for c in truth["categories"]]
    fields = ("map", "ap50", "ap75", SENSITIVITY, "precision", "recall", "tp", "fp", "fn", *names)
    qualitative = {}
    for backend, rows in results.items():
        target = output / ("exp-004" if backend == "yolo" else "exp-005")
        target.mkdir()
        values[backend] = {arm: {**r["values"], **{c["name"]: c["ap"] for c in r["classes"]}}
                           for arm, r in rows.items()}
        archive.write_csv(target / "contrastes.csv", contrast_rows(values[backend], backend, fields))
        if rows:
            curves_dir = target / "curvas_pr"
            curves_dir.mkdir()
            for arm, result in rows.items():
                plot_pr(result["pr_curves"], curves_dir / f"{arm.upper()}.png")
            archive.write_csv(target / "resumen.csv", [{"arm": arm.upper(), **r["values"], **r["benchmark"]}
                                                      for arm, r in rows.items()])
            archive.write_csv(target / "por_clase.csv", [{"arm": arm.upper(), **c,
                                                         "n_train": splits["train"]["ground_truth_by_category"][str(c["category_id"])]}
                                                        for arm, r in rows.items() for c in r["classes"]])
            history_fields = sorted({key for r in rows.values() for row in r["history"] for key in row
                                     if "loss" in key or key.startswith(("val/", "lr"))})
            archive.write_csv(target / "historial.csv",
                              ({"arm": arm.upper(), "step": row["step"], "epoch": row["step"] / r["stride"],
                                **{key: row.get(key) for key in history_fields}}
                               for arm, r in rows.items() for row in r["history"]),
                              ("arm", "step", "epoch", *history_fields))
            qualitative[backend] = write_error_outputs(rows, truth, dataset / "valid", target)
    write_reports(output, results, plan, names, values, CONTRASTS, pending)
    comparison = {}
    for backend, contrast in (("yolo", "affine_only_100"), ("cascade", "augmentation_100")):
        comparison[backend] = {r["metric"]: r["delta"]
                               for r in contrast_rows(values[backend], backend, fields) if r["contrast"] == contrast}
    archive.write_csv(output / "geometria_comun_100.csv", [{"metric": key, "yolo_e_minus_b": comparison["yolo"][key],
                                                             "cascade_d_minus_b": comparison["cascade"][key]}
                                                            for key in fields])
    write_json(output / "auditoria.json", {
        "snapshot_sha256": manifest["snapshot_sha256"], "base_commit": snapshot["base_commit"],
        "source_sha256": snapshot["sha256"],
        "snapshots": snapshots,
        "valid_coco_sha256": plan["valid_coco_sha256"], "seed": plan["seed"], "slurm_states": states,
        "conditions": {backend: {arm: {**r["audit"], "benchmark_protocol": r["protocol"]} for arm, r in rows.items()}
                       for backend, rows in results.items()},
        "pending": pending,
        "qualitative_valid_images_available": qualitative,
        "latency_comparison": "Descriptiva por trabajo. Las evaluaciones se ejecutan en asignaciones independientes.",
    })
    print(f"{sum(map(len, results.values()))}/{len(plan['conditions'])} condiciones verificadas en {output}")


def write_error_outputs(rows, truth, images_root, target):
    names = {c["id"]: c["name"] for c in truth["categories"]}
    images = {img["id"]: img["file_name"] for img in truth["images"]}
    available = {ident for ident, name in images.items() if (images_root / name).is_file()}
    rare_ids = {ann["image_id"] for ann in truth["annotations"] if names[ann["category_id"]] == RARE_CLASS}
    archive.write_csv(target / "confusiones.csv", (
        {"arm": arm.upper(), "clase_real": names[gt], "clase_predicha": names[pred], "n": count}
        for arm, result in rows.items() for (gt, pred), count in sorted(result["confusions"].items())
    ), ("arm", "clase_real", "clase_predicha", "n"))
    focused = [row for r in rows.values() for row in r["focused"]]
    if focused:
        archive.write_csv(target / "errores_instancias.csv", focused)
    archive.write_csv(target / "ibuphil.csv", (row for r in rows.values() for row in r["rare"]),
                      archive.IBUPHIL_FIELDS)
    archive.write_csv(target / "errores_por_imagen.csv", (
        {"arm": arm.upper(), "image_id": ident, "file_name": images[ident], "rare_gt": int(ident in rare_ids),
         **{event: len(result["examples"].get(ident, {}).get(event, ())) for event in ("tp", "fp", "fn")}}
        for arm, result in rows.items() for ident in sorted(images)
    ))
    if available:
        examples_root = target / "ejemplos"
        examples_root.mkdir()
        for arm, result in rows.items():
            matches = {i: groups for i, groups in result["examples"].items() if i in available}
            save_examples(result["evaluator"], matches, images_root, examples_root / arm.upper(), 2)
            if rare_ids & matches.keys():
                rare = {i: matches[i] for i in rare_ids if i in matches}
                save_examples(result["evaluator"], rare, images_root, examples_root / f"{arm.upper()}-ibuphil", 1)
    return len(available)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "data/pills-sxdht")
    parser.add_argument("--snapshot-sha256", help="SHA-256 del descriptor mostrado al empaquetar")
    parser.add_argument("--recovery-sha256", help="SHA-256 del snapshot de los trabajos YOLO recuperados")
    parser.add_argument("--require-complete", action="store_true", help="Exige las nueve evaluaciones completas")
    args = parser.parse_args()
    analyze(args.imported, args.data, args.output, args.snapshot_sha256, args.recovery_sha256, args.require_complete)
