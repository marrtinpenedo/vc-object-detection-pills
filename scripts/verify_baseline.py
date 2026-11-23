"""Verify completed smoke runs and replay common evaluation without inference."""

import argparse
import csv
import math
from pathlib import Path

from PIL import Image

from scripts.baseline_data import prepared_manifest, read_json, sha256, verify_files
from scripts.baseline_metrics import coco_eval, report


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_figure(path):
    with Image.open(path) as image:
        require(image.format == "PNG", f"{path}: se esperaba PNG")
        image.verify()


def verify_run(path, digest):
    run = read_json(path / "run.json")["run"]
    model = run["backend"]
    require(run["dataset_manifest_sha256"] == digest, f"{path}: datos distintos")
    status = read_json(path / "status.json")
    require(status["status"] == "complete", f"{path}: entrenamiento incompleto")
    for kind in ("best", "last"):
        checkpoint = path / status[f"{kind}_checkpoint"]
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"Falta {checkpoint}")
    require(status["total_elapsed_seconds"] > 0, f"{path}: falta duración")
    require(status["cuda_peak_allocated_bytes"] > 0, f"{path}: falta memoria GPU")
    with (path / "history.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    require(steps == sorted(set(steps)), f"{path}: pasos duplicados o desordenados")
    scored = [row for row in rows if row["val/map"]]
    require(len(scored) == run["config"]["epochs"], f"{path}: falta validar alguna época")
    losses = [float(v) for row in rows for key, v in row.items() if "loss" in key and v]
    require(losses and all(math.isfinite(v) for v in losses), f"{path}: pérdidas ausentes o inválidas")
    require(all(0 <= float(row["val/map"]) <= 1 for row in scored), f"{path}: AP de validación fuera de rango")
    require(any(int(row["step"]) == status["best_step"] for row in scored), f"{path}: best sin validación")
    selected = next(row for row in scored if int(row["step"]) == status["best_step"])
    require(math.isclose(float(selected["val/map"]), status["best_validation_map"], abs_tol=1e-6),
            f"{path}: best no coincide con el historial")
    require(math.isclose(max(float(row["val/map"]) for row in scored), status["best_validation_map"], abs_tol=1e-6),
            f"{path}: selección de checkpoint no máxima")
    if run["config"]["epochs"] <= 2:
        require(run["config"]["epochs"] == 2 and [s["status"] for s in status["sessions"]] == ["paused", "complete"],
                f"{path}: el smoke test requiere pausa y reanudación documentadas")
        require((path / "resume_runtime.json").is_file(), f"{path}: falta registro de reanudación")
        updates = [
            record for record in read_json(path / "parameter_updates.json")
            if len(record["changed"]) == 2 and all(record["changed"].values())
        ]
        require({record["step"] for record in updates} == {int(row["step"]) for row in scored},
                f"{path}: faltan actualizaciones verificadas antes y después de reanudar")
    verify_figure(path / "training.png")
    return model, sha256(path / status["best_checkpoint"])


def verify_evaluation(path, truth, coco_sha, digest, split):
    actual = read_json(path / "metrics.json")
    require(actual["split"] == split and actual["coco_sha256"] == coco_sha, f"{path}: split o COCO distinto")
    require(actual["dataset_manifest_sha256"] == digest, f"{path}: dataset distinto")
    require(actual["image_ids"] == [row["id"] for row in truth["images"]], f"{path}: imágenes distintas")
    predictions = read_json(path / "predictions.json")
    recomputed, curves, _ = report(coco_eval(truth, predictions))
    for key, value in recomputed.items():
        require(actual[key] == value, f"{path}: {key} no coincide al repetir COCOeval")
    require(read_json(path / "pr_curves.json") == curves, f"{path}: curvas PR incoherentes")
    for key in ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"):
        require(actual[key] is None or 0 <= actual[key] <= 1, f"{path}: AP fuera de rango")
    for category in actual["per_class"]:
        ap = category["ap"]
        require((ap is None) == (category["ground_truth"] == 0), f"{path}: AP indefinido incorrecto")
        require(ap is None or 0 <= ap <= 1, f"{path}: AP de clase fuera de rango")
    verify_figure(path / "pr_curves.png")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", nargs=2, type=Path, required=True)
    parser.add_argument("--evaluations", nargs=2, type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    manifest = prepared_manifest(args.data)
    verify_files(args.data, manifest)
    digest = sha256(args.data / "manifest.json")
    coco = args.data / "coco" / f"{args.split}.json"
    truth, coco_sha = read_json(coco), sha256(coco)
    best = dict(verify_run(path, digest) for path in args.runs)
    actual = [verify_evaluation(path, truth, coco_sha, digest, args.split) for path in args.evaluations]
    require(set(best) == {item["backend"] for item in actual} == {"cascade", "yolo"}, "Falta un detector")
    for item in actual:
        require(item["checkpoint_sha256"] == best[item["backend"]], "La evaluación no corresponde al mejor checkpoint")
    a, b = (item["benchmark"] for item in actual)
    if a is not None or b is not None:
        require(a is not None and b is not None, "Mide ambos detectores con el mismo protocolo")
        for key in ("batch", "nominal_shape", "dtype", "warmup_iterations", "iterations", "images", "scope"):
            require(a[key] == b[key], f"Benchmark: {key} distinto")
        for key in ("gpu", "compute_capability", "nvidia_driver", "host"):
            require(actual[0]["hardware"][key] == actual[1]["hardware"][key], f"Benchmark: {key} distinto")
        for item in (a, b):
            require(len(item["samples_ms"]) == item["iterations"], "Benchmark: faltan muestras")
            require(all(math.isfinite(v) and v > 0 for v in item["samples_ms"]), "Benchmark: tiempos inválidos")
    print("Ambos detectores, mismo COCO, métricas reproducibles y artefactos coherentes.")


if __name__ == "__main__":
    main()
