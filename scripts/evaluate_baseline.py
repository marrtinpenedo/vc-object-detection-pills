"""Evaluate a trained checkpoint with the shared COCO detection protocol."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.baseline_data import prepared_manifest, read_json, sha256, verify_files, write_json
from scripts.baseline_metrics import coco_eval, plot_pr, prediction_rows, report, save_examples
from scripts.baseline_run import backend_for, dry_runtime, provenance


def load_image(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"No se pudo decodificar {path}")
    return image


def benchmark(torch, predict, paths, imgsz, warmup, iterations):
    # The exact same ordered in-memory JPEG decodes are used by both backends.
    images = [load_image(path) for path in paths[:min(16, len(paths))]]
    for index in range(warmup):
        predict(images[index % len(images)])
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    start_allocated = torch.cuda.memory_allocated(0)
    samples = []
    for index in range(iterations):
        image = images[index % len(images)]
        torch.cuda.synchronize(0)
        start = perf_counter()
        predict(image)
        torch.cuda.synchronize(0)
        samples.append((perf_counter() - start) * 1000)
    mean = float(np.mean(samples))
    return {
        "batch": 1, "nominal_shape": [imgsz, imgsz], "dtype": "fp32",
        "warmup_iterations": warmup, "iterations": iterations,
        "mean_ms_per_image": mean, "median_ms_per_image": float(np.median(samples)),
        "p95_ms_per_image": float(np.percentile(samples, 95)),
        "fps_from_mean": 1000 / mean, "samples_ms": samples,
        "cuda_allocated_before_bytes": start_allocated,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "images": [path.name for path in paths[:len(images)]],
        "scope": "decoded BGR host array through resize, tensor transfer, model, native postprocess and host Nx6 array",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--examples", type=int, default=5, help="Máximo por tipo de ejemplo")
    args = parser.parse_args()
    if args.examples < 0 or args.iterations < 1 or args.warmup < 0:
        parser.error("Revisa ejemplos, iteraciones y warm-up")
    args.data, args.run, args.output = args.data.resolve(), args.run.resolve(), args.output.resolve()
    if args.output == args.data or args.data in args.output.parents or args.output in args.data.parents:
        parser.error("La evaluación debe escribirse fuera de los datos preparados")
    if args.output.exists():
        parser.error("usa un directorio nuevo para la evaluación")

    manifest = prepared_manifest(args.data)
    verify_files(args.data, manifest)
    identity = sha256(args.data / "manifest.json")
    run = read_json(args.run / "run.json")["run"]
    if run["dataset_manifest_sha256"] != identity:
        raise ValueError("Los datos preparados difieren de los de entrenamiento")
    status = read_json(args.run / "status.json")
    if status["status"] != "complete":
        raise ValueError("Evalúa una ejecución que haya completado su presupuesto")
    weights = args.run / status[f"{args.checkpoint}_checkpoint"]
    if not weights.is_file():
        raise FileNotFoundError(weights)

    os.environ.setdefault("MPLBACKEND", "Agg")
    torch = dry_runtime()
    import cv2

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    backend = backend_for(run["backend"])
    spec = run["config"]
    predict = backend.predictor(spec, manifest, weights, args.output)
    coco_path = args.data / "coco" / f"{args.split}.json"
    truth = read_json(coco_path)
    images_root = args.data / "images" / args.split
    paths = [images_root / image["file_name"] for image in truth["images"]]
    predictions = []
    for image, path in zip(truth["images"], paths):
        decoded = load_image(path)
        if decoded.shape[:2] != (image["height"], image["width"]):
            raise ValueError(f"Dimensiones decodificadas incompatibles: {path}")
        predictions.extend(prediction_rows(image, predict(decoded), manifest["classes"]))
    ev = coco_eval(truth, predictions)
    metrics, curves, examples = report(ev)
    benchmark_result = None
    if args.benchmark:
        benchmark_result = benchmark(torch, predict, paths, spec["imgsz"], args.warmup, args.iterations)
    args.output.mkdir(parents=True)
    write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "pr_curves.json", curves)
    plot_pr(curves, args.output / "pr_curves.png")
    if args.examples:
        save_examples(ev, examples, images_root, args.output / "examples", args.examples)
    write_json(args.output / "metrics.json", {
        **metrics, "backend": run["backend"], "split": args.split,
        "image_ids": [image["id"] for image in truth["images"]],
        "coco_sha256": sha256(coco_path), "dataset_manifest_sha256": identity,
        "checkpoint": str(weights), "checkpoint_sha256": sha256(weights), "checkpoint_kind": args.checkpoint,
        "model_config": spec, "benchmark": benchmark_result, **provenance(torch),
    })
    print(f"{args.split} {run['backend']}: mAP {metrics['map']:.4f}, AP50 {metrics['ap50']:.4f}")


if __name__ == "__main__":
    main()
