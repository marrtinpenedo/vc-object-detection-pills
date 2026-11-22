"""Initialize, fine-tune or resume a single-GPU detection baseline."""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from scripts.baseline_data import prepared_manifest, read_json, sha256, verify_files, write_json
from scripts.baseline_run import (
    MODELS, OVERRIDES, backend_for, dry_runtime, load_config, provenance, save_session,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=MODELS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--data", type=Path, help="Directorio generado por baseline_data")
    parser.add_argument("--weights", type=Path, help="Checkpoint oficial local")
    parser.add_argument("--output", type=Path)
    for key, kind in OVERRIDES.items():
        parser.add_argument(f"--{key}", type=kind)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--initialize", action="store_true", help="Inferencia y backward sin paso de optimizador")
    modes.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", type=int, metavar="EPOCH", help="Pausa de verificación tras guardar")
    return parser, parser.parse_args()


def main():
    parser, args = arguments()
    spec = load_config(args.backend, args.config, {key: getattr(args, key) for key in OVERRIDES})
    if args.check_config:
        print(json.dumps(spec, indent=2))
        return
    if any(value is None for value in (args.data, args.weights, args.output)):
        parser.error("indica --data, --weights y --output")
    if args.pause_after is not None and (args.initialize or not 0 < args.pause_after < spec["epochs"]):
        parser.error("--pause-after requiere una época anterior al presupuesto total")

    # Validate the filename before resolving symlinks to local checkpoint storage.
    if args.weights.name != MODELS[args.backend][1] or not args.weights.is_file():
        parser.error(f"--weights debe apuntar a un {MODELS[args.backend][1]} local")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.weights = args.weights.absolute()
    if args.data == args.output or args.data in args.output.parents or args.output in args.data.parents:
        parser.error("los directorios de datos preparados y de ejecución deben ser independientes")

    manifest = prepared_manifest(args.data)
    verify_files(args.data, manifest)
    declared = {
        "backend": args.backend,
        "config": spec,
        "data": str(args.data),
        "dataset_manifest_sha256": sha256(args.data / "manifest.json"),
        "initial_weights": str(args.weights),
        "initial_weights_sha256": sha256(args.weights),
    }
    if args.resume and not args.output.is_dir():
        parser.error("falta el directorio de la ejecución que se quiere reanudar")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("otra ejecución está escribiendo en este directorio")
        run(args, spec, manifest, declared, parser)


def run(args, spec, manifest, declared, parser):
    run_path = args.output / "run.json"
    if args.resume:
        if not run_path.is_file() or read_json(run_path)["run"] != declared:
            parser.error("--resume requiere los mismos datos, pesos, configuración y ruta de ejecución")
        status_path = args.output / "status.json"
        if status_path.exists() and read_json(status_path)["status"] in ("complete", "initialized"):
            parser.error("esta ejecución terminó o es una prueba de inicialización")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLBACKEND", "Agg")
    torch = dry_runtime()
    backend = backend_for(args.backend)
    provenance_path = args.output / "resume_runtime.json" if args.resume else run_path
    write_json(provenance_path, {
        "run": declared, "invocation": sys.argv,
        "mode": "initialize" if args.initialize else "resume" if args.resume else "train",
        **provenance(torch),
    })
    torch.cuda.reset_peak_memory_stats(0)
    started = perf_counter()
    try:
        result = backend.train(args, spec, manifest)
    except BaseException:
        save_session(args.output, torch, started, "failed", {})
        raise
    else:
        status = result.pop("status", "initialized")
        save_session(args.output, torch, started, status, result)
        print(f"{status}: {args.output}")


if __name__ == "__main__":
    main()
