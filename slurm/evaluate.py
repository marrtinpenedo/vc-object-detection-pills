"""Orquestación de evaluaciones secuenciales en una asignación GPU de SLURM."""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys


BACKENDS = ("cascade", "yolo")
COMPLETED = "COMPLETED|0:0"


def arguments(argv):
    parser = argparse.ArgumentParser(
        prog="slurm/evaluate.sh", description=__doc__,
        epilog="Interfaz anterior: DATOS RUN_CASCADE RUN_YOLO SALIDA_NUEVA",
        allow_abbrev=False,
    )
    parser.add_argument("--data", required=True, help="Directorio de datos preparados")
    parser.add_argument("--output", required=True, help="Directorio nuevo, con su padre ya creado")
    parser.add_argument("--run", required=True, action="append", nargs=2, metavar=("NOMBRE", "RUTA"))
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    benchmark = parser.add_mutually_exclusive_group()
    benchmark.add_argument("--benchmark", dest="benchmark", action="store_true", default=True)
    benchmark.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    parser.add_argument("--warmup", type=int, default=20, help="Calentamientos del benchmark (20)")
    parser.add_argument("--iterations", type=int, default=100, help="Iteraciones del benchmark (100)")
    examples = parser.add_mutually_exclusive_group()
    examples.add_argument("--examples", type=int, help="Máximo por tipo (5); 0 desactiva los ejemplos")
    examples.add_argument("--example-image-id", type=int, action="append", metavar="ID",
                          help="ID COCO de una imagen común a todos los brazos; repetible")
    parser.add_argument("--selection", type=Path, help="Registro certificado de los dos finalistas")
    parser.add_argument("--check-slurm", action="store_true",
                        help="Exige COMPLETED y 0:0 al último trabajo de entrenamiento con sacct")

    legacy = len(argv) == 4 and all(not arg.startswith("-") for arg in argv)
    if legacy:
        data, cascade, yolo, output = argv
        argv = ["--data", data, "--output", output, "--run", "cascade", cascade, "--run", "yolo", yolo]
    args = parser.parse_args(argv)
    names = [name for name, _ in args.run]
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in names) or len(set(names)) != len(names):
        parser.error("los nombres deben ser únicos y usar letras, dígitos, guiones o guiones bajos")
    if not all(path.strip() for path in (args.data, args.output, *(path for _, path in args.run))):
        parser.error("las rutas no pueden estar vacías")
    if args.warmup < 0 or args.iterations < 1 or (args.examples is not None and args.examples < 0):
        parser.error("revisa --warmup, --iterations y --examples")
    ids = args.example_image_id or []
    if any(ident < 0 for ident in ids) or len(ids) != len(set(ids)):
        parser.error("los IDs COCO deben ser enteros no negativos y únicos")
    return parser, args, legacy


def overlaps(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def training_job(status):
    sessions = status.get("sessions", [])
    return sessions[-1].get("job_id") if sessions else None


def completed_slurm(job_id):
    if job_id is None or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(job_id)):
        raise ValueError("Falta un ID SLURM válido para comprobar el entrenamiento")
    state = subprocess.check_output(
        ["sacct", "-j", str(job_id), "--allocations", "--noheader", "--parsable2", "--format=State,ExitCode"],
        text=True,
    ).strip()
    if state != COMPLETED:
        raise ValueError(f"Entrenamiento SLURM {job_id}: se esperaba {COMPLETED}, figura {state or 'sin datos'}")
    return state


def benchmark_signature(metrics, args, shape, images):
    from scripts.verify_baseline import require

    bench = metrics["benchmark"]
    if not args.benchmark:
        require(bench is None, "El benchmark debía estar desactivado")
        return None
    require(bench is not None, "Falta el benchmark")
    require((bench["batch"], bench["nominal_shape"], bench["dtype"], bench["warmup_iterations"],
             bench["iterations"], bench["images"]) == (1, shape, "fp32", args.warmup, args.iterations, images),
            "El benchmark no corresponde al protocolo solicitado")
    times = bench["samples_ms"]
    require(len(times) == args.iterations and all(math.isfinite(t) and t > 0 for t in times),
            "Muestras del benchmark inválidas")
    average = sum(times) / len(times)
    require(math.isclose(bench["mean_ms_per_image"], average)
            and math.isclose(bench["fps_from_mean"], 1000 / average), "Latencia o FPS incoherentes")
    return (bench["scope"], *(metrics["hardware"][key] for key in
                             ("gpu", "compute_capability", "nvidia_driver", "host")))


def paired_examples(args, truth, output, data):
    from PIL import Image

    from scripts.baseline_data import read_json
    from scripts.baseline_metrics import coco_eval, operating_point, save_examples

    evaluation = coco_eval(truth, read_json(output / "predictions.json"))
    _, groups = operating_point(evaluation)
    ids = args.example_image_id
    images = data / "images" / args.split
    folder = output / "examples"
    save_examples(evaluation, {ident: groups[ident] for ident in ids}, images, folder, len(ids))
    for ident in ids:
        target = folder / f"image-{ident}.jpg"
        if not any(groups[ident].values()):
            with Image.open(images / evaluation.cocoGt.imgs[ident]["file_name"]) as image:
                image.convert("RGB").save(target)
        else:
            kind = "failure" if groups[ident]["fn"] or groups[ident]["fp"] else "correct"
            (folder / f"{kind}-{ident}.jpg").rename(target)


def final_contract(args, data, manifest, truth, runs):
    from scripts.baseline_data import read_json, sha256
    from scripts.baseline_metrics import OPERATING_IOU, OPERATING_SCORE
    from scripts.campaign import verify_source
    from scripts.verify_baseline import require

    selection = read_json(args.selection)
    require(selection["c_slurm_accounting_verified"] is True and selection["status"] == "pendiente_sha256_fisico",
            "La selección final requiere la contabilidad C certificada")
    require((args.split, args.checkpoint, args.benchmark, args.warmup, args.iterations, args.check_slurm)
            == ("test", "best", True, 20, 100, True) and set(dict(runs)) == set(BACKENDS)
            and len(runs) == 2, "Protocolo final distinto del fijado")
    require(selection["operating_point"] == {"score": OPERATING_SCORE, "iou": OPERATING_IOU},
            "Punto operativo distinto del fijado")
    require(len(truth["images"]) == manifest["splits"]["test"]["images"] == 45
            and len(truth["categories"]) == len(manifest["classes"]) == 8
            and sha256(data / "manifest.json") == selection["dataset_manifest_sha256"]
            and sha256(data / "coco" / "test.json") == selection["test_coco_expected_sha256_from_manifest"],
            "La partición final difiere de los 45 ejemplos y ocho clases fijados")
    _, snapshot = verify_source(selection["yolo"]["code_snapshot_sha256"])
    require(snapshot["sha256"]["scripts/evaluate_baseline.py"] == selection["evaluation_entrypoint_sha256"],
            "Evaluador distinto del fijado")
    return selection


def check_final_weights(selection, name, path, run, status, checkpoint, checkpoint_sha, slurm_job):
    from scripts.verify_baseline import require

    frozen = selection[name]
    stride = 1 if name == "yolo" else 79
    require((Path(frozen["checkpoint_path"]), frozen["checkpoint_sha256"], frozen["training_job"], frozen["recipe"],
             frozen["validation"]["best_epoch"] * stride) ==
            (checkpoint, checkpoint_sha, slurm_job, run["config"], status["best_step"])
            and checkpoint.is_relative_to(path) and run["backend"] == name,
            f"{name}: el checkpoint físico, su época o su receta difieren del registro fijado")


def evaluate(args, legacy):
    # The launcher may be deployed separately. Use the modules of the submission checkout.
    sys.path.insert(0, str(Path.cwd()))
    from scripts.baseline_data import prepared_manifest, read_json, sha256, verify_files, write_json
    from scripts.verify_baseline import require, verify_evaluation, verify_run

    data = Path(args.data).resolve(strict=True)
    target = Path(args.output)
    require(not target.is_symlink() and not target.exists(), "La salida debe ser un directorio nuevo")
    output = target.resolve()
    require(output.parent.is_dir(), "El directorio padre de la salida debe existir")
    runs = [(name, Path(path).resolve(strict=True)) for name, path in args.run]
    require(data.is_dir() and all(path.is_dir() for _, path in runs),
            "Los datos y las ejecuciones deben ser directorios")
    require(not any(overlaps(output, path) for path in (data, *(path for _, path in runs))),
            "La salida debe quedar separada de los datos y los entrenamientos")
    require(not any(overlaps(data, path) for _, path in runs), "Los datos y entrenamientos deben estar separados")

    job_id = os.environ.get("SLURM_JOB_ID")
    require(bool(job_id), "Falta SLURM_JOB_ID. Envía el lanzador mediante sbatch")
    envs = Path(os.environ["STORE"]) / "pills-baseline/envs"
    manifest = prepared_manifest(data)
    verify_files(data, manifest)
    digest = sha256(data / "manifest.json")
    coco = data / "coco" / f"{args.split}.json"
    truth, coco_sha = read_json(coco), sha256(coco)
    require(set(args.example_image_id or []) <= {image["id"] for image in truth["images"]},
            f"Hay IDs de ejemplo que no pertenecen a {args.split}")
    images = [image["file_name"] for image in truth["images"][:16]]
    selection = final_contract(args, data, manifest, truth, runs) if args.selection else None
    selection_hash = sha256(args.selection) if selection else None

    entries = {}
    for name, path in runs:
        training = read_json(path / "run.json")
        run = training["run"]
        backend = run["backend"]
        require(backend in BACKENDS, f"{name}: backend desconocido {backend!r}")
        require(not legacy or name == backend, f"{name}: backend incorrecto en la interfaz posicional")
        python = envs / backend / "bin/python"
        require(python.is_file() and os.access(python, os.X_OK), f"No hay intérprete ejecutable: {python}")
        status = read_json(path / "status.json")
        require(status["status"] == "complete", f"{name}: entrenamiento incompleto")
        require(run["dataset_manifest_sha256"] == digest, f"{name}: datos distintos a los de entrenamiento")
        relative = Path(status[f"{args.checkpoint}_checkpoint"])
        selected = path / relative
        require(not relative.is_absolute() and selected.resolve().is_relative_to(path)
                and selected.is_file() and selected.stat().st_size > 0,
                f"{name}: checkpoint {args.checkpoint} ausente o fuera del entrenamiento")
        checkpoint_sha = verify_run(path, digest)[1] if legacy else sha256(selected)
        train_job = training_job(status)
        if selection:
            check_final_weights(selection, name, path, run, status, selected, checkpoint_sha, train_job)
        entries[name] = {
            "run": str(path), "backend": backend, "python": str(python),
            "checkpoint": str(selected), "checkpoint_sha256": checkpoint_sha,
            "training_git_head": training.get("git_head"), "training_job_id": train_job,
            "training_slurm": completed_slurm(train_job) if args.check_slurm else None,
            "imgsz": run["config"]["imgsz"] if args.benchmark else None,
        }
    require(not args.benchmark or len({row["imgsz"] for row in entries.values()}) == 1,
            "Usa la misma resolución para comparar latencia o desactiva el benchmark")

    output.mkdir()
    (output / "eval.job").write_text(f"{job_id}\n", encoding="utf-8")
    if any(row["backend"] == "yolo" for row in entries.values()):
        settings = output / ".ultralytics"
        settings.mkdir()
        os.environ["YOLO_CONFIG_DIR"] = str(settings)
    entrypoint = Path(os.environ["PILLS_EVAL_ENTRYPOINT"])
    driver = Path(__file__).resolve()
    examples = 0 if args.example_image_id else 5 if args.examples is None else args.examples
    write_json(output / "launch.json", {
        "job_id": job_id, "job_name": os.environ.get("SLURM_JOB_NAME"), "submission": str(Path.cwd()),
        "launcher_sha256": {"evaluate.sh": sha256(entrypoint), "evaluate.py": sha256(driver)},
        "data": str(data), "dataset_manifest_sha256": digest, "coco_sha256": coco_sha,
        "split": args.split, "checkpoint_kind": args.checkpoint, "benchmark": args.benchmark,
        "warmup": args.warmup, "iterations": args.iterations, "examples": examples,
        "example_image_ids": args.example_image_id, "runs": entries,
        **({"selection_sha256": selection_hash} if selection else {}),
    })
    common = ["--data", str(data), "--split", args.split, "--checkpoint", args.checkpoint,
              "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--examples", str(examples)]
    if args.benchmark:
        common.append("--benchmark")
    signature = None
    for name, entry in entries.items():
        if args.check_slurm:
            (output / f"{name}.train.sacct").write_text(f"{entry['training_slurm']}\n", encoding="utf-8")
        arm = output / name
        subprocess.run([entry["python"], "-m", "scripts.evaluate_baseline", *common,
                        "--run", entry["run"], "--output", str(arm)], check=True)
        metrics = verify_evaluation(arm, truth, coco_sha, digest, args.split)
        require(metrics["backend"] == entry["backend"] and metrics["checkpoint_kind"] == args.checkpoint
                and metrics["checkpoint"] == entry["checkpoint"]
                and metrics["checkpoint_sha256"] == entry["checkpoint_sha256"],
                f"{name}: la evaluación no corresponde al checkpoint seleccionado")
        require(metrics["hardware"]["job_id"] == job_id, f"{name}: asignación SLURM distinta")
        if selection:
            require(metrics["model_config"] == selection[name]["recipe"] and sha256(args.selection) == selection_hash,
                    f"{name}: la receta o el registro cambiaron durante la evaluación")
        current = benchmark_signature(metrics, args, [entry["imgsz"]] * 2, images)
        require(signature is None or signature == current, f"{name}: benchmark no comparable")
        signature = current
        if args.example_image_id:
            paired_examples(args, truth, arm, data)
    (output / "verified.ok").touch(exist_ok=False)
    if legacy:
        print("Ambos detectores, mismo COCO, métricas reproducibles y artefactos coherentes.")
    else:
        print(f"{len(entries)} evaluaciones verificadas en {output}")


def main():
    parser, args, legacy = arguments(sys.argv[1:])
    try:
        evaluate(args, legacy)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode if error.returncode > 0 else 128 - error.returncode,
                    f"Error: proceso de evaluación o verificación fallido ({error.returncode}).\n")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
