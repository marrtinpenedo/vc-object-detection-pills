"""Exporta los artefactos seleccionados de dos baselines a un ZIP."""

import argparse
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile


TRAINING_FILES = ("run.json", "status.json", "history.csv", "training.png")
EVALUATION_FILES = ("metrics.json", "predictions.json", "pr_curves.json", "pr_curves.png")
ATTRIBUTION = (
    b"Images: pills-sxdht by Mohamed Attia, distributed in Roboflow 100. "
    b"https://universe.roboflow.com/roboflow-100/pills-sxdht | "
    b"CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ | "
    b"Modified with bounding boxes and TP, FP and FN labels."
)


def path_argument(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("La ruta no puede estar vacía")
    return Path(value)


def directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Se requiere un directorio existente sin enlace simbólico: {path}")
    return path.resolve(strict=True)


def artifacts(folder, names, prefix):
    for name in names:
        source = folder / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Se requiere un archivo existente sin enlace simbólico: {source}")
        yield source, f"{prefix}/{name}"


def examples(folder):
    if not folder.exists() and not folder.is_symlink():
        return
    names = sorted(path.name for path in directory(folder).iterdir())
    for label in ("correct", "failure"):
        selected = [name for name in names if re.fullmatch(rf"{label}-[0-9]+\.jpe?g", name)]
        yield from selected[:5]


def export(root, cascade, yolo, evaluation, output):
    root = directory(root)
    inputs = [directory(path) for path in (cascade, yolo, evaluation)]
    if root == root.parent or any(path == root or not path.is_relative_to(root) for path in inputs):
        raise ValueError("Los tres directorios de entrada deben estar dentro de la raíz de resultados")
    for index, path in enumerate(inputs):
        if any(path.is_relative_to(other) for i, other in enumerate(inputs) if i != index):
            raise ValueError("Los directorios de entrada deben ser distintos y no contenerse entre sí")

    output = output.parent.resolve(strict=True) / output.name
    if (output.suffix != ".zip" or not output.parent.is_relative_to(root)
            or any(output.is_relative_to(path) for path in inputs)):
        raise ValueError("La salida debe ser un ZIP dentro de la raíz y fuera de los directorios de entrada")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"La salida ya existe: {output}")

    entries = []
    for detector, training in zip(("cascade", "yolo"), inputs[:2]):
        evaluated = directory(inputs[2] / detector)
        entries.extend(artifacts(training, TRAINING_FILES, f"{detector}/training"))
        entries.extend(artifacts(evaluated, EVALUATION_FILES, f"{detector}/evaluation"))
        sample_dir = evaluated / "examples"
        entries.extend(artifacts(sample_dir, examples(sample_dir), f"{detector}/evaluation/examples"))

    with TemporaryDirectory(dir=output.parent) as temporary:
        archive = Path(temporary) / output.name
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            bundle.comment = ATTRIBUTION
            for source, name in entries:
                bundle.write(source, name)
        with ZipFile(archive) as bundle:
            if bundle.namelist() != [name for _, name in entries] or bundle.testzip() is not None:
                raise BadZipFile("El contenido del ZIP no coincide con la selección o está dañado")
        os.link(archive, output)
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-root", "cascade-run", "yolo-run", "evaluation", "output"):
        parser.add_argument(f"--{name}", type=path_argument, required=True)
    args = parser.parse_args()
    try:
        count = export(args.results_root, args.cascade_run, args.yolo_run, args.evaluation, args.output)
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{args.output}: {count} archivos verificados")


if __name__ == "__main__":
    main()
