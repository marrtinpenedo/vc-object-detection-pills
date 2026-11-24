"""Copia una descarga local de EXP-002 a la estructura analizable."""
import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results"
SEED = 2026
DETECTORS = ("yolo", "cascade")
ARMS = ("control", "variante")
RUNS = tuple((detector, arm) for detector in DETECTORS for arm in ARMS)
RUNTIMES = {
    "yolo": ("ultralytics_runtime.json",),
    "cascade": ("cascade_runtime.json", "detectron2.yaml"),
}
PROVENANCE = ("source-commit", "inputs.sha256", "manifest.json")
TRAINING = ("run.json", "status.json", "history.csv")
EVALUATION = ("metrics.json", "predictions.json", "pr_curves.json")
INVENTORY = Path("provenance/downloaded.tsv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checksum_entries(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest), f"Huella inválida: {line}")
        yield digest, Path(name.lstrip("*"))


def run_name(detector, arm):
    return f"{detector}-{arm}-s{SEED}"


def required_paths():
    yield from (Path("provenance") / name for name in PROVENANCE)
    for detector, arm in RUNS:
        run = Path(detector) / arm
        for name in (*TRAINING, *RUNTIMES[detector], "config.json", "slurm.job"):
            yield run / "training" / name
        for name in EVALUATION:
            yield run / "evaluation" / name


def evaluation_folder(source, selected):
    if selected:
        require(Path(selected).name == selected, "--evaluation debe ser un nombre de carpeta dentro de la descarga")
        return source / selected
    candidates = sorted(path for path in source.glob("eval-valid-*") if path.is_dir())
    require(len(candidates) == 1,
            "Se requiere una carpeta eval-valid-* única o --evaluation NOMBRE para elegir la tanda")
    return candidates[0]


def download_entries(source, evaluation, manifest):
    for name in PROVENANCE:
        original = manifest if name == "manifest.json" and manifest is not None else source / name
        yield original, Path("provenance") / name, True
    for name in ("code-config.sha256", "preflight.ok"):
        yield source / name, Path("provenance") / name, False
    for detector in DETECTORS:
        for suffix in ("tests", "initialize"):
            name = f"{detector}-{suffix}.log"
            yield source / "preflight" / name, Path("provenance/preflight") / name, False
        yield (source / "preflight" / detector / "initialization.json",
               Path("provenance/preflight") / detector / "initialization.json", False)
    for detector, arm in RUNS:
        name = run_name(detector, arm)
        run = Path(detector) / arm
        for original, target in (("config.json", "config.json"), ("job", "slurm.job")):
            yield source / f"{name}.{original}", run / "training" / target, True
        for artifact in (*TRAINING, *RUNTIMES[detector]):
            yield source / name / artifact, run / "training" / artifact, True
        for artifact in ("training.png", "resume_runtime.json"):
            yield source / name / artifact, run / "training" / artifact, False
        for artifact in (*EVALUATION, "pr_curves.png"):
            yield evaluation / name / artifact, run / "evaluation" / artifact, artifact != "pr_curves.png"
        for example in sorted((evaluation / name / "examples").glob("*")):
            if re.fullmatch(r"(correct|failure)-[0-9]+\.jpe?g", example.name):
                yield example, run / "evaluation/examples" / example.name, False


def import_results(source, results, evaluation=None, manifest=None):
    source = source.resolve(strict=True)
    require(source.is_dir(), f"Se esperaba una carpeta de descarga: {source}")
    selected = evaluation_folder(source, evaluation)
    require(selected.is_dir() and selected.resolve() == selected, f"Carpeta de evaluación inválida: {selected}")
    results = results.resolve()
    target = results / f"s{SEED}"
    require(not target.exists() and not target.is_symlink(), f"Ya existe una importación en {target}")
    if manifest is not None:
        manifest = manifest.absolute()
        require(not (source / "manifest.json").exists() and not (source / "manifest.json").is_symlink(),
                "La descarga ya contiene manifest.json")

    entries, missing = [], []
    for path, local, required in download_entries(source, selected, manifest):
        if path.is_file() and path.resolve() == path:
            entries.append((path, local))
        elif required or path.exists() or path.is_symlink():
            missing.append(str(path.relative_to(source) if path.is_relative_to(source) else path))
    require(not missing, "Archivos ausentes, inválidos o enlaces simbólicos:\n  " + "\n  ".join(missing))
    results.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".import-exp002-", dir=results) as temporary:
        staged = Path(temporary) / f"s{SEED}"
        hashes = []
        for original, local in entries:
            target_file = staged / local
            digest = sha256(original)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target_file)
            require(sha256(target_file) == digest, f"El archivo cambió durante la copia: {original}")
            origin = "local-reconstruction/manifest.json" if original == manifest else original.relative_to(source)
            hashes.append((origin, local, digest))
        if manifest is not None:
            provenance = staged / "provenance"
            expected = [digest for digest, path in checksum_entries(provenance / "inputs.sha256")
                        if path.name == "manifest.json"]
            require(expected == [sha256(provenance / "manifest.json")],
                    "El manifiesto reconstruido difiere de inputs.sha256")
        with (staged / INVENTORY).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("downloaded_path", "local_path", "sha256"))
            writer.writerows(hashes)
        staged.rename(target)
    return target, len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="descarga con la estructura original de la tanda")
    parser.add_argument("--evaluation", help="nombre de una carpeta eval-valid-* si hay varias")
    parser.add_argument("--manifest", type=Path, help="manifiesto reconstruido localmente si falta en la descarga")
    parser.add_argument("--results", type=Path, default=RESULTS, help="directorio local de resultados")
    options = parser.parse_args()
    try:
        destination, count = import_results(options.source, options.results, options.evaluation, options.manifest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{count} artefactos importados en {destination}")


if __name__ == "__main__":
    main()
