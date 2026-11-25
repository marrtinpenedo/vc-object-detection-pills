"""Inventory the CESGA download and export the evidence retained in Git."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from scripts.baseline_data import DESTINATION, read_json
from scripts.baseline_metrics import coco_eval, operating_point, save_examples
from scripts.verify_baseline import require

from .analyze import common, plot
from .provenance import A_ARMS, B_ARMS, C_ARMS, SEEDS, digest, training_phase, verify_evidence


COHORTS = {"a": (SEEDS[:1], A_ARMS), "b": (SEEDS[1:], B_ARMS), "c": (SEEDS, C_ARMS)}
TRAIN = ("run.json", "status.json", "history.csv", "ultralytics_runtime.json", "parameters.json", "weight_transfer.json")
EVALUATION = ("metrics.json", "predictions.json", "pr_curves.json", "pr_curves.png")
EXAMPLE_IDS = (57, 61, 80)


def inventory(phase, *, raw=False):
    names = ["source-commit", "code-config.sha256", "inputs.sha256", "data/manifest.json"]
    if phase == "c":
        names.extend(f"smoke/train/{file}" for file in (*TRAIN, "staged_transfer.json"))
        if raw:
            names.extend(("smoke/verified.ok", "queue/exp007-yolo-smoke-permanent-s2026.json"))
    seeds, arms = COHORTS[phase]
    for seed in seeds:
        ev = f"eval-{phase}-s{seed}"
        names.append(f"{ev}/launch.json")
        if raw:
            names.extend((f"{ev}/eval.job", f"{ev}/verified.ok"))
        for arm in arms:
            output = f"{ev}/{arm}"
            names.extend(f"{output}/{file}" for file in EVALUATION)
            if raw:
                names.append(f"{ev}/{arm}.train.sacct")
                names.extend(f"{output}/examples/image-{ident}.jpg" for ident in EXAMPLE_IDS)
            if training_phase(seed, arm) == phase:
                slot = f"s{seed}/{arm}"
                names.append(f"{slot}/config.json")
                names.extend(f"{slot}/train/{file}" for file in (*TRAIN, "training.png"))
                audit = "parameter_updates.json" if arm == "control" else "staged_transfer.json"
                names.append(f"{slot}/train/{audit}")
                if raw:
                    names.extend((f"{slot}/train.job", f"{slot}/train.path"))
    return sorted(names)


def example_name(ident, matches):
    kind = "failure" if matches["fp"] or matches["fn"] else "correct" if matches["tp"] else "empty"
    return f"{kind}-{ident}.jpg"


def export_examples(source, destination, truth, images, extra_ids):
    ev = coco_eval(truth, read_json(source / "predictions.json"))
    _, groups = operating_point(ev)
    save_examples(ev, {ident: groups[ident] for ident in extra_ids}, images, destination, len(extra_ids))
    for ident in EXAMPLE_IDS:
        shutil.copyfile(source / f"examples/image-{ident}.jpg", destination / example_name(ident, groups[ident]))
    for ident in extra_ids:
        if not any(groups[ident].values()):
            from PIL import Image

            with Image.open(images / ev.cocoGt.imgs[ident]["file_name"]) as image:
                image.convert("RGB").save(destination / example_name(ident, groups[ident]))


def archive(phase, source, output, data, examples, extra_ids=()):
    require(not (source / "evidence.sha256").exists(), "Indicar una descarga bruta")
    output = output.resolve()
    require(not output.exists() and not output.is_relative_to(source.resolve()), "La salida debe ser nueva e independiente")
    hashes = verify_evidence(source)
    required = set(inventory(phase, raw=True))
    missing = required - hashes.keys()
    require(not missing or (phase == "a" and all(name.endswith("/training.png") for name in missing)),
            f"Faltan archivos verificados en la descarga: {sorted(missing)}")
    seeds, arms = COHORTS[phase]
    allowed = {f"eval-{phase}-s{seed}/{arm}" for seed in seeds for arm in arms}
    require(len(examples) == len(set(examples)) and set(examples) <= allowed, "Selección de ejemplos inválida")
    counts = Counter(Path(name).parent for name in examples)
    require(all(count >= 2 for count in counts.values()), "Seleccionar al menos dos brazos de cada evaluación elegida")
    require(not extra_ids or (examples and len(extra_ids) == len(set(extra_ids))
                              and not set(extra_ids).intersection(EXAMPLE_IDS)), "IDs adicionales inválidos")
    if examples:
        _, inputs = common.campaign_identity(source)
        truth = common.validation_truth(data, source, inputs)
        images = data / "valid"
        if extra_ids:
            files = read_json(source / "data/manifest.json")["files"]
            names = {row["id"]: row["file_name"] for row in truth["images"]}
            require(set(extra_ids) <= names.keys(), "La escena adicional no pertenece a validación")
            for ident in extra_ids:
                name = names[ident]
                require(digest(images / name) == files[f"images/valid/{name}"],
                        f"La imagen adicional no coincide con el manifiesto: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "archive"
        for name in inventory(phase):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in missing:
                plot({Path(name).parts[1]: common.read_history(source / Path(name).with_name("history.csv"))}, target)
            else:
                shutil.copyfile(source / name, target)
        for name in examples:
            export_examples(source / name, stage / name / "examples", truth, images, extra_ids)
        sums = sorted(p for p in stage.rglob("*") if p.is_file())
        (stage / "evidence.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(stage).as_posix()}\n" for p in sums))
        verify_evidence(stage)
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("inventory")
    listing.add_argument("phase", choices=COHORTS)
    export = sub.add_parser("archive")
    export.add_argument("phase", choices=COHORTS)
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--data", type=Path, default=DESTINATION)
    export.add_argument("--examples", action="append", default=[], metavar="EVAL/BRAZO")
    export.add_argument("--extra-image-id", type=int, action="append", default=[])
    args = parser.parse_args()
    try:
        if args.action == "inventory":
            print("\n".join(inventory(args.phase, raw=True)))
        else:
            archive(args.phase, args.source, args.output, args.data, args.examples, args.extra_image_id)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
