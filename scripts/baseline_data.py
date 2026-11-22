"""Validate the pinned COCO release and build a portable, isolated dataset."""

import argparse
import hashlib
import json
import math
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.download_dataset import ANNOTATIONS, DESTINATION, REPO_ID, REVISION, SPLITS, validate

ACTIVE_IDS = tuple(range(1, 9))
INSTANCE_COUNTS = {"train": 351, "valid": 98, "test": 47}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def index_records(rows, kind):
    indexed = {}
    for row in rows:
        ident = row.get("id")
        if type(ident) is not int or ident < 0 or ident in indexed:
            raise ValueError(f"{kind}: ID inválido o repetido {ident!r}")
        indexed[ident] = row
    return indexed


def class_mapping(categories, annotations):
    used = {row["category_id"] for row in annotations}
    if not used or not used <= categories.keys():
        raise ValueError("Categorías activas vacías o no declaradas")
    return [
        {"original_id": original, "internal_id": internal, "name": categories[original]["name"]}
        for internal, original in enumerate(sorted(used))
    ]


def coco_to_yolo(box, width, height):
    if not isinstance(box, (tuple, list)) or len(box) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in box
    ):
        raise ValueError(f"caja COCO inválida {box!r}")
    x, y, w, h = box
    if width <= 0 or height <= 0 or min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(f"caja degenerada o fuera de límites {box!r} en {width}×{height}")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def convert_split(coco, classes):
    images = index_records(coco["images"], "imagen")
    declared = index_records(coco["categories"], "categoría")
    annotations = index_records(coco["annotations"], "anotación")
    internal = {c["original_id"]: c["internal_id"] for c in classes}
    labels = {image_id: [] for image_id in images}
    for image in images.values():
        if any(type(image.get(key)) is not int or image[key] <= 0 for key in ("width", "height")):
            raise ValueError(f"imagen {image['id']}: dimensiones COCO inválidas")
    seen = defaultdict(set)
    converted = []
    for new_id, (_, ann) in enumerate(sorted(annotations.items()), start=1):
        image_id, category_id = ann["image_id"], ann["category_id"]
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"anotación {ann['id']}: image_id desconocido {image_id!r}")
        if type(category_id) is not int or category_id not in internal or category_id not in declared:
            raise ValueError(f"anotación {ann['id']}: category_id inválido {category_id!r}")
        area = ann.get("area")
        if type(area) not in (int, float) or not math.isfinite(area) or area <= 0 or ann.get("iscrowd") != 0:
            raise ValueError(f"anotación {ann['id']}: área inválida o crowd no representable en YOLO")
        image = images[image_id]
        try:
            box = coco_to_yolo(ann["bbox"], image["width"], image["height"])
        except ValueError as error:
            raise ValueError(f"anotación {ann['id']} de {image['file_name']}: {error}") from error
        label = (internal[category_id], *box)
        # Ultralytics carga los TXT como float32 y elimina filas duplicadas.
        key = struct.pack("5f", *label)
        if key in seen[image_id]:
            raise ValueError(f"anotación {ann['id']}: caja y clase duplicadas para {image['file_name']}")
        seen[image_id].add(key)
        labels[image_id].append(label)
        # COCOeval utiliza 0 como centinela en sus matrices de matches.
        converted.append({**ann, "id": new_id})
    return {
        **coco,
        "info": coco.get("info", {}),
        "images": [image for _, image in sorted(images.items())],
        "categories": [declared[c["original_id"]] for c in classes],
        "annotations": converted,
    }, labels


def validate_image(path, record):
    width, height = record["width"], record["height"]
    with path.open("rb") as source:
        source.seek(-2, 2)
        if source.read() != b"\xff\xd9":
            raise ValueError(f"{path}: JPEG sin marcador de fin")
    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != (width, height) or image.getexif().get(274, 1) != 1:
            raise ValueError(f"{path}: formato, dimensiones u orientación incompatibles con COCO")
        image.load()


def prepared_manifest(root):
    manifest = read_json(root / "manifest.json")
    if (
        manifest["dataset"] != REPO_ID or manifest["revision"] != REVISION
        or [c["original_id"] for c in manifest["classes"]] != list(ACTIVE_IDS)
        or [c["internal_id"] for c in manifest["classes"]] != list(range(len(ACTIVE_IDS)))
    ):
        raise ValueError(f"{root}: manifest de la revisión o categorías incorrectas")
    return manifest


def verify_files(root, manifest):
    for relative, digest in manifest["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Archivo preparado ausente o modificado: {path}")
    for split in SPLITS:
        expected = {
            name for name in manifest["files"] if name.startswith((f"images/{split}/", f"labels/{split}/"))
        }
        observed = {
            file.relative_to(root).as_posix()
            for kind in ("images", "labels")
            for file in (root / kind / split).rglob("*")
            if file.is_file()
        }
        if observed != expected:
            raise ValueError(f"{split}: archivos inesperados en el dataset preparado")


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("El dataset original y el preparado deben estar en directorios independientes")
    validate(source)
    raw = {split: read_json(source / split / ANNOTATIONS) for split in SPLITS}
    declarations = {split: index_records(coco["categories"], split) for split, coco in raw.items()}
    categories = declarations["train"]
    if any(table != categories for table in declarations.values()):
        raise ValueError("Las declaraciones de categorías difieren entre particiones")
    classes = class_mapping(categories, [ann for coco in raw.values() for ann in coco["annotations"]])
    if set(categories) != set(range(9)) or categories[0]["name"] != "pills" or tuple(
        item["original_id"] for item in classes
    ) != ACTIVE_IDS:
        raise ValueError("La revisión esperada declara IDs 0–8 y utiliza IDs 1–8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".baseline-data-") as temporary:
        staged = Path(temporary)
        (staged / "coco").mkdir()
        summary = {}
        for split, coco in raw.items():
            try:
                prepared, labels = convert_split(coco, classes)
            except ValueError as error:
                raise ValueError(f"{split}: {error}") from error
            n_images, n_instances = len(prepared["images"]), len(prepared["annotations"])
            if n_images != SPLITS[split] or n_instances != INSTANCE_COUNTS[split]:
                raise ValueError(f"{split}: recuentos inesperados de imágenes o instancias")
            if sum(map(len, labels.values())) != n_instances:
                raise ValueError(f"{split}: se perdieron anotaciones durante la conversión")
            images_dir, labels_dir = staged / "images" / split, staged / "labels" / split
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            for image in prepared["images"]:
                name = image["file_name"]
                original = source / split / name
                validate_image(original, image)
                shutil.copyfile(original, images_dir / name)
                label_file = labels_dir / Path(name).with_suffix(".txt")
                label_file.write_text(
                    "".join(
                        f"{cls} " + " ".join(f"{value:.17g}" for value in coords) + "\n"
                        for cls, *coords in labels[image["id"]]
                    ),
                    encoding="utf-8",
                )
            write_json(staged / "coco" / f"{split}.json", prepared)
            counts = Counter(ann["category_id"] for ann in prepared["annotations"])
            summary[split] = {
                "images": n_images,
                "instances": n_instances,
                "ground_truth_by_category": {str(cid): counts[cid] for cid in ACTIVE_IDS},
                "source_annotations_sha256": sha256(source / split / ANNOTATIONS),
            }

        # JSON is valid YAML. An omitted path makes Ultralytics use the YAML's directory.
        write_json(staged / "dataset.yaml", {
            "train": "images/train", "val": "images/valid", "test": "images/test",
            "names": [c["name"] for c in classes],
        })
        manifest = {
            "dataset": REPO_ID,
            "revision": REVISION,
            "classes": classes,
            "splits": summary,
            "files": {
                file.relative_to(staged).as_posix(): sha256(file)
                for file in sorted(staged.rglob("*")) if file.is_file()
            },
        }
        write_json(staged / "manifest.json", manifest)
        if output.exists():
            if prepared_manifest(output) != manifest:
                raise ValueError(f"{output}: contenido diferente. Elige otro directorio de salida")
            verify_files(output, manifest)
        else:
            staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.data, args.output)
    print(json.dumps({"classes": manifest["classes"], "splits": manifest["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
