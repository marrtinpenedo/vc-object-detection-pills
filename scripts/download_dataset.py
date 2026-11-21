"""Descarga y extrae el dataset COCO original de Hugging Face."""

import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

REPO_ID = "Francesco/pills-sxdht"
REVISION = "f599e8469b24e0474723fed566bca04c59c477f0"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "pills-sxdht"
ARCHIVE_ROOT = PurePosixPath(
    "home/zuppif/Documents/Work/RoboFlow/ODinW-RF100-challenge/rf100/pills-sxdht"
)
SPLITS = {"train": 316, "valid": 90, "test": 45}
DOCUMENTS = {"README.dataset.txt", "README.roboflow.txt"}
ANNOTATIONS = "_annotations.coco.json"


def validate(root):
    """Comprueba la estructura y las referencias COCO de la revisión fijada."""
    for document in DOCUMENTS:
        if not (root / document).is_file():
            raise ValueError(f"Falta {document}")

    for split, expected_count in SPLITS.items():
        folder = root / split
        coco = json.loads((folder / ANNOTATIONS).read_text(encoding="utf-8"))
        images = coco["images"]
        names = {image["file_name"] for image in images}
        image_ids = {image["id"] for image in images}
        files = {
            path.name for path in folder.glob("*.jpg")
            if path.is_file() and path.stat().st_size > 0
        }
        if (
            len(images) != expected_count
            or len(names) != expected_count
            or len(image_ids) != expected_count
            or files != names
        ):
            raise ValueError(f"{split}: las imágenes en disco y el JSON no coinciden")

        category_ids = {category["id"] for category in coco["categories"]}
        if not coco["annotations"] or any(
            annotation["image_id"] not in image_ids
            or annotation["category_id"] not in category_ids
            for annotation in coco["annotations"]
        ):
            raise ValueError(f"{split}: anotaciones vacías o referencias COCO inválidas")


def extract(archive, root):
    """Copia archivos regulares sin aplicar rutas, enlaces ni permisos del tar."""
    extracted = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Ruta insegura en el archivo: {member.name}")
            relative = path.relative_to(ARCHIVE_ROOT)
            if len(relative.parts) == 1:
                if member.isdir() and relative.name in SPLITS:
                    continue
                allowed = relative.name in DOCUMENTS
            else:
                allowed = (
                    len(relative.parts) == 2
                    and relative.parts[0] in SPLITS
                    and (relative.name == ANNOTATIONS or relative.suffix == ".jpg")
                )
            # El tar repite cada JSON como enlace duro a su propia ruta.
            if (
                allowed and member.islnk() and relative.name == ANNOTATIONS
                and member.linkname == member.name and relative in extracted
            ):
                continue
            if not allowed or not member.isfile() or relative in extracted:
                raise ValueError(f"Entrada inesperada en el archivo: {member.name}")

            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            extracted.add(relative)


def main():
    if DESTINATION.exists() or DESTINATION.is_symlink():
        try:
            validate(DESTINATION)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"{DESTINATION} ya existe y no supera la comprobación ({error}). "
                "Muévelo a otra ubicación antes de repetir la descarga."
            ) from error
        print(f"Dataset ya disponible en {DESTINATION}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Instala huggingface_hub con python -m pip install huggingface_hub") from error

    try:
        archive = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename="dataset.tar.gz", revision=REVISION
        )
    except Exception as error:
        raise RuntimeError(f"No se pudo descargar dataset.tar.gz de {REPO_ID}: {error}") from error

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pills-sxdht-", dir=DESTINATION.parent) as temporary:
        staged = Path(temporary) / DESTINATION.name
        extract(archive, staged)
        validate(staged)
        staged.rename(DESTINATION)
    print(f"Dataset descargado y extraído en {DESTINATION}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError) as error:
        sys.exit(f"Error: {error}")
