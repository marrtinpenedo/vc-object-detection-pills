"""Import one selective campaign archive and verify its byte inventory."""
import argparse
import shutil
import tarfile
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.baseline_data import read_json

archived = import_module("experiments.exp-002-photometric.import_results")
require, sha256 = archived.require, archived.sha256


def verify_inventory(root):
    inventory_path = root / "export.sha256.json"
    require(inventory_path.is_file() and not inventory_path.is_symlink(), "Inventario ausente")
    inventory = read_json(inventory_path)
    observed = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"Enlace no permitido en la importación: {path}")
        if path.is_file() and path != inventory_path:
            name = path.relative_to(root).as_posix()
            require(inventory.get(name) == sha256(path), f"Huella incorrecta: {name}")
            observed.add(name)
    require(observed == inventory.keys(), f"Faltan entradas: {sorted(inventory.keys() - observed)}")


def import_results(source, target):
    target = target.absolute()
    require(not target.exists() and not target.is_symlink(), f"El destino ya existe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".import-campaign-", dir=target.parent) as tmp:
        staged = Path(tmp)
        with tarfile.open(source, "r:gz") as tar:
            names = set()
            for entry in tar:
                name = Path(entry.name)
                require(entry.isfile() and not name.is_absolute() and ".." not in name.parts
                        and name not in names and name.as_posix() == entry.name, f"Entrada inválida: {entry.name}")
                names.add(name)
                dest = staged / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as stream, dest.open("xb") as out:
                    shutil.copyfileobj(stream, out)
        verify_inventory(staged)
        staged.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(import_results(args.source, args.target))
