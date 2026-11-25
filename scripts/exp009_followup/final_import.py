"""Adapt a raw final evaluation directory tar to the campaign importer."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from scripts.campaign_analysis.import_results import import_results, require, sha256
from scripts.exp009_followup.final_export import FILES

ROOT = "exp-009-final-test"
RECEIPT = "directory-import.json"


def normalize(source, target):
    files, ignored, observed = {}, [], set()
    with tarfile.open(source) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and path.parts and path.parts[0] == ROOT and ".." not in path.parts
                    and path.as_posix() == member.name and path not in observed
                    and (member.isdir() or member.isfile()), f"Entrada de directorio inválida: {member.name}")
            observed.add(path)
            if member.isdir():
                continue
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if name in FILES:
                with archive.extractfile(member) as stream:
                    files[name] = stream.read()
            else:
                ignored.append(name)
    require(set(files) == set(FILES), f"Faltan artefactos: {sorted(set(FILES) - files.keys())}")
    receipt = {"kind": "local_projection", "source_archive_sha256": sha256(source), "archive_root": ROOT,
               "local_inventory_only": True, "ignored_members": sorted(ignored)}
    files[RECEIPT] = (json.dumps(receipt, indent=2) + "\n").encode()
    inventory = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files["export.sha256.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    with TemporaryDirectory(prefix="final-directory-import-") as temporary:
        projected = Path(temporary) / "official-layout.tar.gz"
        with tarfile.open(projected, "w:gz") as archive:
            for name, content in sorted(files.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))
        return import_results(projected, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(normalize(args.source, args.target))
