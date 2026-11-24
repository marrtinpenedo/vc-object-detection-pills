"""Authorize the scientific files of the two EXP-007 revisions."""

import hashlib
from pathlib import Path
import re
import subprocess


ORIGINAL = "ed7f9a6ef75afa9c66673bf8f03d34be2fa0c228"
ORIGINAL_CODE_RECEIPT = "81234d7e0e579b7bc58bfc1302996bed1f4028c2fccf7b621985625d977e3d46"
ORIGINAL_INPUT_RECEIPT = "ed65c83f0ef917f52fb99b10994ec2e5bca2719be01b67f552740bf7e7fe289f"
NEW_CONFIGS = (20, 25, 30)
SEEDS = (2026, 2027, 2028)
A_ARMS = ("control", "freeze-5", "freeze-10", "freeze-15")
B_ARMS = ("control", "freeze-15")
C_ARMS = (*A_ARMS, *(f"freeze-{duration}" for duration in NEW_CONFIGS))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt(path):
    hashes = {}
    for line in path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or name in hashes:
            raise ValueError(f"Huella inválida o duplicada: {path}: {line}")
        hashes[name] = sha
    return hashes


def verify_evidence(source):
    sums = source / "evidence.sha256"
    archived = sums.is_file()
    hashes = receipt(sums if archived else source / "downloaded.sha256")
    if archived and set(hashes) != {p.relative_to(source).as_posix()
                                    for p in source.rglob("*") if p.is_file() and p != sums}:
        raise ValueError(f"El inventario del archivo no coincide con sus huellas: {source}")
    for name, sha in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest(source / relative) != sha:
            raise ValueError(f"Evidencia incompatible: {source}/{name}")
    return hashes


def historical_receipt(original, historical_evidence):
    original_receipt = historical_evidence / "code-config.sha256"
    if ((historical_evidence / "source-commit").read_text().strip() != ORIGINAL
            or digest(original_receipt) != ORIGINAL_CODE_RECEIPT
            or digest(historical_evidence / "inputs.sha256") != ORIGINAL_INPUT_RECEIPT):
        raise ValueError("La procedencia histórica no coincide con la fase A autorizada")
    old = receipt(original_receipt)
    verify_script_inventory(original, old)
    for name, sha in old.items():
        if digest(original / name) != sha:
            raise ValueError(f"Checkout histórico incompatible: {name}")
    return old


def verify_script_inventory(checkout, hashes):
    actual = {p.relative_to(checkout).as_posix() for p in (checkout / "scripts").glob("*.py")}
    if actual != script_hashes(hashes).keys():
        raise ValueError(f"Inventario de scripts incompatible: {checkout}")


def verify_revisions(original, extended, historical_evidence):
    old = historical_receipt(original, historical_evidence)
    verify_script_inventory(extended, old)
    new = old.copy()
    for name, sha in old.items():
        candidate = extended / name
        if name == "scripts/baseline_run.py":
            expected = (original / name).read_bytes().replace(
                b'0 <= freeze < spec["epochs"]', b'0 <= freeze <= spec["epochs"]', 1,
            ).replace(b'un entero entre 0 y epochs - 1', b'un entero entre 0 y epochs', 1)
            if candidate.read_bytes() != expected:
                raise ValueError(f"Delta de entrenamiento no autorizado: {name}")
            new[name] = digest(candidate)
        elif digest(candidate) != sha:
            raise ValueError(f"Delta de código o configuración no autorizado: {name}")
    baseline = (original / "configs/baseline/yolo.json").read_bytes().rstrip()
    for duration in NEW_CONFIGS:
        name = f"configs/experiments/yolo-freeze-{duration}.json"
        expected = baseline[:-2] + f',\n  "freeze_backbone_epochs": {duration}\n}}\n'.encode()
        if (extended / name).read_bytes() != expected:
            raise ValueError(f"Configuración de ampliación no autorizada: {name}")
        new[name] = digest(extended / name)
    return old, new


def script_hashes(hashes):
    return {name: sha for name, sha in hashes.items() if name.startswith("scripts/")}


def verify_commit(git_repo, revision, expected):
    """Independently tie a deployment and its receipt to immutable Git blobs."""
    git = ["git", "-C", str(git_repo)]
    files = subprocess.check_output([*git, "ls-tree", "-r", "--name-only", revision, "--", "scripts"],
                                    text=True).splitlines()
    scripts = {name for name in files if name.count("/") == 1 and name.endswith(".py")}
    if scripts != script_hashes(expected).keys():
        raise ValueError(f"Inventario de scripts distinto en {revision}")
    for name, sha in expected.items():
        blob = subprocess.check_output([*git, "show", f"{revision}:{name}"])
        if hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError(f"Huella Git incompatible en {revision}: {name}")


def training_phase(seed, arm):
    if seed == SEEDS[0]:
        return "a" if arm in A_ARMS else "c"
    return "b" if arm in B_ARMS else "c"
