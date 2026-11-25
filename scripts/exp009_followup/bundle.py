"""Bundle B, C or D with the original, byte-for-byte runtime and a separate controller."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from scripts.baseline_data import read_json, sha256
from scripts.campaign import REPO, attempt_dir, condition_config, require
from scripts.campaign_analysis.import_results import verify_inventory
from scripts.exp009_followup.recipes import RUNTIME_PLAN, verify_plan

SOURCE = Path(__file__).resolve().parent
A_EVIDENCE = REPO / "experiments/exp-009-final-optimization/results/evidence"
D_EVIDENCE = REPO / "experiments/exp-008-cascade-freezing/results/evidence"


def plan_path(phase):
    return REPO / "configs/campaign" / f"exp-009-{phase.lower()}.json"


def check_reference(plan, imported):
    verify_inventory(imported)
    reference = plan["reference"]
    manifest = read_json(imported / "manifest.json")
    original = manifest["plan"]
    verify_plan(plan, original, condition_config)
    return check_pins(imported, manifest, reference["arms"], reference["snapshot_sha256"],
                      reference["inventory_sha256"])


def confirmation_references(plan, a, b):
    ref = plan["reference"]
    manifests = {}
    for phase, imported in (("A", a), ("B", b)):
        verify_inventory(imported)
        manifest = read_json(imported / "manifest.json")
        cohort = ref["cohorts"][phase]
        if phase == "A":
            verify_plan(plan, manifest["plan"], condition_config)
        else:
            require(manifest["plan"] == read_json(plan_path("B"))
                    and manifest["controller_sha256"] == cohort["controller_sha256"],
                    "Manifiesto o controlador de referencia B incompatible")
        pins = {name: row for name, row in ref["arms"].items() if row["phase"] == phase}
        manifests[phase] = check_pins(imported, manifest, pins, ref["snapshot_sha256"], cohort["inventory_sha256"])
    return manifests


def check_pins(imported, manifest, arms, snapshot, inventory):
    require(sha256(imported / "snapshot.json") == manifest["snapshot_sha256"] == snapshot
            and sha256(imported / "export.sha256.json") == inventory,
            "Inventario o snapshot de referencia distinto del oficial")
    ledger, states = (read_json(imported / file) for file in ("jobs.json", "slurm-states.json"))
    conditions = {row["name"]: row for row in manifest["plan"]["conditions"]}
    for name, arm in arms.items():
        require(conditions[name] == arm["condition"] and manifest["recipes"][name] == arm["config"],
                f"Receta histórica distinta: {name}")
        train_name, eval_name = f"train-{name}", f"eval-{name}"
        preflight_name = manifest["nodes"][train_name]["requires"]
        parent = None
        for node, key in ((preflight_name, "preflight_job"), (train_name, "training_job"),
                          (eval_name, "evaluation_job")):
            ident = arm[key]
            entries = [e for e in ledger[node] if e["id"] == ident]
            require(len(entries) == 1 and states.get(ident) == "COMPLETED", f"Trabajo de referencia incompleto: {node}")
            entry, = entries
            path = attempt_dir(imported, node, entry["attempt"])
            require(entry["parent_id"] == parent and
                    read_json(path / "done.json") == {"job_id": ident, "snapshot_sha256": snapshot},
                    f"Cadena de referencia incompatible: {node}")
            parent = ident
            if node == train_name:
                require((path / "training/best.sha256").read_text().strip() == arm["best_sha256"],
                        f"Checkpoint best de referencia distinto: {name}")
        evaluation = path / "evaluation"
        metrics = read_json(evaluation / "metrics.json")
        require(sha256(evaluation / "predictions.json") == arm["predictions_sha256"]
                and metrics["checkpoint_sha256"] == arm["best_sha256"]
                and metrics["split"] == "valid" and metrics["checkpoint_kind"] == "best",
                f"Evaluación de referencia distinta: {name}")
    return manifest


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundle(phase, reference, output, reference_b=None):
    spec = plan_path(phase)
    plan = read_json(spec)
    require((phase == "C") == (reference_b is not None), "C requiere las importaciones de A y B")
    manifest = (confirmation_references(plan, reference, reference_b)["A"] if phase == "C"
                else check_reference(plan, reference))
    original = (D_EVIDENCE / "exp-008-cascade-freezing.tar.gz" if phase == "D"
                else A_EVIDENCE / "exp-009-a.tar.gz")
    if phase != "D" and not original.is_file():
        original = REPO / "results/exp-009-a.tar.gz"
    with tarfile.open(original) as tar:
        runtime = {}
        for member in tar:
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and member.name not in runtime and path.as_posix() == member.name,
                    f"Entrada inválida en el bundle histórico: {member.name}")
            with tar.extractfile(member) as stream:
                runtime[member.name] = stream.read()
    snapshot = runtime["snapshot.json"]
    require(digest(snapshot) == plan["reference"]["snapshot_sha256"], "El snapshot histórico no coincide")
    require({file: digest(data) for file, data in runtime.items() if file != "snapshot.json"}
            == json.loads(snapshot)["sha256"], "Contenido histórico alterado")
    require(json.loads(runtime[RUNTIME_PLAN[phase]]) == manifest["plan"],
            "El plan del bundle y la evidencia histórica difieren")
    controller = {name: (SOURCE / name).read_bytes() for name in ("controller.py", "campaign.sh", "recipes.py")}
    controller["plan.json"] = spec.read_bytes()
    receipt = (json.dumps({name: digest(data) for name, data in sorted(controller.items())}, indent=2) + "\n").encode()
    sidecar = output.with_name(output.name + ".sha256")
    require(not sidecar.exists(), f"Ya existe {sidecar}")
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_stream:
        with tarfile.open(fileobj=gzip_stream, mode="w") as tar:
            files = {**controller, "controller.json": receipt, **{f"runtime/{n}": d for n, d in runtime.items()}}
            for filename, data in sorted(files.items()):
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"{output} · snapshot nativo {digest(snapshot)} · controlador {digest(receipt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("B", "C", "D"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-b", type=Path)
    args = parser.parse_args()
    bundle(args.phase, args.reference, args.output, args.reference_b)
