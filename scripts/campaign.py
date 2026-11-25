"""Bundle, launch, monitor and export reproducible GPU campaigns."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.baseline_data import read_json, sha256, write_json
from scripts.baseline_run import REPO, load_config

PLAN = Path("configs/campaign/exp-003-005.json")
EXP008 = Path("configs/campaign/exp-008.json")
EXP009_A = Path("configs/campaign/exp-009-a.json")
BATCH = Path("slurm/campaign.sh")
SNAPSHOT = Path("snapshot.json")
LEDGER = Path("jobs.json")
YOLO_RECOVERY_FILES = {"scripts/campaign.py", "tests/test_geometry.py"}
BACKENDS = ("yolo", "cascade")
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "REQUEUE_FED",
          "RESIZING", "RESV_DEL_HOLD", "SUSPENDED", "SPECIAL_EXIT", "SIGNALING", "STAGE_OUT"}
FAILED = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY",
          "PREEMPTED", "REVOKED", "TIMEOUT"}
EXPORT_FILES = {
    "preflight": ("run.json", "status.json", "initialization.json", "ultralytics_runtime.json",
                  "cascade_runtime.json", "detectron2.yaml", "weight_transfer.json", "cascade_freezing.json", "smoke.json",
                  "parameters.json", "staged_transfer.json", "history.csv"),
    "train": ("run.json", "status.json", "history.csv", "ultralytics_runtime.json", "cascade_runtime.json",
              "detectron2.yaml", "weight_transfer.json", "parameter_updates.json", "best.sha256", "cascade_freezing.json",
              "parameters.json", "staged_transfer.json"),
    "eval": ("metrics.json", "predictions.json", "pr_curves.json"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bundle(output, plan_path=PLAN):
    sources = [plan_path, BATCH, Path("tests/test_geometry.py"),
               *(Path("configs/baseline") / f"{backend}.json" for backend in BACKENDS),
               *(file.relative_to(REPO) for file in sorted((REPO / "scripts").glob("*.py")))]
    identity = {"base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "sha256": {name.as_posix(): sha256(REPO / name) for name in sorted(sources)}}
    with TemporaryDirectory() as directory:
        receipt = Path(directory) / SNAPSHOT
        write_json(receipt, identity)
        with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name in sorted(sources):
                archive.add(REPO / name, arcname=name.as_posix())
            archive.add(receipt, arcname=SNAPSHOT.as_posix())
        print(f"Snapshot {sha256(receipt)} · tar SHA-256 {sha256(output)} · {output}")


def verify_source(expected=None):
    descriptor = REPO / SNAPSHOT
    digest = sha256(descriptor)
    require(expected is None or digest == expected, "El descriptor del snapshot cambió")
    identity = read_json(descriptor)
    for name, expected_hash in identity["sha256"].items():
        path = REPO / name
        require(path.resolve().is_relative_to(REPO) and path.is_file() and sha256(path) == expected_hash,
                f"Archivo alterado en el snapshot: {name}")
    # provenance() hashes every Python script. Refuse undeclared scripts in the deployed copy.
    require({f"scripts/{p.name}" for p in (REPO / "scripts").glob("*.py")} ==
            {name for name in identity["sha256"] if name.startswith("scripts/")},
            "La copia desplegada contiene scripts fuera del snapshot")
    return digest, identity


def condition_config(plan, condition):
    backend = condition["backend"]
    overrides = {"epochs": condition["epochs"], "seed": plan["seed"]}
    if "freeze_backbone_epochs" in condition:
        require((plan.get("experiment"), backend) in {("EXP-008", "cascade"), ("EXP-009", "yolo")},
                "Freezing con backend incompatible")
        overrides["freeze_backbone_epochs"] = condition["freeze_backbone_epochs"]
    spec = load_config(backend, None, overrides)
    if condition.get("geometric"):
        from scripts.geometric import validate_geometric

        validate_geometric(plan["geometric"])
        spec["geometric"] = plan["geometric"]
    if condition.get("mosaic"):
        probability = plan["mosaic_probability"]
        require(backend == "yolo" and condition["geometric"] and 0 < probability <= 1,
                "Mosaic requiere YOLO y la política geométrica")
        spec["mosaic"] = probability
    return spec


def graph(plan):
    nodes = {}
    for backend in plan["packages"]:
        nodes[f"preflight-{backend}"] = {"stage": "preflight", "backend": backend, "requires": None,
                                        "probes": [c["name"] for c in plan["conditions"]
                                                   if c["backend"] == backend and c["epochs"] == 30]}
    if plan.get("experiment") == "EXP-008":
        nodes["preflight-cascade"]["probes"] = ["e030-f000", "smoke"]
    if plan.get("experiment") == "EXP-009":
        nodes["preflight-yolo"]["probes"] = ["f000-control", "smoke-temporal", "smoke-permanent"]
    for condition in plan["conditions"]:
        name, backend = condition["name"], condition["backend"]
        nodes[f"train-{name}"] = {"stage": "train", "backend": backend, "condition": name,
                                  "requires": f"preflight-{backend}"}
        nodes[f"eval-{name}"] = {"stage": "eval", "backend": backend, "condition": name,
                                 "requires": f"train-{name}"}
    return nodes


def deployment(work, plan_path=PLAN):
    digest, _ = verify_source()
    plan = read_json(REPO / plan_path)
    data = work / plan["dataset_relpath"]
    require(sha256(data / "manifest.json") == plan["dataset_manifest_sha256"]
            and sha256(data / "coco/valid.json") == plan["valid_coco_sha256"], "Datos preparados inesperados")
    for backend, weights in plan["weights"].items():
        require((work / f"envs/{backend}/bin/python").is_file(), f"Falta el entorno {backend} en {work}")
        require(sha256(work / weights["path"]) == weights["sha256"], f"Pesos COCO inesperados: {backend}")
    return {"snapshot_sha256": digest, "work": str(work), "data": str(data), "plan": plan,
            "nodes": graph(plan),
            "recipes": {c["name"]: condition_config(plan, c) for c in plan["conditions"]}}


def attempt_dir(root, node, attempt):
    return root / "jobs" / node / f"attempt-{attempt:02d}"


def attempt_snapshot(manifest, record):
    return record.get("snapshot_sha256", manifest["snapshot_sha256"])


def snapshot_file(root, manifest, digest):
    require(re.fullmatch(r"[0-9a-f]{64}", digest), "SHA-256 de snapshot inválido")
    path = root / SNAPSHOT if digest == manifest["snapshot_sha256"] else root / "snapshots" / f"{digest}.json"
    require(sha256(path) == digest, f"Descriptor de snapshot incompatible: {path}")
    return path


def verify_yolo_recovery(original, fixed):
    before, after = original["sha256"], fixed["sha256"]
    require(original["base_commit"] == fixed["base_commit"] and before.keys() == after.keys(),
            "La recuperación YOLO requiere el mismo commit base y el mismo inventario")
    changes = {name for name in before if before[name] != after[name]}
    require(changes == YOLO_RECOVERY_FILES, f"Cambios incompatibles con esta recuperación YOLO: {sorted(changes)}")


def register_yolo_recovery(root, manifest, deployment_manifest, ledger, states, failed_preflight):
    original = manifest["snapshot_sha256"]
    fixed = deployment_manifest["snapshot_sha256"]
    require({**deployment_manifest, "snapshot_sha256": original} == manifest,
            "La recuperación no admite cambios de recetas, grafo, entradas ni rutas")
    verify_yolo_recovery(read_json(snapshot_file(root, manifest, original)), read_json(REPO / SNAPSHOT))
    records = ledger.get("preflight-yolo", [])
    require(any(r["id"] == failed_preflight and attempt_snapshot(manifest, r) == original
                and states.get(failed_preflight) == "FAILED" for r in records),
            f"SLURM no confirma el preflight YOLO original fallido {failed_preflight}")
    for name, node in manifest["nodes"].items():
        if node["backend"] != "yolo":
            continue
        attempts = ledger.get(name, [])
        require(attempts, f"No consta un envío inicial para {name}")
        for record in attempts:
            digest = attempt_snapshot(manifest, record)
            require(digest in (original, fixed), f"Se registró otro parche para {name}")
            if digest == original:
                require(states.get(record["id"]) in FAILED
                        and not (attempt_dir(root, name, record["attempt"]) / "done.json").exists(),
                        f"El intento original de {name} debe estar fallido/cancelado y sin recibo completo")
    target = root / "snapshots" / f"{fixed}.json"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        staged = target.with_suffix(".tmp")
        staged.write_bytes((REPO / SNAPSHOT).read_bytes())
        staged.replace(target)
    snapshot_file(root, manifest, fixed)


def job_states(ledger):
    ids = {a["id"] for attempts in ledger.values() for a in attempts if a.get("id")}
    if not ids:
        return {}
    completed = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(ids)), "-o", "JobIDRaw,State"], text=True)
    queued = subprocess.check_output(["squeue", "--me", "-h", "-o", "%i|%T"], text=True)
    states = {}
    for line in (completed + "\n" + queued).splitlines():
        job, _, state = line.partition("|")
        if job in ids and state.strip():
            states[job] = state.strip().split()[0].rstrip("+")
    return states


def disposition(root, node, record, states, manifest):
    ident = record.get("id")
    if not ident:
        return "FAILED" if record.get("confirmed_not_submitted") else "UNRECORDED"
    state = states.get(ident, "UNKNOWN")
    if state == "COMPLETED":
        receipt = attempt_dir(root, node, record["attempt"]) / "done.json"
        require(receipt.is_file() and read_json(receipt)
                == {"job_id": ident, "snapshot_sha256": attempt_snapshot(manifest, record)},
                f"SLURM indica COMPLETED pero falta un recibo válido: {node}, job {ident}")
        return "COMPLETE"
    if state in FAILED:
        return "FAILED"
    if state in ACTIVE:
        return "ACTIVE"
    return "UNKNOWN"


def launch(work, root, retry=False, failed_yolo_preflight=None, plan_path=PLAN):
    current = deployment(work, plan_path)
    manifest = current
    recovery = failed_yolo_preflight is not None
    require(not recovery or (root / "manifest.json").is_file(), "La recuperación YOLO requiere una campaña existente")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = root / "manifest.json"
        if receipt.exists():
            manifest = read_json(receipt)
            if not recovery:
                require(manifest == current, "La campaña ya está ligada a otro snapshot o entradas")
            snapshot_file(root, manifest, manifest["snapshot_sha256"])
            require(sha256(root / "prepared-manifest.json") == manifest["plan"]["dataset_manifest_sha256"],
                    "Manifiesto de datos registrado incompatible")
        else:
            (root / "configs").mkdir(exist_ok=True)
            (root / "logs").mkdir(exist_ok=True)
            for name, spec in manifest["recipes"].items():
                write_json(root / "configs" / f"{name}.json", spec)
            (root / SNAPSHOT).write_bytes((REPO / SNAPSHOT).read_bytes())
            data_manifest = Path(manifest["data"]) / "manifest.json"
            (root / "prepared-manifest.json").write_bytes(data_manifest.read_bytes())
            write_json(receipt, manifest)
        ledger_path = root / LEDGER
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
        for name, spec in manifest["recipes"].items():
            require(read_json(root / "configs" / f"{name}.json") == spec, f"Receta materializada alterada: {name}")
        states = job_states(ledger)
        if recovery:
            register_yolo_recovery(root, manifest, current, ledger, states, failed_yolo_preflight)
        digest = current["snapshot_sha256"]
        for name, node in manifest["nodes"].items():
            if recovery and node["backend"] != "yolo":
                continue
            attempts = ledger.setdefault(name, [])
            if attempts:
                state = disposition(root, name, attempts[-1], states, manifest)
                if state != "FAILED" or not retry:
                    print(f"{name}: {state} {attempts[-1].get('id', attempts[-1]['job_name'])}")
                    continue
            dependencies = []
            parent = node["requires"]
            if parent:
                upstream = ledger.get(parent, [])
                parent_state = disposition(root, parent, upstream[-1], states, manifest) if upstream else "UNKNOWN"
                if parent_state not in {"ACTIVE", "COMPLETE"}:
                    print(f"{name}: a la espera de {parent} ({parent_state})")
                    continue
                require(attempt_snapshot(manifest, upstream[-1]) == digest,
                        f"{name} requiere un padre del mismo snapshot: {parent}")
                # A completed dependency can already have aged out of the controller.
                if parent_state == "ACTIVE":
                    dependencies = [f"--dependency=afterok:{upstream[-1]['id']}", "--kill-on-invalid-dep=yes"]
            attempt = len(attempts) + 1
            output = attempt_dir(root, name, attempt)
            output.mkdir(parents=True)
            location = hashlib.sha256(str(root).encode()).hexdigest()[:8]
            job_name = f"pills-{location}-{digest[:8]}-{name}-{attempt:02d}"
            entry = {"attempt": attempt, "id": None, "job_name": job_name, "snapshot_sha256": digest,
                     "parent_id": ledger[parent][-1]["id"] if parent else None,
                     "afterok": bool(dependencies)}
            attempts.append(entry)
            # Persist intent before sbatch. A transport failure can be ambiguous.
            write_json(ledger_path, ledger)
            campaign_mode = {EXP008: "exp008", EXP009_A: "exp009-a"}.get(plan_path)
            campaign_option = [campaign_mode] if campaign_mode else []
            command = ["sbatch", "--parsable", "--export=NONE", "--no-requeue", f"--job-name={job_name}",
                       "--time=" + ("06:00:00" if node["stage"] == "train" else "01:00:00"),
                       f"--output={root}/logs/%x-%j.out", f"--error={root}/logs/%x-%j.err",
                       *dependencies, str(REPO / BATCH), *campaign_option, "job", str(work), node["backend"], str(root),
                       name, str(attempt)]
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
            if result.returncode or not re.fullmatch(r"\d+(?:;[\w.-]+)?", result.stdout.strip()):
                entry["submission_error"] = (result.stderr or result.stdout).strip()
                write_json(ledger_path, ledger)
                # Acceptance is unknown. Leave this node blocked and continue with independent nodes.
                print(f"{name}: envío sin ID confirmado ({job_name})", file=sys.stderr)
                continue
            entry["id"] = result.stdout.strip().split(";")[0]
            write_json(ledger_path, ledger)
            states[entry["id"]] = "PENDING"
            print(f"{name}: {entry['id']}")
        require(not any(attempts and disposition(root, name, attempts[-1], states, manifest) == "UNRECORDED"
                        for name, attempts in ledger.items() if not recovery or manifest["nodes"][name]["backend"] == "yolo"),
                "Hay envíos sin ID confirmado. Consulta status y reconcilia el recibo antes de reintentar.")


def run_module(module, *args, **kwargs):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=REPO, check=True, **kwargs)


def job(work, backend, root, name, attempt, plan_path=PLAN):
    ident = os.environ.get("SLURM_JOB_ID")
    require(ident and ident.isdecimal(), "Usa sbatch para los trabajos GPU")
    manifest = read_json(root / "manifest.json")
    require(work == Path(manifest["work"]), "Ruta de trabajo distinta de la registrada")
    node = manifest["nodes"][name]
    require(node["backend"] == backend, "Intérprete de otra arquitectura")
    ledger = read_json(root / LEDGER)
    entry = ledger[name][attempt - 1]
    require(entry["attempt"] == attempt and entry["id"] in (None, ident), "El envío difiere del registro de intentos")
    digest = attempt_snapshot(manifest, entry)
    snapshot_file(root, manifest, digest)
    verify_source(digest)
    plan, data = manifest["plan"], Path(manifest["data"])
    require(plan == read_json(REPO / plan_path) and manifest["nodes"] == graph(plan),
            "El manifiesto de lanzamiento difiere del snapshot")
    require(data == work / plan["dataset_relpath"]
            and sha256(data / "manifest.json") == plan["dataset_manifest_sha256"]
            and sha256(data / "coco/valid.json") == plan["valid_coco_sha256"],
            "Datos preparados alterados después del lanzamiento")
    output = attempt_dir(root, name, attempt)
    require(not (output / "done.json").exists(), f"El intento ya tiene salida completa: {output}")
    if backend == "yolo":
        settings = output / "ultralytics-config"
        settings.mkdir(exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(settings)
    package, version = plan["packages"][backend]
    weights = work / plan["weights"][backend]["path"]
    metadata = {"job_id": ident, "snapshot_sha256": digest}
    # An eligible launch follows the controller dependency. Verify the exact parent output too.
    parent_dir = None
    if node["requires"]:
        parent = next(a for a in ledger[node["requires"]] if a["id"] == entry["parent_id"])
        parent_dir = attempt_dir(root, node["requires"], parent["attempt"])
        require(attempt_snapshot(manifest, parent) == digest and read_json(parent_dir / "done.json") ==
                {"job_id": parent["id"], "snapshot_sha256": digest},
                "Recibo del trabajo previo inválido")
    if node["stage"] == "preflight":
        from importlib.metadata import version as package_version

        require(package_version(package) == version, f"Se requiere {package}=={version}")
        if backend == "cascade":
            import detectron2._C  # noqa: F401
        if plan_path != EXP008:
            checked_env = {**os.environ, "PYTHONPATH": str(REPO / "tests")}
            native_test = ("YoloGeometryLoaderTest" if backend == "yolo" else "CascadeGeometryLoaderTest")
            run_module("test_geometry", native_test, env=checked_env)
        for condition in node["probes"]:
            if condition == "smoke" and plan_path == EXP008:
                require(sha256(weights) == plan["weights"][backend]["sha256"], "Pesos COCO alterados")
                run_module("scripts.cascade_staged_smoke", "--data", data, "--weights", weights, "--output", output / "smoke")
                require(read_json(output / "smoke/smoke.json")["status"] == "complete", "Smoke sin transición")
            elif plan_path == EXP009_A and condition in ("smoke-temporal", "smoke-permanent"):
                yolo_stage_smoke(manifest, condition, data, weights, output / condition)
            else:
                train(root, manifest, condition, backend, data, weights, output / condition, initialize=True)
    elif node["stage"] == "train":
        train(root, manifest, node["condition"], backend, data, weights, output / "training")
    else:
        protocol = plan["evaluation"]
        run_module("scripts.evaluate_baseline", "--data", data, "--run", parent_dir / "training",
                   "--output", output / "evaluation", "--split", protocol["split"],
                   "--checkpoint", protocol["checkpoint"], "--benchmark",
                   "--warmup", protocol["warmup"], "--iterations", protocol["iterations"])
        metrics = read_json(output / "evaluation/metrics.json")
        require(metrics["checkpoint_sha256"] == (parent_dir / "training/best.sha256").read_text().strip(),
                "El checkpoint evaluado difiere del publicado por el entrenamiento")
    write_json(output / "done.json", metadata)


def yolo_smoke_recipe(manifest, name):
    duration = {"smoke-temporal": 1, "smoke-permanent": 3}[name]
    return {**manifest["recipes"]["f010-geom"], "epochs": 3, "freeze_backbone_epochs": duration}


def yolo_stage_smoke(manifest, name, data, weights, output):
    require(sha256(weights) == manifest["plan"]["weights"]["yolo"]["sha256"], "Pesos COCO del smoke alterados")
    spec = yolo_smoke_recipe(manifest, name)
    config = output.with_suffix(".json")
    write_json(config, spec)
    run_module("scripts.train_baseline", "yolo", "--data", data, "--weights", weights, "--config", config,
               "--output", output)
    audit = read_json(output / "staged_transfer.json")
    duration = spec["freeze_backbone_epochs"]
    require(read_json(output / "status.json")["status"] == "complete"
            and [row["epoch"] for row in audit["epochs"]] == [1, 2, 3]
            and [row["backbone_frozen"] for row in audit["epochs"]] == [epoch <= duration for epoch in (1, 2, 3)]
            and audit["thaw_epoch"] == (2 if duration == 1 else None), f"Smoke YOLO incompleto: {name}")


def train(root, manifest, condition, backend, data, weights, output, initialize=False):
    require(sha256(weights) == manifest["plan"]["weights"][backend]["sha256"], "Pesos COCO alterados")
    config = root / "configs" / f"{condition}.json"
    definition = next(row for row in manifest["plan"]["conditions"] if row["name"] == condition)
    require(read_json(config) == manifest["recipes"][condition] == condition_config(manifest["plan"], definition),
            "Receta alterada desde el envío")
    options = ["--initialize"] if initialize else []
    run_module("scripts.train_baseline", backend, "--data", data, "--weights", weights, "--config", config,
               "--output", output, *options)
    status = read_json(output / "status.json")
    if initialize:
        probe = read_json(output / "initialization.json")
        require(status["status"] == "initialized" and probe["classes"] == 8
                and "nonzero" in probe["gradients"].values(), "Sondeo incompleto")
    else:
        require(status["status"] == "complete", "Entrenamiento incompleto")
        (output / "best.sha256").write_text(sha256(output / status["best_checkpoint"]) + "\n")


def status(root):
    manifest = read_json(root / "manifest.json")
    ledger = read_json(root / LEDGER)
    states = job_states(ledger)
    for name in manifest["nodes"]:
        attempts = ledger.get(name, [])
        for record in attempts:
            ident = record.get("id")
            print(f"{name:<22} {record['attempt']:02d} {ident or '-':>10} "
                  f"{states.get(ident, '—'):<18} {disposition(root, name, record, states, manifest)} "
                  f"snapshot={attempt_snapshot(manifest, record)[:12]}")
            if not ident:
                print(f"  {record['job_name']} {record.get('submission_error', '')}")
        if not attempts:
            print(f"{name:<22} sin envío")


def collect(root, output):
    manifest = read_json(root / "manifest.json")
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        ledger = read_json(root / LEDGER)
        states = job_states(ledger)
        files = [root / name for name in ("manifest.json", SNAPSHOT, LEDGER, "prepared-manifest.json")]
        digests = {attempt_snapshot(manifest, record) for attempts in ledger.values() for record in attempts}
        files += [snapshot_file(root, manifest, digest) for digest in sorted(digests - {manifest["snapshot_sha256"]})]
        files += sorted((root / "configs").glob("*.json"))
        for name, attempts in ledger.items():
            stage = manifest["nodes"][name]["stage"]
            for record in attempts:
                state = disposition(root, name, record, states, manifest)
                if state not in {"COMPLETE", "FAILED"}:
                    continue
                attempt = attempt_dir(root, name, record["attempt"])
                files += [attempt / "done.json"]
                folders = (manifest["nodes"][name]["probes"] if stage == "preflight"
                           else ["training" if stage == "train" else "evaluation"])
                files += [attempt / folder / key for folder in folders for key in EXPORT_FILES[stage]]
        with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            hashes = {}
            def add(relative, content):
                hashes[relative] = hashlib.sha256(content).hexdigest()
                entry = tarfile.TarInfo(relative)
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))

            add("slurm-states.json", (json.dumps(states, indent=2) + "\n").encode())
            for path in files:
                if path.is_file():
                    add(path.relative_to(root).as_posix(), path.read_bytes())
            for attempts in ledger.values():
                for record in attempts:
                    ident = record.get("id")
                    if states.get(ident) in FAILED:
                        for suffix in ("out", "err"):
                            log = root / "logs" / f"{record['job_name']}-{ident}.{suffix}"
                            if log.is_file():
                                with log.open("rb") as source:
                                    source.seek(max(0, log.stat().st_size - 24000))
                                    add(log.relative_to(root).as_posix(), source.read())
            inventory = (json.dumps(hashes, indent=2) + "\n").encode()
            entry = tarfile.TarInfo("export.sha256.json")
            entry.size = len(inventory)
            archive.addfile(entry, io.BytesIO(inventory))
    print(output)


def reconcile(root, name, job_id, confirmed_not_submitted):
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / LEDGER
        ledger = read_json(path)
        record = ledger[name][-1]
        manifest = read_json(root / "manifest.json")
        require(disposition(root, name, record, {}, manifest) == "UNRECORDED", "El último intento ya tiene recibo")
        if confirmed_not_submitted:
            record["confirmed_not_submitted"] = True
        else:
            require(job_id.isdecimal(), "ID SLURM inválido")
            require(all(r.get("id") != job_id for attempts in ledger.values() for r in attempts),
                    "Ese ID ya está registrado")
            records = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job_id,
                                               "-o", "JobIDRaw,JobName%100"], text=True)
            queued = subprocess.check_output(["squeue", "--me", "-h", "-o", "%i|%j"], text=True)
            require(f"{job_id}|{record['job_name']}" in (records + "\n" + queued).splitlines(),
                    "SLURM no confirma ese ID con el nombre del envío")
            record["id"] = job_id
        write_json(path, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    campaign = parser.add_mutually_exclusive_group()
    campaign.add_argument("--exp008", action="store_true", help="Campaña de freezing Cascade")
    campaign.add_argument("--exp009-a", action="store_true", help="Fase A de optimización final YOLO")
    subs = parser.add_subparsers(dest="mode", required=True)
    pack = subs.add_parser("bundle")
    pack.add_argument("output", type=Path)
    subs.add_parser("plan")
    for name in ("launch", "recover-yolo", "status", "collect", "reconcile"):
        sub = subs.add_parser(name)
        sub.add_argument("--work", required=True, type=Path)
        sub.add_argument("--root", type=Path)
        if name == "launch":
            sub.add_argument("--retry-failed", action="store_true")
        if name == "recover-yolo":
            sub.add_argument("--failed-preflight", required=True, help="ID del preflight YOLO del snapshot original")
        if name == "collect":
            sub.add_argument("output", type=Path)
        if name == "reconcile":
            sub.add_argument("node")
            resolution = sub.add_mutually_exclusive_group(required=True)
            resolution.add_argument("--job-id")
            resolution.add_argument("--confirmed-not-submitted", action="store_true")
    sub = subs.add_parser("job")
    sub.add_argument("work", type=Path)
    sub.add_argument("backend", choices=BACKENDS)
    sub.add_argument("root", type=Path)
    sub.add_argument("name")
    sub.add_argument("attempt", type=int)
    args = parser.parse_args()
    plan_path = EXP009_A if args.exp009_a else EXP008 if args.exp008 else PLAN
    if args.mode == "bundle":
        bundle(args.output, plan_path)
        return
    if args.mode == "plan":
        plan = read_json(REPO / plan_path)
        print(json.dumps({"plan": plan, "nodes": graph(plan)}, indent=2))
        return
    work = args.work.resolve()
    default_root = {EXP008: "results/exp-008-freezing", EXP009_A: "results/exp-009-a"}.get(
        plan_path, "results/campaign-003-005")
    root = args.root.resolve() if args.root else work / default_root
    if args.mode == "launch":
        launch(work, root, args.retry_failed, plan_path=plan_path)
    elif args.mode == "recover-yolo":
        require(plan_path == PLAN and args.failed_preflight.isdecimal(), "Recuperación YOLO incompatible")
        launch(work, root, retry=True, failed_yolo_preflight=args.failed_preflight)
    elif args.mode == "status":
        status(root)
    elif args.mode == "collect":
        collect(root, args.output)
    elif args.mode == "reconcile":
        reconcile(root, args.node, args.job_id, args.confirmed_not_submitted)
    else:
        job(work, args.backend, root, args.name, args.attempt, plan_path)


if __name__ == "__main__":
    main()
