"""Submit and recover the EXP-007 SLURM chain. Run only on the CESGA login node."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import shlex
import subprocess

from scripts.baseline_data import read_json, write_json
from scripts.baseline_run import REPO, load_config

from .analyze import ALL_ARMS, common, require, verify_permanent_smoke, verify_stages
from .provenance import (
    A_ARMS, B_ARMS, C_ARMS, ORIGINAL, SEEDS, digest, receipt, script_hashes, training_phase, verify_revisions,
)


MODULE = "experiments.exp-007-yolo-staged-transfer.campaign"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "STAGE_OUT", "REQUEUED"}
FAILED = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED",
          "SPECIAL_EXIT", "TIMEOUT"}


@dataclass(frozen=True)
class Task:
    phase: str
    seed: int
    arm: str = ""

    @property
    def name(self):
        if self.phase == "smoke":
            return "exp007-yolo-smoke-permanent-s2026"
        if self.arm:
            return f"exp007-yolo-{self.arm}-s{self.seed}"
        return f"exp007-eval-{self.phase}-s{self.seed}"


def slurm(*args):
    return subprocess.check_output(args, text=True).strip()


def state(job, name):
    rows = slurm("sacct", "-X", "-n", "-P", "-j", job, "-o", "JobName%100,State,ExitCode").splitlines()
    require(len(rows) <= 1, f"SLURM devuelve varias filas para {job}")
    if not rows:
        rows = slurm("squeue", "-h", "-j", job, "-o", "%j|%T").splitlines()
        require(len(rows) == 1, f"SLURM aún no identifica {job}. Verificar antes de reenviar.")
        actual, status = rows[0].split("|")
        require(actual == name and status in ACTIVE, f"Trabajo sin identificar: {job}, {rows[0]}")
        return "active"
    actual, status, code = rows[0].split("|")
    require(actual == name, f"El trabajo {job} corresponde a {actual}, no a {name}")
    status = status.split()[0].rstrip("+")
    if status == "COMPLETED" and code == "0:0":
        return "complete"
    if status in ACTIVE:
        return "active"
    require(status in FAILED or status == "COMPLETED", f"Estado de SLURM desconocido: {job} {rows[0]}")
    return "failed"


def tasks():
    yield Task("smoke", 2026, "freeze-30")
    for seed in SEEDS[1:]:
        yield from (Task("b", seed, arm) for arm in B_ARMS)
        yield Task("b", seed)
    for seed in SEEDS:
        yield from (Task("c", seed, arm) for arm in C_ARMS if training_phase(seed, arm) == "c")
        yield Task("c", seed)


class Campaign:
    def __init__(self, original, historical, extension, revision, python, weights, retry):
        self.original, self.historical, self.extension = original, historical, extension
        self.revisions = {"a": ORIGINAL, "b": ORIGINAL, "c": revision, "smoke": revision}
        self.roots = {"a": historical, "b": historical, "c": extension, "smoke": extension}
        self.python, self.weights, self.retry = python, weights, retry
        self.queue = extension / "queue"
        self.data = historical / "data"
        old, new = verify_revisions(original, REPO, historical)
        self.codes = {phase: script_hashes(new if phase in {"c", "smoke"} else old) for phase in self.revisions}
        self.manifests = new
        self.inputs = {Path(path).name: sha for path, sha in receipt(historical / "inputs.sha256").items()}
        for file in (self.data / "manifest.json", self.data / "coco/valid.json", weights):
            require(digest(file) == self.inputs[file.name], f"Huella de entrada incompatible: {file}")

    def slot(self, task):
        root = self.roots[task.phase]
        if task.phase == "smoke":
            return root / "smoke"
        return root / f"s{task.seed}" / task.arm if task.arm else root / f"eval-{task.phase}-s{task.seed}"

    def checkout(self, task):
        # Evaluations share the historical evaluator even when checkpoints were trained in C.
        return REPO if task.arm and task.phase in {"c", "smoke"} else self.original

    def config(self, task):
        if task.phase == "smoke":
            return {**load_config("yolo", None, {}), "epochs": 3, "freeze_backbone_epochs": 3}
        variant = None if task.arm == "control" else self.checkout(task) / f"configs/experiments/yolo-{task.arm}.json"
        return load_config("yolo", variant, {"seed": task.seed})

    def records(self, task):
        return self.queue / f"{task.name}.json"

    def legacy_job_file(self, task):
        return self.slot(task) / ("train.job" if task.arm else "eval.job")

    def verify_training(self, task, job):
        slot = self.slot(task)
        directory = (slot / "train").resolve(strict=True)
        run = read_json(directory / "run.json")
        spec = run["run"]
        status = read_json(directory / "status.json")
        require(run["mode"] == "train" and run["git_head"] in (None, self.revisions[task.phase])
                and run["code_sha256"] == self.codes[task.phase]
                and run["hardware"]["job_id"] == status["sessions"][-1]["job_id"] == job,
                f"Procedencia o ID incompatible: {slot}")
        require(Path((slot / "train.path").read_text().strip()).resolve() == directory
                and spec["backend"] == "yolo" and spec["config"] == read_json(slot / "config.json") == self.config(task)
                and spec["dataset_manifest_sha256"] == self.inputs["manifest.json"]
                and spec["initial_weights_sha256"] == self.inputs["yolov10s.pt"]
                and status["status"] == "complete", f"Receta, datos o estado incompatible: {slot}")
        checkpoint = (directory / status["best_checkpoint"]).resolve(strict=True)
        require(checkpoint.is_relative_to(directory) and checkpoint.is_file() and checkpoint.stat().st_size > 0,
                f"Checkpoint inválido: {checkpoint}")
        if task.phase == "smoke":
            require((slot / "verified.ok").is_file(), "El smoke terminó sin evidencia de verificación")
            verify_permanent_smoke(slot)
        else:
            history = common.read_history(directory / "history.csv")
            duration = ALL_ARMS[task.arm]
            verify_stages(read_json(directory / "staged_transfer.json") if duration else None,
                          read_json(directory / "parameters.json"), read_json(directory / "ultralytics_runtime.json"),
                          history, duration)
        return checkpoint

    def verify_evaluation(self, task, job):
        directory = self.slot(task)
        launch = read_json(directory / "launch.json")
        arms = {"a": A_ARMS, "b": B_ARMS, "c": C_ARMS}[task.phase]
        require((directory / "verified.ok").is_file() and launch["job_id"] == job
                and (directory / "eval.job").read_text().strip() == job and set(launch["runs"]) == set(arms)
                and launch["split"] == "valid" and launch["checkpoint_kind"] == "best"
                and launch["dataset_manifest_sha256"] == self.inputs["manifest.json"]
                and launch["coco_sha256"] == self.inputs["valid.json"]
                and (launch["benchmark"], launch["warmup"], launch["iterations"]) == (True, 20, 100)
                and launch["example_image_ids"] == [57, 61, 80]
                and launch["launcher_sha256"] == {name: digest(self.original / "slurm" / name)
                                                  for name in ("evaluate.sh", "evaluate.py")},
                f"Evaluación no verificada o incompatible: {directory}")
        for arm in arms:
            trained = Task(training_phase(task.seed, arm), task.seed, arm)
            train_job = self.legacy_job_file(trained).read_text().strip()
            best = self.verify_training(trained, train_job)
            entry = launch["runs"][arm]
            metrics = read_json(directory / arm / "metrics.json")
            require(metrics["code_sha256"] == self.codes["a"]
                    and metrics["git_head"] in (None, ORIGINAL)
                    and metrics["hardware"]["job_id"] == job
                    and entry["training_job_id"] == train_job and entry["training_slurm"] == "COMPLETED|0:0"
                    and Path(entry["run"]).resolve() == self.slot(trained).resolve() / "train"
                    and Path(entry["checkpoint"]).resolve() == best
                    and metrics["checkpoint_sha256"] == entry["checkpoint_sha256"] == digest(best),
                    f"Checkpoint reevaluado incompatible: {directory}/{arm}")

    def prepare_campaign(self):
        self.queue.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("source-commit", f"{self.revisions['c']}\n"),
            ("code-config.sha256", "".join(f"{sha}  {name}\n" for name, sha in self.manifests.items())),
            ("inputs.sha256", (self.historical / "inputs.sha256").read_text()),
        ):
            path = self.extension / name
            if path.exists():
                require(path.read_text() == content, f"La campaña existente usa otra revisión o entradas: {path}")
            else:
                path.write_text(content)
        (self.extension / "slurm").mkdir(exist_ok=True)
        # A one-file data receipt permits analysis without copying the prepared dataset.
        manifest = self.extension / "data/manifest.json"
        if not manifest.exists():
            manifest.parent.mkdir(exist_ok=True)
            manifest.write_bytes((self.data / "manifest.json").read_bytes())
        require(digest(manifest) == self.inputs["manifest.json"], "Manifiesto de extensión distinto")

    def command(self, task):
        slot = self.slot(task)
        if not task.arm:
            arms = B_ARMS if task.phase == "b" else C_ARMS
            return ["slurm/evaluate.sh", "--data", str(self.data), "--output", str(slot), "--split", "valid",
                    "--checkpoint", "best", "--check-slurm", "--example-image-id", "57", "--example-image-id", "61",
                    "--example-image-id", "80",
                    *(value for arm in arms for value in
                      ("--run", arm, str(self.slot(Task(training_phase(task.seed, arm), task.seed, arm)) / "train")))]
        slot.mkdir(parents=True)
        write_json(slot / "config.json", self.config(task))
        (slot / "train.path").write_text(f"{slot / 'train'}\n")
        argv = ["slurm/yolo.sh", "--data", str(self.data), "--weights", str(self.weights),
                "--config", str(slot / "config.json"), "--output", str(slot / "train")]
        if task.phase == "smoke":
            check = ["env", "-u", "PYTHONHOME", "-u", "PYTHONPATH", "PYTHONNOUSERSITE=1", str(self.python),
                     "-m", MODULE, "verify-smoke", str(slot)]
            wrapped = f"{shlex.join(['bash', '-l', *argv])} && {shlex.join(check)}"
            return ["--partition=short", "--nodes=1", "--ntasks=1", "--gres=gpu:a100:1", "--cpus-per-task=32",
                    "--mem=64G", "--wrap", wrapped]
        return argv

    def job(self, task):
        log = self.records(task)
        existing = self.legacy_job_file(task)
        if log.is_file():
            record = read_json(log)
            require(record["job"], f"Envío sin confirmar: {log}. Consultar SLURM antes de reconciliar el ID.")
            if existing.is_file():
                require(existing.read_text().strip() == record["job"], f"IDs contradictorios: {log}")
            return record["job"]
        return existing.read_text().strip() if existing.is_file() else None

    def active_external_jobs(self):
        active = set()
        for task in tasks():
            job = self.job(task)
            queued = slurm("squeue", "--me", "--name", task.name, "-h", "-o", "%i").splitlines()
            require(not set(queued) - {job}, f"Trabajo activo sin registro para {task.name}: {queued}")
            if job and not self.records(task).exists() and state(job, task.name) == "active":
                require(task.phase == "b", f"Trabajo activo de la ampliación fuera del registro: {task.name}")
                slot = self.slot(task)
                if task.arm:
                    require(read_json(slot / "config.json") == self.config(task)
                            and Path((slot / "train.path").read_text().strip()).resolve() == slot.resolve() / "train",
                            f"Trabajo previo activo con receta o ruta distinta: {slot}")
                active.add(job)
        require(len(active) <= 1, "Hay varios trabajos previos activos. Resolver su solapamiento antes de encolar.")
        return active

    def schedule(self, task, dependencies, external):
        log, slot = self.records(task), self.slot(task)
        job = self.job(task)
        if job:
            current = state(job, task.name)
            if task.arm and log.is_file() and current in {"active", "complete"}:
                job_file = self.legacy_job_file(task)
                if not job_file.exists():
                    job_file.write_text(f"{job}\n")
            if current == "complete":
                (self.verify_training if task.arm else self.verify_evaluation)(task, job)
                if not log.exists():
                    write_json(log, {"job": job, "afterok": [], "adopted": True})
                print(f"REUSE {task.name} {job}", flush=True)
                return None
            if current == "active":
                # A prior external job must finish before the new chain, never wait on itself.
                if log.is_file():
                    recorded = set(read_json(log)["afterok"])
                    require(dependencies - {job} <= recorded, f"{task.name} sigue en cola con dependencias antiguas")
                else:
                    require(job in external, f"Trabajo previo no inventariado: {task.name}")
                print(f"QUEUED {task.name} {job}", flush=True)
                return job
            require(self.retry, f"{task.name} {job} falló. Revisar su log y usar --retry-failed.")
        outstanding = slurm("squeue", "--me", "--name", task.name, "-h", "-o", "%i")
        require(not outstanding, f"Trabajo activo sin registro para {task.name}: {outstanding}")
        if job:
            archive = self.extension / "attempts" / f"{task.name}-{job}"
            archive.mkdir(parents=True)
            if slot.exists():
                slot.rename(archive / "output")
            if log.exists():
                log.rename(archive / "submission.json")
        require(not slot.exists(), f"Salida sin ID conocido: {slot}. Conservarla y reconciliar antes de reenviar.")
        argv = self.command(task)
        deps = sorted(dependencies | external)
        logs = self.roots[task.phase] / "slurm"
        command = ["sbatch", "--parsable", "--export=ALL", "--time=01:00:00", "--kill-on-invalid-dep=yes",
                   f"--job-name={task.name}", f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
        if deps:
            command.append(f"--dependency=afterok:{':'.join(deps)}")
        checkout = self.checkout(task)
        command.extend(argv)
        record = {"job": None, "afterok": deps, "command": command, "checkout": str(checkout),
                  "submitted_at": datetime.now(timezone.utc).isoformat()}
        write_json(log, record)
        env = {**os.environ, "PILLS_YOLO_CONFIG_DIR": str(slot / "ultralytics")}
        env.pop("PILLS_EVAL_LAUNCHER_DIR", None)
        output = subprocess.check_output(command, cwd=checkout, env=env, text=True).strip()
        job = output.split(";")[0]
        require(re.fullmatch(r"[0-9]+", job), f"Respuesta de sbatch inesperada: {output}; revisar {log}")
        record["job"] = job
        write_json(log, record)
        if task.arm:
            self.legacy_job_file(task).write_text(f"{job}\n")
        print(f"SUBMIT {task.name} {job}", flush=True)
        return job

    def submit(self):
        self.prepare_campaign()
        a = Task("a", 2026)
        self.verify_evaluation(a, self.legacy_job_file(a).read_text().strip())
        external = self.active_external_jobs()
        live = set()
        for task in tasks():
            job = self.schedule(task, live, external)
            if job:
                live.add(job)
        print(f"Registro de la campaña: {self.queue}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    smoke = sub.add_parser("verify-smoke")
    smoke.add_argument("slot", type=Path)
    submit = sub.add_parser("submit")
    submit.add_argument("--revision", required=True)
    submit.add_argument("--original-checkout", type=Path, default=Path.home() / f"vc-object-detection-pills-{ORIGINAL[:12]}")
    submit.add_argument("--historical", type=Path)
    submit.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "verify-smoke":
            verify_permanent_smoke(args.slot)
            (args.slot / "verified.ok").touch(exist_ok=False)
        else:
            require(re.fullmatch(r"[0-9a-f]{40}", args.revision) and args.revision != ORIGINAL,
                    "Indicar el SHA completo de la revisión nueva desplegada")
            work = Path(os.environ["STORE"]) / "pills-baseline"
            historical = (args.historical or work / "results/exp-007-yolo-staged-HhuHld").resolve(strict=True)
            extension = historical.with_name(historical.name + "-extension").resolve()
            require(not extension.is_relative_to(historical) and not historical.is_relative_to(extension),
                    "Las dos campañas deben estar en directorios separados")
            original = args.original_checkout.resolve(strict=True)
            require(REPO != original, "Ejecutar la ampliación desde el checkout nuevo")
            for candidate in (original, REPO):
                if (candidate / ".git").exists():
                    expected = ORIGINAL if candidate == original else args.revision
                    head = subprocess.check_output(["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True).strip()
                    require(head == expected, f"Revisión desplegada distinta: {candidate}")
                elif candidate == REPO:
                    require((candidate / ".exp007-revision").read_text().strip() == args.revision,
                            "La revisión indicada difiere del recibo escrito al desplegar")
            with (historical / ".extension-submit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                Campaign(original, historical, extension, args.revision, work / "envs/yolo/bin/python",
                         work / "checkpoints/yolov10s.pt", args.retry_failed).submit()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
