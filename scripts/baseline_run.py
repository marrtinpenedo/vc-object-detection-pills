"""Configuration, provenance and training-history helpers for both baselines."""

import csv
import importlib
import importlib.metadata
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

from scripts.baseline_data import read_json, sha256, write_json
from scripts.download_dataset import SPLITS

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "cascade": ("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", "model_final_480dd8.pkl"),
    "yolo": ("yolov10s.pt", "yolov10s.pt"),
}
OVERRIDES = {"epochs": int, "batch": int, "imgsz": int, "workers": int, "seed": int, "lr": float}


class PauseTraining(BaseException):
    """An intentional pause that bypasses the trainer's error logging."""


class UpdateCheck:
    """Check one backbone and one classifier tensor after a session's first epoch."""

    def __init__(self, model, output, prefixes):
        self.output = output / "parameter_updates.json"
        self.samples = []
        for prefix in prefixes:
            name, parameter = next(
                (name, p) for name, p in model.named_parameters()
                if name.startswith(prefix) and name.endswith(".weight") and p.requires_grad
            )
            self.samples.append((name, parameter, parameter.detach().cpu().clone()))

    def after_epoch(self, step):
        if not self.samples:
            return
        changed = {
            name: not before.equal(parameter.detach().cpu())
            for name, parameter, before in self.samples
        }
        if not all(changed.values()):
            raise RuntimeError(f"Se esperaba actualizar los parámetros de control: {changed}")
        records = read_json(self.output) if self.output.is_file() else []
        write_json(self.output, [*records, {"step": step, "changed": changed}])
        self.samples.clear()


def backend_for(name):
    return importlib.import_module(f"scripts.baseline_{name}")


def publish_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def load_config(backend, path, overrides):
    shape = read_json(REPO / "configs" / "baseline" / f"{backend}.json")
    spec = read_json(path) if path else shape.copy()
    if spec.keys() - {"photometric", "geometric", "mosaic"} != shape.keys():
        raise ValueError(f"Claves de configuración diferentes de {sorted(shape)}")
    if "mosaic" in spec:
        probability = spec["mosaic"]
        if (backend != "yolo" or "geometric" not in spec or type(probability) not in (int, float)
                or not 0 < probability <= 1):
            raise ValueError("mosaic requiere YOLO, geometría y una probabilidad en (0, 1]")
    if "geometric" in spec:
        from scripts.geometric import validate_geometric

        if "photometric" in spec:
            raise ValueError("La receta admite una sola política de augmentación")
        validate_geometric(spec["geometric"])
    if "photometric" in spec:
        from scripts.photometric import validate_photometric

        validate_photometric(spec["photometric"])
    for key, value in overrides.items():
        if value is not None:
            spec[key] = value
    for key, default in shape.items():
        value = spec[key]
        if type(default) in (int, float):
            types = (int,) if type(default) is int else (int, float)
            if type(value) not in types or not math.isfinite(value) or value < 0:
                raise ValueError(f"Valor numérico inválido para {key}: {value!r}")
        elif value != default:
            raise ValueError(f"{key} debe ser {default!r} para este baseline")
    if (
        not 0 < spec["batch"] <= SPLITS["train"] or spec["epochs"] < 1 or spec["lr"] <= 0
        or spec["imgsz"] < 32 or spec["imgsz"] % 32 or spec["seed"] >= 2**32
        or spec["eval_every_epochs"] != 1 or spec["checkpoint_every_epochs"] != 1
    ):
        raise ValueError("Revisa lote, épocas, LR, tamaño múltiplo de 32, semilla y frecuencia por época")
    for key in ("momentum", "warmup_momentum", "warmup_factor", "final_lr_factor", "nms_iou"):
        if key in spec and not 0 < spec[key] <= 1:
            raise ValueError(f"{key} debe estar en (0, 1]")
    return spec


def dry_runtime():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Ejecuta en un nodo de cómputo con exactamente una GPU CUDA visible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def hardware(torch):
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_runtime": torch.version.cuda, "nvidia_driver": driver, "cpu": platform.processor() or platform.machine(),
        "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
    }


def provenance(torch):
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    packages = {}
    for package in importlib.metadata.distributions():
        name = package.metadata["Name"]
        if name:
            packages[name] = package.version
    return {
        "hardware": hardware(torch), "git_head": git,
        "packages": dict(sorted(packages.items())),
        "code_sha256": {
            path.relative_to(REPO).as_posix(): sha256(path)
            for path in sorted((REPO / "scripts").glob("*.py"))
        },
    }


def parameters(model):
    return {
        "trainable": [{"name": name, "shape": list(p.shape), "numel": p.numel()}
                      for name, p in model.named_parameters() if p.requires_grad],
        "frozen": [{"name": name, "shape": list(p.shape), "numel": p.numel()}
                   for name, p in model.named_parameters() if not p.requires_grad],
        "frozen_bn_buffers": [name for name, _ in model.named_buffers()
                              if any(name.startswith(f"{key}.") for key, mod in model.named_modules()
                                     if type(mod).__name__ == "FrozenBatchNorm2d")],
    }


def check_gradients(torch, model):
    per_parameter = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"Parámetro congelado con gradiente: {name}")
            continue
        grad = parameter.grad
        if grad is None:
            per_parameter[name] = "absent"
        elif not torch.isfinite(grad).all().item():
            raise RuntimeError(f"Gradiente no finito: {name}")
        else:
            per_parameter[name] = "nonzero" if torch.count_nonzero(grad).item() else "zero"
    if "nonzero" not in per_parameter.values():
        raise RuntimeError("La pasada de retropropagación no produjo gradientes útiles")
    return per_parameter


def history_plots(rows, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, fields, heading in (
        (axes[0], {key for row in rows for key in row if "loss" in key}, "Pérdidas"),
        (axes[1], {"val/map", "val/ap50"}, "Detección en validación"),
    ):
        for field in sorted(fields):
            points = [(row["step"], row[field]) for row in rows if field in row and row[field] is not None]
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="." if field.startswith("val/") else None, label=field)
        ax.set(title=heading, xlabel="Paso (iteración o época según el detector)")
        if ax.lines:
            ax.legend(fontsize="small")
    axes[1].set_ylim(0, 1.03)
    fig.savefig(output / "training.png", dpi=140)
    plt.close(fig)


def save_history(rows, output, prefer_last_tie):
    fields = ["step", *sorted({key for row in rows for key in row} - {"step"})]
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    history_plots(rows, output)
    scored = [row for row in rows if "val/map" in row]
    return max(
        scored, key=lambda row: (row["val/map"], row["step"] if prefer_last_tie else -row["step"]),
        default=None,
    )


def save_session(output, torch, started, status, metadata):
    path = output / "status.json"
    old = read_json(path) if path.exists() else {}
    sessions = old.get("sessions", [])
    sessions.append({
        "status": status,
        "elapsed_seconds": perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    })
    write_json(path, {
        **metadata, "status": status, "sessions": sessions,
        "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in sessions),
        "cuda_peak_allocated_bytes": max(item["cuda_peak_allocated_bytes"] for item in sessions),
    })
