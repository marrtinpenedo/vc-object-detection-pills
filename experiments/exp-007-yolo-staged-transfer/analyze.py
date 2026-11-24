"""Verifica evidencias de EXP-007 y aplica sus criterios preestablecidos."""

import argparse
from importlib import import_module
import math
from pathlib import Path
import re

from scripts.baseline_data import DESTINATION, read_json, sha256, write_json
from scripts.baseline_run import REPO, load_config
from scripts.verify_baseline import require, verify_evaluation

# Reuse the history, benchmark and COCO reconstruction of the preceding YOLO comparison.
common = import_module("experiments.exp-006-yolo-warmup.analyze")
ARMS = {"control": 0, "freeze-5": 5, "freeze-10": 10, "freeze-15": 15}
REGIONS = {"backbone", "neck", "one2many", "one2one"}
DFL = "model.23.dfl.conv.weight"
ENVIRONMENT_HARDWARE = ("gpu", "cuda_runtime", "nvidia_driver")


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_stages(audit, inventory, runtime, history, duration):
    frozen = {row["name"]: row["numel"] for row in inventory["frozen"]}
    trainable = {row["name"]: row["numel"] for row in inventory["trainable"]}
    if duration == 0:
        require(set(frozen) == {DFL}, "El control debe conservar solo DFL congelada")
        return
    require(audit["freeze_backbone_epochs"] == runtime["freeze_backbone_epochs"] == duration
            and audit["thaw_epoch"] == duration + 1
            and audit["optimizer_groups_verified_at_thaw"] is True,
            "Duración, transición de descongelación o grupos AdamW incorrectos")
    require([item["index"] for item in audit["modules"]] == list(range(24))
            and [item["part"] for item in audit["modules"]] == ["backbone"] * 11 + ["neck"] * 12 + ["detector"]
            and audit["modules"][-1]["type"] == "v10Detect", "Delimitación del backbone inconsistente")
    expected_names = {name for name in frozen | trainable
                      if re.match(r"model\.(?:[0-9]|10)\.", name)}
    total = sum(frozen.values()) + sum(trainable.values())
    require(set(audit["backbone_parameters"]) == expected_names == frozen.keys() - {DFL}
            and audit["internally_frozen"] == [DFL] and audit["frozen_after_thaw"] == frozen[DFL]
            and audit["parameters_total"] == total and audit["frozen_initial"] == sum(frozen.values())
            and math.isclose(audit["frozen_initial_pct"], 100 * sum(frozen.values()) / total),
            "El inventario de parámetros de la fase inicial no coincide")
    require(audit["backbone_buffers"] and all(re.match(r"model\.(?:[0-9]|10)\.", name)
                                             for name in audit["backbone_buffers"])
            and audit["backbone_batch_norm"], "Faltan buffers o Batch Normalization del backbone")
    groups = audit["optimizer_groups"]
    membership = {name: group["index"] for group in groups for name in group["parameters"]}
    require(audit["optimizer"] == runtime["optimizer"] == "AdamW"
            and [group["index"] for group in groups] == [0, 1, 2]
            and len(membership) == len(frozen | trainable) == sum(len(group["parameters"]) for group in groups)
            and membership.keys() == (frozen | trainable).keys()
            and all((index == 0) == ("bias" in name) for name, index in membership.items())
            and {membership[name] for name in expected_names} == {0, 1, 2}
            and [{key: group[key] for key in ("initial_lr", "weight_decay", "betas")} for group in groups]
            == runtime["optimizer_initial_groups"], "Pertenencia o configuración AdamW incompatible")
    samples = audit["samples"]
    require(samples.keys() == REGIONS and all(name in membership for name in samples.values())
            and samples["backbone"] in expected_names
            and samples["neck"].startswith("model.13.")
            and samples["one2many"].startswith("model.23.cv3.")
            and samples["one2one"].startswith("model.23.one2one_cv3."), "Muestras de actualización incorrectas")
    events = audit["epochs"]
    require([row["epoch"] for row in events] == [row["step"] for row in history], "Faltan épocas verificadas")
    before, internal = audit["initial_backbone_sha256"], audit["internal_sha256"]
    require(valid_digest(before) and valid_digest(internal), "Huellas iniciales inválidas")
    batches = runtime["iterations_per_epoch"]
    for event, row in zip(events, history):
        epoch = row["step"]
        fixed = epoch <= duration
        active = REGIONS - {"backbone"} if fixed else REGIONS
        digest = event["backbone_sha256"]
        require(event["backbone_frozen"] is fixed and event["backbone_bn_training"] is not fixed
                and event["backbone_bn_steps"] == (0 if fixed else batches)
                and event["frozen_gradients_absent"] is True
                and valid_digest(digest) and (before == digest) is fixed and event["internal_sha256"] == internal,
                f"Pesos, gradientes o Batch Normalization incompatibles en época {epoch}")
        require(event["nonzero_gradients"] == event["updated"] == dict.fromkeys(active, True)
                and event["adamw_steps"] == {
                    name: batches * (max(0, epoch - duration) if name == "backbone" else epoch) for name in REGIONS
                }
                and event["scheduler_epoch"] == epoch - 1 and event["ema_updates"] == epoch * batches
                and len(event["lr"]) == len(common.LR)
                and all(math.isclose(rate, row[key], rel_tol=5e-6) for rate, key in zip(event["lr"], common.LR)),
                f"Actualizaciones o continuidad incompatibles en época {epoch}")
        before = digest


def read_cohort(source, phase, seed, arms, identity, truth):
    commit, inputs = identity
    evaluation = source / f"eval-{phase}-s{seed}"
    require((evaluation / "verified.ok").is_file(), "Falta el sello de verificación del lanzador")
    job = (evaluation / "eval.job").read_text().strip()
    launch = read_json(evaluation / "launch.json")
    require(launch["job_id"] == job and set(launch["runs"]) == set(arms)
            and launch["dataset_manifest_sha256"] == inputs["manifest.json"]
            and launch["coco_sha256"] == inputs["valid.json"] and launch["split"] == "valid"
            and launch["checkpoint_kind"] == "best" and launch["example_image_ids"] == [57, 61, 80]
            and (launch["benchmark"], launch["warmup"], launch["iterations"]) == (True, 20, 100)
            and launch["launcher_sha256"] == {name: sha256(REPO / "slurm" / name) for name in ("evaluate.sh", "evaluate.py")},
            "Procedencia o inventario del lanzamiento incompatible")
    code = {path.relative_to(REPO).as_posix(): sha256(path) for path in sorted((REPO / "scripts").glob("*.py"))}
    results, histories, comparability = {}, {}, []
    environment = None
    for arm in arms:
        slot = source / f"s{seed}" / arm
        training = slot / "train"
        output = evaluation / arm
        original = Path((slot / "train.path").read_text().strip())
        run = read_json(training / "run.json")
        runtime = read_json(training / "ultralytics_runtime.json")
        status = read_json(training / "status.json")
        metrics = verify_evaluation(output, truth, inputs["valid.json"], inputs["manifest.json"], "valid")
        recorded = launch["runs"][arm]
        config = None if arm == "control" else REPO / f"configs/experiments/yolo-{arm}.json"
        expected = load_config("yolo", config, {"seed": seed})
        require(run["run"]["config"] == metrics["model_config"] == read_json(slot / "config.json") == expected,
                f"{arm}: recetas distintas")
        train_job = (slot / "train.job").read_text().strip()
        require(status["status"] == "complete" and run["mode"] == "train"
                and status["sessions"][-1]["job_id"] == run["hardware"]["job_id"] == recorded["training_job_id"] == train_job
                and (evaluation / f"{arm}.train.sacct").read_text().strip() == recorded["training_slurm"] == "COMPLETED|0:0",
                f"{arm}: estado de entrenamiento incompatible")
        require(run["run"]["backend"] == recorded["backend"] == metrics["backend"] == "yolo"
                and run["code_sha256"] == metrics["code_sha256"] == code
                and all(item["git_head"] in (None, commit) for item in (run, metrics))
                and run["run"]["dataset_manifest_sha256"] == inputs["manifest.json"]
                and run["run"]["initial_weights_sha256"] == inputs["yolov10s.pt"]
                and Path(recorded["run"]) == original and metrics["hardware"]["job_id"] == job
                and original / status["best_checkpoint"] == Path(metrics["checkpoint"]) == Path(recorded["checkpoint"])
                and valid_digest(metrics["checkpoint_sha256"])
                and metrics["checkpoint_sha256"] == recorded["checkpoint_sha256"] and metrics["checkpoint_kind"] == "best",
                f"{arm}: procedencia de datos, checkpoint o código incompatible")
        transferred = read_json(training / "weight_transfer.json")["reinitialized"]
        require(len(transferred) == 12 and sum(key.startswith("model.23.cv3.") for key in transferred) == 6
                and sum(key.startswith("model.23.one2one_cv3.") for key in transferred) == 6,
                f"{arm}: adaptación de las dos ramas incompatible")
        args = runtime["args"]
        require(runtime["iterations_per_epoch"] == 40 and runtime["effective_batch"] == 8
                and runtime["nominal_max_iterations"] == 1200 and runtime["warmup_iterations_end_index"] == 100
                and args["seed"] == seed and args["optimizer"] == "AdamW" and args["epochs"] == 30
                and args["imgsz"] == 640 and args["batch"] == 8 and not args["amp"]
                and args["cos_lr"] and args["lr0"] == 0.001 and args["lrf"] == 0.05
                and not args["augment"] and args["warmup_epochs"] == 1
                and args["warmup_momentum"] == 0.8 and args["warmup_bias_lr"] == 0.001
                and args["momentum"] == 0.9 and args["weight_decay"] == 0.0005,
                f"{arm}: configuración efectiva incompatible")
        history = common.read_history(training / "history.csv")
        audit_file = training / "staged_transfer.json"
        if ARMS[arm] == 0:
            require(not audit_file.exists() and "freeze_backbone_epochs" not in runtime, "El control ejecutó ajuste por etapas")
            updates = read_json(training / "parameter_updates.json")
            require(len(updates) == 1 and updates[0]["step"] == 1 and len(updates[0]["changed"]) == 2
                    and all(updates[0]["changed"].values()), "Faltan las actualizaciones iniciales del control")
        verify_stages(read_json(audit_file) if ARMS[arm] else None,
                      read_json(training / "parameters.json"), runtime, history, ARMS[arm])
        require(all((output / "examples" / f"image-{ident}.jpg").is_file() for ident in (57, 61, 80)),
                "Faltan los ejemplos emparejados")
        vals = common.values(metrics, history, status)
        results[arm] = {
            **vals, **{key: metrics[key] for key in ("ap_small", "ap_medium", "ap_large")},
            "train_seconds": status["total_elapsed_seconds"], "train_gpu_peak_bytes": status["cuda_peak_allocated_bytes"],
        }
        require(results[arm]["train_seconds"] > 0 and results[arm]["train_gpu_peak_bytes"] > 0,
                f"{arm}: faltan tiempo o memoria de entrenamiento")
        histories[arm] = history
        comparable_args = {key: Path(value) if key in {"data", "model"} else value
                           for key, value in args.items() if key not in {"name", "project", "save_dir"}}
        bench = metrics["benchmark"]
        require(bench["images"] == [row["file_name"] for row in truth["images"][:16]], "Orden de imágenes distinto")
        arm_environment = tuple((item["packages"], tuple(item["hardware"][key] for key in ENVIRONMENT_HARDWARE))
                                for item in (run, metrics))
        if environment is None:
            environment = arm_environment
        require(arm_environment == environment, f"Semilla {seed}, {arm}: entorno de entrenamiento o evaluación distinto")
        comparability.append((comparable_args, metrics["hardware"], bench["images"], bench["scope"],
                              runtime["optimizer_initial_groups"]))
    require(all(item == comparability[0] for item in comparability), "Argumentos o benchmark distintos")
    require(all([tuple(row[key] for key in common.LR) for row in history]
                == [tuple(row[key] for key in common.LR) for row in histories["control"]]
                for history in histories.values()), "Trayectorias del LR distintas dentro de la pareja")
    return results, histories, environment


def deltas(control, variant):
    return {k: variant[k] - value if variant[k] is not None and value is not None else None for k, value in control.items()}


def eligible(delta):
    return meets(delta["recall"], -0.02) and (
        (meets(delta["map"], 0.01) and meets(delta["map_without_ibuphil"], 0))
        or (meets(delta["map_without_ibuphil"], 0.02) and meets(delta["map"], 0))
    )


def meets(value, bound):
    return value >= bound or math.isclose(value, bound, rel_tol=0, abs_tol=1e-12)


def select(results):
    candidates = [arm for arm in ARMS if arm != "control" and eligible(deltas(results["control"], results[arm]))]
    return max(candidates, key=lambda arm: (
        *(results[arm][key] for key in ("map", "map_without_ibuphil", "recall")), -ARMS[arm]
    ), default=None)


def adoption(all_pairs, confirmation):
    total = all_pairs["delta"]
    return (total["map"]["mean"] > 0 and all_pairs["map_wins"] >= 2
            and confirmation["delta"]["map"]["mean"] > 0
            and meets(total["map_without_ibuphil"]["mean"], 0) and meets(total["recall"]["mean"], -0.02))


def transition_windows(histories):
    keys = ("step", "val/map", "val/cls_loss", "train/cls_loss")
    return {
        arm: {name: [{key: row[key] for key in keys} for row in histories[name][duration - 1:duration + 1]]
              for name in ("control", arm)}
        for arm, duration in ARMS.items() if duration and arm in histories
    }


def plot(histories, output):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10), layout="constrained")
    groups = (("val/map",), common.LOSSES[:3], common.LOSSES[3:], common.LR)
    labels = ("mAP COCO", "Pérdida train", "Pérdida valid", "LR al fin de época")
    styles = ("-", "--", ":")
    for panel, (ax, keys, label) in enumerate(zip(axes, groups, labels)):
        for color, (arm, rows) in enumerate(histories.items()):
            for index, key in enumerate(keys):
                ax.plot([r["step"] for r in rows], [r[key] for r in rows], styles[index],
                        color=f"C{color}", label=arm if panel == 0 else None)
            if ARMS[arm]:
                ax.axvline(ARMS[arm] + 1, lw=1, alpha=0.45, color=f"C{color}")
        if panel:
            ax.legend(handles=[Line2D([], [], color="0.3", ls=style, label=key)
                               for key, style in zip(keys, styles)], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    for ax in axes[1:3]:
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, 1)
    axes[0].legend(ncols=4)
    axes[-1].set_xlabel("Época")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Evidencias de la fase A")
    parser.add_argument("--confirmation", type=Path, help="Evidencias de la fase B")
    parser.add_argument("--data", type=Path, default=DESTINATION, help="Datos COCO originales")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        identity = common.campaign_identity(args.source)
        baseline = read_json(REPO / "experiments/baseline/results/yolo/training/run.json")["run"]
        require(identity[1]["yolov10s.pt"] == baseline["initial_weights_sha256"]
                and identity[1]["manifest.json"] == baseline["dataset_manifest_sha256"],
                "Los pesos COCO o el dataset preparado difieren de la baseline")
        truth = common.validation_truth(args.data, args.source, identity[1])
        exploration, history, exploration_environment = read_cohort(args.source, "a", 2026, tuple(ARMS), identity, truth)
        selected = select(exploration)
        result = {
            "source_commit": identity[0], "coco_verified": True,
            "per_class": [{"category_id": row["id"], "name": row["name"],
                           "support": sum(ann["category_id"] == row["id"] for ann in truth["annotations"])}
                          for row in truth["categories"]],
            "exploration": {"seed": 2026, "arms": exploration,
                            "deltas": {arm: deltas(exploration["control"], exploration[arm]) for arm in ARMS if ARMS[arm]},
                            "selected": selected},
            "confirmation": None,
        }
        plots = {2026: history}
        if args.confirmation:
            require(selected is not None and common.campaign_identity(args.confirmation) == identity,
                    "La fase B requiere una variante elegible y la misma revisión y entradas")
            pairs = {}
            for seed in (2027, 2028):
                values, plots[seed], environment = read_cohort(
                    args.confirmation, "b", seed, ("control", selected), identity, truth,
                )
                require(environment == exploration_environment,
                        f"Semilla {seed}: entorno de confirmación distinto al de exploración")
                pairs[seed] = {**values, "delta": deltas(values["control"], values[selected])}
            confirm = common.aggregate(pairs, selected)
            all_pairs = common.aggregate({
                2026: {"control": exploration["control"], selected: exploration[selected],
                       "delta": result["exploration"]["deltas"][selected]}, **pairs
            }, selected)
            result["confirmation"] = {
                "pairs": pairs, "confirmation_only": confirm, "all_pairs": all_pairs,
                "adopt": adoption(all_pairs, confirm),
            }
        result["transition_windows"] = {seed: transition_windows(histories) for seed, histories in plots.items()}
        args.output.mkdir(parents=True)
        write_json(args.output / "summary.json", result)
        for seed, histories in plots.items():
            plot(histories, args.output / f"history-s{seed}.png")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
