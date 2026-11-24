"""Resume comparaciones emparejadas de EXP-006 a partir de sus artefactos."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from scripts.baseline_data import ANNOTATIONS, convert_split, prepared_manifest, read_json, sha256, write_json
from scripts.baseline_run import REPO, load_config


LOSSES = tuple(f"{split}/{kind}_loss" for split in ("train", "val") for kind in ("box", "cls", "dfl"))
LR = tuple(f"lr/pg{index}" for index in range(3))
SEEDS = {2026, 2027, 2028}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_history(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and {"step", "val/map", *LOSSES} <= set(reader.fieldnames),
                f"{path}: faltan columnas de época, mAP o pérdidas")
        keys = ("val/map", *LOSSES, *(key for key in LR if key in reader.fieldnames))
        rows = [{"step": int(row["step"]), **{key: float(row[key]) for key in keys}} for row in reader]
    require([row["step"] for row in rows] == list(range(1, 31)), f"{path}: se esperan las 30 épocas ordenadas")
    require(all(math.isfinite(row[key]) and row[key] >= 0 for row in rows for key in keys)
            and all(row["val/map"] <= 1 for row in rows), f"{path}: valores inválidos")
    return rows


def history_summary(rows, best_epoch):
    early = rows[:5]
    peak = max(early, key=lambda row: row["val/cls_loss"])
    best_map = max(row["val/map"] for row in rows)
    require(type(best_epoch) is int and 1 <= best_epoch <= len(rows)
            and rows[best_epoch - 1]["val/map"] == best_map, "La época de best discrepa del historial")
    return {
        "peak_val_cls_loss_1_5": peak["val/cls_loss"],
        "peak_val_cls_loss_epoch": peak["step"],
        "zero_map_epochs_1_5": sum(row["val/map"] == 0 for row in early),
        "zero_map_epochs_1_30": sum(row["val/map"] == 0 for row in rows),
        "mean_map_1_5": mean(row["val/map"] for row in early),
        "best_epoch": best_epoch,
        "best_map_history": best_map,
        "last_map_history": rows[-1]["val/map"],
        "best_minus_last_map": best_map - rows[-1]["val/map"],
    }


def campaign_identity(source):
    commit = (source / "source-commit").read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Commit de la tanda inválido")
    inputs = {Path(name).name: digest for digest, name in
              (line.split(maxsplit=1) for line in (source / "inputs.sha256").read_text().splitlines())}
    require(set(inputs) == {"manifest.json", "valid.json", "yolov10s.pt"}
            and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in inputs.values())
            and sha256(source / "data/manifest.json") == inputs["manifest.json"],
            "Huellas de entrada o manifiesto inconsistentes")
    return commit, inputs


def validation_truth(dataset, source, inputs):
    manifest = prepared_manifest(source / "data")
    annotations = dataset / "valid" / ANNOTATIONS
    require(sha256(annotations) == manifest["splits"]["valid"]["source_annotations_sha256"],
            "Las anotaciones originales de validación difieren del manifiesto")
    truth, _ = convert_split(read_json(annotations), manifest["classes"])
    with TemporaryDirectory() as temporary:
        reconstructed = Path(temporary) / "valid.json"
        write_json(reconstructed, truth)
        require(sha256(reconstructed) == inputs["valid.json"] == manifest["files"]["coco/valid.json"],
                "El COCO de validación reconstruido difiere del evaluado")
    return truth


def read_arm(source, seed, arm, round_name, identity, truth=None):
    commit, inputs = identity
    training = source / f"s{seed}" / arm / "train"
    evaluation = source / f"eval-{round_name}-s{seed}" / arm
    run, runtime, status = (read_json(training / file) for file in
                            ("run.json", "ultralytics_runtime.json", "status.json"))
    if truth is None:
        metrics = read_json(evaluation / "metrics.json")
    else:
        from scripts.verify_baseline import verify_evaluation

        metrics = verify_evaluation(evaluation, truth, inputs["valid.json"], inputs["manifest.json"], "valid")
    config = None if arm == "control" else REPO / f"configs/experiments/yolo-{arm}.json"
    expected = load_config("yolo", config, {"seed": seed})
    require(run["run"]["config"] == metrics["model_config"]
            == read_json(training.parent / "config.json") == expected, f"{arm}: configuración distinta")
    args = runtime["args"]
    require(args["warmup_epochs"] == expected["warmup_epochs"] and args["seed"] == seed
            and args["epochs"] == 30 and args["batch"] == 8 and args["lr0"] == 0.001
            and args["cos_lr"] and args["lrf"] == 0.05 and not args["amp"]
            and args["warmup_bias_lr"] == expected["warmup_bias_lr"]
            and args["warmup_momentum"] == expected["warmup_momentum"]
            and runtime["iterations_per_epoch"] == 40 and runtime["optimizer"] == "AdamW"
            and runtime["warmup_iterations_end_index"] == max(round(expected["warmup_epochs"] * 40), 100),
            f"{arm}: configuración efectiva o warm-up distinto")
    job = (training.parent / "train.job").read_text().strip()
    accounting = (evaluation.parent / f"{arm}.train.sacct").read_text().strip()
    require(status["status"] == "complete" and run["hardware"]["job_id"] == job
            and accounting == "COMPLETED|0:0", f"{arm}: ejecución incompleta o ID de entrenamiento distinto")
    eval_job = (evaluation.parent / "eval.job").read_text().strip()
    original = (training.parent / "train.path").read_text().strip()
    require((metrics["backend"], metrics["split"], metrics["checkpoint_kind"]) == ("yolo", "valid", "best")
            and (evaluation.parent / "verified.ok").is_file()
            and all(item["git_head"] in (None, commit) for item in (run, metrics))
            and metrics["hardware"]["job_id"] == eval_job
            and Path(metrics["checkpoint"]) == Path(original) / status["best_checkpoint"]
            and metrics["dataset_manifest_sha256"] == run["run"]["dataset_manifest_sha256"] == inputs["manifest.json"]
            and metrics["coco_sha256"] == inputs["valid.json"]
            and run["run"]["initial_weights_sha256"] == inputs["yolov10s.pt"]
            and metrics["code_sha256"] == run["code_sha256"], f"{arm}: procedencia de evaluación incompatible")
    return run, metrics, status, read_history(training / "history.csv")


def values(metrics, rows, status):
    point = metrics["operating_point"]
    bench = metrics["benchmark"]
    require((point["score"], point["iou"]) == (0.25, 0.50), "Punto operativo incompatible")
    require((bench["warmup_iterations"], bench["iterations"], bench["batch"], bench["dtype"], bench["nominal_shape"])
            == (20, 100, 1, "fp32", [640, 640]), "Benchmark incompatible")
    require(len(bench["samples_ms"]) == 100 and all(math.isfinite(t) and t > 0 for t in bench["samples_ms"]),
            "Muestras de latencia inválidas")
    average = mean(bench["samples_ms"])
    require(math.isclose(bench["mean_ms_per_image"], average)
            and math.isclose(bench["fps_from_mean"], 1000 / average), "Resumen de latencia inválido")
    classes = metrics["per_class"]
    require(len(classes) == 8 and {row["category_id"] for row in classes} == set(range(1, 9))
            and all(row["ground_truth"] > 0 and 0 <= row["ap"] <= 1 for row in classes)
            and all(0 <= metrics[key] <= 1 for key in ("map", "ap50", "ap75")),
            "Categorías, soporte o AP inválidos")
    rare = next(row for row in classes if row["category_id"] == 2)
    require(rare["name"] == "Ibuphil 600 mg" and rare["ground_truth"] == 1, "Soporte de Ibuphil distinto")
    summary = history_summary(rows, status["best_step"])
    require(math.isclose(summary["best_map_history"], status["best_validation_map"], abs_tol=5e-7),
            "El mAP de best discrepa del historial")
    return {
        **{key: metrics[key] for key in ("map", "ap50", "ap75")},
        **{f"ap_class_{row['category_id']}": row["ap"] for row in classes},
        "map_without_ibuphil": mean(row["ap"] for row in classes if row["category_id"] != 2),
        **{key: point[key] for key in ("precision", "recall", "tp", "fp", "fn")},
        **{key: bench[key] for key in ("mean_ms_per_image", "median_ms_per_image", "p95_ms_per_image", "fps_from_mean")},
        **summary,
    }


def compare(source, seed, variant, round_name, identity, truth=None):
    result, histories, evidence = {}, {}, []
    for arm in ("control", variant):
        run, metrics, status, rows = read_arm(source, seed, arm, round_name, identity, truth)
        classes = {row["category_id"]: (row["name"], row["ground_truth"]) for row in metrics["per_class"]}
        result[arm] = values(metrics, rows, status)
        histories[arm] = rows
        evidence.append((
            *(metrics[key] for key in ("dataset_manifest_sha256", "coco_sha256", "image_ids", "parameters", "code_sha256")),
            run["run"]["initial_weights_sha256"], run["packages"], metrics["packages"], classes,
            *(run["hardware"][key] for key in ("gpu", "cuda_runtime", "nvidia_driver")),
            *(metrics["hardware"][key] for key in ("gpu", "host", "job_id")),
            *(metrics["benchmark"][key] for key in ("images", "scope")),
        ))
    require(evidence[0] == evidence[1], f"Semilla {seed}: brazos con procedencia o protocolo distintos")
    result["delta"] = {key: value - result["control"][key]
                       if value is not None and result["control"][key] is not None else None
                       for key, value in result[variant].items()}
    return {"round": round_name, "classes": classes, **result}, histories


def aggregate(pairs, variant):
    result = {}
    for arm in ("control", variant, "delta"):
        columns = next(iter(pairs.values()))[arm]
        result[arm] = {}
        for key in columns:
            data = [pair[arm][key] for pair in pairs.values() if pair[arm][key] is not None]
            result[arm][key] = {
                "n": len(data), "mean": mean(data) if data else None,
                "sample_sd": stdev(data) if len(data) > 1 else None,
            }
    result["map_wins"] = sum(pair["delta"]["map"] > 0 for pair in pairs.values())
    return result


def plot(histories, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10), layout="constrained")
    groups = (("val/map",), LOSSES[:3], LOSSES[3:], LR)
    for ax, keys, label in zip(axes, groups, ("mAP COCO", "Pérdidas de train", "Pérdidas de valid", "LR al fin de época")):
        ax.axvspan(1, 5, color="0.94")
        for arm, rows in histories.items():
            for index, key in enumerate(keys):
                if key in rows[0]:
                    ax.plot([row["step"] for row in rows], [row[key] for row in rows],
                            "-" if arm == "control" else "--", color=f"C{index}", label=f"{arm} · {key}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncols=2, fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[-1].set_xlabel("Época")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="dataset COCO original para verificar las métricas de validación")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("warmup-3", "warmup-5"), required=True)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SEMILLA", "RONDA"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    try:
        identity = campaign_identity(args.source)
        truth = validation_truth(args.data, args.source, identity[1]) if args.data is not None else None
        pairs, histories = {}, {}
        for seed_arg, round_name in args.pair:
            seed = int(seed_arg)
            valid_rounds = ({"a", "b"} if args.variant == "warmup-5" else {"b"}) if seed == 2026 else {"c"}
            require(seed in SEEDS and seed not in pairs and round_name in valid_rounds,
                    "Semilla duplicada o ronda incompatible")
            pairs[seed], histories[seed] = compare(args.source, seed, args.variant, round_name, identity, truth)
        confirmed = {seed: value for seed, value in pairs.items() if seed != 2026}
        report = {
            "source_commit": identity[0], "variant": args.variant,
            "coco_verified": truth is not None,
            "pairs": pairs, "all_pairs": aggregate(pairs, args.variant),
            "confirmation_only": aggregate(confirmed, args.variant) if confirmed else None,
        }
        args.output.mkdir(parents=True)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.plots:
            for seed, rows in histories.items():
                plot(rows, args.output / f"history-s{seed}.png")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
