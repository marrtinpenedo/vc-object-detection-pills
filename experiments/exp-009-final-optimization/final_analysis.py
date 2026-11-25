"""Audit the two frozen final predictions and reproduce COCO locally."""
import argparse
import hashlib
import json
import tarfile
from importlib import import_module
from pathlib import Path
from statistics import mean

from scripts.baseline_data import ACTIVE_IDS, ANNOTATIONS, convert_split, read_json, sha256, write_json
from scripts.campaign_analysis.analyze import COCO_METRICS, archive
from scripts.campaign_analysis.import_results import require, verify_inventory
from scripts.campaign_analysis.reporting import RARE_CLASS, SENSITIVITY, save, write_csv
from scripts.exp009_followup.final_export import FILES
from scripts.exp009_followup.final_import import RECEIPT
from scripts.verify_baseline import verify_evaluation

confirmation = import_module("experiments.exp-009-final-optimization.confirmation")
presentation = import_module("experiments.exp-009-final-optimization.final_reports")
MODELS = ("yolo", "cascade")


def test_truth(dataset, manifest_file, selection):
    expected = selection["test_coco_expected_sha256_from_manifest"]
    require(sha256(manifest_file) == selection["dataset_manifest_sha256"], "Manifiesto preparado inesperado")
    manifest = read_json(manifest_file)
    archive.verify_split_isolation(manifest)
    categories = manifest["classes"]
    split = manifest["splits"]["test"]
    source = dataset / "test" / ANNOTATIONS
    require(sha256(source) == split["source_annotations_sha256"]
            and [row["original_id"] for row in categories] == list(ACTIVE_IDS),
            "Anotaciones de origen o clases distintas")
    truth, _ = convert_split(read_json(source), categories)
    encoded = (json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(encoded).hexdigest() == manifest["files"]["coco/test.json"] == expected
            and len(truth["images"]) == split["images"] == 45
            and len(truth["annotations"]) == split["instances"]
            and len(truth["categories"]) == 8, "COCO de test no coincide con la partición congelada")
    return truth


def verify_launch(root, selection_path, deployment, snapshot, *, accounting_path=None, allow_missing_accounting=False):
    verify_inventory(root)
    inventory = read_json(root / "export.sha256.json")
    directory_archive = RECEIPT in inventory
    receipt_name = RECEIPT if directory_archive else "final-eval-sacct.txt"
    require(set(inventory) == set(FILES) | {receipt_name}, "Export final incompleto o inesperado")
    source = read_json(root / RECEIPT) if directory_archive else {"kind": "selective_export"}
    if directory_archive:
        require(source["kind"] == "local_projection" and source["local_inventory_only"] is True,
                "Falta acreditar que el inventario se hizo localmente")
        require(accounting_path is not None or allow_missing_accounting,
                "Adjunta el recibo de sacct o solicita expresamente una reproducción sin contabilidad")
    else:
        accounting_path = root / "final-eval-sacct.txt"
    selection = read_json(selection_path)
    require(selection["status"] == "pendiente_sha256_fisico" and selection["c_slurm_accounting_verified"],
            "El registro local de validación no está certificado")
    with tarfile.open(deployment) as package:
        def packaged(name):
            return package.extractfile(name).read()

        require(packaged("seleccion-valid.json") == selection_path.read_bytes(),
                "La selección local difiere de la transferida")
        launch_code = {name: hashlib.sha256(packaged(name)).hexdigest() for name in ("evaluate.sh", "evaluate.py")}
    launch = read_json(root / "launch.json")
    job = (root / "eval.job").read_text().strip()
    require(job.isdecimal() and launch["job_id"] == job, "ID de la asignación final distinto")
    accounting = (confirmation.verify_slurm_accounting(accounting_path, {"final": {"id": job}})
                  if accounting_path else None)
    require(launch["selection_sha256"] == sha256(selection_path) and launch["launcher_sha256"] == launch_code
            and launch["dataset_manifest_sha256"] == selection["dataset_manifest_sha256"]
            and launch["coco_sha256"] == selection["test_coco_expected_sha256_from_manifest"]
            and (launch["split"], launch["checkpoint_kind"], launch["benchmark"], launch["warmup"], launch["iterations"])
            == ("test", "best", True, 20, 100) and set(launch["runs"]) == set(MODELS),
            "Protocolo de la asignación final incompatible")
    require(sha256(snapshot) == selection["yolo"]["code_snapshot_sha256"], "Snapshot incorrecto")
    hashes = read_json(snapshot)["sha256"]
    require(hashes["scripts/evaluate_baseline.py"] == selection["evaluation_entrypoint_sha256"],
            "Código del evaluador incompatible")
    return selection, launch, accounting, source, {key: value for key, value in hashes.items() if key.startswith("scripts/")}


def verify_detector(root, name, selection, launch, code, truth):
    pinned = selection[name]
    info = launch["runs"][name]
    require((info["backend"], info["checkpoint"], info["checkpoint_sha256"], info["training_job_id"],
             info["training_slurm"], info["imgsz"]) ==
            (name, pinned["checkpoint_path"], pinned["checkpoint_sha256"], pinned["training_job"], "COMPLETED|0:0", 640)
            and (root / f"{name}.train.sacct").read_text().strip() == "COMPLETED|0:0",
            f"Recibo de entrenamiento incompatible: {name}")
    metrics = verify_evaluation(root / name, truth, launch["coco_sha256"], launch["dataset_manifest_sha256"], "test")
    require((metrics["backend"], metrics["checkpoint"], metrics["checkpoint_sha256"], metrics["checkpoint_kind"],
             metrics["model_config"], metrics["hardware"]["job_id"], metrics["code_sha256"]) ==
            (name, pinned["checkpoint_path"], pinned["checkpoint_sha256"], "best", pinned["recipe"], launch["job_id"], code)
            and metrics["images"] == 45 and metrics["parameters"]["minimum_export_score"] == .001
            and metrics["parameters"]["max_detections"] == [1, 10, 100]
            and {key: metrics["operating_point"][key] for key in ("score", "iou")} == selection["operating_point"],
            f"Predicciones, checkpoint o protocolo incompatible: {name}")
    times, benchmark = archive.benchmark_metrics(metrics, [row["file_name"] for row in truth["images"]])
    require(benchmark["protocol"], f"Benchmark distinto del fijado: {name}")
    curves = read_json(root / name / "pr_curves.json")
    classes = []
    for item in metrics["per_class"]:
        rates = curves["classes"][str(item["category_id"])]
        classes.append({**item, **{metric: mean(rates[iou]) if item["ap"] is not None else None
                                  for metric, iou in (("ap50", "0.50"), ("ap75", "0.75"))}})
    common = [c["ap"] for c in classes if c["name"] != RARE_CLASS and c["ap"] is not None]
    return {**{key: metrics[key] for key in COCO_METRICS}, **metrics["operating_point"],
            SENSITIVITY: mean(common) if common else None, "supported_classes_without_ibuphil": len(common),
            **times}, classes, benchmark


def plot(path, classes):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    for offset, name, color in ((-.17, "yolo", "#174c80"), (.17, "cascade", "#b05c12")):
        bars = [100 * row["ap"] if row["ap"] is not None else float("nan") for row in classes[name]]
        ax.bar([idx + offset for idx in range(len(bars))], bars, width=.34, label=name.upper(), color=color)
    labels = [r["name"] if r["ground_truth"] else f"{r['name']} (sin GT)" for r in classes["yolo"]]
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 100), ylabel="AP COCO (%)",
           title="Evaluación final · AP por clase")
    ax.tick_params(axis="x", rotation=20)
    fig.legend(loc="outside upper center", ncols=2)
    save(fig, path)


def analyze(root, selection_path, deployment, snapshot, manifest_file, dataset, output, *,
            accounting_path=None, allow_missing_accounting=False):
    selection, launch, accounting, source, code = verify_launch(
        root, selection_path, deployment, snapshot, accounting_path=accounting_path,
        allow_missing_accounting=allow_missing_accounting)
    truth = test_truth(dataset, manifest_file, selection)
    result = {name: verify_detector(root, name, selection, launch, code, truth) for name in MODELS}
    left, right = result["yolo"], result["cascade"]
    require(archive.benchmark_comparable(left[2], right[2]), "Latencias sin asignación o alcance comunes")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "resumen.csv", [{"backend": name, **row[0]} for name, row in result.items()])
    write_csv(output / "por_clase.csv", [{"backend": name, **item} for name, row in result.items() for item in row[1]])
    metrics = (*COCO_METRICS, SENSITIVITY, "precision", "recall", "mean_ms_per_image", "p95_ms_per_image")
    write_csv(output / "comparacion.csv", [
        {"metric": key, "yolo": left[0][key], "cascade": right[0][key],
         "yolo_minus_cascade": left[0][key] - right[0][key] if None not in (left[0][key], right[0][key]) else None}
        for key in metrics])
    write_json(output / "auditoria.json", {
        "split": "test", "image_ids": [row["id"] for row in truth["images"]], "job_id": launch["job_id"],
        "slurm": accounting, "import_source": source, "archive_inventory_sha256": sha256(root / "export.sha256.json"),
        "selection_sha256": sha256(selection_path), "deployment_sha256": sha256(deployment),
        "snapshot_sha256": sha256(snapshot), "dataset_manifest_sha256": launch["dataset_manifest_sha256"],
        "coco_sha256": launch["coco_sha256"], "launch": launch, "benchmark": left[2],
    })
    plot(output / "ap_por_clase.png", {name: result[name][1] for name in MODELS})
    presentation.write(output, root, truth, selection, result, dataset, read_json(manifest_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection_path", "deployment", "snapshot", "manifest_file", "dataset", "output"):
        parser.add_argument(name, type=Path)
    accounting = parser.add_mutually_exclusive_group()
    accounting.add_argument("--slurm-accounting", type=Path)
    accounting.add_argument("--allow-missing-accounting", action="store_true")
    args = parser.parse_args()
    analyze(args.root, args.selection_path, args.deployment, args.snapshot, args.manifest_file, args.dataset, args.output,
            accounting_path=args.slurm_accounting, allow_missing_accounting=args.allow_missing_accounting)
