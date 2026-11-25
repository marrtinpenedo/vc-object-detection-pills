"""Native three-minibatch CUDA probe using COCO weights and the real Cascade hooks."""

import argparse
from pathlib import Path

from scripts.baseline_data import prepared_manifest, sha256, verify_files, write_json
from scripts.baseline_run import dry_runtime, load_config, provenance


def smoke(data, weights, output):
    torch = dry_runtime()
    from detectron2.engine import hooks
    from detectron2.utils.env import seed_all_rng
    from detectron2.utils.logger import setup_logger

    from scripts.baseline_cascade import (
        CascadeFreezeHook, CascadeTrainer, CheckedWeights, configure, register_training_data,
    )
    from scripts.cascade_staged import CascadeFreeze, require

    class SmokeTrainer(CascadeTrainer):
        def build_hooks(self):
            return [hooks.LRScheduler(), CascadeFreezeHook()]

    manifest = prepared_manifest(data)
    verify_files(data, manifest)
    spec = load_config("cascade", None, {"epochs": 3, "workers": 0})
    seed_all_rng(spec["seed"])
    torch.use_deterministic_algorithms(False)
    setup_logger(name="detectron2")
    output.mkdir()
    write_json(output / "run.json", {**provenance(torch), "weights_sha256": sha256(weights)})
    cfg, _, _ = configure(spec, len(manifest["classes"]), manifest["splits"]["train"]["images"], weights, output)
    cfg.SOLVER.MAX_ITER = 3
    cfg.SOLVER.WARMUP_ITERS = 1
    register_training_data(cfg, data, manifest)
    trainer = SmokeTrainer(cfg, 0)
    coco = CheckedWeights(trainer.model, pretrained=True)
    coco.load(cfg.MODEL.WEIGHTS)
    write_json(output / "weight_transfer.json", coco.transfer)
    require(trainer.model.roi_heads.num_classes == 8 and not cfg.MODEL.MASK_ON, "cabeza del smoke incorrecta")
    trainer.freeze_audit = CascadeFreeze(trainer, output, epochs=1, steps_per_epoch=1)
    trainer.train()
    report = trainer.freeze_audit.report
    require(report["frozen_steps"] == 1 and report["transition"]["step"] == 2
            and [epoch["frozen"] for epoch in report["epochs"]] == [True, False, False],
            "el smoke no recorrió las dos fases")
    write_json(output / "smoke.json", {"steps": 3, "transition_step": 2, "dataset_split": "train",
                                      "status": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.data, args.weights, args.output)
