"""Audit ResNet freezing after Detectron2 has built the single SGD optimizer."""

from collections import Counter
from hashlib import sha256

from scripts.baseline_data import write_json
from scripts.baseline_run import check_gradients

BOTTOM_UP = "backbone.bottom_up."
REST = ("backbone.fpn_lateral", "backbone.fpn_output", "proposal_generator.", "roi_heads.box_predictor.")


def thaw_iteration(epochs, steps_per_epoch, max_iter):
    boundary = epochs * steps_per_epoch
    return boundary if boundary < max_iter else None


def digest(tensors):
    stream = sha256()
    for name, tensor in sorted(tensors):
        stream.update(name.encode() + b"\0")
        stream.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return stream.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(f"Freezing Cascade: {message}")


class CascadeFreeze:
    def __init__(self, trainer, output, epochs, steps_per_epoch):
        from torch.optim import SGD

        self.model = trainer.model
        self.bottom_up = self.model.backbone.bottom_up
        self.optimizer = trainer.optimizer
        self.scheduler = trainer.scheduler
        require(isinstance(self.optimizer, SGD), "se requiere SGD")
        require(type(self.model.backbone).__name__ == "FPN" and type(self.bottom_up).__name__ == "ResNet"
                and tuple(len(getattr(self.bottom_up, name)) for name in ("res2", "res3", "res4", "res5"))
                == (3, 4, 6, 3), "se requiere ResNet-50 + FPN con sus cuatro etapas")
        norms = [type(module).__name__ for module in self.bottom_up.modules()
                 if "BatchNorm" in type(module).__name__]
        require(trainer.cfg.MODEL.RESNETS.NORM == "FrozenBN" and norms and set(norms) == {"FrozenBatchNorm2d"}, "normalización distinta del FrozenBN de la baseline")
        self.parameters = dict(self.bottom_up.named_parameters())
        model_parameters = dict(self.model.named_parameters())
        self.others = tuple(p for name, p in model_parameters.items() if not name.startswith(BOTTOM_UP))
        group_members = Counter(id(p) for group in self.optimizer.param_groups for p in group["params"])
        require(self.parameters and all(p.requires_grad and group_members[id(p)] == 1 for p in self.parameters.values()),
                "ResNet debe entrar en SGD antes de congelarse")
        require(all(any(name.startswith(prefix) and p.requires_grad and group_members[id(p)] == 1
                        for name, p in model_parameters.items()) for prefix in REST), "faltan módulos activos en SGD")

        self.path = output / "cascade_freezing.json"
        self.steps = steps_per_epoch
        self.boundary = thaw_iteration(epochs, steps_per_epoch, trainer.cfg.SOLVER.MAX_ITER)
        self.frozen = True
        self.bottom_up.requires_grad_(False)
        self.bottom_up.eval()
        self.initial = self.weights()
        self.buffers = digest(self.bottom_up.named_buffers())
        self.report = {
            "freeze_epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "first_unfrozen_step": None if self.boundary is None else self.boundary + 1,
            "backbone_parameters": list(self.parameters),
            "backbone_scalars": sum(p.numel() for p in self.parameters.values()),
            "normalization": "FrozenBatchNorm2d",
            "initial_weights_sha256": self.initial, "initial_buffers_sha256": self.buffers,
            "frozen_steps": 0, "first_step": None, "transition": None, "epochs": [],
        }
        write_json(self.path, self.report)

    def weights(self):
        return digest(self.parameters.items())

    def check_frozen(self):
        require(all(not p.requires_grad and p.grad is None and p not in self.optimizer.state
                    for p in self.parameters.values()), "ResNet congelada tiene gradiente o estado SGD")

    def before_step(self, trainer):
        step = trainer.iter
        require(trainer.optimizer is self.optimizer and trainer.scheduler is self.scheduler,
                "se sustituyó el optimizador o el scheduler")
        self.scheduler_before = self.scheduler.last_epoch
        self.lr_before = self.optimizer.param_groups[0]["lr"]
        if self.frozen and self.boundary is not None and step >= self.boundary:
            require(step == self.boundary, "se omitió la primera iteración descongelada")
            self.check_frozen()
            require(self.weights() == self.initial, "ResNet cambió antes de descongelarse")
            self.momentum_before = self.momentum_count(self.others)
            require(self.momentum_before > 0, "las ramas activas perdieron su estado SGD antes de descongelar")
            self.bottom_up.requires_grad_(True)
            self.bottom_up.train(self.model.training)
            self.frozen = False
        elif self.frozen and self.bottom_up.training:
            # Detectron2 restores train() recursively after periodic inference.
            self.bottom_up.eval()

    def momentum_count(self, parameters):
        return sum("momentum_buffer" in self.optimizer.state.get(p, {}) for p in parameters)

    def after_step(self, trainer):
        import torch

        step = trainer.iter + 1
        require(self.scheduler.last_epoch == self.scheduler_before + 1, "el scheduler perdió continuidad")
        if self.frozen:
            self.check_frozen()
            self.report["frozen_steps"] += 1

        milestone = step == 1 or (self.boundary is not None and step == self.boundary + 1)
        if milestone:
            gradients = check_gradients(torch, self.model)
            expected = REST if self.frozen else (*REST, BOTTOM_UP)
            require(all(any(name.startswith(prefix) and value == "nonzero" for name, value in gradients.items())
                        for prefix in expected), "faltan gradientes útiles en módulos activos")
            state_count = self.momentum_count(self.parameters.values())
            others_after = self.momentum_count(self.others)
            current = self.weights()
            require((current == self.initial and not state_count) if self.frozen
                    else (current != self.initial and state_count > 0 and others_after >= self.momentum_before),
                    "huella ResNet o momento SGD incoherentes con la fase")
            self.report["first_step" if step == 1 else "transition"] = {
                "step": step, "epoch": trainer.iter // self.steps + 1,
                "scheduler_before": self.scheduler_before, "scheduler_after": self.scheduler.last_epoch,
                "lr": self.lr_before, "backbone_momentum_buffers": state_count, "weights_sha256": current,
                "other_momentum_before": 0 if self.frozen else self.momentum_before,
                "other_momentum_after": others_after,
            }

        if step % self.steps == 0:
            weights, buffers = self.weights(), digest(self.bottom_up.named_buffers())
            require(buffers == self.buffers, "cambiaron los buffers FrozenBN")
            require(not self.frozen or weights == self.initial, "ResNet congelada cambió de pesos")
            self.report["epochs"].append({"epoch": step // self.steps, "step": step, "frozen": self.frozen,
                                          "weights_sha256": weights, "buffers_sha256": buffers})
        if milestone or step % self.steps == 0:
            write_json(self.path, self.report)
