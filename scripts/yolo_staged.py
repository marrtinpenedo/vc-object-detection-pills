"""Freeze and audit the YOLOv10-S training backbone inside a single optimizer run."""

from hashlib import sha256

import torch
from torch import nn

from scripts.baseline_data import write_json


BACKBONE = ("Conv", "Conv", "C2f", "Conv", "C2f", "SCDown", "C2f", "SCDown", "C2fCIB", "SPPF", "PSA")
NECK = ("Upsample", "Concat", "C2f", "Upsample", "Concat", "C2f", "Conv", "Concat", "C2f", "SCDown", "Concat", "C2fCIB")
TYPES = BACKBONE + NECK + ("v10Detect",)
DETECT = len(TYPES) - 1
ROUTES = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10], DETECT: [16, 19, 22]}
PREFIXES = {
    "backbone": "model.0.", "neck": "model.13.",
    "one2many": f"model.{DETECT}.cv3.", "one2one": f"model.{DETECT}.one2one_cv3.",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(f"Ajuste por etapas YOLOv10-S: {message}")


def layout(model):
    """Check the S architecture embedded in the weights and the parsed graph."""
    require(model.yaml.get("scale") == "s", "la escala del modelo no es s")
    backbone, head = model.yaml["backbone"], model.yaml["head"]
    expected = tuple(row[2].removeprefix("nn.") for row in backbone + head)
    require(len(backbone) == len(BACKBONE) and expected == TYPES
            and tuple(type(m).__name__ for m in model.model) == TYPES,
            "los módulos del backbone, neck o detector no coinciden con YOLOv10-S")
    for index, module in enumerate(model.model):
        require((module.i, module.f) == (index, ROUTES.get(index, -1)), f"conexiones inesperadas en model.{index}")
    return [
        {"index": i, "type": kind, "part": "backbone" if i < len(BACKBONE) else "neck" if i < DETECT else "detector"}
        for i, kind in enumerate(TYPES)
    ]


def is_backbone(name):
    parts = name.split(".", 2)
    return len(parts) == 3 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < len(BACKBONE)


def byte_digest(tensors):
    # One device-to-host transfer per fingerprint, at setup/epoch boundaries only.
    bytes_on_device = torch.cat([t.detach().contiguous().reshape(-1).view(torch.uint8) for _, t in tensors])
    return sha256(bytes_on_device.cpu().numpy().tobytes()).hexdigest()


class StagedBackbone:
    def __init__(self, trainer, output, duration):
        self.path = output / "staged_transfer.json"
        self.duration = duration
        model = trainer.model
        architecture = layout(model)
        names = dict(model.named_parameters())
        self.backbone = {name: p for name, p in names.items() if is_backbone(name)}
        self.internal = {name: p for name, p in names.items() if not p.requires_grad}
        require(set(self.internal) == {f"model.{DETECT}.dfl.conv.weight"}, "congelación interna inesperada")
        self.parameters = names
        self.original = {name: p.requires_grad for name, p in self.backbone.items()}
        require(all(self.original.values()), "el backbone ya contiene parámetros congelados")
        self.buffers = {name: value for name, value in model.named_buffers() if is_backbone(name)}
        self.tensors = sorted((self.backbone | self.buffers).items())
        self.internal_tensors = sorted(self.internal.items())
        self.bn = {name: value for name, value in model.named_modules()
                   if is_backbone(name) and isinstance(value, nn.BatchNorm2d)}
        require(self.bn and all(m.track_running_stats for m in self.bn.values()), "faltan Batch Normalization con estadísticas")
        self.optimizer, self.scheduler, self.ema = trainer.optimizer, trainer.scheduler, trainer.ema
        require(isinstance(self.optimizer, torch.optim.AdamW) and trainer.accumulate == 1 and self.ema.updates == 0,
                "se esperaban AdamW, acumulación unitaria y EMA nueva")
        self.groups = self.optimizer_groups(model)
        self.samples = {region: next(name for name in names if name.startswith(prefix) and name.endswith(".weight"))
                        for region, prefix in PREFIXES.items()}
        self.last_steps = dict.fromkeys(self.samples, 0)
        self.active_states = {}
        self.frozen = True
        self.previous_digest = byte_digest(self.tensors)
        self.initial_internal = byte_digest(self.internal_tensors)
        for p in self.backbone.values():
            p.requires_grad_(False)
            p.grad = None
        for bn in self.bn.values():
            bn.eval()
        total = sum(p.numel() for p in names.values())
        frozen = sum(p.numel() for p in (*self.backbone.values(), *self.internal.values()))
        self.audit = {
            "freeze_backbone_epochs": duration, "modules": architecture,
            "backbone_parameters": list(self.backbone), "backbone_buffers": list(self.buffers),
            "backbone_batch_norm": list(self.bn), "internally_frozen": list(self.internal),
            "parameters_total": total, "frozen_initial": frozen, "frozen_initial_pct": 100 * frozen / total,
            "frozen_after_thaw": sum(p.numel() for p in self.internal.values()),
            "optimizer": type(self.optimizer).__name__, "optimizer_groups": self.groups,
            "samples": self.samples, "initial_backbone_sha256": self.previous_digest,
            "internal_sha256": self.initial_internal, "thaw_epoch": None,
            "optimizer_groups_verified_at_thaw": False, "epochs": [],
        }
        self.check_phase()
        self.persist()

    def optimizer_groups(self, model):
        names = {id(p): name for name, p in self.parameters.items()}
        modules = dict(model.named_modules())
        norms = tuple(value for name, value in vars(nn).items() if "Norm" in name)
        groups = [[names[id(p)] for p in group["params"]] for group in self.optimizer.param_groups]
        assigned = {name: index for index, group in enumerate(groups) for name in group}
        require(len(assigned) == sum(map(len, groups)) == len(names) and len(groups) == 3,
                "grupos AdamW incompletos o duplicados")
        for name in names.values():
            owner = modules[name.rsplit(".", 1)[0]]
            if "bias" in name:
                index = 0
            elif isinstance(owner, norms) or "logit_scale" in name:
                index = 2
            else:
                index = 1
            require(assigned[name] == index, f"grupo AdamW incorrecto para {name}")
        return [
            {"index": index, "parameters": members,
             **{key: group[key] for key in ("initial_lr", "weight_decay", "betas")}}
            for index, (members, group) in enumerate(zip(groups, self.optimizer.param_groups))
        ]

    def persist(self):
        write_json(self.path, self.audit)

    def check_continuity(self, trainer):
        require(trainer.optimizer is self.optimizer and trainer.scheduler is self.scheduler and trainer.ema is self.ema,
                "se reemplazó AdamW, el scheduler o EMA")
        for region, state in self.active_states.items():
            name = self.samples[region]
            require(self.optimizer.state.get(self.parameters[name]) is state, f"se reemplazó el estado AdamW de {name}")

    def frozen_parameters(self):
        return (self.backbone | self.internal) if self.frozen else self.internal

    def check_phase(self):
        frozen = self.frozen_parameters()
        require(all(p.requires_grad == (name not in frozen) for name, p in self.parameters.items()),
                "estado entrenable incompatible con la fase")
        require(all(p.grad is None and not self.optimizer.state.get(p) for p in frozen.values()),
                "parámetros congelados con gradiente residual o estado AdamW")
        require(all(bn.training != self.frozen for bn in self.bn.values()), "modo Batch Normalization incorrecto")

    def begin_epoch(self, trainer, epoch, *, probe=False):
        require(epoch == len(self.audit["epochs"]) + 1, "orden de épocas inesperado")
        self.check_continuity(trainer)
        if not probe:
            require(self.scheduler.last_epoch == epoch - 1, "contador coseno inesperado")
        if epoch == self.duration + 1:
            require(byte_digest(self.tensors) == self.audit["initial_backbone_sha256"],
                    "cambios en el backbone antes de descongelar")
            require(self.optimizer_groups(trainer.model) == self.groups,
                    "cambiaron los grupos AdamW antes de descongelar")
            for name, value in self.original.items():
                self.backbone[name].requires_grad_(value)
            self.frozen = False
            self.audit["thaw_epoch"] = epoch
            self.audit["optimizer_groups_verified_at_thaw"] = True
        for bn in self.bn.values():
            bn.train(not self.frozen)
        self.check_phase()
        self.ema_before = self.ema.updates
        self.before = {region: self.parameters[name].detach().clone() for region, name in self.samples.items()
                       if self.parameters[name].requires_grad}
        self.bn_before = torch.stack([m.num_batches_tracked for m in self.bn.values()])
        self.gradients = None

    def before_optimizer_step(self):
        # The disabled GradScaler (FP32) and native optimizer clear gradients after each step.
        require(all(p.grad is None for p in self.frozen_parameters().values()),
                "un parámetro congelado recibió gradientes")
        if self.gradients is None:
            grads = [self.parameters[self.samples[region]].grad for region in self.before]
            require(all(grad is not None for grad in grads), "faltan gradientes de control")
            checks = torch.stack([torch.isfinite(g).all() & torch.any(g != 0) for g in grads]).tolist()
            self.gradients = dict(zip(self.before, checks))
            require(all(checks), f"gradientes de control nulos o no finitos: {self.gradients}")

    def end_epoch(self, trainer, epoch):
        self.check_continuity(trainer)
        self.check_phase()
        require(self.gradients and self.ema.updates - self.ema_before == len(trainer.train_loader),
                "faltan gradientes o actualizaciones de EMA")
        require(self.scheduler.last_epoch == epoch - 1, "el contador del scheduler cambió durante la época")
        changes = torch.stack([(self.parameters[self.samples[region]].detach() != before).any()
                               for region, before in self.before.items()]).tolist()
        require(all(changes), "algún parámetro de control entrenable no se actualizó")
        steps = dict.fromkeys(self.samples, 0)
        for region in self.before:
            state = self.optimizer.state[self.parameters[self.samples[region]]]
            steps[region] = int(state["step"].item())
            require(steps[region] - self.last_steps[region] == len(trainer.train_loader),
                    f"contador AdamW discontinuo en {region}")
            self.active_states[region] = state
        self.last_steps = steps
        backbone = byte_digest(self.tensors)
        internal = byte_digest(self.internal_tensors)
        require((backbone == self.previous_digest) == self.frozen and internal == self.initial_internal,
                "los parámetros o buffers no respetaron la política de congelación")
        self.previous_digest = backbone
        bn_after = torch.stack([m.num_batches_tracked for m in self.bn.values()])
        delta = bn_after - self.bn_before
        expected = 0 if self.frozen else len(trainer.train_loader)
        require(bool((delta == expected).all().item()), "las Batch Normalization no respetaron la fase")
        self.audit["epochs"].append({
            "epoch": epoch, "backbone_frozen": self.frozen, "backbone_sha256": backbone,
            "internal_sha256": internal, "frozen_gradients_absent": True,
            "backbone_bn_training": not self.frozen, "backbone_bn_steps": expected,
            "nonzero_gradients": self.gradients, "updated": dict(zip(self.before, changes)),
            "adamw_steps": steps, "scheduler_epoch": self.scheduler.last_epoch,
            "ema_updates": self.ema.updates, "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
        })
        self.persist()

    def finish_probe(self, trainer):
        self.before_optimizer_step()
        self.check_continuity(trainer)
        self.check_phase()
        require(self.frozen and self.ema.updates == 0 and byte_digest(self.tensors) == self.previous_digest
                and byte_digest(self.internal_tensors) == self.initial_internal,
                "la inicialización modificó parámetros o buffers congelados")
        self.audit["probe"] = {"backbone_frozen": True, "nonzero_gradients": self.gradients, "optimizer_steps": 0}
        self.persist()
