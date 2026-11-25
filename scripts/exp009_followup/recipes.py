"""Recipes and dependency graphs for independent EXP-009 phases."""
SMOKE = "smoke-photometric"
RUNTIME_PLAN = {"B": "configs/campaign/exp-009-a.json", "C": "configs/campaign/exp-009-a.json",
                "D": "configs/campaign/exp-008.json"}
CONFIRMATION_ARMS = dict(zip("ABCDEF", ("f000-control", "f000-geom", "f080-control", "f080-geom",
                                        "f080-control-photo", "f080-geom-photo")))
CONFIRMATION_SEEDS = (2027, 2028)


def confirmation_conditions():
    return [{"name": f"{arm.lower()}-s{seed}", "id": arm, "backend": "yolo", "epochs": 100,
             "reference": name, "seed": seed}
            for seed in CONFIRMATION_SEEDS for arm, name in CONFIRMATION_ARMS.items()]


def recipe(plan, condition):
    """Add only the intervention (B) or change only the seed (D)."""
    base = plan["reference"]["arms"][condition["reference"]]["config"]
    if plan["phase"] == "B":
        return {**base, "photometric": plan["photometric"]}
    return {**base, "seed": condition["seed"]}


def smoke_recipe(plan):
    return {**recipe(plan, plan["conditions"][-1]), "epochs": 3, "freeze_backbone_epochs": 1}


def graph(plan):
    backend, = plan["packages"]
    conditions = plan["conditions"]
    preflight = f"preflight-{backend}"
    probes = [conditions[0]["name"]]
    if plan["phase"] in ("B", "C"):
        probes += [SMOKE]
    nodes = {preflight: {"stage": "preflight", "backend": backend, "requires": None, "probes": probes}}
    for condition in conditions:
        name = condition["name"]
        nodes[f"train-{name}"] = {"stage": "train", "backend": backend, "condition": name, "requires": preflight}
        nodes[f"eval-{name}"] = {"stage": "eval", "backend": backend, "condition": name, "requires": f"train-{name}"}
    return nodes


def verify_plan(plan, historical, historical_recipe):
    fixed = ("dataset", "dataset_relpath", "dataset_manifest_sha256", "valid_coco_sha256",
             "evaluation", "weights", "packages", "seed")
    if not all(plan[key] == historical[key] for key in fixed):
        raise ValueError("El protocolo o las entradas difieren de la campaña original")
    phase = plan["phase"]
    references = plan["reference"]["arms"]
    old_arms = {row["name"]: row for row in historical["conditions"]}
    if phase in ("B", "C"):
        if (any(plan[k] != historical[k] for k in ("geometric", "mosaic_probability"))
                or plan["photometric"] != {"probability": .5, "brightness": .1, "contrast": .1}):
            raise ValueError("Las políticas de geometría y fotometría difieren de A y B")
    for name, reference in references.items():
        original = name.removesuffix("-photo") if phase == "C" else name
        old = old_arms[original]
        config = historical_recipe(historical, old)
        if name != original:
            old = {**old, "name": name, "reference": original, "seed": 2026}
            config = {**config, "photometric": plan["photometric"]}
        if reference["condition"] != old or reference["config"] != config:
            raise ValueError(f"Receta de referencia incompatible: {name}")
    if phase == "B":
        expected_names = [f"f{freeze:03d}-{geom}" for freeze in (0, 80) for geom in ("control", "geom")]
        if list(references) != expected_names:
            raise ValueError("El factorial B requiere los cuatro controles A y la política EXP-002")
        expected = [{**old_arms[name], "name": name + "-photo", "reference": name, "seed": 2026}
                    for name in expected_names]
    elif phase == "C":
        if list(references) != list(CONFIRMATION_ARMS.values()) or plan["arm_ids"] != CONFIRMATION_ARMS:
            raise ValueError("C requiere las seis referencias de A y B")
        cohorts = plan["reference"]["cohorts"]
        sources = {name: "B" if name.endswith("-photo") else "A" for name in references}
        if set(cohorts) != {"A", "B"} or sources != {name: row["phase"] for name, row in references.items()}:
            raise ValueError("Procedencia de las referencias de C incompatible")
        expected = confirmation_conditions()
    else:
        if phase != "D" or set(references) != {"e100-f000"}:
            raise ValueError("D requiere el control de 100 épocas de EXP-008")
        expected = [{**old_arms["e100-f000"], "name": f"e100-f000-s{seed}",
                     "reference": "e100-f000", "seed": seed} for seed in (2027, 2028)]
    if plan["conditions"] != expected:
        raise ValueError("Matriz de entrenamientos incompatible")
