"""Audit geometric label acceptance using train boxes only, without scoring any model."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np

from scripts.baseline_data import read_json, sha256
from scripts.campaign import PLAN, REPO
from scripts.geometric import GeometrySampler


def mosaic_boxes(tiles, size, rng):
    """Replay the box placement and clipping of a square 4-tile Mosaic."""
    x, y = (int(rng.uniform(size / 2, size * 1.5)) for _ in range(2))
    offsets = ((x - size, y - size), (x, y - size), (x - size, y), (x, y))
    raw = np.concatenate([boxes + [dx, dy, dx, dy] for boxes, (dx, dy) in zip(tiles, offsets)])
    clipped = raw.clip(0, 2 * size)
    retained = (clipped[:, 2:] > clipped[:, :2]).all(axis=1)
    return clipped[retained], int((~retained).sum()), int((raw[retained] != clipped[retained]).any(axis=1).sum())


def audit(source, repetitions):
    recipe = read_json(REPO / PLAN)
    data = read_json(source)
    sizes = {(img["width"], img["height"]) for img in data["images"]}
    if sizes != {(640, 640)}:
        raise ValueError("La simulación de Mosaic requiere las imágenes cuadradas originales a 640")
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        by_image[ann["image_id"]].append([x, y, x + w, y + h])
    orig = [np.asarray(by_image[img["id"]], dtype=float).reshape(-1, 4) for img in data["images"]]
    side = 640
    draws = repetitions * len(orig)
    seed = recipe["seed"]
    base = recipe["geometric"]
    angles = tuple(dict.fromkeys((base["degrees"], 15)))
    results = {}
    rng = Random(seed)
    for subset in ("plain", "mosaic"):
        # Set the gate to one to isolate clipping rejection from its nominal probability.
        samplers = {angle: GeometrySampler({**base, "degrees": angle, "probability": 1}, seed) for angle in angles}
        accepted = dict.fromkeys(angles, 0)
        flipped = dict.fromkeys(angles, 0)
        inflation = {angle: [] for angle in angles}
        removed, clipped, empty = 0, 0, 0
        for _ in range(repetitions):
            for source_boxes in orig:
                if subset == "plain":
                    boxes, size = source_boxes, side
                else:
                    other = [orig[rng.randrange(len(orig))] for _ in range(3)]
                    boxes, removed_here, clipped_here = mosaic_boxes([source_boxes, *other], side, rng)
                    removed += removed_here
                    clipped += clipped_here
                    empty += not len(boxes)
                    size = 2 * side
                for angle, sampler in samplers.items():
                    sample = sampler.sample(size, size, boxes)
                    if sample is None:
                        continue
                    matrix, after = sample
                    accepted[angle] += 1
                    flipped[angle] += matrix[0, 0] < 0
                    # Divide out scale to isolate inflation of axis-aligned boxes due to rotation.
                    ratio = np.prod(after[:, 2:] - after[:, :2], axis=1) / (
                        np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) * abs(np.linalg.det(matrix[:, :2]))
                    )
                    inflation[angle].extend(ratio)
        results[subset] = {
            "geometry": {angle: {"reject_pct_given_attempt": 100 * (draws - accepted[angle]) / draws,
                                 "applies_pct": 100 * accepted[angle] / draws * base["probability"],
                                 "flip_pct": 100 * flipped[angle] / draws * base["probability"],
                                 "rotation_area_ratio_median_p95": np.percentile(inflation[angle], [50, 95]).tolist()}
                         for angle in angles},
            "mosaic_boxes_removed": removed, "mosaic_boxes_partly_clipped": clipped, "mosaics_empty": empty,
        }
    all_boxes = np.concatenate(orig)
    wh = all_boxes[:, 2:] - all_boxes[:, :2]
    return {"train_source_sha256": sha256(source), "images": len(orig), "boxes": len(all_boxes),
            "draws_per_subset": draws, "seed": seed,
            "boxes_under_coco_small_area": int((wh.prod(axis=1) < 32**2).sum()),
            "boxes_enters_coco_small_area_at_half_size": int(((wh.prod(axis=1) >= 32**2)
                                                           & (wh.prod(axis=1) / 4 < 32**2)).sum()),
            **results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO / "data/pills-sxdht/train/_annotations.coco.json")
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions debe ser mayor que cero")
    print(json.dumps(audit(args.train, args.repetitions), indent=2))
