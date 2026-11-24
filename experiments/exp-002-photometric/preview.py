"""Render the original and both endpoint gains for one train image per class."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from scripts.baseline_data import read_json
from scripts.baseline_run import REPO, load_config
from scripts.photometric import adjust_luminance


def render(train, output):
    spec = load_config("yolo", REPO / "configs/experiments/yolo-photometric.json", {})["photometric"]
    dataset = read_json(train / "_annotations.coco.json")
    images = {row["id"]: row["file_name"] for row in dataset["images"]}
    labels = defaultdict(list)
    first_image = {}
    for ann in sorted(dataset["annotations"], key=lambda row: (row["image_id"], row["id"])):
        labels[ann["image_id"]].append(ann["bbox"])
        first_image.setdefault(ann["category_id"], ann["image_id"])
    names = {category["id"]: category["name"] for category in dataset["categories"]}
    gains = [(1 + sign * spec["brightness"], 1 + sign * spec["contrast"]) for sign in (-1, 1)]

    fig, axes = plt.subplots(len(first_image), 3, figsize=(7.8, 19), layout="constrained")
    for row, (category, image_id) in zip(axes, sorted(first_image.items())):
        with Image.open(train / images[image_id]) as source:
            rgb = np.asarray(source.convert("RGB"))
        bgr = rgb[..., ::-1]
        variants = [rgb, *(adjust_luminance(bgr, *factors)[..., ::-1] for factors in gains)]
        for ax, image in zip(row, variants):
            ax.imshow(image)
            for x, y, w, h in labels[image_id]:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffffff", linewidth=1))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        row[0].set_ylabel(f"{names[category]}\nImagen {image_id}", fontsize=9)
    for ax, title in zip(axes[0], ("Original", "Brillo y contraste −10 %", "Brillo y contraste +10 %")):
        ax.set_title(title, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO / "data/pills-sxdht/train")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/preview.jpg")
    options = parser.parse_args()
    render(options.train, options.output)
