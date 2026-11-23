"""Genera una figura legible a partir del historial CSV de YOLOv10-S."""

import argparse
import csv
from io import BytesIO
import math
from pathlib import Path


TRAIN_LOSSES = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
VAL_LOSSES = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
AP = ("val/map", "val/ap50")


def read_history(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("El historial no contiene épocas")
    epochs = [int(row["step"]) for row in rows]
    series = {key: [float(row[key]) for row in rows] for key in TRAIN_LOSSES + VAL_LOSSES + AP}
    if any(not math.isfinite(value) for values in series.values() for value in values):
        raise ValueError("El historial contiene valores no finitos")
    if any(value <= 0 for key in TRAIN_LOSSES + VAL_LOSSES for value in series[key]):
        raise ValueError("La escala logarítmica requiere pérdidas estrictamente positivas")
    return epochs, series


def render(epochs, series):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (losses, validation, ap) = plt.subplots(
        3, 1, sharex=True, figsize=(8.5, 8.2), layout="constrained"
    )
    fig.suptitle("YOLOv10-S · Historial de entrenamiento", fontsize=15)
    colors = ("#245a83", "#b25420", "#16715e")
    for key, color in zip(TRAIN_LOSSES, colors):
        losses.plot(epochs, series[key], color=color, label=key)
    losses.plot(epochs, series["val/box_loss"], "--", color=colors[0], label="val/box_loss")
    losses.set(ylabel="Pérdida (escala log.)", yscale="log")
    losses.grid(axis="y", which="minor", alpha=0.12)
    losses.legend(loc="upper right", ncols=2)

    for key, color in zip(VAL_LOSSES[1:], colors[1:]):
        validation.plot(epochs, series[key], color=color, label=key)
    cls_loss = series["val/cls_loss"]
    peak = max(cls_loss)
    epoch = epochs[cls_loss.index(peak)]
    label = f"{peak:,.10g}".replace(",", " ")
    validation.scatter(epoch, peak, color=colors[1], zorder=3)
    validation.annotate(f"Época {epoch}: {label}", (epoch, peak), xytext=(15, -5),
                        textcoords="offset points", va="center", color=colors[1],
                        arrowprops={"arrowstyle": "-", "color": colors[1]})
    validation.set(ylabel="Pérdida (escala log.)", yscale="log")
    validation.legend(loc="upper right")

    ap.plot(epochs, series["val/map"], color=colors[0], label="mAP@0.5:0.95")
    ap.plot(epochs, series["val/ap50"], "--", color=colors[2], label="AP50")
    ap.set(xlabel="Época", ylabel="AP de validación", ylim=(0, 1))
    ap.legend(loc="lower right")
    for axis in (losses, validation, ap):
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    png = BytesIO()
    fig.savefig(png, format="png", dpi=160)
    plt.close(fig)
    return png.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.history.strip() or not args.output.strip():
        parser.error("Las rutas no pueden estar vacías")
    source, output = Path(args.history), Path(args.output)
    try:
        if output.suffix != ".png":
            raise ValueError("La salida debe tener extensión .png")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"La salida ya existe: {output}")
        png = render(*read_history(source))
        with output.open("xb") as destination:
            destination.write(png)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
