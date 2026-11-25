"""Shared luminance-only augmentation for uint8 BGR training images."""

import math
import random

import numpy as np

BGR_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def validate_photometric(spec):
    keys = {"probability", "brightness", "contrast"}
    if not isinstance(spec, dict) or spec.keys() != keys:
        raise ValueError(f"photometric requiere las claves {sorted(keys)}")
    for key, value in spec.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"photometric.{key} debe estar en [0, 1]")


def adjust_luminance(image, brightness, contrast):
    """Scale luma about its image mean without changing channel ratios."""
    luma = image @ BGR_LUMA
    center = luma.mean()
    target = brightness * (center + contrast * (luma - center))
    gain = np.divide(target, luma, out=np.ones_like(luma), where=luma > 0)
    # Limit the shared gain to preserve channel ratios even at the gamut limits.
    peak = np.maximum(image.max(axis=2), 1).astype(np.float32)
    gain = np.clip(gain, 1 / peak, 255 / peak)
    return np.rint(image * gain[..., None]).astype(np.uint8)


class PhotometricSampler:
    """Keep a separate random stream per loader worker."""

    def __init__(self, spec, seed):
        self.spec = spec
        self.seed = seed
        self.worker_seed = None
        self.rng = None

    def worker_rng(self, worker):
        seed = self.seed if worker is None else worker.seed
        if self.rng is None or seed != self.worker_seed:
            self.worker_seed = seed
            self.rng = random.Random(seed)
        return self.rng

    def sample(self, worker=None):
        rng = self.worker_rng(worker)
        if rng.random() >= self.spec["probability"]:
            return None
        return (
            rng.uniform(1 - self.spec["brightness"], 1 + self.spec["brightness"]),
            rng.uniform(1 - self.spec["contrast"], 1 + self.spec["contrast"]),
        )
