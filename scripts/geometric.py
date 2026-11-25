"""Shared train-only affine augmentation for detection boxes in pixel coordinates."""

import math

import numpy as np

from scripts.photometric import PhotometricSampler


def validate_geometric(spec):
    limits = {"probability": 1, "degrees": 15, "translate": 0.1, "scale": 0.2, "flip": 1}
    if not isinstance(spec, dict) or spec.keys() != limits.keys():
        raise ValueError(f"geometric requiere las claves {sorted(limits)}")
    for key, limit in limits.items():
        value = spec[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"geometric.{key} debe estar en [0, {limit}]")


def affine(width, height, angle, scale, dx, dy, flip):
    angle = math.radians(angle)
    c, s = scale * math.cos(angle), scale * math.sin(angle)
    linear = np.array([[c, -s], [s, c]]) @ np.diag([-1 if flip else 1, 1])
    center = np.array([width / 2, height / 2])
    return np.column_stack((linear, center + [dx, dy] - linear @ center))


def transform_coords(coords, matrix):
    return coords @ matrix[:, :2].T + matrix[:, 2]


def transform_boxes(boxes, matrix):
    corners = np.stack((boxes[:, [0, 1]], boxes[:, [0, 3]],
                        boxes[:, [2, 1]], boxes[:, [2, 3]]), axis=1)
    transformed = transform_coords(corners, matrix)
    return np.column_stack((transformed.min(axis=1), transformed.max(axis=1)))


def warp(image, matrix):
    import cv2

    height, width = image.shape[:2]
    pixels = matrix.copy()
    # Boxes use pixel edges; OpenCV uses pixel centers at integer coordinates.
    pixels[:, 2] += matrix[:, :2] @ [0.5, 0.5] - 0.5
    return cv2.warpAffine(image, pixels, (width, height), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


class GeometrySampler(PhotometricSampler):
    """Use the same per-worker RNG policy as the photometric experiment."""

    def sample(self, width, height, boxes, worker=None):
        rng = self.worker_rng(worker)
        p = self.spec
        if rng.random() >= p["probability"]:
            return None
        matrix = affine(width, height, rng.uniform(-p["degrees"], p["degrees"]),
                        rng.uniform(1 - p["scale"], 1 + p["scale"]),
                        width * rng.uniform(-p["translate"], p["translate"]),
                        height * rng.uniform(-p["translate"], p["translate"]),
                        rng.random() < p["flip"])
        transformed = transform_boxes(np.asarray(boxes, dtype=float).reshape(-1, 4), matrix)
        if (not np.isfinite(transformed).all()
                or np.any(transformed[:, :2] < 0) or np.any(transformed[:, 2:] > [width, height])
                or np.any(transformed[:, :2] >= transformed[:, 2:])):
            return None
        return matrix, transformed
