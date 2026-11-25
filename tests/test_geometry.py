"""Lightweight affine checks and native loader contracts for the campaign."""
import sys
import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
from PIL import Image

from scripts.baseline_run import REPO, load_config
from scripts.geometric import GeometrySampler, affine, transform_boxes, validate_geometric

FLIP = {"probability": 1, "degrees": 0, "translate": 0, "scale": 0, "flip": 1}
BOX = [8, 12, 24, 44]
FLIPPED = [40, 12, 56, 44]


def jpeg(path, channel):
    pixels = np.full((64, 64, 3), 16, dtype=np.uint8)
    pixels[12:44, 8:24, channel % 3] = 80 + channel * 50
    Image.fromarray(pixels).save(path)


class GeometryTest(unittest.TestCase):
    def test_flip_rotation_and_immutable_source(self):
        boxes = np.array([BOX], dtype=float)
        np.testing.assert_array_equal(transform_boxes(boxes, affine(64, 64, 0, 1, 0, 0, True)), [FLIPPED])
        np.testing.assert_allclose(transform_boxes(boxes, affine(64, 64, 90, 1, 0, 0, False)),
                                   [[20, 8, 52, 24]])
        np.testing.assert_array_equal(boxes, [BOX])

    def test_reproducible_worker_stream_and_reject_entire_crop(self):
        spec = {**FLIP, "translate": 0.04, "degrees": 8, "scale": 0.1}
        first, second = (GeometrySampler(spec, 2026) for _ in range(2))
        worker = SimpleNamespace(seed=20)
        for _ in range(20):
            a, b = (r.sample(64, 64, [BOX], worker) for r in (first, second))
            if a is None:
                self.assertIsNone(b)
            else:
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
        # Any rotation or movement of the full-image box is unsafe here.
        wide = GeometrySampler({**FLIP, "degrees": 8}, 20)
        self.assertIsNone(wide.sample(64, 64, [[0, 0, 64, 64]]))
        self.assertIsNone(wide.sample(64, 64, [[float("nan"), 0, 1, 1]]))
        self.assertEqual(wide.sample(64, 64, [])[1].shape, (0, 4))

    def test_reject_invalid_policy(self):
        for invalid in ({**FLIP, "scale": 1}, {**FLIP, "flip": True}, {**FLIP, "mosaic": 0.3}):
            with self.assertRaises(ValueError):
                validate_geometric(invalid)


@unittest.skipUnless(find_spec("ultralytics"), "Ultralytics ausente")
class YoloGeometryLoaderTest(unittest.TestCase):
    def test_native_mosaic_and_affine_only_in_train(self):
        import cv2
        import torch
        from ultralytics.cfg import get_cfg

        from scripts.baseline_yolo import YoloTrainer, check_version, ultralytics_args
        from scripts.evaluate_baseline import load_image

        check_version()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, count in (("train", 4), ("valid", 1)):
                images, labels = root / "images" / split, root / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                for index in range(count):
                    jpeg(images / f"{index}.jpg", index)
                    (labels / f"{index}.txt").write_text(f"{index} 0.25 0.4375 0.25 0.5\n")
            spec = load_config("yolo", None, {"imgsz": 64, "workers": 0})
            cli = SimpleNamespace(data=root, output=root / "output", resume=False, weights=root / "yolov10s.pt")
            trainer = object.__new__(YoloTrainer)
            trainer.args = get_cfg(overrides=ultralytics_args(cli, spec, None))
            trainer.model = SimpleNamespace(stride=torch.tensor([32]))
            trainer.data = {"names": dict(enumerate("abcd")), "nc": 4, "channels": 3}
            trainer.classes = [{"original_id": index} for index in range(1, 5)]
            trainer.summary = {split: {"images": count, "instances": count,
                                      "ground_truth_by_category": {str(i): int(i <= count) for i in range(1, 5)}}
                               for split, count in (("train", 4), ("valid", 1))}
            trainer.photometric = None
            trainer.geometric = FLIP

            def dataset(split, mode):
                return trainer.build_dataset(str(root / "images" / split), mode, batch=2)

            affine_only = dataset("train", "train")
            trainer.mosaic_probability = 1.0
            combined, validation = dataset("train", "train"), dataset("valid", "val")
            names = lambda ds: [type(t).__name__ for t in ds.transforms.transforms]
            self.assertEqual(names(combined), ["YoloMosaic", "YoloGeometry", "LetterBox", "Format"])
            self.assertEqual(names(validation), ["LetterBox", "Format"])
            self.assertEqual(combined.transforms.transforms[-1].bgr, 0.0)
            source_labels = deepcopy(combined.labels)
            original = [load_image(root / f"images/train/{i}.jpg") for i in range(4)]
            plain = affine_only[0]
            rgb = lambda im: np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
            np.testing.assert_array_equal(plain["img"].numpy(), rgb(original[0][:, ::-1]))
            torch.testing.assert_close(plain["bboxes"], torch.tensor([[.75, .4375, .25, .5]]))
            val = validation[0]
            np.testing.assert_array_equal(val["img"].numpy(), rgb(original[0]))
            torch.testing.assert_close(val["bboxes"], torch.tensor([[.25, .4375, .25, .5]]))
            self.assertEqual(val["cls"].flatten().tolist(), [0])
            def fixed_mosaic(center, ds=combined):
                mosaic = ds.transforms.transforms[0].mosaic
                # BaseMixTransform draws the gate first, then Mosaic draws y and x.
                # Format also draws a gate for BGR/RGB. All four draws have different roles.
                with patch.object(mosaic, "get_indexes", return_value=[1, 2, 3]) as indexes, \
                     patch("ultralytics.data.augment.random") as rng:
                    rng.uniform.side_effect = [0.5, center, center, 0.5]
                    result = ds[0]
                indexes.assert_called_once_with()
                self.assertEqual(rng.uniform.call_args_list,
                                 [call(0, 1), call(32, 96), call(32, 96), call(0, 1)])
                return result

            augmented = fixed_mosaic(64)
            self.assertEqual(augmented["cls"].flatten().tolist(), [0, 1, 2, 3])
            torch.testing.assert_close(augmented["bboxes"], torch.tensor([
                [.875, .21875, .125, .25], [.375, .21875, .125, .25],
                [.875, .71875, .125, .25], [.375, .71875, .125, .25],
            ]))
            canvas = np.vstack((np.hstack(original[:2]), np.hstack(original[2:])))
            self.assertEqual(tuple(augmented["img"].shape), (3, 64, 64))
            self.assertEqual(augmented["img"].dtype, torch.uint8)
            np.testing.assert_array_equal(augmented["img"].numpy(),
                                          rgb(cv2.resize(cv2.flip(canvas, 1), (64, 64))))
            self.assertEqual(set(augmented), set(plain))
            mixed = combined.collate_fn([augmented, plain])
            self.assertEqual(tuple(mixed["img"].shape), (2, 3, 64, 64))
            self.assertEqual(mixed["batch_idx"].tolist(), [0, 0, 0, 0, 1])
            clipped = fixed_mosaic(32)
            boxes = clipped["bboxes"]
            self.assertEqual(clipped["cls"].flatten().tolist(), [1, 3])
            torch.testing.assert_close(boxes, torch.tensor([[.625, .046875, .125, .09375],
                                                            [.625, .46875, .125, .25]]))
            self.assertTrue(torch.isfinite(boxes).all())
            self.assertTrue((boxes[:, 2:] > 0).all())
            self.assertTrue((boxes[:, :2] - boxes[:, 2:] / 2 >= -1e-6).all())
            self.assertTrue((boxes[:, :2] + boxes[:, 2:] / 2 <= 1 + 1e-6).all())
            for original_label, label in zip(source_labels, combined.labels):
                for key in ("cls", "bboxes"):
                    np.testing.assert_array_equal(label[key], original_label[key])

            from scripts.photometric import adjust_luminance

            trainer.photometric = {"probability": 0.5, "brightness": 0.1, "contrast": 0.1}
            joint = dataset("train", "train")
            self.assertEqual(names(joint), ["YoloMosaic", "YoloGeometry", "YoloPhotometric", "LetterBox", "Format"])
            photo = joint.transforms.transforms[2]
            with patch.object(photo.sampler, "sample", return_value=(1.1, 1.1)) as sample:
                composed = fixed_mosaic(64, joint)
            sample.assert_called_once()
            expected = adjust_luminance(cv2.flip(canvas, 1), 1.1, 1.1)
            np.testing.assert_array_equal(composed["img"].numpy(), rgb(cv2.resize(expected, (64, 64))))
            torch.testing.assert_close(composed["bboxes"], augmented["bboxes"])
            torch.testing.assert_close(composed["cls"], augmented["cls"])
            val_joint = dataset("valid", "val")
            self.assertEqual(names(val_joint), ["LetterBox", "Format"])
            np.testing.assert_array_equal(val_joint[0]["img"].numpy(), val["img"].numpy())


@unittest.skipUnless(find_spec("detectron2"), "Detectron2 ausente")
class CascadeGeometryLoaderTest(unittest.TestCase):
    def test_native_loader_and_logical_epoch_budget(self):
        import torch
        from detectron2.config import CfgNode
        from detectron2.data import DatasetCatalog
        from detectron2.structures import BoxMode

        from scripts.baseline_cascade import CascadeTrainer, configure
        from scripts.evaluate_baseline import load_image

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "probe.jpg"
            jpeg(image, 0)
            spec = load_config("cascade", None, {"imgsz": 64, "batch": 1, "workers": 0})
            cfg, _, _ = configure(spec, 1, 1, root / "unused.pkl", root)
            record = {"file_name": str(image), "width": 64, "height": 64, "image_id": 1,
                      "annotations": [{"bbox": BOX, "bbox_mode": BoxMode.XYXY_ABS, "category_id": 0}]}
            original = deepcopy(record)
            name = "campaign_geometry_probe"
            DatasetCatalog.register(name, lambda: [record])
            self.addCleanup(DatasetCatalog.remove, name)
            cfg.DATASETS.TRAIN = cfg.DATASETS.TEST = (name,)
            cfg.INPUT.GEOMETRIC = CfgNode(FLIP)
            cfg.freeze()
            train = next(iter(CascadeTrainer.build_train_loader(cfg)))[0]
            val = next(iter(CascadeTrainer.build_test_loader(cfg, name)))[0]
            decoded = load_image(image)
            np.testing.assert_array_equal(train["image"].numpy(), decoded[:, ::-1].transpose(2, 0, 1))
            np.testing.assert_array_equal(val["image"].numpy(), decoded.transpose(2, 0, 1))
            torch.testing.assert_close(train["instances"].gt_boxes.tensor, torch.tensor([FLIPPED]).float())
            self.assertEqual(train["instances"].gt_classes.tolist(), [0])
            self.assertEqual(record, original)
            for epochs in (30, 100):
                recipe = load_config("cascade", None, {"epochs": epochs})
                settings, stride, _ = configure(recipe, 8, 316, root / "unused.pkl", root)
                self.assertEqual((stride, settings.SOLVER.MAX_ITER, settings.TEST.EVAL_PERIOD,
                                  settings.SOLVER.CHECKPOINT_PERIOD, settings.SOLVER.WARMUP_ITERS),
                                 (79, 79 * epochs, 79, 79, 79))
                self.assertEqual(settings.SOLVER.LR_SCHEDULER_NAME, "WarmupCosineLR")


if __name__ == "__main__":
    name = sys.argv[1]
    suite = unittest.defaultTestLoader.loadTestsFromName(name, module=sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.testsRun and result.wasSuccessful() and not result.skipped else 1)
