"""Test the shared operator and exercise the train-only hooks with CPU loaders."""

import random
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scripts.baseline_run import REPO, load_config
from scripts.photometric import PhotometricSampler, adjust_luminance, validate_photometric


def image_fixture():
    rgb = np.full((64, 64, 3), [70, 112, 160], dtype=np.uint8)
    rgb[16:48, 16:48] = [140, 70, 130]
    return rgb


def expected_train_images(image, recipe):
    sampler = PhotometricSampler(recipe["photometric"], recipe["seed"])
    images = []
    for _ in range(8):
        gains = sampler.sample()
        images.append(image if gains is None else adjust_luminance(image, *gains))
    return images


class PhotometricTest(unittest.TestCase):
    def test_opt_in_config_changes_only_the_photometric_setting(self):
        expected = {"probability": 0.5, "brightness": 0.1, "contrast": 0.1}
        for backend in ("yolo", "cascade"):
            baseline = load_config(backend, None, {})
            variant = load_config(backend, REPO / f"configs/experiments/{backend}-photometric.json", {})
            self.assertEqual(variant.pop("photometric"), expected)
            self.assertEqual(variant, baseline)
        exp001 = load_config("yolo", REPO / "configs/experiments/yolo-lr-0005.json", {})
        self.assertNotIn("photometric", exp001)
        self.assertEqual(exp001["lr"], 0.0005)

    def test_invalid_config_fails(self):
        for invalid in (None, {}, {"probability": 0.5, "brightness": -0.1, "contrast": 0.1},
                        {"probability": True, "brightness": 0.1, "contrast": 0.1},
                        {"probability": 0.5, "brightness": 0.1, "contrast": 0.1, "saturation": 0}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_photometric(invalid)

    def test_luma_changes_and_channel_ratios_survive_clipping(self):
        image = image_fixture()[..., ::-1].copy()
        image[0, :4] = [[0, 80, 90], [250, 255, 100], [0, 0, 0], [3, 1, 1]]
        original = image.copy()
        np.testing.assert_array_equal(adjust_luminance(image, 1.0, 1.0), image)
        for gains in ((0.9, 0.9), (1.1, 1.1)):
            result = adjust_luminance(image, *gains)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.any(result != image))
            a, b = image.astype(np.int32), result.astype(np.int32)
            a_peak, b_peak = a.max(axis=2, keepdims=True), b.max(axis=2, keepdims=True)
            # Two rounded terms may each contribute up to half a unit of error.
            self.assertTrue(np.all(np.abs(b * a_peak - a * b_peak) <= a_peak))
            self.assertTrue(np.array_equal(a_peak == 0, b_peak == 0))
            np.testing.assert_array_equal(image, original)

    def test_brightness_and_contrast_have_separate_effects(self):
        image = np.array([[[80] * 3, [120] * 3]], dtype=np.uint8)
        self.assertEqual(adjust_luminance(image, 1.1, 1.0)[0, :, 0].tolist(), [88, 132])
        self.assertEqual(adjust_luminance(image, 1.0, 1.1)[0, :, 0].tolist(), [78, 122])

    def test_probability_and_worker_streams(self):
        spec = {"probability": 0.5, "brightness": 0.1, "contrast": 0.1}
        first, repeat = PhotometricSampler(spec, 2026), PhotometricSampler(spec, 2026)
        state = random.getstate()
        draws = [first.sample() for _ in range(1000)]
        self.assertEqual(draws, [repeat.sample() for _ in draws])
        picked = [pair for pair in draws if pair is not None]
        self.assertTrue(450 < len(picked) < 550)
        self.assertTrue(all(0.9 <= gain <= 1.1 for pair in picked for gain in pair))
        self.assertIsNone(PhotometricSampler({**spec, "probability": 0}, 2026).sample())
        self.assertIsNotNone(PhotometricSampler({**spec, "probability": 1}, 2026).sample())
        worker = SimpleNamespace(seed=2027)
        self.assertEqual(first.sample(worker), PhotometricSampler(spec, 2027).sample())
        self.assertEqual(state, random.getstate())


@unittest.skipUnless(find_spec("ultralytics"), "Ultralytics no disponible")
class YoloPhotometricLoaderTest(unittest.TestCase):
    def test_train_pixels_change_but_labels_and_validation_do_not(self):
        import torch
        from ultralytics.cfg import get_cfg

        from scripts.baseline_yolo import YoloTrainer, check_version, ultralytics_args
        from scripts.evaluate_baseline import load_image

        check_version()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = {"images": 1, "instances": 1, "ground_truth_by_category": {"1": 1}}
            for split in ("train", "valid"):
                images = root / "images" / split
                images.mkdir(parents=True)
                labels = root / "labels" / split
                labels.mkdir(parents=True)
                Image.fromarray(image_fixture()).save(images / "sample.jpg")
                (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            recipe = load_config("yolo", None, {"imgsz": 64, "workers": 0})
            variant_recipe = load_config("yolo", REPO / "configs/experiments/yolo-photometric.json", {})
            cli = SimpleNamespace(data=root, output=root / "output", resume=False, weights=root / "yolov10s.pt")
            args = get_cfg(overrides=ultralytics_args(cli, recipe, None))
            trainer = object.__new__(YoloTrainer)
            trainer.args = args
            trainer.model = SimpleNamespace(stride=torch.tensor([32]))
            trainer.data = {"names": {0: "sample"}, "nc": 1, "channels": 3}
            trainer.summary = {"train": specs, "valid": specs}
            trainer.classes = [{"original_id": 1}]
            trainer.photometric = None

            def dataloader(split, mode):
                return trainer.get_dataloader(str(root / "images" / split), batch_size=1, rank=-1, mode=mode)

            control = dataloader("train", "train")
            control_valid = dataloader("valid", "val")
            trainer.photometric = variant_recipe["photometric"]
            variant = dataloader("train", "train")
            variant_valid = dataloader("valid", "val")

            self.assertFalse(args.augment)
            plain = ["LetterBox", "Format"]
            for loader, expected in ((control, plain), (control_valid, plain), (variant_valid, plain),
                                     (variant, ["YoloPhotometric", *plain])):
                self.assertEqual([type(item).__name__ for item in loader.dataset.transforms.transforms], expected)
            original = load_image(root / "images/train/sample.jpg")
            expected_control = original[..., ::-1].transpose(2, 0, 1)
            control_batch = next(iter(control))
            np.testing.assert_array_equal(control_batch["img"][0].numpy(), expected_control)
            torch.testing.assert_close(control_batch["bboxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
            self.assertEqual(control_batch["cls"].flatten().tolist(), [0])
            self.assertEqual(control_batch["batch_idx"].flatten().tolist(), [0])

            changed = []
            for expected in expected_train_images(original, variant_recipe):
                batch = next(iter(variant))
                np.testing.assert_array_equal(batch["img"][0].numpy(), expected[..., ::-1].transpose(2, 0, 1))
                changed.append(not torch.equal(batch["img"], control_batch["img"]))
                for key in ("bboxes", "cls", "batch_idx"):
                    self.assertTrue(torch.equal(batch[key], control_batch[key]), key)
            self.assertEqual(set(changed), {False, True})
            for loader in (control_valid, variant_valid):
                batch = next(iter(loader))
                self.assertTrue(torch.equal(batch["img"], control_batch["img"]))
                for key in ("bboxes", "cls", "batch_idx"):
                    self.assertTrue(torch.equal(batch[key], control_batch[key]), key)


@unittest.skipUnless(find_spec("detectron2"), "Detectron2 no disponible")
class CascadePhotometricLoaderTest(unittest.TestCase):
    def test_train_pixels_change_but_labels_and_validation_do_not(self):
        import torch
        from detectron2.config import CfgNode
        from detectron2.data import DatasetCatalog, DatasetMapper
        from detectron2.data.detection_utils import read_image
        from detectron2.structures import BoxMode

        from scripts.baseline_cascade import CascadeTrainer, configure

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.jpg"
            Image.fromarray(image_fixture()).save(image_path)
            recipe = load_config("cascade", None, {"imgsz": 64, "batch": 1, "workers": 0})
            variant_recipe = load_config("cascade", REPO / "configs/experiments/cascade-photometric.json", {})
            cfg, _, _ = configure(recipe, 1, 1, root / "unused.pkl", root)
            train, valid = "photometric_loader_train", "photometric_loader_valid"
            record = {
                "file_name": str(image_path), "height": 64, "width": 64, "image_id": 1,
                "annotations": [{"bbox": [16, 16, 48, 48], "bbox_mode": BoxMode.XYXY_ABS, "category_id": 0}],
            }
            annotated = deepcopy(record)
            for name in (train, valid):
                DatasetCatalog.register(name, lambda: [record])
                self.addCleanup(DatasetCatalog.remove, name)
            cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = (train,), (valid,)
            self.assertEqual([type(item).__name__ for item in DatasetMapper(cfg, is_train=True).augmentations.augs],
                             ["ResizeShortestEdge"])
            self.assertFalse(cfg.INPUT.CROP.ENABLED)
            self.assertEqual(cfg.INPUT.RANDOM_FLIP, "none")
            cfg.freeze()
            variant_cfg = cfg.clone()
            variant_cfg.defrost()
            variant_cfg.INPUT.PHOTOMETRIC = CfgNode(variant_recipe["photometric"])
            variant_cfg.freeze()

            original = read_image(str(image_path), format="BGR")
            control = next(iter(CascadeTrainer.build_train_loader(cfg)))[0]
            np.testing.assert_array_equal(control["image"].numpy(), original.transpose(2, 0, 1))
            instances = control["instances"]
            torch.testing.assert_close(instances.gt_boxes.tensor, torch.tensor([[16.0, 16.0, 48.0, 48.0]]))
            self.assertEqual(instances.gt_classes.tolist(), [0])

            variant = iter(CascadeTrainer.build_train_loader(variant_cfg))
            changed = []
            for expected in expected_train_images(original, variant_recipe):
                batch = next(variant)[0]
                np.testing.assert_array_equal(batch["image"].numpy(), expected.transpose(2, 0, 1))
                changed.append(not torch.equal(control["image"], batch["image"]))
                self.assertTrue(torch.equal(batch["instances"].gt_boxes.tensor, instances.gt_boxes.tensor))
                self.assertTrue(torch.equal(batch["instances"].gt_classes, instances.gt_classes))
            self.assertEqual(set(changed), {False, True})
            for evaluation_cfg in (cfg, variant_cfg):
                validation = next(iter(CascadeTrainer.build_test_loader(evaluation_cfg, valid)))[0]
                self.assertTrue(torch.equal(control["image"], validation["image"]))
            self.assertEqual(record, annotated)


if __name__ == "__main__":
    unittest.main()
