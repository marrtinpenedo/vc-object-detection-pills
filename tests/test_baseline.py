"""Synthetic checks for the baseline pipeline."""

import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock, patch

from scripts.baseline_data import class_mapping, coco_to_yolo, convert_split, index_records
from scripts.baseline_metrics import coco_eval, prediction_rows, report, validate_predictions
from scripts.baseline_run import load_config


def fixture():
    categories = [{"id": 0, "name": "unused"}, {"id": 3, "name": "first"}, {"id": 9, "name": "second"}]
    images = [
        {"id": 0, "file_name": "one.jpg", "width": 100, "height": 200},
        {"id": 1, "file_name": "two.jpg", "width": 100, "height": 200},
    ]
    annotations = [
        {"id": 0, "image_id": 0, "category_id": 9, "bbox": [20, 40, 60, 120], "area": 7200, "iscrowd": 0}
    ]
    return {
        "info": {}, "categories": categories, "images": images, "annotations": annotations,
    }, [
        {"original_id": 3, "internal_id": 0, "name": "first"},
        {"original_id": 9, "internal_id": 1, "name": "second"},
    ]


class ConversionTest(unittest.TestCase):
    def test_mapping_uses_union_sorted_and_is_reversible(self):
        raw, expected = fixture()
        all_annotations = [{"category_id": 9}, {"category_id": 3}, {"category_id": 9}]
        self.assertEqual(class_mapping(index_records(raw["categories"], "category"), all_annotations), expected)

    def test_conversion_preserves_native_coco_and_counts(self):
        raw, classes = fixture()
        before = deepcopy(raw)
        coco, labels = convert_split(raw, classes)
        self.assertEqual(labels, {0: [(1, 0.5, 0.5, 0.6, 0.6)], 1: []})
        self.assertEqual(coco["annotations"], [{**raw["annotations"][0], "id": 1}])
        self.assertEqual(coco["images"], raw["images"])
        self.assertEqual(raw, before)
        self.assertEqual([row["id"] for row in coco["categories"]], [3, 9])

    def test_box_edges_and_numeric_errors(self):
        self.assertEqual(coco_to_yolo([0, 0, 100, 200], 100, 200), (0.5, 0.5, 1, 1))
        cases = [[0, 0, 0, 2], [1, -1, 2, 3], [99, 0, 2, 1], [0, 0, 2, 201],
                 [float("nan"), 0, 1, 1], [True, 0, 1, 1], [1, 2, 3], [0, 0, "1", 1]]
        for box in cases:
            with self.subTest(box=box), self.assertRaises(ValueError):
                coco_to_yolo(box, 100, 200)

    def test_rejects_unknown_and_duplicate_ids_or_boxes(self):
        raw, classes = fixture()
        variations = [
            [{"image_id": 999}],
            [{"category_id": 0}],
            [{"category_id": 9.0}],
            [{"id": -1}],
            [{"iscrowd": 1}],
            [{}, {}],
            [{}, {"id": 1}],
        ]
        original = raw["annotations"][0]
        for overrides in variations:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                convert_split({**raw, "annotations": [{**original, **row} for row in overrides]}, classes)


class EvaluationTest(unittest.TestCase):
    def test_float32_box_preserves_clipped_image_edges(self):
        import numpy as np

        image = {"id": 9, "file_name": "edge.jpg", "width": 640, "height": 640}
        classes = [{"internal_id": 0, "original_id": 1}]
        truth = {"images": [image], "categories": [{"id": 1}]}
        detection = np.array(
            [119.85000610351562, 34.116180419921875, 640, 486.21728515625, 0.1357076, 0],
            dtype=np.float32,
        )
        for axis, extent, row in (("x", 2, detection), ("y", 3, detection[[1, 0, 3, 2, 4, 5]])):
            with self.subTest(axis=axis):
                predictions = prediction_rows(image, [row], classes)
                validate_predictions(truth, predictions)
                x, y, width, height = predictions[0]["bbox"]
                self.assertEqual([x, y, x + width, y + height], list(map(float, row[:4])))
                self.assertLessEqual(x + width, image["width"])
                self.assertLessEqual(y + height, image["height"])

                outside = predictions[0]["bbox"].copy()
                outside[extent] += 0.01
                with self.assertRaisesRegex(ValueError, "fuera de los límites"):
                    validate_predictions(truth, [{**predictions[0], "bbox": outside}])

    def test_perfect_detection_uses_original_ids_and_absent_ap_is_null(self):
        raw, classes = fixture()
        coco, _ = convert_split(raw, classes)
        predictions = prediction_rows(coco["images"][0], [[20, 40, 80, 160, 0.9, 1]], classes)
        self.assertEqual(predictions[0]["category_id"], 9)
        self.assertEqual(predictions[0]["bbox"], [20, 40, 60, 120])
        metrics, curves, _ = report(coco_eval(coco, predictions))
        for field in ("map", "ap50", "ap75", "ap_medium"):
            self.assertAlmostEqual(metrics[field], 1)
        self.assertIsNone(metrics["per_class"][0]["ap"])
        self.assertIsNone(metrics["ap_small"])
        self.assertAlmostEqual(metrics["per_class"][1]["ap"], 1)
        self.assertEqual(metrics["operating_point"]["tp"], 1)
        self.assertEqual(metrics["operating_point"]["precision"], 1)
        self.assertEqual(curves["classes"]["3"]["0.50"], [None] * 101)

    def test_empty_and_false_positive_evaluation(self):
        raw, classes = fixture()
        coco, _ = convert_split(raw, classes)
        empty, _, _ = report(coco_eval(coco, []))
        self.assertEqual(empty["map"], 0)
        self.assertIsNone(empty["operating_point"]["precision"])
        wrong = prediction_rows(coco["images"][0], [[20, 40, 80, 160, 0.9, 0]], classes)
        metrics, _, _ = report(coco_eval(coco, wrong))
        self.assertEqual(metrics["map"], 0)
        self.assertEqual(metrics["operating_point"]["fp"], 1)
        self.assertEqual(metrics["operating_point"]["fn"], 1)

    def test_clipping_and_evaluation_rejects_foreign_ids(self):
        raw, classes = fixture()
        coco, _ = convert_split(raw, classes)
        clipped = prediction_rows(coco["images"][0], [[-5, 30, 110, 210, 0.7, 1]], classes)
        self.assertEqual(clipped[0]["bbox"], [0, 30, 100, 170])
        for changes in ({"image_id": 404}, {"category_id": 0}, {"bbox": [0, 0, 100.5, 200]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                coco_eval(coco, [{**clipped[0], **changes}])


class ConfigurationTest(unittest.TestCase):
    def test_defaults_and_smoke_override(self):
        for model, batch in (("cascade", 4), ("yolo", 8)):
            spec = load_config(model, None, {"epochs": 2})
            self.assertEqual((spec["epochs"], spec["batch"], spec["imgsz"]), (2, batch, 640))
            with self.assertRaises(ValueError):
                load_config(model, None, {"imgsz": 601})


@unittest.skipUnless(find_spec("detectron2"), "requires the Cascade environment")
class CascadeInitializationTest(unittest.TestCase):
    def test_training_forward_has_event_storage(self):
        import numpy as np
        import torch
        from detectron2.utils.events import get_event_storage
        from PIL import Image

        from scripts import baseline_cascade

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                self.roi_heads = torch.nn.Module()
                self.roi_heads.num_classes = 1
                self.roi_heads.box_predictor = torch.nn.Linear(1, 1, bias=False)
                self.proposal_generator = torch.nn.Linear(1, 1, bias=False)

            def forward(self, batch):
                get_event_storage().put_scalar("probe/inputs", len(batch))
                return {"loss": sum(parameter.sum() for parameter in self.parameters())}

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            image = output / "probe.png"
            Image.new("RGB", (2, 2)).save(image)
            trainer = Mock(model=ProbeModel(), data_loader=[[{"file_name": str(image)}]])
            args = SimpleNamespace(
                output=output, data=output, weights=output / "model_final_480dd8.pkl",
                initialize=True, resume=False, pause_after=None,
            )
            manifest = {"classes": [1], "splits": {"train": {"images": 1}}}
            with patch.object(torch, "use_deterministic_algorithms"), patch.multiple(
                baseline_cascade,
                seed_all_rng=DEFAULT,
                register_training_data=DEFAULT,
                CascadeTrainer=Mock(return_value=trainer),
                CheckedWeights=DEFAULT,
                make_predictor=Mock(return_value=lambda image: np.empty((0, 6))),
            ):
                baseline_cascade.train(args, load_config("cascade", None, {}), manifest)
            trainer.train.assert_not_called()
            report = baseline_cascade.read_json(output / "initialization.json")
            self.assertEqual(report["gradients"], {
                "backbone.weight": "nonzero",
                "roi_heads.box_predictor.weight": "nonzero",
                "proposal_generator.weight": "nonzero",
            })
