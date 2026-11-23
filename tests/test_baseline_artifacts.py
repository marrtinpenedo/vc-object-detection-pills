"""Pruebas locales para la exportación y la figura de resultados."""

from importlib.util import find_spec
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile


BASELINE = Path(__file__).resolve().parents[1] / "experiments" / "baseline"
HISTORY = BASELINE / "results/yolo/training/history.csv"


class ExportResultsTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "results"
        self.cascade = self.root / "cascade-run"
        self.yolo = self.root / "yolo-run"
        self.evaluation = self.root / "evaluation"
        self.output = self.root / "bundle.zip"
        self.expected = []
        for backend, run in (("cascade", self.cascade), ("yolo", self.yolo)):
            run.mkdir(parents=True)
            for name in ("run.json", "status.json", "history.csv", "training.png", "last.pt"):
                (run / name).write_bytes(name.encode())
                if name != "last.pt":
                    self.expected.append(f"{backend}/training/{name}")
            evaluated = self.evaluation / backend
            samples = evaluated / "examples"
            samples.mkdir(parents=True)
            for name in ("metrics.json", "predictions.json", "pr_curves.json", "pr_curves.png"):
                (evaluated / name).write_bytes(name.encode())
                self.expected.append(f"{backend}/evaluation/{name}")
            for kind in ("correct", "failure"):
                for index in range(7):
                    name = f"{kind}-{index}.jpg"
                    (samples / name).write_bytes(name.encode())
                    if index < 5:
                        self.expected.append(f"{backend}/evaluation/examples/{name}")
            (samples / "unrelated.jpg").touch()
            (samples / "nested").mkdir()
            (samples / "nested/correct-0.jpg").touch()

    def run_export(self, **overrides):
        options = {
            "results-root": self.root, "cascade-run": self.cascade, "yolo-run": self.yolo,
            "evaluation": self.evaluation, "output": self.output,
        }
        options.update(overrides)
        command = [sys.executable, str(BASELINE / "export_results.py")]
        for name, value in options.items():
            command.extend([f"--{name}", str(value)])
        return subprocess.run(command, capture_output=True, text=True)

    def test_archive_contains_only_selected_files(self):
        result = self.run_export()
        self.assertEqual(result.returncode, 0, result.stderr)
        with ZipFile(self.output) as archive:
            self.assertEqual(archive.namelist(), self.expected)
            self.assertEqual(len(self.expected), 36)
            self.assertIsNone(archive.testzip())
            self.assertIn(b"Mohamed Attia", archive.comment)
            self.assertIn(b"CC BY 4.0", archive.comment)
            for name in archive.namelist():
                self.assertEqual(archive.read(name), Path(name).name.encode())

    def test_rejects_empty_or_missing_paths(self):
        options = ("results-root", "cascade-run", "yolo-run", "evaluation", "output")
        for name in options:
            with self.subTest(name=name):
                result = self.run_export(**{name: " "})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("vacía", result.stderr)
        self.assertNotEqual(self.run_export(evaluation=self.root / "missing").returncode, 0)
        (self.cascade / "run.json").unlink()
        self.assertNotEqual(self.run_export().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_rejects_external_or_overlapping_paths_and_file_links(self):
        external = self.root.parent / "external"
        external.mkdir()
        self.assertNotEqual(self.run_export(**{"yolo-run": external}).returncode, 0)
        self.assertNotEqual(self.run_export(**{"yolo-run": self.cascade}).returncode, 0)
        self.assertNotEqual(self.run_export(output=external / "bundle.zip").returncode, 0)
        (self.root / "linked").symlink_to(external, target_is_directory=True)
        self.assertNotEqual(self.run_export(output=self.root / "linked/bundle.zip").returncode, 0)
        (external / "run.json").touch()
        run_json = self.cascade / "run.json"
        run_json.unlink()
        run_json.symlink_to(external / "run.json")
        self.assertNotEqual(self.run_export().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_preserves_existing_output(self):
        self.output.write_bytes(b"existing")
        result = self.run_export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ya existe", result.stderr)
        self.assertEqual(self.output.read_bytes(), b"existing")


class PlotYoloHistoryTest(unittest.TestCase):
    def run_plot(self, history, output):
        return subprocess.run(
            [sys.executable, str(BASELINE / "plot_yolo_history.py"),
             "--history", str(history), "--output", str(output)],
            capture_output=True, text=True,
        )

    @unittest.skipUnless(find_spec("matplotlib"), "Matplotlib unavailable")
    def test_renders_original_csv_with_outlier_and_ap(self):
        from PIL import Image

        read_history = runpy.run_path(str(BASELINE / "plot_yolo_history.py"))["read_history"]
        epochs, series = read_history(HISTORY)
        self.assertEqual(epochs, list(range(1, 31)))
        self.assertEqual(series["val/cls_loss"][1], 50_317_300)
        self.assertEqual(len(series["val/map"]), 30)
        self.assertEqual(len(series["val/ap50"]), 30)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "plot.png"
            result = self.run_plot(HISTORY, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                image.verify()

    def test_rejects_empty_missing_and_existing_paths(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "plot.png"
            for history, target in (("", output), (HISTORY, ""), (output, output)):
                with self.subTest(history=history, target=target):
                    self.assertNotEqual(self.run_plot(history, target).returncode, 0)
                    self.assertFalse(output.exists())
            output.write_bytes(b"existing")
            self.assertNotEqual(self.run_plot(HISTORY, output).returncode, 0)
            self.assertEqual(output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
