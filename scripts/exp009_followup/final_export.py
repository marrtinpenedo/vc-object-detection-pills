"""Export the completed final evaluation and its SLURM accounting without weights or images."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

FILES = (
    "launch.json", "eval.job", "verified.ok", "yolo.train.sacct", "cascade.train.sacct",
    *(f"{name}/{file}" for name in ("yolo", "cascade")
      for file in ("metrics.json", "predictions.json", "pr_curves.json", "pr_curves.png")),
)


def export(root, output):
    job = (root / "eval.job").read_text().strip()
    launch = json.loads((root / "launch.json").read_text())
    if not job.isdecimal() or launch["job_id"] != job or launch["split"] != "test" or not launch.get("selection_sha256"):
        raise ValueError("El directorio no pertenece a la evaluación final fijada")
    accounting = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", job, "-o", "JobIDRaw,State,ExitCode"], text=True)
    if accounting.strip().removesuffix("|") != f"{job}|COMPLETED|0:0":
        raise ValueError(f"Se esperaba {job}|COMPLETED|0:0; figura {accounting.strip()}")
    entries = {name: (root / name).read_bytes() for name in FILES}
    entries["final-eval-sacct.txt"] = accounting.encode()
    inventory = {name: hashlib.sha256(content).hexdigest() for name, content in entries.items()}
    entries["export.sha256.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, content in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.root, args.output)
