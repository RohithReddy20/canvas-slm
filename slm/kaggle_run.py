"""Run a job script in a Kaggle kernel on the account's free GPUs (two T4s).

    python slm/kaggle_run.py start --job scripts.sh --name scripts-t4 --input runs/colab/real-test-v1.jsonl
    python slm/kaggle_run.py status --name scripts-t4
    python slm/kaggle_run.py collect --name scripts-t4

The kernel script carries the slm/ code. The v7 data comes from its Kaggle dataset,
small private inputs from a private dataset, and models stay in the kernel's output
on Kaggle. Only logs and reports come back.
"""

import argparse
import base64
import io
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parent / "runs" / "kaggle"
OWNER = "rohithresearch"
DATASET = f"{OWNER}/canvas-270m-v7-accuracy-workflow"
INPUTS = f"{OWNER}/canvas-slm-inputs"
INPUT_LIMIT = 5 * 2**20
SMALL = r"(\.log|job-status\.json|report[\w-]*\.json|training\.json|autotune\.json)$"

RUNNER = '''import base64, io, json, os, shutil, subprocess, tarfile, time, zipfile
from pathlib import Path

CODE, JOB, EXTRA = {code!r}, {job!r}, {extra!r}
DATA_FILES = ("examples.jsonl", "sessions.jsonl", "selection-receipt.json")
code, data, inputs = Path("/tmp/slm"), Path("/tmp/data/v7"), Path("/tmp/input")
for folder in (code, data, inputs):
    folder.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:xz").extractall(code)
mounted = Path("/kaggle/input")
for bundle in mounted.rglob("canvas-training.bundle"):
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if Path(name).name in DATA_FILES:
                (data / Path(name).name).write_bytes(archive.read(name))
    receipt = bundle.with_name("selection-receipt.json")
    if not (data / receipt.name).exists() and receipt.exists():
        shutil.copy(receipt, data)
for path in mounted.rglob("*"):
    if path.is_file() and "canvas-slm-inputs" in str(path):
        shutil.copy(path, inputs / path.name)
print("data", sorted(p.name for p in data.iterdir()), "inputs", sorted(p.name for p in inputs.iterdir()), flush=True)
working = Path("/kaggle/working")
environment = {{**os.environ, "WORK": "/tmp/work", "DATA": str(data), "RUNS": str(working / "runs"),
               "INPUT": str(inputs), **EXTRA}}
Path("/tmp/work").mkdir(exist_ok=True)
started = time.time()
with open(working / "job.log", "w") as log:
    job = subprocess.Popen(["bash", str(code / JOB)], env=environment, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    for line in job.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
# Exit cleanly either way, so the kernel keeps its output; job-status.json has the result.
(working / "job-status.json").write_text(json.dumps(
    {{"job": JOB, "exit_code": job.wait(), "minutes": round((time.time() - started) / 60, 1)}}))
'''


def kaggle(*args, check=True):
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    output = "\n".join(
        line for line in (result.stdout + result.stderr).splitlines() if "outdated `kaggle`" not in line
    )
    if check and result.returncode:
        raise SystemExit(f"`kaggle {' '.join(args[:2])}` failed:\n{output[-2000:]}")
    return output


def code_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        for path in sorted(ROOT.iterdir()):
            if path.suffix in (".py", ".sh"):
                archive.add(path, arcname=path.name)
    return base64.b64encode(buffer.getvalue()).decode()


def upload_inputs(paths):
    """Version the private inputs dataset with these files and wait until it is ready."""
    if sum(p.stat().st_size for p in paths) > INPUT_LIMIT:
        raise SystemExit("Inputs are limited to 5 MB; large files stay off this connection.")
    with tempfile.TemporaryDirectory() as folder:
        for path in paths:
            (Path(folder) / path.name).write_bytes(path.read_bytes())
        (Path(folder) / "dataset-metadata.json").write_text(json.dumps(
            {"title": "canvas slm inputs", "id": INPUTS, "licenses": [{"name": "CC0-1.0"}]}))
        exists = "ready" in kaggle("datasets", "status", INPUTS, check=False)
        if exists:
            kaggle("datasets", "version", "-p", folder, "-m", "inputs", "-q")
        else:
            kaggle("datasets", "create", "-p", folder, "-q")
    for _ in range(60):
        time.sleep(10)
        if "ready" in kaggle("datasets", "status", INPUTS, check=False):
            return
    raise SystemExit(f"{INPUTS} did not become ready.")


def start(args):
    if args.input:
        upload_inputs([Path(p).resolve() for p in args.input])
    slug = f"canvas-slm-{args.name}"
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "run.py").write_text(RUNNER.format(code=code_archive(), job=args.job, extra=dict(args.env)))
        (Path(folder) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "machine_shape": args.accelerator,
            "dataset_sources": [DATASET, *([INPUTS] if args.input or args.with_inputs else [])],
            "kernel_sources": [f"{OWNER}/{s}" for s in args.reuse], "competition_sources": [],
            "model_sources": [],
        }))
        print(kaggle("kernels", "push", "-p", folder, "--accelerator", args.accelerator,
                     "-t", str(args.timeout)))


def status(args):
    print(kaggle("kernels", "status", f"{OWNER}/canvas-slm-{args.name}"))


def collect(args):
    target = LOCAL / args.name
    target.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{OWNER}/canvas-slm-{args.name}", "-p", str(target),
           "--file-pattern", SMALL, "-o", "-q")
    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(target)}  {path.stat().st_size / 1e3:.0f} kB")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    begin = commands.add_parser("start")
    begin.add_argument("--job", default="scripts.sh")
    begin.add_argument("--input", nargs="*", default=[], help="small files for the inputs dataset")
    begin.add_argument("--with-inputs", action="store_true", help="mount the inputs dataset as it is")
    begin.add_argument("--reuse", nargs="*", default=[], help="earlier kernels whose output to mount")
    begin.add_argument("--accelerator", default="NvidiaTeslaT4")
    begin.add_argument("--env", nargs="*", default=[], type=lambda x: tuple(x.split("=", 1)),
                       help="NAME=VALUE pairs for the job")
    begin.add_argument("--timeout", type=int, default=12 * 3600)
    for command in (begin, commands.add_parser("status"), commands.add_parser("collect")):
        command.add_argument("--name", required=True)
    args = parser.parse_args()
    {"start": start, "status": status, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
