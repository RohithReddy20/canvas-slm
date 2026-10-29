"""Run job.sh on a Colab GPU VM, signed in with the account kept under COLAB_HOME.

    python slm/colab_run.py start --gpu A100 --bundle <v7 Kaggle dataset folder>
    python slm/colab_run.py status
    python slm/colab_run.py watch      # waits for the end, collects results, stops the VM
    python slm/colab_run.py collect [--adapters q17 ...]
    python slm/colab_run.py stop

The job runs in a kernel cell, which keeps the VM assigned; progress is read by
downloading its log, since file transfer survives the dropped kernel connections
that `colab exec` meets while the GPU is busy. run.sh releases the VM itself after
a grace period, so it never keeps spending units unattended.
"""

import argparse
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parent / "runs" / "colab"
REMOTE = "/content/slm"
RUNS = "/content/runs"
MARK = "@@result "
COLAB_HOME = Path(os.environ.get("COLAB_HOME", Path.home() / ".colab-docextract"))
DATA_FILES = ("examples.jsonl", "sessions.jsonl", "selection-receipt.json")


def colab(*args, stdin=None, timeout=300, check=True):
    environment = {**os.environ, "HOME": str(COLAB_HOME), "REQUEST_TIMEOUT": "120"}
    try:
        result = subprocess.run(
            ["colab", *args],
            input=stdin,
            text=True,
            env=environment,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"no reply from `colab {args[0]}` in {timeout} s") from None
        return ""
    output = result.stdout + result.stderr
    if check and result.returncode:
        raise RuntimeError(f"`colab {args[0]}` failed:\n{output[-3000:]}")
    return output


def remote(session, code, timeout=180):
    output = colab("exec", "-s", session, stdin=code, timeout=timeout)
    line = next((x for x in reversed(output.splitlines()) if x.startswith(MARK)), None)
    if line is None:
        raise RuntimeError(f"no result from the VM:\n{output[-3000:]}")
    return json.loads(line.removeprefix(MARK))


def download(session, path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    colab("download", "-s", session, path, str(partial), timeout=900, check=False)
    if partial.exists():
        partial.replace(target)
        return True
    return False


def source_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted([*ROOT.glob("*.py"), *ROOT.glob("*.sh")]):
            archive.add(path, arcname=path.name)
    return buffer.getvalue()


def data_archive(bundle):
    target = LOCAL / "data.tar.xz"
    if not target.exists():
        files = []
        for name in DATA_FILES:
            found = [p for p in (bundle / name, bundle / "x" / name) if p.exists()]
            if not found:
                raise SystemExit(f"{name} is not in {bundle}")
            files.append(found[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w:xz") as archive:
            for path in files:
                archive.add(path, arcname=path.name)
    return target


def start(args):
    print(colab("usage").strip(), flush=True)
    data = data_archive(args.bundle)
    if args.reuse:
        # A finished job leaves a timer that releases the VM; it is stopped first.
        state = remote(
            args.session,
            f"""
import json, os, subprocess
subprocess.run(["pkill", "-f", "unassign"], check=False)
print({MARK!r} + json.dumps({{"data": os.path.exists("/content/data/v7/examples.jsonl")}}))
""",
        )
    else:
        print(colab("new", "-s", args.session, "--gpu", args.gpu, timeout=1200).strip())
        state = {"data": False}
    source = LOCAL / "source.tar.gz"
    source.write_bytes(source_archive())
    colab("upload", "-s", args.session, str(source), "/content/source.tar.gz", timeout=600)
    if not state["data"]:
        colab("upload", "-s", args.session, str(data), "/content/data.tar.xz", timeout=1800)
    print(
        remote(
            args.session,
            f"""
import json, os, tarfile, time
os.makedirs({RUNS!r}, exist_ok=True)
if os.path.exists({RUNS!r} + "/run.log"):
    os.rename({RUNS!r} + "/run.log", {RUNS!r} + f"/run-{{int(time.time())}}.log")
os.makedirs({REMOTE!r}, exist_ok=True)
with tarfile.open("/content/source.tar.gz") as archive:
    archive.extractall({REMOTE!r}, filter="data")
print({MARK!r} + json.dumps({{"files": sorted(os.listdir({REMOTE!r}))}}))
""",
        )
    )
    cell = f"""
import os, subprocess
with open({RUNS!r} + "/run.log", "a") as log:
    subprocess.Popen(
        ["bash", {REMOTE + "/run.sh"!r}], stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
        env={{**os.environ, "DEADLINE_MINUTES": "{args.deadline}", "PYTHONUNBUFFERED": "1"}},
    ).wait()
"""
    environment = {**os.environ, "HOME": str(COLAB_HOME), "REQUEST_TIMEOUT": "120"}
    for _ in range(3):
        # The CLI call holding the cell may lose its connection; the cell runs on.
        holder = subprocess.Popen(
            ["colab", "exec", "-s", args.session, "--timeout", str(args.deadline * 60 + 3600)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=environment,
            start_new_session=True,
        )
        holder.stdin.write(cell)
        holder.stdin.close()
        for _ in range(12):
            time.sleep(20)
            if download(args.session, f"{RUNS}/run.log", LOCAL / "run.log"):
                print(f"started; log at {RUNS}/run.log", flush=True)
                return
            if holder.poll() is not None:
                break
        holder.kill()
        print("the job did not start; trying again", flush=True)
    raise SystemExit("the job did not start")


def status(args, lines=12):
    if not download(args.session, f"{RUNS}/run.log", LOCAL / "run.log"):
        print("could not read the log", flush=True)
        return None
    log = (LOCAL / "run.log").read_text(errors="replace").replace("\r", "\n").splitlines()
    print("\n".join(line[:240] for line in log[-lines:]), flush=True)
    end = next((x for x in reversed(log) if x.startswith(("== done", "== exit"))), None)
    return end


def collect(args):
    names = " ".join(f"{name}/adapter" for name in args.adapters)
    remote(
        args.session,
        f"""
import glob, json, os, subprocess
os.chdir({RUNS!r})
files = [p for p in glob.glob("*.log") + glob.glob("*/*.json") if os.path.isfile(p)]
subprocess.run(["tar", "-czf", "/content/results.tar.gz", *files, *{names.split()!r}], check=True)
print({MARK!r} + json.dumps({{"files": len(files), "bytes": os.path.getsize("/content/results.tar.gz")}}))
""",
        timeout=600,
    )
    target = LOCAL / "results.tar.gz"
    if not download(args.session, "/content/results.tar.gz", target):
        raise SystemExit("could not download the results")
    with tarfile.open(target) as archive:
        archive.extractall(LOCAL / "results", filter="data")
    print(f"results in {LOCAL / 'results'}", flush=True)


def watch(args):
    while True:
        end = status(args, lines=3)
        if end:
            print(end, flush=True)
            collect(args)
            colab("stop", "-s", args.session, check=False)
            print("VM stopped", flush=True)
            return
        time.sleep(args.every * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=("start", "status", "watch", "collect", "stop"))
    parser.add_argument("--session", default="canvas-slm")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--deadline", type=int, default=300, help="minutes")
    parser.add_argument("--every", type=float, default=5, help="minutes between log reads")
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--reuse", action="store_true", help="start on the session's VM")
    args = parser.parse_args()
    if args.command == "start":
        if args.bundle is None:
            raise SystemExit("start needs --bundle")
        start(args)
    elif args.command == "status":
        status(args, lines=30)
    elif args.command == "watch":
        watch(args)
    elif args.command == "collect":
        collect(args)
    else:
        print(colab("stop", "-s", args.session, check=False).strip())


if __name__ == "__main__":
    main()
