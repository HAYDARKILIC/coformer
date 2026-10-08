#!/usr/bin/env python3
"""Run the CoFormer experiment notebook to completion on a GPU machine with no interaction.

Usage (Linux with an NVIDIA GPU of at least 38 GB, e.g. an A100 40 GB), from the repository root:

    python3 scripts/run_headless.py --state-zip CoFormer_state.zip   # first run: unpack a state
    python3 scripts/run_headless.py                                  # later: resume
    python3 scripts/run_headless.py --workdir .  # with ./CoFormer_run already present
                                                 # (e.g. a copy of results/CoFormer)

What it does:
  1. Checks for a CUDA GPU and installs the notebook runner (nbclient, ipykernel) if missing.
  2. Unpacks the state zip into ./CoFormer_run, which is where the notebook looks when it is not
     on Colab. An existing ./CoFormer_run is never overwritten.
  3. Executes notebooks/CoFormer_experiments.ipynb with an unlimited session budget
     (COFORMER_SESSION_HOURS), prints progress every 10 minutes and saves each executed notebook
     in logs/.
  4. If the kernel dies or a cell fails, it runs the notebook again (every finished run is
     checkpointed, so nothing is repeated), up to --max-rounds times, and stops early if a round
     makes no progress.
  5. When the second follow-up study (60 runs) and family H (59 runs) are complete, it writes
     CoFormer_state_complete.zip into the working directory.
"""
import argparse, json, os, shutil, subprocess, sys, threading, time, zipfile

PROBE_FP = "99f69cb71d"                 # fingerprint of the stored study (Cell 10)
NEED = {"fs2": 60, "famH": 59}


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def counts(base):
    out = {}
    for key, fn in (("fs2", f"results_fs2_{PROBE_FP}.csv"), ("famH", f"results_famH_{PROBE_FP}.csv")):
        p = os.path.join(base, "checkpoints", fn)
        try:
            with open(p) as fh:
                out[key] = max(0, sum(1 for _ in fh) - 1)
        except FileNotFoundError:
            out[key] = 0
    return out


def complete(c):
    return all(c[k] >= NEED[k] for k in NEED)


def ensure_packages():
    try:
        import nbclient, nbformat, ipykernel  # noqa: F401
    except ImportError:
        log("installing nbclient, nbformat, ipykernel ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbclient", "nbformat",
                        "ipykernel"], check=True)
        subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name",
                        "python3"], check=False)


def unpack_state(zip_path, base):
    if os.path.isdir(os.path.join(base, "checkpoints")):
        log(f"{base} already holds a state; the zip is NOT unpacked over it.")
        return
    tmp = base + "_unpack_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    src = None
    for root, dirs, _ in os.walk(tmp):
        if "checkpoints" in dirs and os.path.exists(os.path.join(root, "checkpoints", "results.csv")):
            src = root
            break
    if src is None:
        sys.exit("The zip holds no folder with checkpoints/results.csv; is it the CoFormer state?")
    shutil.move(src, base)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"state unpacked into {base}")


def zip_state(base, out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("checkpoints", "outputs"):
            for root, _, files in os.walk(os.path.join(base, sub)):
                for fn in files:
                    if fn.endswith(".tmp"):
                        continue
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.join("CoFormer", os.path.relpath(full, base)))
    log(f"final state written to {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--notebook", default=os.path.join(here, "..", "notebooks", "CoFormer_experiments.ipynb"))
    ap.add_argument("--state-zip", default=None, help="zip of the CoFormer folder (first run only)")
    ap.add_argument("--workdir", default=os.getcwd(), help="directory that holds CoFormer_run/ (default: current directory)")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)   # tests only
    a = ap.parse_args()

    try:
        import torch
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit("No CUDA GPU visible. This runner is for a GPU machine.")
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    except ImportError:
        sys.exit("PyTorch with CUDA is required (pip install torch).")
    ensure_packages()
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, DeadKernelError

    base = os.path.join(a.workdir, "CoFormer_run")
    if a.state_zip:
        unpack_state(a.state_zip, base)
    if not os.path.exists(os.path.join(base, "checkpoints", "results.csv")):
        sys.exit(f"No state in {base}. Pass --state-zip <CoFormer backup zip> on the first run.")
    os.makedirs(os.path.join(a.workdir, "logs"), exist_ok=True)
    os.environ["COFORMER_SESSION_HOURS"] = "10000"     # no session limit on a dedicated machine

    stop = threading.Event()
    def monitor():
        while not stop.wait(600):
            c = counts(base)
            log(f"progress: follow-up study 2 {c['fs2']}/60, family H {c['famH']}/59")
    threading.Thread(target=monitor, daemon=True).start()

    prev = None
    for rnd in range(1, a.max_rounds + 1):
        c = counts(base)
        log(f"round {rnd}: follow-up study 2 {c['fs2']}/60, family H {c['famH']}/59")
        nb = nbformat.read(a.notebook, as_version=4)
        client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": a.workdir}})
        def _start(cell, cell_index, **kw):
            if cell.cell_type == "code":
                head = "".join(cell.source).strip().split("\n")[0][:90]
                log(f"  cell {cell_index}: {head}")
        client.on_cell_start = _start
        err = None
        try:
            client.execute()
        except (CellExecutionError, DeadKernelError) as e:
            err = e
        except Exception as e:                                  # kernel crash, CUDA error, ...
            err = e
        out_nb = os.path.join(a.workdir, "logs", f"executed_round{rnd}.ipynb")
        nbformat.write(nb, out_nb)
        if err is not None:
            log(f"round {rnd} stopped with an error (details in {out_nb}):\n{str(err)[-3000:]}")
        c = counts(base)
        if complete(c):
            log("ALL STAGES COMPLETE.")
            if err is not None:
                log(f"NOTE: a cell failed after the stages finished; the runs are complete, but "
                    f"send {out_nb} as well.")
            stop.set()
            zip_state(base, os.path.join(a.workdir, "CoFormer_state_complete.zip"))
            return 0
        if prev is not None and c == prev and err is not None:
            stop.set()
            sys.exit(f"No progress in round {rnd} and the round failed; stopping. "
                     f"Send {out_nb} and headless.log.")
        prev = c
    stop.set()
    c = counts(base)
    if complete(c):
        zip_state(base, os.path.join(a.workdir, "CoFormer_state_complete.zip"))
        return 0
    sys.exit(f"Stopped after {a.max_rounds} rounds: {c}. Send the logs/ folder and headless.log.")


if __name__ == "__main__":
    sys.exit(main())
