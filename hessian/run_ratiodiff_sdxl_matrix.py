#!/usr/bin/env python3
"""Plan first; explicit --execute runs pilots -> configured runs -> reports.

No GPU reservation, process killing, auto-lease, or background launch at setup.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import signal
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.train_ratiodiff_sdxl import load_config, validate_config, atomic_json, digest

METRICS = ("pickscore", "hpsv2", "aesthetics_clip", "imagereward")


def run_group(jobs):
    """On any failure stop only our own child workers, not unrelated GPU jobs."""
    from contextlib import ExitStack
    import time
    with ExitStack() as stack:
        processes = []
        stage_logs = {}
        try:
            for command, env, log in jobs:
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(log.open("a", encoding="utf-8"))
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                processes.append(process)
                stage_logs[process.pid] = log
            pending = list(processes)
            while pending:
                for process in pending[:]:
                    code = process.poll()
                    if code is not None:
                        pending.remove(process)
                        if code:
                            log = stage_logs[process.pid]
                            try:
                                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                                tail = "\n".join(lines[-80:])
                            except OSError as error:
                                tail = f"<could not read log: {error}>"
                            raise RuntimeError(
                                f"Worker pid={process.pid} exit={code}; log={log}\n"
                                f"--- stage log tail ---\n{tail}\n--- end stage log tail ---"
                            )
                if pending:
                    time.sleep(1)
        finally:
            for process in processes:
                if process.poll() is None:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()


def find_resume(run):
    valid = [p for p in run.glob("checkpoint-*") if (p / "state.json").is_file()]
    return max(valid, key=lambda p: int(p.name.split("-")[-1])) if valid else None


def main(default_config="ratiodiff_xl_h100_1gpu.json"):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name(default_config)))
    p.add_argument("--gpus", default="0", help="Comma-separated physical GPU indices; H100 release defaults to one GPU")
    p.add_argument("--execute", action="store_true", help="Without this flag, print plan only")
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--eval-python", default=sys.executable, help="Existing working five-metric eval environment")
    args = p.parse_args()
    config_path = str(Path(args.config).resolve())
    c = load_config(config_path)
    variants = c["variants"]
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or not all(g.strip().isdigit() for g in gpus):
        raise ValueError("Provide distinct GPU indices")
    steps, accumulation, _ = validate_config(c, len(gpus))
    project = Path(__file__).resolve().parents[1]
    trainer = str(project / "hessian/train_ratiodiff_sdxl.py")
    evaluator = str(project / "hessian/evaluate_ratiodiff_sdxl.py")
    print(json.dumps(dict(model_family=c["model_family"], variants=variants, pairs=c["samples"], epochs=c["epochs"],
        optimizer_steps_per_run=steps, gpus=gpus, micro_batch=c["micro_batch"],
        accumulation=accumulation, effective_batch=c["effective_batch"],
        winner_auxiliary=c["loss"].get("winner_enabled", False), calibration_window=[c["calibration_start"], c["calibration_end"]],
        phases=["unit tests", "local asset preflight", f"{len(variants)} x 8-step pilots + infer + five metrics",
                f"{len(variants)} fresh full-budget runs + infer + five metrics", "comparison + compact archive"],
        mode="EXECUTE" if args.execute else "PLAN ONLY; no GPU accessed",
        hyperparameters="unvalidated starting values; pilot is not evidence of image quality"), indent=2))
    if not args.execute:
        return
    if os.name != "posix":
        raise RuntimeError("Execute on Linux GPU host; planning/testing also works on Windows")
    import fcntl
    root = Path(c["work_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This matrix already has a running pipeline")
    def status(message):
        line = datetime.now(timezone.utc).isoformat() + " | " + message
        print(line, flush=True)
        with (root / "pipeline-status.log").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1")
    if c["local_files_only"]:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    previous_config = root / "matrix_config.json"
    if previous_config.exists() and json.loads(previous_config.read_text()) != c:
        raise ValueError("Existing matrix has different config; use a new work_root")
    atomic_json(previous_config, c)
    identity = dict(world_size=len(gpus), source_hashes={name: digest(project/name) for name in
        ("ratiodiff_sdxl.py", "hessian/train_ratiodiff_sdxl.py", "hessian/streaming_pickapic.py",
         "hessian/evaluate_ratiodiff_sdxl.py", "modal_evaluate.py")})
    identity_path = root / "matrix_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Code/topology changed; use a new work_root to prevent stale result reuse")
    atomic_json(identity_path, identity)
    try:
        status("unit_tests_start")
        subprocess.run([sys.executable, "-m", "pytest", str(project/"tests/test_ratiodiff_sdxl.py"),
                        str(project/"tests/test_ratiodiff_pipeline.py"), "-q"], check=True, env=env)
        status("unit_tests_completed")
        subprocess.run([sys.executable, trainer, "--config", config_path, "--preflight"],
                       check=True, env=dict(env, WORLD_SIZE=str(len(gpus))))
        # Validate all held-out prompts before allocating models.
        subprocess.run([args.eval_python, evaluator, "prepare", "--config", config_path,
                        "--variant", "full", "--pilot"], check=True, env=env)
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                            "--format=csv"], text=True)
        print(inventory)
        atomic_json(root / "launch_inventory.json", dict(inventory=inventory, gpus=gpus))
        for pilot in ([True] if args.pilot_only else [True, False]):
            phase = "pilot" if pilot else "full"
            flags = ["--pilot"] if pilot else []
            for variant in variants:
                run = root / phase / "runs" / variant
                logdir = root / phase / "logs" / variant
                if not (run / "completed.json").exists():
                    status(f"{phase}_{variant}_training_start")
                    if len(gpus) == 1:
                        command = [sys.executable, trainer, "--config", config_path,
                                   "--variant", variant, *flags]
                    else:
                        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                   f"--nproc_per_node={len(gpus)}", trainer, "--config", config_path,
                                   "--variant", variant, *flags]
                    resume = find_resume(run)
                    if resume:
                        command += ["--resume", str(resume)]
                    run_group([(command, env, logdir/"train.log")])
                    status(f"{phase}_{variant}_training_completed")
                complete = root / phase / "eval" / variant / "complete.json"
                if complete.exists():
                    status(f"{phase}_{variant}_eval_reused")
                    continue
                def eval_command(stage, *extra):
                    return [args.eval_python, evaluator, stage, "--config", config_path,
                            "--variant", variant, *flags, *extra]
                status(f"{phase}_{variant}_generation_start")
                run_group([(eval_command("generate", "--rank", str(i), "--world-size", str(len(gpus))),
                            dict(env, CUDA_VISIBLE_DEVICES=g), logdir/f"generate_gpu{g}.log")
                           for i, g in enumerate(gpus)])
                for start in range(0, len(METRICS), len(gpus)):
                    status(f"{phase}_{variant}_scoring_{start}")
                    run_group([(eval_command("score", "--metric", metric), dict(env, CUDA_VISIBLE_DEVICES=gpus[i]),
                                logdir/f"score_{metric}.log") for i, metric in enumerate(METRICS[start:start+len(gpus)])])
                run_group([(eval_command("report"), env, logdir/"report.log")])
                status(f"{phase}_{variant}_eval_completed")
        phase = "pilot" if args.pilot_only else "full"
        summary = []
        for variant in variants:
            path = root/phase/"eval"/variant/"report/summary.json"
            summary.extend(dict(variant=variant, **r) for r in json.loads(path.read_text(encoding="utf-8")))
        atomic_json(root/phase/"comparison.json", summary)
        with (root/phase/"comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        # No model weights, optimizer states, dataset or full generated image folders.
        archive = root / f"ratiodiff_{c['model_family']}_{c['samples']}_{phase}_compact.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            for path in root.rglob("*"):
                rel = path.relative_to(root)
                if not path.is_file() or path == archive or "images" in rel.parts:
                    continue
                if any(part.startswith("checkpoint-") for part in rel.parts) or "unet" in rel.parts:
                    continue
                if path.suffix in (".json", ".jsonl", ".csv", ".md", ".log", ".jpg"):
                    tf.add(path, arcname=str(rel))
        status(f"completed archive={archive} sha256={digest(archive)}")
    except BaseException as exc:
        status(f"failed {type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
