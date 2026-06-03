#!/usr/bin/env python3
"""Run the selected quality-budget attack matrix at formal sample counts."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from selected_attack_matrix import (  # noqa: E402
    SelectedAttack,
    attack_provenance_for,
    baseline_provenance_for,
    default_count_for,
    heldout_count_for,
    selected_for_methods,
)


DEFAULT_ROOT = Path("/data2/liyanlei/stego_attack_data/attack_runs/selected_quality_budget_20260527")
PYTHON = Path("/data2/liyanlei/envs/stego_attack/bin/python")
ENV_BIN = Path("/data2/liyanlei/envs/stego_attack/bin")
HF_HOME = Path("/data2/liyanlei/huggingface")
TORCH_HOME = Path("/data2/liyanlei/torch")


@dataclass(frozen=True)
class Job:
    spec: SelectedAttack
    count: int

    @property
    def name(self) -> str:
        return f"{self.spec.method}_{self.spec.name_part}_{self.count}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--methods", default="", help="Comma-separated method subset. Default: all selected methods.")
    parser.add_argument("--method-counts", default="", help="Comma-separated overrides, e.g. gsd_cifar10=100,mas_grdh=100.")
    parser.add_argument("--count", type=int, default=None, help="Override every selected method count.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_method_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not raw.strip():
        return counts
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid method-count item: {item!r}")
        method, value = item.split("=", 1)
        counts[method.strip()] = int(value.strip())
    return counts


def result_total(out_dir: Path) -> int:
    total = 0
    for name in ("identity_results.csv", "identity_failures.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            total += sum(1 for _ in csv.DictReader(handle))
    return total


def lock_dir(out_dir: Path) -> Path:
    return out_dir.with_name(out_dir.name + ".running")


def try_lock(out_dir: Path) -> bool:
    try:
        lock_dir(out_dir).mkdir()
        return True
    except FileExistsError:
        return False


def unlock(out_dir: Path) -> None:
    shutil.rmtree(lock_dir(out_dir), ignore_errors=True)


def base_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HF_HOME"] = str(HF_HOME)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    env["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
    env["DIFFUSERS_CACHE"] = str(HF_HOME / "diffusers")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["TORCH_HOME"] = str(TORCH_HOME)
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    return env


def attack_args(spec: SelectedAttack) -> list[str]:
    if spec.attack == "resize":
        return ["--attack-kind", "resize", "--resize-factor", spec.factor]
    if spec.attack in {"jpeg", "mblur", "gblur"}:
        return ["--attack-kind", spec.attack, "--attack-factor", spec.factor]
    if spec.attack == "regen_vae":
        if spec.method == "gsd_cifar10":
            return ["--attack-kind", "regen_vae", "--regen-quality", spec.factor]
        return ["--attack-kind", "regen_vae", "--attack-factor", spec.factor]
    if spec.attack == "unmarker":
        stage, profile, iterations = spec.factor.split("_", 2)
        return [
            "--attack-kind",
            "unmarker",
            "--unmarker-stage",
            stage,
            "--unmarker-profile",
            profile,
            "--unmarker-iterations",
            iterations,
        ]
    if spec.attack == "ads":
        return ["--attack-kind", f"ads_{spec.factor}"]
    raise ValueError(f"unsupported selected attack: {spec.attack}")


def command_for(job: Job, out_dir: Path, force: bool) -> list[str]:
    spec = job.spec
    force_flag = ["--force"] if force else []
    save_flag = ["--save-images"]
    if spec.method == "cross":
        return [
            str(PYTHON),
            "scripts/run_cross_identity.py",
            "--count",
            str(job.count),
            "--num-steps",
            "50",
            *attack_args(spec),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if spec.method == "gsd_cifar10":
        return [
            str(PYTHON),
            "scripts/run_gsd_identity.py",
            "--count",
            str(job.count),
            "--timesteps",
            "1000",
            "--device",
            "cuda",
            *attack_args(spec),
            *save_flag,
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if spec.method == "mas_grdh":
        return [
            str(PYTHON),
            "scripts/run_mas_grdh_identity.py",
            "--count",
            str(job.count),
            "--dpm-steps",
            "20",
            "--scale",
            "5.0",
            "--gpu",
            "cuda:0",
            *attack_args(spec),
            *save_flag,
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if spec.method == "mddm_128_pilot":
        return [
            str(PYTHON),
            "scripts/run_mddm_identity.py",
            "--count",
            str(job.count),
            "--steps",
            "20",
            "--guidance-scale",
            "1.0",
            "--payload-bytes",
            "128",
            *attack_args(spec),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if spec.method == "pulsar":
        return [
            str(PYTHON),
            "scripts/run_pulsar_identity.py",
            "--count",
            str(job.count),
            "--steps",
            "50",
            "--region-estimate-samples",
            "1",
            "--hist-bins",
            "100",
            "--sample-dtype",
            "uint8",
            *attack_args(spec),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    raise ValueError(f"unsupported selected method: {spec.method}")


def make_jobs(args: argparse.Namespace) -> list[Job]:
    methods = {item.strip() for item in args.methods.split(",") if item.strip()} or None
    overrides = parse_method_counts(args.method_counts)
    jobs = []
    for spec in selected_for_methods(methods):
        count = args.count if args.count is not None else overrides.get(spec.method, default_count_for(spec.method))
        jobs.append(Job(spec=spec, count=int(count)))
    return jobs


def write_matrix_manifest(root: Path, jobs: list[Job]) -> None:
    if not jobs:
        raise ValueError("selected attack queue has no jobs")
    rows = [
        {
            "method": job.spec.method,
            "attack": job.spec.attack,
            "factor": job.spec.factor,
            "label": job.spec.label,
            "metric": job.spec.metric,
            "provenance": job.spec.provenance,
            "baseline_provenance": baseline_provenance_for(job.spec.method),
            "attack_provenance": attack_provenance_for(job.spec),
            "count": job.count,
            "paper_heldout_count": heldout_count_for(job.spec.method) if job.count == default_count_for(job.spec.method) else "",
            "excluded_calibration_samples": 10 if job.count == default_count_for(job.spec.method) else "",
            "output_dir": str(root / job.name),
            "note": job.spec.note,
        }
        for job in jobs
    ]
    path = root / "selected_attack_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    logs = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args)
    write_matrix_manifest(root, jobs)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    print(f"[selected-queue] root={root}")
    print(f"[selected-queue] jobs={len(jobs)} gpus={','.join(gpus)}")
    for job in jobs:
        print(f"[selected-queue] job={job.name} count={job.count}")
    if args.dry_run:
        for job in jobs:
            print("COMMAND " + " ".join(command_for(job, root / job.name, args.force)))
        return 0

    pending = jobs[:]
    active: dict[str, tuple[Job, subprocess.Popen[bytes], object]] = {}
    failures: list[tuple[Job, int]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = None
            out_dir = None
            while pending:
                candidate = pending.pop(0)
                candidate_out = root / candidate.name
                if not args.force and result_total(candidate_out) >= candidate.count:
                    continue
                if try_lock(candidate_out):
                    job = candidate
                    out_dir = candidate_out
                    break
            if job is None or out_dir is None:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for(job, out_dir, args.force)
            log_path = logs / f"{job.name}.log"
            log_handle = log_path.open("ab")
            log_handle.write(("COMMAND " + " ".join(cmd) + "\n").encode("utf-8"))
            log_handle.flush()
            print(f"[selected-queue] START gpu={gpu} job={job.name}", flush=True)
            proc = subprocess.Popen(
                cmd,
                cwd=WORKSPACE_ROOT,
                env=base_env(gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (job, proc, log_handle)

        time.sleep(args.poll_seconds)
        for gpu, (job, proc, log_handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log_handle.close()
            total = result_total(root / job.name)
            print(f"[selected-queue] DONE gpu={gpu} job={job.name} exit={code} records={total}", flush=True)
            if code != 0:
                failures.append((job, code))
            unlock(root / job.name)
            del active[gpu]

    if failures:
        for job, code in failures:
            print(f"[selected-queue] FAILED job={job.name} exit={code}", flush=True)
        return 1
    print("[selected-queue] all jobs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
