#!/usr/bin/env python3
"""Run 10-sample attack calibration jobs for quality-budget comparison."""

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
DEFAULT_ROOT = Path("/data2/liyanlei/stego_attack_data/attack_runs/quality_budget_20260527")
PYTHON = Path("/data2/liyanlei/envs/stego_attack/bin/python")
ENV_BIN = Path("/data2/liyanlei/envs/stego_attack/bin")
HF_HOME = Path("/data2/liyanlei/huggingface")
TORCH_HOME = Path("/data2/liyanlei/torch")

METHODS = ["gsd_cifar10", "mas_grdh", "mddm_128_pilot", "pulsar"]
HEAVY_METHODS = ["cross"]
ATTACK_FACTORS = {
    "resize": ["0.25", "0.375", "0.5", "0.75", "1.25", "1.5"],
    "jpeg": ["20", "30", "40", "50", "70", "80", "90", "95"],
    "mblur": ["0.25", "0.5", "0.75", "3", "5", "7", "9", "11", "13"],
    "gblur": ["0.25", "0.5", "0.75", "1", "1.5", "2", "3", "5", "7", "9", "11", "13"],
    "regen_vae": ["6", "5", "4", "3", "2", "1"],
    "unmarker": ["high_smoke_25", "high_smoke_100"],
    "ads_fgsm": ["0.005", "0.01", "0.02", "0.04", "0.06", "0.08"],
    "ads_qdir": ["0.005", "0.01", "0.02", "0.04", "0.06", "0.08", "0.10"],
    "ecrs_fast": ["0.75", "1.0", "1.25", "1.5", "2.0", "2.5", "3.0", "3.5"],
    "ecrs_diff": ["0.10", "0.25", "0.40", "0.60", "0.85", "1.10", "1.50", "2.20", "3.00", "4.00"],
}


@dataclass(frozen=True)
class Job:
    method: str
    attack: str
    factor: str
    count: int

    @property
    def safe_factor(self) -> str:
        return self.factor.replace(".", "_")

    @property
    def name(self) -> str:
        return f"{self.method}_{self.attack}_{self.safe_factor}_{self.count}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--attacks", default="resize,jpeg,mblur,gblur,regen_vae,ads_fgsm,ads_qdir,ecrs_fast,ecrs_diff")
    parser.add_argument("--include-heavy", action="store_true", help="Include CRoSS calibration jobs.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
    env["ADS_LOCAL_FILES_ONLY"] = "1"
    env["ADS_TIMESTEP"] = "1"
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    return env


def unmarker_args(factor: str) -> list[str]:
    parts = factor.split("_")
    if len(parts) != 3:
        raise ValueError(f"unmarker factor must be <stage>_<profile>_<iterations>, got {factor!r}")
    stage, profile, iterations = parts
    return ["--unmarker-stage", stage, "--unmarker-profile", profile, "--unmarker-iterations", iterations]


def attack_args(job: Job) -> list[str]:
    if job.attack == "resize":
        return ["--attack-kind", "resize", "--resize-factor", job.factor]
    if job.attack in {"jpeg", "mblur", "gblur"}:
        return ["--attack-kind", job.attack, "--attack-factor", job.factor]
    if job.attack == "regen_vae":
        if job.method == "gsd_cifar10":
            return ["--attack-kind", "regen_vae", "--regen-quality", job.factor]
        return ["--attack-kind", "regen_vae", "--attack-factor", job.factor]
    if job.attack == "unmarker":
        return ["--attack-kind", "unmarker", *unmarker_args(job.factor)]
    if job.attack in {"ads_fgsm", "ads_qdir", "ecrs_fast", "ecrs_diff"}:
        return ["--attack-kind", job.attack, "--attack-factor", job.factor]
    raise ValueError(f"unsupported attack: {job.attack}")


def command_for(job: Job, out_dir: Path, force: bool) -> list[str]:
    force_flag = ["--force"] if force else []
    save_flag = ["--save-images"]
    if job.method == "cross":
        return [
            str(PYTHON),
            "scripts/run_cross_identity.py",
            "--count",
            str(job.count),
            "--num-steps",
            "50",
            *attack_args(job),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if job.method == "gsd_cifar10":
        return [
            str(PYTHON),
            "scripts/run_gsd_identity.py",
            "--count",
            str(job.count),
            "--timesteps",
            "1000",
            "--device",
            "cuda",
            *attack_args(job),
            *save_flag,
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if job.method == "mas_grdh":
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
            *attack_args(job),
            *save_flag,
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if job.method == "pulsar":
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
            *attack_args(job),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    if job.method == "mddm_128_pilot":
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
            *attack_args(job),
            "--output-dir",
            str(out_dir),
            *force_flag,
        ]
    raise ValueError(f"unsupported method: {job.method}")


def make_jobs(methods: list[str], attacks: list[str], count: int) -> list[Job]:
    jobs = []
    for attack in attacks:
        if attack not in ATTACK_FACTORS:
            raise ValueError(f"unsupported attack: {attack}")
        for factor in ATTACK_FACTORS[attack]:
            for method in methods:
                jobs.append(Job(method=method, attack=attack, factor=factor, count=count))
    return jobs


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    logs = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if args.include_heavy:
        methods.extend(method for method in HEAVY_METHODS if method not in methods)
    attacks = [attack.strip() for attack in args.attacks.split(",") if attack.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    pending = make_jobs(methods, attacks, args.count)
    active: dict[str, tuple[Job, subprocess.Popen[bytes], object]] = {}
    failures: list[tuple[Job, int]] = []
    print(f"[quality-queue] root={root}")
    print(f"[quality-queue] methods={','.join(methods)} attacks={','.join(attacks)} jobs={len(pending)}")

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
            print(f"[quality-queue] START gpu={gpu} job={job.name}", flush=True)
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
            print(f"[quality-queue] DONE gpu={gpu} job={job.name} exit={code} records={total}", flush=True)
            if code != 0:
                failures.append((job, code))
            unlock(root / job.name)
            del active[gpu]

    if failures:
        for job, code in failures:
            print(f"[quality-queue] FAILED job={job.name} exit={code}", flush=True)
        return 1
    print("[quality-queue] all jobs finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
