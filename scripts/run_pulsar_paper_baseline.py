#!/usr/bin/env python3
"""Run paper-style Pulsar gateway-sanitization attacks.

This runner intentionally uses the official raw Pulsar generate/reveal path
instead of the region/ECC identity protocol. It is for checking whether basic
Pulsar behavior is in the same regime as ADS-style papers that report BER-based
failure under resize/JPEG/blur/ADS attacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from attack_common import attack_roundtrip_tensor_minus1_1  # noqa: E402
from pulsar_native_utils import (  # noqa: E402
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_HOME,
    DEFAULT_PULSAR_REF,
    configure_sage,
    ensure_hf_cache,
    load_official_pulsar,
)


PROTOCOL_SEED = "pulsar-paper-baseline-v1-20260528"
PAPER_SUCCESS_BER_THRESHOLD = 0.48
DEFAULT_ATTACKS = "identity,jpeg90,jpeg70,resize224,blur0.5,blur1.0,diffusion1step,ads_fgsm0.01,ads_qdir0.01"
PULSAR_REPOS = {
    "church": "google/ddpm-church-256",
    "celebahq": "google/ddpm-celebahq-256",
    "bedroom": "google/ddpm-bedroom-256",
    "cat": "google/ddpm-cat-256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/data2/liyanlei/stego_attack_data/attack_runs/pulsar_paper_baseline_10_20260528",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--model", default="church")
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--payload-bytes", type=int, default=8192)
    parser.add_argument("--attacks", default=DEFAULT_ATTACKS)
    parser.add_argument("--ber-threshold", type=float, default=PAPER_SUCCESS_BER_THRESHOLD)
    parser.add_argument("--key", default="E" * 64)
    parser.add_argument("--seed-prefix", default="")
    parser.add_argument("--reference-dir", default=str(DEFAULT_PULSAR_REF))
    parser.add_argument("--hf-cache-dir", default=str(DEFAULT_HF_HOME))
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--sage-bin", default="/data2/liyanlei/envs/stego_attack/bin/sage")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def derive_bytes(sample_index: int, length: int) -> bytes:
    return hashlib.shake_256(
        f"{PROTOCOL_SEED}|{sample_index:06d}|{length}".encode("utf-8")
    ).digest(length)


def attack_specs(specs: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for raw in specs.split(","):
        spec = raw.strip().lower()
        if not spec:
            continue
        normalized = spec.replace("-", "_")
        if spec == "identity":
            out.append({"label": "identity", "kind": "identity", "factor": None})
        elif spec in {"diffusion", "diffusion1step", "diffusion_1step"}:
            out.append({"label": "diffusion_1step_t1_eps0", "kind": "diffusion_1step", "factor": 0.0})
        elif spec == "resize224":
            out.append({"label": "resize_224", "kind": "resize", "factor": 224 / 256})
        elif spec.startswith("resize"):
            target = float(spec.removeprefix("resize"))
            factor = target / 256 if target > 4 else target
            out.append({"label": f"resize_{target:g}".replace(".", "_"), "kind": "resize", "factor": factor})
        elif spec.startswith("jpeg"):
            quality = float(spec.removeprefix("jpeg").removeprefix("_q").removeprefix("q"))
            out.append({"label": f"jpeg_q{quality:g}".replace(".", "_"), "kind": "jpeg", "factor": quality})
        elif spec.startswith("blur") or spec.startswith("gblur"):
            sigma = float(spec.removeprefix("gblur").removeprefix("blur"))
            out.append({"label": f"blur_sigma{sigma:g}".replace(".", "_"), "kind": "gblur", "factor": sigma})
        elif normalized.startswith("ads_fgsm") or normalized.startswith("ads_qdir"):
            variant = "fgsm" if normalized.startswith("ads_fgsm") else "qdir"
            raw_eps = normalized.removeprefix(f"ads_{variant}").removeprefix("_").removeprefix("eps")
            epsilon = float(raw_eps) if raw_eps else 0.01
            kind = f"ads_{variant}"
            out.append({"label": f"{kind}_eps{epsilon:g}".replace(".", "_"), "kind": kind, "factor": epsilon})
        elif normalized.startswith("ads"):
            raise ValueError(f"unknown ADS attack spec: {raw}; use ads_fgsm or ads_qdir")
        else:
            raise ValueError(f"unknown attack spec: {raw}")
    if not out:
        raise ValueError("no attacks requested")
    return out


def bit_errors(expected: bytes, recovered: bytes) -> tuple[int, int, float]:
    bit_count = max(len(expected), len(recovered)) * 8
    errors = 0
    for a, b in zip(expected, recovered):
        errors += (a ^ b).bit_count()
    if len(expected) != len(recovered):
        errors += abs(len(expected) - len(recovered)) * 8
    ber = errors / bit_count if bit_count else 0.0
    return errors, bit_count, ber


def tensor_to_uint8(tensor_bchw: torch.Tensor) -> np.ndarray:
    array = (
        ((tensor_bchw.detach().float().cpu()[0] / 2.0 + 0.5).clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return array


def psnr_uint8(reference: np.ndarray, attacked: np.ndarray) -> float:
    diff = reference.astype(np.float64) - attacked.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_uint8(reference: np.ndarray, attacked: np.ndarray) -> float | str:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(reference, attacked, channel_axis=2, data_range=255))
    except Exception:
        return ""


def make_lpips(device: torch.device | str):
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def lpips_tensor(model, reference: torch.Tensor, attacked: torch.Tensor) -> float:
    with torch.no_grad():
        return float(model(reference.detach(), attacked.detach()).item())


def apply_attack(hidden: torch.Tensor, spec: dict[str, object]) -> torch.Tensor:
    kind = str(spec["kind"])
    if kind == "identity":
        return hidden
    if kind == "resize":
        return attack_roundtrip_tensor_minus1_1(hidden, "resize", resize_factor=float(spec["factor"]))
    if kind == "jpeg":
        return attack_roundtrip_tensor_minus1_1(hidden, "jpeg", attack_factor=float(spec["factor"]))
    if kind == "gblur":
        return attack_roundtrip_tensor_minus1_1(hidden, "gblur", attack_factor=float(spec["factor"]))
    if kind == "diffusion_1step":
        return attack_roundtrip_tensor_minus1_1(hidden, "ads_qdir", attack_factor=0.0)
    if kind in {"ads_fgsm", "ads_qdir"}:
        return attack_roundtrip_tensor_minus1_1(hidden, kind, attack_factor=float(spec["factor"]))
    raise ValueError(f"unsupported attack kind: {kind}")


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else ""


def summarize(rows: list[dict[str, object]], out_dir: Path, ber_threshold: float) -> None:
    labels = sorted({str(row["attack_label"]) for row in rows})
    summary_rows = []
    for label in labels:
        subset = [row for row in rows if row["attack_label"] == label]
        failures = sum(1 for row in subset if str(row["paper_failure"]).lower() == "true")
        exact = sum(1 for row in subset if str(row["exact_match"]).lower() == "true")
        psnrs = [float(row["psnr"]) for row in subset if row["psnr"] != "inf"]
        ssims = [float(row["ssim"]) for row in subset if row["ssim"] != ""]
        lpips_values = [float(row["lpips"]) for row in subset if row["lpips"] != ""]
        bers = [float(row["ber"]) for row in subset]
        summary_rows.append(
            {
                "attack_label": label,
                "records": len(subset),
                "paper_failures": failures,
                "paper_failure_rate": failures / len(subset) if subset else "",
                "exact_matches": exact,
                "exact_match_rate": exact / len(subset) if subset else "",
                "ber_mean": mean(bers),
                "psnr_mean": "inf" if label == "identity" else mean(psnrs),
                "ssim_mean": mean(ssims),
                "lpips_mean": mean(lpips_values),
            }
        )

    csv_path = out_dir / "paper_baseline_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = out_dir / "paper_baseline_summary.md"
    lines = [
        "# Pulsar Paper-Style Resize/JPEG Baseline",
        "",
        f"Paper success threshold: BER <= {ber_threshold:g}.",
        "",
        "| Attack | Records | Paper failures | Failure rate | Exact matches | BER mean | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {attack_label} | {records} | {paper_failures} | {paper_failure_rate:.6f} | "
            "{exact_matches} | {ber_mean:.6f} | {psnr} | {ssim} | {lpips} |".format(
                **row,
                psnr=format_float(row["psnr_mean"]),
                ssim=format_float(row["ssim_mean"]),
                lpips=format_float(row["lpips_mean"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    if value == "":
        return ""
    if value == "inf" or value == float("inf"):
        return "inf"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    ensure_hf_cache(args.hf_cache_dir, args.hf_endpoint)
    configure_sage(sage_bin=args.sage_bin, reference_dir=args.reference_dir)
    pulsar = load_official_pulsar(args.reference_dir)

    from diffusers import DDIMScheduler, DDPMScheduler

    scheduler_cls = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}[args.scheduler]
    repo = PULSAR_REPOS.get(args.model, args.model)
    specs = attack_specs(args.attacks)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "paper_baseline_results.csv"
    if args.force and results_csv.exists():
        results_csv.unlink()

    stego = pulsar.Pulsar(
        repo=repo,
        scheduler=scheduler_cls,
        num_inference_steps=args.steps,
        seed=b"0",
        key=args.key.encode("utf-8")[:64].ljust(64, b"\x00"),
        benchmarks=False,
    )
    lpips_model = make_lpips(stego.device)

    all_rows: list[dict[str, object]] = []
    for sample_index in range(args.start_index, args.count):
        started = time.perf_counter()
        seed = f"{args.seed_prefix}{sample_index}".encode("utf-8")
        message = derive_bytes(sample_index, args.payload_bytes)
        stego.modify_randomness(new_seed=seed)
        stego.generate_state = {}
        generated = stego.generate(message, use_ecc=False)
        last = stego.scheduler.num_inference_steps - 1
        hidden = generated["samples"][last]["hidden"]
        all0 = generated["samples"][last]["all0"]
        all1 = generated["samples"][last]["all1"]
        hidden_u8 = tensor_to_uint8(hidden)
        sample_rows = []
        for spec in specs:
            attacked = apply_attack(hidden, spec)
            reveal = stego.reveal(attacked, all0, all1, use_ecc=False)
            recovered = reveal["ecc"]["decoded"][: len(message)]
            errors, bit_count, ber = bit_errors(message, recovered)
            attacked_u8 = tensor_to_uint8(attacked)
            psnr = psnr_uint8(hidden_u8, attacked_u8)
            row = {
                "method": "pulsar_raw_paper",
                "sample_index": sample_index,
                "model_repo": repo,
                "scheduler": args.scheduler,
                "steps": args.steps,
                "payload_bytes": len(message),
                "attack_label": spec["label"],
                "attack_kind": spec["kind"],
                "attack_factor": spec["factor"] if spec["factor"] is not None else "",
                "bit_errors": errors,
                "bit_count": bit_count,
                "ber": ber,
                "bit_accuracy": 1.0 - ber,
                "paper_success": ber <= args.ber_threshold,
                "paper_failure": ber > args.ber_threshold,
                "exact_match": recovered == message,
                "psnr": "inf" if math.isinf(psnr) else psnr,
                "ssim": ssim_uint8(hidden_u8, attacked_u8),
                "lpips": lpips_tensor(lpips_model, hidden, attacked),
                "runtime_s": time.perf_counter() - started,
            }
            sample_rows.append(row)
        append_rows(results_csv, sample_rows)
        all_rows.extend(sample_rows)
        compact = ", ".join(
            f"{row['attack_label']}:ber={float(row['ber']):.4f}"
            for row in sample_rows
        )
        print(f"[pulsar-paper] {sample_index + 1}/{args.count} {compact}", flush=True)

    manifest = {
        "method": "pulsar_raw_paper",
        "protocol_id": "pulsar_paper_ads_v2_20260604",
        "protocol_seed": PROTOCOL_SEED,
        "count": args.count,
        "model_repo": repo,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "payload_bytes": args.payload_bytes,
        "attacks": args.attacks,
        "ber_threshold": args.ber_threshold,
        "key": "E*64" if args.key == "E" * 64 else "custom",
        "results_csv": str(results_csv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summarize(all_rows, out, args.ber_threshold)
    print(f"results={results_csv}")


if __name__ == "__main__":
    main()
