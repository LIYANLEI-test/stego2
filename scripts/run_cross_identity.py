#!/usr/bin/env python3
"""Run the official CRoSS demo as a resumable image-payload identity baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from identity_common import (  # noqa: E402
    DEFAULT_HF_HOME,
    DEFAULT_PROTOCOL_DIR,
    DEFAULT_RUN_ROOT,
    PROTOCOL_ID,
    append_csv_row,
    append_failure,
    ensure_hf_cache,
    image_metrics,
    load_done,
    load_image_payloads,
    summarize_exception,
    traceback_summary,
    utc_now,
)
from attack_common import ADS_ATTACK_KINDS, attack_roundtrip_array_rgb, attack_suffix  # noqa: E402


DEFAULT_CROSS_REF = WORKSPACE_ROOT / "references" / "CRoSS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "cross"))
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--reference-dir", default=str(DEFAULT_CROSS_REF))
    parser.add_argument("--python-bin", default="/data2/liyanlei/envs/stego_attack/bin/python")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--private-key", default="Effiel tower")
    parser.add_argument("--public-key", default="a tree")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--attack-kind",
        default="identity",
        choices=["identity", "resize", "storage", "jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS],
    )
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--attack-factor", type=float, default=None)
    parser.add_argument("--hf-cache-dir", default=str(DEFAULT_HF_HOME))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_cross_demo(args: argparse.Namespace, reference_dir: Path):
    env = os.environ.copy()
    ensure_hf_cache(args.hf_cache_dir, args.hf_endpoint)
    if str(reference_dir) not in sys.path:
        sys.path.insert(0, str(reference_dir))
    old_cwd = Path.cwd()
    try:
        os.chdir(reference_dir)
        import demo as cross_demo
    finally:
        os.chdir(old_cwd)
    if hasattr(cross_demo.ldm_stable, "enable_attention_slicing"):
        cross_demo.ldm_stable.enable_attention_slicing()
    if hasattr(cross_demo.ldm_stable, "enable_vae_slicing"):
        cross_demo.ldm_stable.enable_vae_slicing()
    return cross_demo


def run_cross(cross_demo, ode, args: argparse.Namespace, secret_path: Path, sample_dir: Path) -> dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_gt, _, _ = cross_demo.load_image(str(secret_path), 0, 0, 0, 0, resize=True)
    image_gt_latent = ode.image2latent(image_gt)
    cv2.imwrite(str(sample_dir / "gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
    image_hide_latent = ode.invert(args.public_key, latent_noise, is_forward=False)
    image_hide = ode.latent2image(image_hide_latent)
    cv2.imwrite(str(sample_dir / "hide.png"), cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR))

    attacked_path = ""
    image_for_reveal = image_hide
    if args.attack_kind != "identity":
        image_for_reveal = attack_roundtrip_array_rgb(
            np.asarray(image_hide, dtype=np.uint8),
            args.attack_kind,
            resize_factor=args.resize_factor,
            attack_factor=args.attack_factor,
        )
        attacked_path = str(sample_dir / f"hide_{attack_suffix(args.attack_kind, args.resize_factor, args.attack_factor)}.png")
        cv2.imwrite(attacked_path, cv2.cvtColor(image_for_reveal, cv2.COLOR_RGB2BGR))

    image_hide_latent_reveal = ode.image2latent(image_for_reveal)
    latent_noise = ode.invert(args.public_key, image_hide_latent_reveal, is_forward=True)
    image_reverse_latent = ode.invert(args.private_key, latent_noise, is_forward=False)
    image_reverse = ode.latent2image(image_reverse_latent)
    cv2.imwrite(str(sample_dir / "reverse.png"), cv2.cvtColor(image_reverse, cv2.COLOR_RGB2BGR))
    return {
        "stego_path": str(sample_dir / "hide.png"),
        "attacked_path": attacked_path,
        "recovered_path": str(sample_dir / "reverse.png"),
        "gt_path": str(sample_dir / "gt.png"),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    sample_root = out / "samples"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "identity_results.csv"
    failures_path = out / "identity_failures.csv"
    if args.force:
        csv_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    done = load_done(csv_path)
    done.update(load_done(failures_path))

    protocol_dir = Path(args.protocol_dir).resolve()
    payloads = load_image_payloads(protocol_dir / "image_payloads_100.csv")
    reference_dir = Path(args.reference_dir).resolve()
    if not (reference_dir / "demo.py").exists():
        raise FileNotFoundError(f"CRoSS demo.py not found: {reference_dir / 'demo.py'}")
    cross_demo = load_cross_demo(args, reference_dir)
    ode = cross_demo.ODESolve(cross_demo.ldm_stable, args.num_steps)

    for sample_index in range(args.start_index, args.count):
        if sample_index in done:
            continue
        started = time.perf_counter()
        stage = "init"
        secret_path: Path | None = None
        recovered_path: Path | None = None
        stego_path: Path | None = None
        attacked_path: Path | None = None
        try:
            stage = "payload"
            secret_path = Path(payloads[sample_index]["secret_image_path"]).resolve()
            sample_dir = sample_root / f"{sample_index:06d}"
            stage = "official_demo"
            paths = run_cross(cross_demo, ode, args, secret_path, sample_dir)
            recovered_path = Path(paths["recovered_path"])
            stego_path = Path(paths["stego_path"])
            attacked_path = Path(paths["attacked_path"]) if paths["attacked_path"] else None
            gt_path = Path(paths["gt_path"])
            stage = "metrics"
            recovered_metrics = image_metrics(secret_path, recovered_path)
            gt_metrics = image_metrics(secret_path, gt_path)
            attack_metrics = image_metrics(stego_path, attacked_path) if attacked_path else {}
            row = {
                "method": "cross",
                "variant": "official_demo",
                "sample_index": sample_index,
                "secret_image_path": str(secret_path),
                "gt_path": str(gt_path),
                "stego_path": str(stego_path),
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                "attack_factor": args.attack_factor if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae"} else "",
                "attacked_path": str(attacked_path) if attacked_path else "",
                "attack_mse": attack_metrics.get("mse", ""),
                "attack_psnr": attack_metrics.get("psnr", ""),
                "attack_ssim": attack_metrics.get("ssim", ""),
                "recovered_path": str(recovered_path),
                "private_key": args.private_key,
                "public_key": args.public_key,
                "num_steps": args.num_steps,
                "recovery_mse": recovered_metrics["mse"],
                "recovery_mae": recovered_metrics["mae"],
                "recovery_psnr": recovered_metrics["psnr"],
                "recovery_ssim": recovered_metrics["ssim"],
                "gt_mse": gt_metrics["mse"],
                "gt_psnr": gt_metrics["psnr"],
                "exact_match": recovered_metrics["mse"] == 0.0,
                "runtime_s": time.perf_counter() - started,
            }
            append_csv_row(csv_path, row)
            print(
                f"[cross] {sample_index + 1}/{args.count} "
                f"psnr={float(row['recovery_psnr']):.3f} ssim={row['recovery_ssim']} runtime={row['runtime_s']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            append_failure(
                failures_path,
                {
                    "method": "cross",
                    "variant": "official_demo",
                    "sample_index": sample_index,
                    "attack_kind": args.attack_kind,
                    "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                    "attack_factor": args.attack_factor if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae"} else "",
                    "image_path": str(secret_path) if secret_path else "",
                    "stego_path": str(stego_path) if stego_path else "",
                    "attacked_path": str(attacked_path) if attacked_path else "",
                    "recovered_path": str(recovered_path) if recovered_path else "",
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error_summary": summarize_exception(exc),
                    "traceback_summary": traceback_summary(),
                    "runtime_s": time.perf_counter() - started,
                    "created_at_utc": utc_now(),
                },
            )
            print(f"[cross] {sample_index + 1}/{args.count} FAILED stage={stage}: {exc}", flush=True)

    manifest = {
        "method": "cross",
        "protocol_id": PROTOCOL_ID,
        "reproduction_label": "native_official",
        "reference_checkout": str(reference_dir),
        "official_script": str(reference_dir / "demo.py"),
        "protocol_image_payload_file": str(protocol_dir / "image_payloads_100.csv"),
        "count": args.count,
        "private_key": args.private_key,
        "public_key": args.public_key,
        "num_steps": args.num_steps,
        "attack_kind": args.attack_kind,
        "resize_factor": args.resize_factor if args.attack_kind == "resize" else None,
        "attack_factor": args.attack_factor if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae"} else None,
        "results_csv": str(csv_path),
        "failures_csv": str(failures_path),
        "created_at_utc": utc_now(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"results={csv_path}")


if __name__ == "__main__":
    main()
