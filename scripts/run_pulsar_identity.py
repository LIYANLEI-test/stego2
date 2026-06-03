#!/usr/bin/env python3
"""Run Pulsar identity recovery with deterministic method-capacity payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import png

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from pulsar_native_utils import (  # noqa: E402
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_HOME,
    DEFAULT_PULSAR_REF,
    configure_sage,
    ensure_hf_cache,
    load_official_pulsar,
)
from attack_common import ADS_ATTACK_KINDS, attack_roundtrip_file, attack_suffix  # noqa: E402


PROTOCOL_SEED = "stego-attack-native-identity-v1-20260522"
PULSAR_REPOS = {
    "church": "google/ddpm-church-256",
    "celebahq": "google/ddpm-celebahq-256",
    "bedroom": "google/ddpm-bedroom-256",
    "cat": "google/ddpm-cat-256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data2/liyanlei/stego_attack_data/identity_runs/native_identity_20260522/pulsar")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--model", default="bedroom")
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--region-estimate-samples", type=int, default=1)
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument("--key", default="stego-attack-workspace-pulsar-key")
    parser.add_argument("--reference-dir", default=str(DEFAULT_PULSAR_REF))
    parser.add_argument("--hf-cache-dir", default=str(DEFAULT_HF_HOME))
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--sage-bin", default="/data2/liyanlei/envs/stego_attack/bin/sage")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--attack-kind",
        default="identity",
        choices=["identity", "resize", "storage", "jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS],
    )
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--attack-factor", type=float, default=None)
    parser.add_argument("--sample-dtype", default="uint16", choices=["uint8", "uint16"])
    parser.add_argument(
        "--preserve-sample-dtype-attack",
        action="store_true",
        help="Apply resize/blur attacks directly on the saved PNG bit depth instead of converting through RGB uint8.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry sample indices already recorded in identity_failures.csv.",
    )
    return parser.parse_args()


def derive_bytes(sample_index: int, length: int) -> bytes:
    return hashlib.shake_256(
        f"{PROTOCOL_SEED}|pulsar|{sample_index:06d}|{length}".encode("utf-8")
    ).digest(length)


def bit_accuracy(expected: bytes, recovered: bytes) -> tuple[int, int, float]:
    n = min(len(expected), len(recovered))
    bit_count = n * 8
    bit_errors = 0
    for a, b in zip(expected[:n], recovered[:n]):
        bit_errors += (a ^ b).bit_count()
    if len(expected) != len(recovered):
        bit_errors += abs(len(expected) - len(recovered)) * 8
        bit_count = max(len(expected), len(recovered)) * 8
    acc = 1.0 - (bit_errors / bit_count if bit_count else 0.0)
    return bit_errors, bit_count, acc


def load_done(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["sample_index"]) for row in csv.DictReader(handle)}


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_failure(csv_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "method",
        "sample_index",
        "model_repo",
        "scheduler",
        "steps",
        "region_estimate_samples",
        "hist_bins",
        "payload_bytes",
        "payload_sha256",
        "attack_kind",
        "resize_factor",
        "attack_factor",
        "sample_dtype",
        "image_path",
        "attacked_path",
        "stage",
        "error_type",
        "error_summary",
        "traceback_summary",
        "runtime_s",
        "created_at_utc",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def summarize_exception(exc: Exception, max_chars: int = 1000) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def odd_kernel(name: str, value: float | None) -> int:
    if value is None:
        raise ValueError(f"{name} requires attack_factor")
    kernel = int(round(value))
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return kernel


def read_png_rgb(path: Path, dtype: type[np.integer]) -> np.ndarray:
    with path.open("rb") as handle:
        width, height, rows, info = png.Reader(handle).asDirect()
        planes = int(info.get("planes", 3))
        image = np.vstack([np.asarray(row, dtype=dtype) for row in rows])
    return image.reshape((height, width, planes))[:, :, :3]


def write_png_rgb(path: Path, image: np.ndarray, dtype: type[np.integer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype_max = np.iinfo(dtype).max
    image = np.clip(np.rint(image), 0, dtype_max).astype(dtype)
    with path.open("wb") as handle:
        writer = png.Writer(
            width=image.shape[1],
            height=image.shape[0],
            bitdepth=int(np.log2(dtype_max)) + 1,
            greyscale=False,
        )
        writer.write(handle, image.reshape(-1, image.shape[1] * image.shape[2]).tolist())


def native_bitdepth_attack_roundtrip_file(
    input_path: Path,
    output_path: Path,
    attack_kind: str,
    dtype: type[np.integer],
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
) -> None:
    if attack_kind == "storage":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return
    if attack_kind not in {"resize", "mblur", "gblur"}:
        raise ValueError(f"native bit-depth attack does not support {attack_kind}")

    import cv2

    image = read_png_rgb(input_path, dtype)
    if attack_kind == "resize":
        if resize_factor <= 0:
            raise ValueError(f"resize factor must be positive, got {resize_factor}")
        height, width = image.shape[:2]
        attack_width = max(1, int(round(width * resize_factor)))
        attack_height = max(1, int(round(height * resize_factor)))
        attacked = cv2.resize(image, (attack_width, attack_height), interpolation=cv2.INTER_LINEAR)
        attacked = cv2.resize(attacked, (width, height), interpolation=cv2.INTER_LINEAR)
    elif attack_kind == "mblur":
        attacked = cv2.medianBlur(image, odd_kernel("median blur kernel", attack_factor))
    else:
        kernel = odd_kernel("gaussian blur kernel", attack_factor)
        attacked = cv2.GaussianBlur(image, (kernel, kernel), 0)
    write_png_rgb(output_path, attacked, dtype)


def main() -> None:
    args = parse_args()
    ensure_hf_cache(args.hf_cache_dir, args.hf_endpoint)
    configure_sage(sage_bin=args.sage_bin, reference_dir=args.reference_dir)
    pulsar = load_official_pulsar(args.reference_dir)

    from diffusers import DDIMScheduler, DDPMScheduler

    scheduler_cls = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}[args.scheduler]
    repo = PULSAR_REPOS.get(args.model, args.model)
    out = Path(args.output_dir).resolve()
    image_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_images or args.attack_kind != "identity":
        image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out / "identity_results.csv"
    failures_path = out / "identity_failures.csv"
    if args.force and csv_path.exists():
        csv_path.unlink()
    if args.force and failures_path.exists():
        failures_path.unlink()
    done = load_done(csv_path)
    if not args.retry_failures:
        done.update(load_done(failures_path))

    stego = pulsar.Pulsar(
        repo=repo,
        scheduler=scheduler_cls,
        num_inference_steps=args.steps,
        seed=b"0",
        key=args.key.encode("utf-8").ljust(64, b"\x00")[:64],
        benchmarks=False,
    )
    sample_dtype = np.uint8 if args.sample_dtype == "uint8" else np.uint16

    for sample_index in range(args.start_index, args.count):
        if sample_index in done:
            continue
        started = time.perf_counter()
        capacity = None
        message = b""
        stage = "init"
        image_path = ""
        attacked_path = ""
        try:
            stage = "seed"
            seed = f"{sample_index}".encode("utf-8")
            stego.modify_randomness(new_seed=seed)
            stego.generate_state = {}
            stage = "estimate_regions"
            capacity = int(
                stego.estimate_regions(
                    n_to_gen=args.region_estimate_samples,
                    n_hist_bins=args.hist_bins,
                    end_to_end=False,
                )
            )
            message = derive_bytes(sample_index, capacity)
            stage = "generate_with_regions"
            generated = stego.generate_with_regions(message)
            last = stego.scheduler.num_inference_steps - 1
            hidden = generated["samples"][last]["hidden"]
            if args.save_images or args.attack_kind != "identity":
                stage = "save_reload_image"
                image_path = str(image_dir / f"{sample_index:06d}.png")
                stego.save_sample(hidden, image_path, dtype=sample_dtype)
                load_path = image_path
                if args.attack_kind == "storage":
                    stage = "storage_attack"
                    attacked_path = str(image_dir / f"{sample_index:06d}_storage.png")
                    shutil.copy2(image_path, attacked_path)
                    load_path = attacked_path
                elif args.attack_kind != "identity":
                    stage = f"{args.attack_kind}_attack"
                    suffix = attack_suffix(args.attack_kind, args.resize_factor, args.attack_factor)
                    attacked_path = str(image_dir / f"{sample_index:06d}_{suffix}.png")
                    if args.preserve_sample_dtype_attack and args.attack_kind in {"storage", "resize", "jpeg", "mblur", "gblur"}:
                        native_bitdepth_attack_roundtrip_file(
                            Path(image_path),
                            Path(attacked_path),
                            args.attack_kind,
                            sample_dtype,
                            resize_factor=args.resize_factor,
                            attack_factor=args.attack_factor,
                        )
                    else:
                        attack_roundtrip_file(
                            Path(image_path),
                            Path(attacked_path),
                            args.attack_kind,
                            resize_factor=args.resize_factor,
                            attack_factor=args.attack_factor,
                        )
                    load_path = attacked_path
                stage = "load_attacked_image"
                hidden = stego.load_sample(load_path, dtype=sample_dtype)
            stage = "reveal_with_regions"
            recovered = stego.reveal_with_regions(hidden)
            recovered = recovered[: len(message)]
            bit_errors, bit_count, bit_acc = bit_accuracy(message, recovered)
            row = {
                "method": "pulsar",
                "sample_index": sample_index,
                "model_repo": repo,
                "scheduler": args.scheduler,
                "steps": args.steps,
                "region_estimate_samples": args.region_estimate_samples,
                "hist_bins": args.hist_bins,
                "payload_bytes": len(message),
                "payload_bits": len(message) * 8,
                "payload_sha256": hashlib.sha256(message).hexdigest(),
                "recovered_sha256": hashlib.sha256(recovered).hexdigest(),
                "bit_errors": bit_errors,
                "bit_count": bit_count,
                "bit_accuracy": bit_acc,
                "exact_match": recovered == message,
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                "attack_factor": args.attack_factor
                if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
                else "",
                "sample_dtype": args.sample_dtype,
                "image_path": image_path,
                "attacked_path": attacked_path,
                "runtime_s": time.perf_counter() - started,
            }
            append_row(csv_path, row)
            print(
                f"[pulsar] {sample_index + 1}/{args.count} "
                f"cap={capacity} acc={bit_acc:.6f} exact={row['exact_match']} runtime={row['runtime_s']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": "pulsar",
                "sample_index": sample_index,
                "model_repo": repo,
                "scheduler": args.scheduler,
                "steps": args.steps,
                "region_estimate_samples": args.region_estimate_samples,
                "hist_bins": args.hist_bins,
                "payload_bytes": capacity if capacity is not None else "",
                "payload_sha256": hashlib.sha256(message).hexdigest() if message else "",
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                    "attack_factor": args.attack_factor
                    if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
                    else "",
                "sample_dtype": args.sample_dtype,
                "image_path": image_path,
                "attacked_path": attacked_path,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_summary": summarize_exception(exc),
                "traceback_summary": traceback.format_exc(limit=4).strip()[-2000:],
                "runtime_s": time.perf_counter() - started,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_failure(failures_path, failure)
            print(
                f"[pulsar] {sample_index + 1}/{args.count} FAILED "
                f"stage={stage} cap={capacity} error={failure['error_summary']}",
                flush=True,
            )

    manifest = {
        "method": "pulsar",
        "protocol_id": "native_identity_v1_20260522",
        "protocol_seed": PROTOCOL_SEED,
        "count": args.count,
        "model_repo": repo,
        "scheduler": args.scheduler,
        "steps": args.steps,
        "region_estimate_samples": args.region_estimate_samples,
        "hist_bins": args.hist_bins,
        "attack_kind": args.attack_kind,
        "resize_factor": args.resize_factor if args.attack_kind == "resize" else None,
        "attack_factor": args.attack_factor
        if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
        else None,
        "sample_dtype": args.sample_dtype,
        "message_rule": "SHAKE256(protocol_seed | pulsar | sample_index | capacity_bytes)",
        "results_csv": str(csv_path),
        "failures_csv": str(failures_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"results={csv_path}")


if __name__ == "__main__":
    main()
