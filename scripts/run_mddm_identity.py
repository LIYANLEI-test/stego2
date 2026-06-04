#!/usr/bin/env python3
"""Run MDDM third-party identity recovery with protocol payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MDDM_REF = WORKSPACE_ROOT / "references" / "MDDM-thirdparty"
if str(MDDM_REF) not in sys.path:
    sys.path.insert(0, str(MDDM_REF))
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from attack_common import ADS_ATTACK_KINDS, ECRS_ATTACK_KINDS, apply_attack_pil, attack_suffix  # noqa: E402


DEFAULT_PROTOCOL_DIR = Path("/data2/liyanlei/stego_attack_data/protocols/native_identity_v1_20260522")
PROTOCOL_SEED = "stego-attack-native-identity-v1-20260522"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data2/liyanlei/stego_attack_data/identity_runs/native_identity_20260522/mddm")
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--ecc-mode", default="none", choices=["none", "rep3", "hamming74"])
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=None,
        help="Override MDDM payload length and derive printable ASCII payloads from the protocol seed.",
    )
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--hf-cache-dir", default="/data2/liyanlei/huggingface")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument(
        "--attack-kind",
        default="identity",
        choices=["identity", "resize", "storage", "jpeg", "mblur", "gblur", "regen_vae", "unmarker", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS],
    )
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--attack-factor", type=float, default=None)
    parser.add_argument("--unmarker-stage", default="high", choices=["high", "low"])
    parser.add_argument("--unmarker-profile", default="smoke", choices=["smoke", "paper_like"])
    parser.add_argument("--unmarker-iterations", type=int, default=25)
    parser.add_argument("--unmarker-reference-dir", default=str(WORKSPACE_ROOT / "references" / "ai-watermark"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def printable_ascii_payload(method: str, index: int, length: int) -> str:
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    raw = hashlib.shake_256(
        f"{PROTOCOL_SEED}|{method}|{index:06d}|{length}".encode("utf-8")
    ).digest(length)
    return "".join(chr(alphabet[byte % len(alphabet)]) for byte in raw)


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["sample_index"]) for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_failure(path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "method",
        "sample_index",
        "attack_kind",
        "resize_factor",
        "attack_factor",
        "image_path",
        "attacked_path",
        "stage",
        "error_type",
        "error_summary",
        "traceback_summary",
        "runtime_s",
        "created_at_utc",
    ]
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_reader = csv.DictReader(handle)
            existing_names = existing_reader.fieldnames or []
            if existing_names != fieldnames:
                existing_rows = list(existing_reader)
                with path.open("w", encoding="utf-8", newline="") as rewrite:
                    writer = csv.DictWriter(rewrite, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(
                        {key: old_row.get(key, "") for key in fieldnames} for old_row in existing_rows
                    )
    with path.open("a", encoding="utf-8", newline="") as handle:
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


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    image_dir = out / "images"
    record_dir = out / "records"
    image_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "identity_results.csv"
    failures_path = out / "identity_failures.csv"
    if args.force and csv_path.exists():
        csv_path.unlink()
    if args.force and failures_path.exists():
        failures_path.unlink()
    done = load_done(csv_path)
    done.update(load_done(failures_path))

    protocol_dir = Path(args.protocol_dir).resolve()
    messages = load_jsonl(protocol_dir / "mddm_messages_500.jsonl")
    prompts = load_prompts(protocol_dir / "prompts_500.txt")

    os.environ.setdefault("HF_HOME", args.hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_cache_dir) / "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(Path(args.hf_cache_dir) / "diffusers"))
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("SD_MODEL_ID", args.model_id)

    from backend.pipeline import MDDMService

    service = MDDMService(image_dir=image_dir, record_dir=record_dir)
    for sample_index in range(args.start_index, args.count):
        if sample_index in done:
            continue
        started = time.perf_counter()
        stage = "init"
        record: dict[str, object] = {}
        attacked_path = ""
        try:
            stage = "payload"
            if args.payload_bytes is None:
                message = str(messages[sample_index]["payload_text"])
            else:
                message = printable_ascii_payload("mddm", sample_index, args.payload_bytes)
            prompt = prompts[sample_index % len(prompts)]
            stage = "generate"
            generated = service.generate(
                prompt=prompt,
                hidden_message=message,
                seed=sample_index,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=None,
                ecc_mode=args.ecc_mode,
                extra_metadata={"protocol_id": "native_identity_v1_20260522", "sample_index": sample_index},
            )
            stage = "decode"
            record_path = record_dir / f"{generated['image_id']}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            image_override = None
            if args.attack_kind != "identity":
                stage = f"{args.attack_kind}_attack"
                from PIL import Image

                image_override = apply_attack_pil(
                    Image.open(record["image_path"]),
                    args.attack_kind,
                    resize_factor=args.resize_factor,
                    attack_factor=args.attack_factor,
                    unmarker_stage=args.unmarker_stage,
                    unmarker_profile=args.unmarker_profile,
                    unmarker_iterations=args.unmarker_iterations,
                    unmarker_reference_dir=Path(args.unmarker_reference_dir).resolve(),
                )
                suffix = attack_suffix(args.attack_kind, args.resize_factor, args.attack_factor)
                attacked_path = str(image_dir / f"{generated['image_id']}_{suffix}.png")
                image_override.save(attacked_path)
            stage = "decode"
            decoded = service.decode(record, image_override=image_override)
            metrics = decoded["metrics"]
            row = {
                "method": "mddm",
                "sample_index": sample_index,
                "prompt": prompt,
                "seed": sample_index,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "ecc_mode": args.ecc_mode,
                "payload_bytes": len(message.encode("ascii")),
                "payload_bits": metrics["payload_bits"],
                "encoded_bits": metrics["encoded_bits"],
                "bit_errors": metrics["bit_errors"],
                "bit_accuracy": metrics["bit_accuracy"],
                "ber": metrics["ber"],
                "exact_match": metrics["exact_match"],
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                "attack_factor": args.attack_factor
                if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS}
                else "",
                "unmarker_stage": args.unmarker_stage if args.attack_kind == "unmarker" else "",
                "unmarker_profile": args.unmarker_profile if args.attack_kind == "unmarker" else "",
                "unmarker_iterations": args.unmarker_iterations if args.attack_kind == "unmarker" else "",
                "image_path": record["image_path"],
                "attacked_path": attacked_path,
                "record_path": str(record_path),
                "runtime_s": time.perf_counter() - started,
            }
            append_row(csv_path, row)
            print(
                f"[mddm] {sample_index + 1}/{args.count} "
                f"acc={row['bit_accuracy']:.6f} exact={row['exact_match']} runtime={row['runtime_s']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            append_failure(
                failures_path,
                {
                    "method": "mddm",
                    "sample_index": sample_index,
                    "attack_kind": args.attack_kind,
                    "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                    "attack_factor": args.attack_factor
                    if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS}
                    else "",
                    "unmarker_stage": args.unmarker_stage if args.attack_kind == "unmarker" else "",
                    "unmarker_profile": args.unmarker_profile if args.attack_kind == "unmarker" else "",
                    "unmarker_iterations": args.unmarker_iterations if args.attack_kind == "unmarker" else "",
                    "image_path": str(record.get("image_path", "")),
                    "attacked_path": attacked_path,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error_summary": summarize_exception(exc),
                    "traceback_summary": traceback.format_exc(limit=4).strip()[-2000:],
                    "runtime_s": time.perf_counter() - started,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[mddm] {sample_index + 1}/{args.count} FAILED stage={stage}: {exc}", flush=True)

    manifest = {
        "method": "mddm",
        "protocol_id": "native_identity_v1_20260522",
        "reference": str(MDDM_REF),
        "count": args.count,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "ecc_mode": args.ecc_mode,
        "payload_bytes_override": args.payload_bytes,
        "attack_kind": args.attack_kind,
        "resize_factor": args.resize_factor if args.attack_kind == "resize" else None,
        "attack_factor": args.attack_factor
        if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS}
        else None,
        "unmarker_stage": args.unmarker_stage if args.attack_kind == "unmarker" else None,
        "unmarker_profile": args.unmarker_profile if args.attack_kind == "unmarker" else None,
        "unmarker_iterations": args.unmarker_iterations if args.attack_kind == "unmarker" else None,
        "model_id": args.model_id,
        "payload_file": str(protocol_dir / "mddm_messages_500.jsonl"),
        "prompt_file": str(protocol_dir / "prompts_500.txt"),
        "results_csv": str(csv_path),
        "failures_csv": str(failures_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"results={csv_path}")


if __name__ == "__main__":
    main()
