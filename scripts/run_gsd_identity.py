#!/usr/bin/env python3
"""Run GSD identity recovery with protocol-controlled native DCT payloads."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.utils as tvu
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from identity_common import (  # noqa: E402
    DEFAULT_PROTOCOL_DIR,
    DEFAULT_RUN_ROOT,
    PROTOCOL_ID,
    append_csv_row,
    append_failure,
    bit_metrics,
    bits_from_payload_row,
    bits_sha256,
    load_done,
    load_jsonl,
    summarize_exception,
    traceback_summary,
    utc_now,
)
from attack_common import ADS_ATTACK_KINDS, ECRS_ATTACK_KINDS, attack_roundtrip_tensor_0_1, attack_suffix  # noqa: E402


DEFAULT_GSD_ROOT = WORKSPACE_ROOT / "references" / "GSD"
DEFAULT_CKPT_CACHE = Path("/data2/liyanlei/stego_attack_models/gsd/ddim_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "gsd_cifar10"))
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--reference-dir", default=str(DEFAULT_GSD_ROOT))
    parser.add_argument("--config", default="cifar10.yml", choices=["cifar10.yml", "celeba64.yml", "celeba-64.yml"])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--sample-type", default="generalized", choices=["generalized", "ddpm_noisy"])
    parser.add_argument("--skip-type", default="uniform", choices=["uniform", "quad"])
    parser.add_argument("--use-ownmodel", action="store_true")
    parser.add_argument("--ckpt-cache", default=str(DEFAULT_CKPT_CACHE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--skip-image", action="store_true", help="Only test official DCT mapping/extraction.")
    parser.add_argument(
        "--attack-kind",
        default="identity",
        choices=[
            "identity",
            "resize",
            "storage",
            "jpeg",
            "mblur",
            "gblur",
            "unmarker",
            "regen_vae",
            *ADS_ATTACK_KINDS,
            *ECRS_ATTACK_KINDS,
        ],
    )
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--attack-factor", type=float, default=None)
    parser.add_argument("--unmarker-stage", default="high", choices=["high", "low"])
    parser.add_argument("--unmarker-profile", default="smoke", choices=["smoke", "paper_like"])
    parser.add_argument("--unmarker-iterations", type=int, default=25)
    parser.add_argument("--unmarker-reference-dir", default=str(WORKSPACE_ROOT / "references" / "ai-watermark"))
    parser.add_argument("--regen-model", default="bmshj2018-factorized")
    parser.add_argument("--regen-quality", type=int, default=3)
    parser.add_argument("--regen-reference-dir", default=str(WORKSPACE_ROOT / "references" / "WatermarkAttacker"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dict_to_namespace(config):
    out = argparse.Namespace()
    for key, value in config.items():
        setattr(out, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return out


def setup_gsd_imports(gsd_root: Path) -> None:
    root = str(gsd_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_ddim_cache_link(cache_root: Path) -> None:
    target_parent = Path("/data/job/ddim")
    target_parent.mkdir(parents=True, exist_ok=True)
    link = target_parent / ".cache"
    if not link.exists():
        link.symlink_to(cache_root.resolve())


def load_config(gsd_root: Path, config_name: str, device: torch.device):
    with (gsd_root / "configs" / config_name).open("r", encoding="utf-8") as handle:
        config = dict_to_namespace(yaml.safe_load(handle))
    config.device = device
    return config


def load_model(args: argparse.Namespace, config, device: torch.device, gsd_root: Path):
    from functions.ckpt_util import get_ckpt_path
    from models.diffusion import Model
    from models.ema import EMAHelper

    model = Model(config)
    if args.use_ownmodel:
        if config.data.dataset == "CELEBA":
            states = torch.load(gsd_root / "out" / "logs" / "bedroom-64" / "ckpt.pth", map_location=device)
        elif config.data.dataset == "CIFAR10":
            states = torch.load(gsd_root / "out" / "logs" / "cifar-32" / "ckpt.pth", map_location=device)
        else:
            states = torch.load(gsd_root / "out" / "logs" / "bedroom-64" / "ckpt.pth", map_location=device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
    else:
        if config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif config.data.dataset == "LSUN":
            name = f"lsun_{config.data.category}"
        else:
            raise ValueError("Public DDPM checkpoint path only supports CIFAR10/LSUN without --use-ownmodel")
        ckpt = get_ckpt_path(f"ema_{name}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


def official_dct_encode(secret: np.ndarray) -> np.ndarray:
    sigma = 1.0
    z_coeff = np.zeros_like(secret, dtype=float)
    coeff_m = (secret.astype(np.float32) * 2 - 1) * sigma
    for i in range(secret.shape[0]):
        for j in range(secret.shape[1]):
            z_coeff[i][j] = cv2.idct(coeff_m[i][j])
    return z_coeff.astype(np.float32)


def official_dct_decode(noise: np.ndarray) -> np.ndarray:
    z_coeff_r = np.zeros_like(noise, dtype=float)
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            z_coeff_r[i][j] = cv2.dct(noise[i][j])
    return np.ceil((np.sign(z_coeff_r) + 1) / 2).astype(np.int64)


def sample_image(runner, x: torch.Tensor, model):
    return runner.sample_image(x, model)


def method_for_config(config_name: str) -> str:
    return "gsd_cifar10" if config_name == "cifar10.yml" else "gsd_celeba64"


def main() -> None:
    args = parse_args()
    method = method_for_config(args.config)
    out = Path(args.output_dir).resolve()
    image_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out / "identity_results.csv"
    failures_path = out / "identity_failures.csv"
    if args.force:
        csv_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    done = load_done(csv_path)
    done.update(load_done(failures_path))

    protocol_dir = Path(args.protocol_dir).resolve()
    messages = load_jsonl(protocol_dir / f"{method}_messages_500.jsonl")

    gsd_root = Path(args.reference_dir).resolve()
    setup_gsd_imports(gsd_root)
    ensure_ddim_cache_link(Path(args.ckpt_cache))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(gsd_root, args.config, device)
    shape = (1, config.data.channels, config.data.image_size, config.data.image_size)

    model = None
    runner = None
    old_cwd = Path.cwd()
    if not args.skip_image:
        os.chdir(gsd_root)
        from runners.diffusion import Diffusion

        runner_args = argparse.Namespace(
            sample_type=args.sample_type,
            skip_type=args.skip_type,
            timesteps=args.timesteps,
            eta=args.eta,
            image_folder=str(image_dir),
        )
        runner = Diffusion(runner_args, config, device=device)
        model = load_model(args, config, device, gsd_root)

    for sample_index in range(args.start_index, args.count):
        if sample_index in done:
            continue
        started = time.perf_counter()
        stage = "init"
        image_path = ""
        attacked_path = ""
        try:
            stage = "payload"
            payload_bits = bits_from_payload_row(messages[sample_index])
            if len(payload_bits) != int(np.prod(shape)):
                raise ValueError(f"Protocol bits length {len(payload_bits)} does not match native GSD shape {shape}")
            secret = np.asarray(payload_bits, dtype=np.int64).reshape(shape)

            stage = "dct_encode"
            z_coeff = official_dct_encode(secret)
            clean_recon = official_dct_decode(z_coeff)
            clean = bit_metrics(payload_bits, clean_recon.reshape(-1).tolist())

            inversion: dict[str, object] | None = None
            if runner is not None and model is not None:
                stage = "sample_image"
                z_s = torch.from_numpy(z_coeff).to(device).float()
                with torch.no_grad():
                    xs = sample_image(runner, z_s, model)
                    x0 = torch.clamp((xs + 1) / 2, 0.0, 1.0)
                    if args.save_images:
                        image_path = str(image_dir / f"stego_{sample_index:06d}.png")
                        tvu.save_image(x0[0], image_path)
                    if args.attack_kind == "unmarker":
                        from unmarker_attack import apply_unmarker_core_tensor

                        with torch.enable_grad():
                            x0_for_recovery = apply_unmarker_core_tensor(
                                x0.detach(),
                                stage=args.unmarker_stage,
                                profile=args.unmarker_profile,
                                max_iterations=args.unmarker_iterations,
                                unmarker_root=Path(args.unmarker_reference_dir).resolve(),
                            )
                        if args.save_images:
                            suffix = attack_suffix(args.attack_kind, args.resize_factor, args.attack_factor)
                            attacked_path = str(image_dir / f"stego_{sample_index:06d}_{suffix}.png")
                            tvu.save_image(x0_for_recovery[0], attacked_path)
                    elif args.attack_kind == "regen_vae":
                        from regen_attack import apply_regen_vae_tensor

                        x0_for_recovery = apply_regen_vae_tensor(
                            x0,
                            model_name=args.regen_model,
                            quality=args.regen_quality,
                            device=device,
                        )
                        if args.save_images:
                            suffix = attack_suffix(args.attack_kind, args.resize_factor, args.regen_quality)
                            attacked_path = str(image_dir / f"stego_{sample_index:06d}_{suffix}.png")
                            tvu.save_image(x0_for_recovery[0], attacked_path)
                    elif args.attack_kind != "identity":
                        x0_for_recovery = attack_roundtrip_tensor_0_1(
                            x0,
                            args.attack_kind,
                            resize_factor=args.resize_factor,
                            attack_factor=args.attack_factor,
                        )
                        if args.save_images:
                            suffix = attack_suffix(args.attack_kind, args.resize_factor, args.attack_factor)
                            attacked_path = str(image_dir / f"stego_{sample_index:06d}_{suffix}.png")
                            tvu.save_image(x0_for_recovery[0], attacked_path)
                    else:
                        x0_for_recovery = x0
                    x_steg = torch.round(x0_for_recovery * 255).clamp(0, 255)
                    x0_dequan = x_steg / 127.5 - 1

                    stage = "ddim_reverse"
                    from functions.denoising import ddim_reverse
                    import runners.diffusion as diffusion_module

                    x_t = ddim_reverse(x0_dequan, diffusion_module.seq, model, runner.betas, eta=args.eta)
                    x_r = x_t[0][-1].detach().cpu().numpy()
                recovered = official_dct_decode(x_r)
                inversion = bit_metrics(payload_bits, recovered.reshape(-1).tolist())

            metrics = inversion or clean
            row = {
                "method": "gsd",
                "variant": method.replace("gsd_", ""),
                "sample_index": sample_index,
                "config": args.config,
                "timesteps": args.timesteps,
                "eta": args.eta,
                "sample_type": args.sample_type,
                "skip_type": args.skip_type,
                "use_ownmodel": args.use_ownmodel,
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                "attack_factor": args.attack_factor if args.attack_kind in {"jpeg", "mblur", "gblur", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS} else "",
                "unmarker_stage": args.unmarker_stage if args.attack_kind == "unmarker" else "",
                "unmarker_profile": args.unmarker_profile if args.attack_kind == "unmarker" else "",
                "unmarker_iterations": args.unmarker_iterations if args.attack_kind == "unmarker" else "",
                "regen_model": args.regen_model if args.attack_kind == "regen_vae" else "",
                "regen_quality": args.regen_quality if args.attack_kind == "regen_vae" else "",
                "payload_bits": len(payload_bits),
                "payload_sha256": bits_sha256(payload_bits),
                "clean_bit_errors": clean["bit_errors"],
                "clean_bit_accuracy": clean["bit_accuracy"],
                "clean_exact_match": clean["exact_match"],
                "bit_errors": metrics["bit_errors"],
                "bit_count": metrics["bit_count"],
                "bit_accuracy": metrics["bit_accuracy"],
                "ber": metrics["ber"],
                "exact_match": metrics["exact_match"],
                "skip_image": args.skip_image,
                "image_path": image_path,
                "attacked_path": attacked_path,
                "runtime_s": time.perf_counter() - started,
            }
            append_csv_row(csv_path, row)
            print(
                f"[gsd:{method}] {sample_index + 1}/{args.count} "
                f"acc={float(row['bit_accuracy']):.6f} exact={row['exact_match']} runtime={row['runtime_s']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            append_failure(
                failures_path,
                {
                    "method": "gsd",
                    "variant": method.replace("gsd_", ""),
                    "sample_index": sample_index,
                    "attack_kind": args.attack_kind,
                    "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                    "attack_factor": args.attack_factor
                    if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS}
                    else "",
                    "image_path": image_path,
                    "attacked_path": attacked_path,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error_summary": summarize_exception(exc),
                    "traceback_summary": traceback_summary(),
                    "runtime_s": time.perf_counter() - started,
                    "created_at_utc": utc_now(),
                },
            )
            print(f"[gsd:{method}] {sample_index + 1}/{args.count} FAILED stage={stage}: {exc}", flush=True)

    if Path.cwd() != old_cwd:
        os.chdir(old_cwd)
    manifest = {
        "method": "gsd",
        "variant": method.replace("gsd_", ""),
        "protocol_id": PROTOCOL_ID,
        "reproduction_label": "native_official",
        "reference_checkout": str(gsd_root),
        "source_function": str(gsd_root / "runners" / "diffusion.py"),
        "protocol_payload_file": str(protocol_dir / f"{method}_messages_500.jsonl"),
        "count": args.count,
        "config": args.config,
        "timesteps": args.timesteps,
        "eta": args.eta,
        "sample_type": args.sample_type,
        "skip_type": args.skip_type,
        "use_ownmodel": args.use_ownmodel,
        "attack_kind": args.attack_kind,
        "resize_factor": args.resize_factor if args.attack_kind == "resize" else None,
        "attack_factor": args.attack_factor if args.attack_kind in {"jpeg", "mblur", "gblur", *ADS_ATTACK_KINDS, *ECRS_ATTACK_KINDS} else None,
        "unmarker_stage": args.unmarker_stage if args.attack_kind == "unmarker" else None,
        "unmarker_profile": args.unmarker_profile if args.attack_kind == "unmarker" else None,
        "unmarker_iterations": args.unmarker_iterations if args.attack_kind == "unmarker" else None,
        "unmarker_reference_dir": str(Path(args.unmarker_reference_dir).resolve()) if args.attack_kind == "unmarker" else None,
        "regen_model": args.regen_model if args.attack_kind == "regen_vae" else None,
        "regen_quality": args.regen_quality if args.attack_kind == "regen_vae" else None,
        "regen_reference_dir": str(Path(args.regen_reference_dir).resolve()) if args.attack_kind == "regen_vae" else None,
        "skip_image": args.skip_image,
        "results_csv": str(csv_path),
        "failures_csv": str(failures_path),
        "created_at_utc": utc_now(),
        "protocol_note": (
            "This runner mirrors GSD sample_reverse_dct: DCT-domain sign payload, DDPM sampling, "
            "image quantization, DDIM reverse, and DCT sign extraction. The local runner supplies "
            "protocol bits in place of sample_reverse_dct's np.random.randint secret and writes per-sample CSV rows."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"results={csv_path}")


if __name__ == "__main__":
    main()
