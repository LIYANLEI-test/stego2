#!/usr/bin/env python3
"""Run MAS/GRDH identity recovery with protocol-controlled native latent payloads."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import autocast


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
    deterministic_u32,
    load_done,
    load_jsonl,
    load_prompts,
    summarize_exception,
    traceback_summary,
    utc_now,
)
from attack_common import ADS_ATTACK_KINDS, attack_roundtrip_tensor_minus1_1, attack_suffix  # noqa: E402


DEFAULT_MAS_ROOT = WORKSPACE_ROOT / "references" / "mas_GRDH"
DEFAULT_CKPT = Path("/data2/liyanlei/stego_attack_models/mas_grdh/v1-5-pruned.ckpt")
DEFAULT_CLIP = Path("/data2/liyanlei/stego_attack_models/mas_grdh/clip/clip-vit-large-patch14-local")
DEFAULT_CONFIG = WORKSPACE_ROOT / "configs" / "mas_grdh_native_ldm.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "mas_grdh"))
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--reference-dir", default=str(DEFAULT_MAS_ROOT))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--clip-dir", default=str(DEFAULT_CLIP))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dpm-steps", type=int, default=20)
    parser.add_argument("--dpm-gen-steps", type=int, default=None)
    parser.add_argument("--dpm-inv-steps", type=int, default=None)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--dpm-order", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--mapping-func", default="ours_mapping")
    parser.add_argument("--bit-num", type=int, default=1)
    parser.add_argument("--attack-layer", default="identity")
    parser.add_argument("--attack-factor", type=float, default=0.0)
    parser.add_argument(
        "--attack-kind",
        default="native",
        choices=["native", "identity", "resize", "storage", "jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS],
    )
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--precision", default="autocast", choices=["full", "autocast"])
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--skip-image", action="store_true", help="Only test official mapping encode/decode.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_config(reference_dir: Path, ckpt: Path, clip_dir: Path, config: Path) -> None:
    from prepare_mas_grdh_native import main as _unused  # noqa: F401
    import prepare_mas_grdh_native

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "prepare_mas_grdh_native.py",
            "--reference-dir",
            str(reference_dir),
            "--ckpt",
            str(ckpt),
            "--clip-dir",
            str(clip_dir),
            "--output-config",
            str(config),
        ]
        prepare_mas_grdh_native.main()
    finally:
        sys.argv = old_argv


def load_official_symbols(reference_dir: Path):
    scripts_dir = reference_dir.resolve() / "scripts"
    ref_root = reference_dir.resolve()
    for path in (str(ref_root), str(scripts_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from omegaconf import OmegaConf
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module
    import robust_eval
    from scripts.txt2img import cal_acc, load_model_and_get_prompt_embedding, load_model_from_config
    from scripts.utils import gray_code

    return {
        "OmegaConf": OmegaConf,
        "DPMSolverSampler": DPMSolverSampler,
        "mapping_module": mapping_module,
        "robust_eval": robust_eval,
        "cal_acc": cal_acc,
        "load_model_and_get_prompt_embedding": load_model_and_get_prompt_embedding,
        "load_model_from_config": load_model_from_config,
        "gray_code": gray_code,
    }


def payload_to_secret(bits: list[int], latent_shape: tuple[int, ...], bit_num: int) -> np.ndarray:
    if bit_num != 1:
        raise ValueError("Protocol mas_grdh payload is 1 bit per latent element; --bit-num must be 1.")
    expected = int(np.prod(latent_shape))
    if len(bits) != expected:
        raise ValueError(f"Protocol bits length {len(bits)} does not match latent shape {latent_shape}")
    return np.asarray(bits, dtype=np.int64).reshape(latent_shape)


def secret_to_bits(secret: np.ndarray) -> list[int]:
    return [int(x) & 1 for x in np.ravel(secret)]


def save_tensor_image(model, tensor: torch.Tensor, path: Path) -> None:
    from scripts.utils import image_grid

    path.parent.mkdir(parents=True, exist_ok=True)
    image_grid(tensor).save(path)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    image_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "identity_results.csv"
    failures_path = out / "identity_failures.csv"
    if args.force:
        csv_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    done = load_done(csv_path)
    done.update(load_done(failures_path))

    protocol_dir = Path(args.protocol_dir).resolve()
    messages = load_jsonl(protocol_dir / "mas_grdh_messages_500.jsonl")
    prompts = load_prompts(protocol_dir / "prompts_500.txt")

    reference_dir = Path(args.reference_dir).resolve()
    ckpt = Path(args.ckpt).resolve()
    clip_dir = Path(args.clip_dir).resolve()
    config_path = Path(args.config).resolve()
    prepare_config(reference_dir, ckpt, clip_dir, config_path)
    symbols = load_official_symbols(reference_dir)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    opt = argparse.Namespace(
        scale=args.scale,
        n_samples=1,
        dpm_gen_steps=args.dpm_gen_steps or args.dpm_steps,
        dpm_inv_steps=args.dpm_inv_steps or args.dpm_steps,
        ddim_eta=args.ddim_eta,
        dpm_order=args.dpm_order,
    )
    bits = args.bit_num
    gray_list = symbols["gray_code"](bits)
    mapping_func = getattr(symbols["mapping_module"], args.mapping_func)(bits=bits)
    attack_func = getattr(symbols["robust_eval"], args.attack_layer)

    if not args.skip_image:
        config = symbols["OmegaConf"].load(str(config_path))
        model = symbols["load_model_from_config"](config, str(ckpt), args.gpu)
        model = model.to(device)
        sampler = symbols["DPMSolverSampler"](model)
        precision_scope = autocast if args.precision == "autocast" else nullcontext
    else:
        model = None
        sampler = None
        precision_scope = nullcontext

    latent_shape = (1, 4, 64, 64)
    width = 512
    height = 512

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
            random_input = payload_to_secret(payload_bits, latent_shape, bits)
            prompt = prompts[sample_index % len(prompts)]

            stage = "mapping_args"
            np.random.seed(sample_index)
            random.seed(sample_index)
            torch.manual_seed(sample_index)
            random_input_ori_sample = None
            if mapping_func.need_uniform_sampler:
                random_input_ori_sample = np.random.rand(*latent_shape)
            if mapping_func.need_gaussian_sampler:
                random_input_ori_sample = np.random.randn(*latent_shape)
            if args.mapping_func == "ours_mapping":
                random_input_args = {
                    "seed_kernel": np.asarray([deterministic_u32("mas_grdh", sample_index, "seed_kernel")], dtype=np.uint32),
                    "seed_shuffle": np.asarray([deterministic_u32("mas_grdh", sample_index, "seed_shuffle")], dtype=np.uint32),
                }
            elif args.mapping_func == "tdsc_mapping":
                random_input_args = {"key": np.asarray([deterministic_u32("mas_grdh", sample_index, "key")], dtype=np.uint32)}
            else:
                random_input_args = {}

            stage = "encode_secret"
            init_latent_np = mapping_func.encode_secret(
                secret_message=random_input,
                ori_sample=random_input_ori_sample,
                **random_input_args,
            ).astype(np.float32)
            init_latent = torch.from_numpy(init_latent_np).to(device)
            clean_recon = mapping_func.decode_secret(pred_noise=init_latent_np, **random_input_args)
            clean = bit_metrics(payload_bits, secret_to_bits(clean_recon))

            inversion: dict[str, object] | None = None
            encoder_decode_error = ""
            recon_error = ""
            if model is not None and sampler is not None:
                with torch.no_grad():
                    with precision_scope("cuda"):
                        stage = "prompt_embedding"
                        c, uc = symbols["load_model_and_get_prompt_embedding"](model, opt, [prompt])
                        shape = init_latent.shape[1:]

                        stage = "dpm_generate"
                        z_0, _ = sampler.sample(
                            steps=opt.dpm_gen_steps,
                            unconditional_conditioning=uc,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=args.scale,
                            eta=args.ddim_eta,
                            order=args.dpm_order,
                            x_T=init_latent,
                            width=width,
                            height=height,
                            DPMencode=False,
                            DPMdecode=True,
                        )
                        x0_samples = model.decode_first_stage(z_0)
                        if args.save_images:
                            image_path = str(image_dir / f"{sample_index:06d}.png")
                            save_tensor_image(model, x0_samples, Path(image_path))

                        stage = "attack_layer"
                        if args.attack_kind == "identity":
                            pass
                        elif args.attack_kind in {
                            "resize",
                            "storage",
                            "jpeg",
                            "mblur",
                            "gblur",
                            "regen_vae",
                            *ADS_ATTACK_KINDS,
                        }:
                            factor = (
                                args.attack_factor
                                if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
                                else None
                            )
                            x0_samples = attack_roundtrip_tensor_minus1_1(
                                x0_samples,
                                args.attack_kind,
                                resize_factor=args.resize_factor,
                                attack_factor=factor,
                            ).to(device)
                            if args.save_images:
                                suffix = attack_suffix(args.attack_kind, args.resize_factor, factor)
                                attacked_path = str(image_dir / f"{sample_index:06d}_{suffix}.png")
                                save_tensor_image(model, x0_samples, Path(attacked_path))
                        else:
                            tmp_name = str(out / "tmp" / f"{sample_index:06d}_{args.attack_layer}")
                            Path(tmp_name).parent.mkdir(parents=True, exist_ok=True)
                            x0_samples = attack_func(x0_samples, factor=args.attack_factor, tmp_image_name=tmp_name).to(device)

                        stage = "vae_encode"
                        init_latent_hat = model.get_first_stage_encoding(model.encode_first_stage(x0_samples))

                        stage = "dpm_invert"
                        z_enc, _ = sampler.sample(
                            steps=opt.dpm_inv_steps,
                            unconditional_conditioning=uc,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=args.scale,
                            eta=args.ddim_eta,
                            order=args.dpm_order,
                            x_T=init_latent_hat,
                            width=width,
                            height=height,
                            DPMencode=True,
                        )
                        encoder_decode_error = float((init_latent - init_latent_hat).abs().mean().detach().cpu())
                        recon_error = float((init_latent - z_enc).abs().mean().detach().cpu())
                        pred_noise = z_enc.detach().cpu().numpy()
                        recon_latent = mapping_func.decode_secret(pred_noise=pred_noise, **random_input_args)
                        if args.mapping_func == "tdsc_mapping":
                            acc = mapping_func._compute_acc(random_input, recon_latent)
                            inversion = {
                                "bit_errors": int(round((1.0 - float(acc)) * len(payload_bits))),
                                "bit_count": len(payload_bits),
                                "bit_accuracy": float(acc),
                                "ber": 1.0 - float(acc),
                                "exact_match": float(acc) == 1.0,
                            }
                        else:
                            official_acc = symbols["cal_acc"](recon_latent, random_input, gray_list=gray_list, bits=bits)
                            inversion = bit_metrics(payload_bits, secret_to_bits(recon_latent))
                            inversion["official_cal_acc"] = float(official_acc)

            metrics = inversion or clean
            row = {
                "method": "mas_grdh",
                "variant": args.mapping_func,
                "sample_index": sample_index,
                "prompt": prompt,
                "dpm_gen_steps": opt.dpm_gen_steps,
                "dpm_inv_steps": opt.dpm_inv_steps,
                "scale": args.scale,
                "attack_layer": args.attack_layer,
                "attack_kind": args.attack_kind,
                "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                "attack_factor": args.attack_factor
                if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
                else "",
                "bit_num": bits,
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
                "official_cal_acc": metrics.get("official_cal_acc", metrics["bit_accuracy"]),
                "encoder_decode_error": encoder_decode_error,
                "recon_error": recon_error,
                "skip_image": args.skip_image,
                "image_path": image_path,
                "attacked_path": attacked_path,
                "runtime_s": time.perf_counter() - started,
            }
            append_csv_row(csv_path, row)
            print(
                f"[mas-grdh] {sample_index + 1}/{args.count} "
                f"acc={float(row['bit_accuracy']):.6f} exact={row['exact_match']} runtime={row['runtime_s']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            append_failure(
                failures_path,
                {
                    "method": "mas_grdh",
                    "variant": args.mapping_func,
                    "sample_index": sample_index,
                    "attack_kind": args.attack_kind,
                    "resize_factor": args.resize_factor if args.attack_kind == "resize" else "",
                    "attack_factor": args.attack_factor
                    if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
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
            print(f"[mas-grdh] {sample_index + 1}/{args.count} FAILED stage={stage}: {exc}", flush=True)

    manifest = {
        "method": "mas_grdh",
        "protocol_id": PROTOCOL_ID,
        "reproduction_label": "native_official",
        "reference_checkout": str(reference_dir),
        "official_script": str(reference_dir / "scripts" / "txt2img.py"),
        "protocol_payload_file": str(protocol_dir / "mas_grdh_messages_500.jsonl"),
        "prompt_file": str(protocol_dir / "prompts_500.txt"),
        "ckpt": str(ckpt),
        "config": str(config_path),
        "count": args.count,
        "dpm_gen_steps": args.dpm_gen_steps or args.dpm_steps,
        "dpm_inv_steps": args.dpm_inv_steps or args.dpm_steps,
        "scale": args.scale,
        "attack_layer": args.attack_layer,
        "attack_kind": args.attack_kind,
        "resize_factor": args.resize_factor if args.attack_kind == "resize" else None,
        "attack_factor": args.attack_factor
        if args.attack_kind in {"jpeg", "mblur", "gblur", "regen_vae", *ADS_ATTACK_KINDS}
        else None,
        "mapping_func": args.mapping_func,
        "bit_num": args.bit_num,
        "skip_image": args.skip_image,
        "results_csv": str(csv_path),
        "failures_csv": str(failures_path),
        "created_at_utc": utc_now(),
        "protocol_note": (
            "This runner mirrors the official txt2img.py encode/generate/attack/invert/decode path. "
            "The local runner supplies deterministic protocol bits in place of txt2img.py's np.random.randint secret."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"results={csv_path}")


if __name__ == "__main__":
    main()
