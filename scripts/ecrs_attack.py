#!/usr/bin/env python3
"""ECRS stego-sanitization candidates.

ECRS means equivalence-class re-randomization sanitizer: keep the visual
semantic representative close to the input while re-keying microstructure that
is likely to carry hidden payload bits. These helpers are image-only and do not
query or use any stego encoder/decoder.
"""

from __future__ import annotations

import hashlib
import math
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter


DEFAULT_ECRS_STRENGTH = 0.5
_BLOCK = 8


def _clamp_strength(strength: float | None) -> float:
    value = DEFAULT_ECRS_STRENGTH if strength is None else float(strength)
    if value < 0:
        raise ValueError(f"ECRS strength must be non-negative, got {strength}")
    return min(value, 4.0)


def _image_seed(image: Image.Image, salt: str) -> int:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(salt.encode("utf-8") + rgb.tobytes()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def _jpeg_view(rgb: Image.Image, quality: int) -> np.ndarray:
    from io import BytesIO

    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(np.clip(quality, 1, 100)))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)


def _resize_view(rgb: Image.Image, factor: float) -> np.ndarray:
    width, height = rgb.size
    small = rgb.resize(
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(small.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def _semantic_consensus(rgb: Image.Image, strength: float) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    blur_radius = 0.25 + 0.55 * min(strength, 1.5)
    blur = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)
    jpeg_quality = int(round(96 - 28 * min(strength, 1.5)))
    jpeg = _jpeg_view(rgb, jpeg_quality)
    resize_factor = max(0.55, 1.0 - 0.16 * min(strength, 1.5))
    resized = _resize_view(rgb, resize_factor)
    # Consensus stays close to the input; it estimates the semantic carrier
    # around which the private residual will be re-keyed.
    return 0.42 * base + 0.28 * blur + 0.18 * jpeg + 0.12 * resized


def _frequency_rekey_residual(residual: np.ndarray, strength: float, seed: int) -> np.ndarray:
    height, width, channels = residual.shape
    pad_h = (math.ceil(height / _BLOCK) * _BLOCK) - height
    pad_w = (math.ceil(width / _BLOCK) * _BLOCK) - width
    padded = np.pad(residual, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    out = np.empty_like(padded, dtype=np.float32)
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((_BLOCK, _BLOCK))
    # Preserve DC and the lowest frequencies. Payload-like carriers tend to use
    # mid/high bands where visual sensitivity is lower.
    mid_mask = (xx + yy >= 3) & (xx + yy <= 10)
    high_mask = xx + yy > 10
    alpha_mid = min(1.0, 0.55 * strength)
    alpha_high = min(1.0, 0.90 * strength)
    atten_high = max(0.55, 1.0 - 0.18 * max(0.0, strength - 0.8))

    for c in range(channels):
        for y in range(0, padded.shape[0], _BLOCK):
            for x in range(0, padded.shape[1], _BLOCK):
                block = padded[y : y + _BLOCK, x : x + _BLOCK, c].astype(np.float32)
                coeff = cv2.dct(block)
                rekey = coeff.copy()
                mid_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=mid_mask.sum())
                high_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=high_mask.sum())
                rekey[mid_mask] = coeff[mid_mask] * mid_sign
                rekey[high_mask] = coeff[high_mask] * high_sign * atten_high
                mixed = coeff.copy()
                mixed[mid_mask] = (1.0 - alpha_mid) * coeff[mid_mask] + alpha_mid * rekey[mid_mask]
                mixed[high_mask] = (1.0 - alpha_high) * coeff[high_mask] + alpha_high * rekey[high_mask]
                out[y : y + _BLOCK, x : x + _BLOCK, c] = cv2.idct(mixed)
    return out[:height, :width, :]


def _private_residual_noise(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    low = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.1, sigmaY=1.1)
    high = noise - low
    std = float(high.std())
    if std < 1.0e-6:
        return np.zeros(shape, dtype=np.float32)
    return high / std


def apply_ecrs_fast_pil(image: Image.Image, strength: float | None = None, seed: int | None = None) -> Image.Image:
    """Apply fast equivalence-class residual re-keying in image/DCT space."""

    value = _clamp_strength(strength)
    rgb = image.convert("RGB")
    base = np.asarray(rgb, dtype=np.float32)
    consensus = _semantic_consensus(rgb, value)
    residual = base - consensus
    rekey_seed = _image_seed(rgb, f"ecrs-fast-{value:.6f}") if seed is None else int(seed)
    rekeyed = _frequency_rekey_residual(residual, value, rekey_seed)
    # At high strength, replace more of the private residual and pull a little
    # toward consensus. This controls payload destruction without a target loss.
    replace = min(1.0, 0.70 * value)
    semantic_pull = min(0.35, 0.10 * value)
    out = consensus + (1.0 - replace) * residual + replace * rekeyed
    out = (1.0 - semantic_pull) * out + semantic_pull * consensus
    noise_amp = 5.5 * max(0.0, value - 0.65) ** 1.15
    if noise_amp > 0:
        out = out + noise_amp * _private_residual_noise(base.shape, rekey_seed + 7919)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _diffusion_reproject_pil(image: Image.Image, strength: float, seed: int | None = None) -> Image.Image:
    # Reuse the cached denoiser loader from the ADS adapter, but do not use its
    # adversarial-gradient update. ECRS-Diff performs stochastic reprojection.
    from ads_attack import _alpha_terms, _load_ads_components, _model_size, _prepare_image, _resolve_device, _tensor_to_pil

    model_id = os.environ.get("ECRS_DENOISER_MODEL_ID", os.environ.get("ADS_DENOISER_MODEL_ID", "google/ddpm-church-256"))
    device = _resolve_device(None)
    local_files_only = _env_bool("ADS_LOCAL_FILES_ONLY", _env_bool("ECRS_LOCAL_FILES_ONLY", False))
    unet, scheduler = _load_ads_components(model_id, str(device), local_files_only)
    size = _model_size(unet)
    x0, original_size = _prepare_image(image, size, device)
    max_timestep = int(scheduler.config.num_train_timesteps) - 1
    default_timestep = int(round(1 + 24 * min(strength, 1.5)))
    timestep = int(os.environ.get("ECRS_TIMESTEP", default_timestep))
    timestep = max(1, min(max_timestep, timestep))
    noise_seed = _image_seed(image, f"ecrs-diff-{strength:.6f}") if seed is None else int(seed)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    sqrt_alpha, sqrt_one_minus_alpha = _alpha_terms(scheduler, timestep, device)
    t = torch.tensor([timestep], device=device, dtype=torch.long)
    x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    with torch.no_grad():
        eps_pred = unet(x_t, t).sample
        x_hat = (x_t - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha
    blend = min(0.78, 0.03 + 0.42 * min(strength, 1.5))
    projected = (1.0 - blend) * x0 + blend * x_hat.clamp(-1, 1)
    projected_pil = _tensor_to_pil(projected, original_size)
    native_blend = min(1.0, 0.18 + 0.70 * min(strength, 1.2))
    if native_blend >= 0.999:
        return projected_pil
    native = np.asarray(image.convert("RGB"), dtype=np.float32)
    reproj = np.asarray(projected_pil.convert("RGB"), dtype=np.float32)
    mixed = (1.0 - native_blend) * native + native_blend * reproj
    return Image.fromarray(np.clip(np.rint(mixed), 0, 255).astype(np.uint8), "RGB")


def apply_ecrs_diff_pil(image: Image.Image, strength: float | None = None, seed: int | None = None) -> Image.Image:
    """Apply ECRS fast re-keying followed by low-noise diffusion reprojection."""

    value = _clamp_strength(strength)
    fast = apply_ecrs_fast_pil(image, strength=0.75 * value, seed=seed)
    return _diffusion_reproject_pil(fast, value, seed=seed)


def apply_ecrs_pil(image: Image.Image, variant: str = "fast", strength: float | None = None, seed: int | None = None) -> Image.Image:
    if variant == "fast":
        return apply_ecrs_fast_pil(image, strength=strength, seed=seed)
    if variant == "diff":
        return apply_ecrs_diff_pil(image, strength=strength, seed=seed)
    raise ValueError(f"unsupported ECRS variant: {variant}")
