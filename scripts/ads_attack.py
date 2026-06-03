#!/usr/bin/env python3
"""Adversarial Diffusion Sanitization attack adapters.

This implements the ADS attack as an image-domain comparison candidate:
one forward noising step, differentiable denoiser reconstruction, an
adversarial update in noisy diffusion space, then one reverse reconstruction.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL_ID = "google/ddpm-church-256"
DEFAULT_TIMESTEP = 1
DEFAULT_SEED = 0


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _load_ads_components(model_id: str, device_name: str, local_files_only: bool):
    try:
        from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise ImportError(
            "ADS attacks require diffusers. Install diffusers in the stego_attack "
            "environment or disable ads_fgsm/ads_qdir candidates."
        ) from exc

    try:
        pipeline = DDPMPipeline.from_pretrained(model_id, local_files_only=local_files_only)
        unet = pipeline.unet
        scheduler = pipeline.scheduler
    except (OSError, ValueError):
        unet = UNet2DModel.from_pretrained(model_id, local_files_only=local_files_only)
        scheduler = DDPMScheduler.from_pretrained(model_id, local_files_only=local_files_only)

    unet = unet.to(device_name)
    unet.eval()
    for param in unet.parameters():
        param.requires_grad_(False)
    return unet, scheduler


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    requested = os.environ.get("ADS_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_size(unet) -> int:
    sample_size = getattr(unet.config, "sample_size", 256)
    if isinstance(sample_size, (tuple, list)):
        return int(sample_size[0])
    return int(sample_size)


def _prepare_image(image: Image.Image, size: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = image.convert("RGB")
    original_size = rgb.size
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32), original_size


def _tensor_to_pil(tensor: torch.Tensor, original_size: tuple[int, int]) -> Image.Image:
    array = (
        ((tensor.detach().float().cpu()[0] / 2.0 + 0.5).clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    image = Image.fromarray(array, "RGB")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.BILINEAR)
    return image


def _alpha_terms(scheduler, timestep: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_timestep = int(scheduler.config.num_train_timesteps) - 1
    if timestep < 0 or timestep > max_timestep:
        raise ValueError(f"ADS timestep must be in [0,{max_timestep}], got {timestep}")
    alpha = scheduler.alphas_cumprod[timestep].to(device=device, dtype=torch.float32).view(1, 1, 1, 1)
    return alpha.sqrt(), (1.0 - alpha).sqrt()


def _ads_update(grad: torch.Tensor, variant: str, epsilon: float) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError(f"ADS epsilon must be non-negative, got {epsilon}")
    if variant == "fgsm":
        return epsilon * grad.sign()
    if variant == "qdir":
        channel_norm = grad.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        return epsilon * grad / channel_norm
    raise ValueError(f"unsupported ADS variant: {variant}")


def apply_ads_pil(
    image: Image.Image,
    variant: str = "qdir",
    epsilon: float = 0.01,
    timestep: int | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> Image.Image:
    """Apply ADS-FGSM or ADS-QDir to a PIL image.

    Environment overrides:
    ADS_DENOISER_MODEL_ID, ADS_TIMESTEP, ADS_SEED, ADS_DEVICE,
    and ADS_LOCAL_FILES_ONLY.
    """

    model_id = model_id or os.environ.get("ADS_DENOISER_MODEL_ID", DEFAULT_MODEL_ID)
    timestep = int(os.environ.get("ADS_TIMESTEP", DEFAULT_TIMESTEP) if timestep is None else timestep)
    seed = int(os.environ.get("ADS_SEED", DEFAULT_SEED) if seed is None else seed)
    device_obj = _resolve_device(device)
    local_files_only = _env_bool("ADS_LOCAL_FILES_ONLY", False)

    unet, scheduler = _load_ads_components(model_id, str(device_obj), local_files_only)
    x0, original_size = _prepare_image(image, _model_size(unet), device_obj)
    sqrt_alpha, sqrt_one_minus_alpha = _alpha_terms(scheduler, timestep, device_obj)

    generator = torch.Generator(device=device_obj)
    generator.manual_seed(seed)
    noise = torch.randn(x0.shape, generator=generator, device=device_obj, dtype=x0.dtype)
    timestep_tensor = torch.tensor([timestep], device=device_obj, dtype=torch.long)

    with torch.enable_grad():
        x_t = (sqrt_alpha * x0 + sqrt_one_minus_alpha * noise).detach().requires_grad_(True)
        eps_pred = unet(x_t, timestep_tensor).sample
        x0_hat = (x_t - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha
        loss = F.mse_loss(x0_hat, x0)
        grad = torch.autograd.grad(loss, x_t)[0]
        x_t_adv = (x_t + _ads_update(grad, variant, float(epsilon))).detach()

    with torch.no_grad():
        eps_adv = unet(x_t_adv, timestep_tensor).sample
        x0_adv = ((x_t_adv - sqrt_one_minus_alpha * eps_adv) / sqrt_alpha).clamp(-1.0, 1.0)
    return _tensor_to_pil(x0_adv, original_size)
