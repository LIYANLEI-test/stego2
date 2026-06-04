#!/usr/bin/env python3
"""Shared image-domain attack helpers."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter


RESIZE_INTERPOLATION = Image.Resampling.BILINEAR
ADS_ATTACK_KINDS = ("ads_fgsm", "ads_qdir")
ADS_DEFAULT_EPSILON = 0.01


def resize_roundtrip_pil(image: Image.Image, factor: float) -> Image.Image:
    """Resize by factor and back to the original size with fixed interpolation."""
    if factor <= 0:
        raise ValueError(f"resize factor must be positive, got {factor}")
    rgb = image.convert("RGB")
    width, height = rgb.size
    attack_width = max(1, int(round(width * factor)))
    attack_height = max(1, int(round(height * factor)))
    attacked = rgb.resize((attack_width, attack_height), RESIZE_INTERPOLATION)
    return attacked.resize((width, height), RESIZE_INTERPOLATION)


def storage_roundtrip_pil(image: Image.Image) -> Image.Image:
    """Save to PNG and reload in RGB image space."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def jpeg_roundtrip_pil(image: Image.Image, quality: float) -> Image.Image:
    """Apply a JPEG compression round trip and return RGB pixels."""
    quality_int = int(round(quality))
    if quality_int < 1 or quality_int > 100:
        raise ValueError(f"jpeg quality must be in [1,100], got {quality}")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality_int)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def median_blur_pil(image: Image.Image, kernel_size: float) -> Image.Image:
    if 0 < kernel_size < 1:
        alpha = float(kernel_size)
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        blurred = cv2.medianBlur(array, 3).astype(np.float32)
        mixed = array.astype(np.float32) * (1.0 - alpha) + blurred * alpha
        return Image.fromarray(np.clip(np.rint(mixed), 0, 255).astype(np.uint8), "RGB")
    kernel = _odd_kernel("median blur kernel", kernel_size)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(cv2.medianBlur(array, kernel), "RGB")


def gaussian_blur_pil(image: Image.Image, kernel_size: float) -> Image.Image:
    if 0 < kernel_size < 3:
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(kernel_size)))
    kernel = _odd_kernel("gaussian blur kernel", kernel_size)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(cv2.GaussianBlur(array, (kernel, kernel), 0), "RGB")


def _odd_kernel(name: str, value: float) -> int:
    kernel = int(round(value))
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return kernel


def attack_suffix(attack_kind: str, resize_factor: float = 1.0, attack_factor: float | None = None) -> str:
    def fmt(value: float) -> str:
        return f"{value:g}".replace(".", "_")

    if attack_kind == "identity":
        return "identity"
    if attack_kind == "storage":
        return "storage"
    if attack_kind == "resize":
        return f"resize_{fmt(resize_factor)}"
    if attack_kind in {"jpeg", "mblur", "gblur"}:
        if attack_factor is None:
            raise ValueError(f"{attack_kind} requires attack_factor")
        return f"{attack_kind}_{fmt(attack_factor)}"
    if attack_kind == "unmarker":
        return "unmarker"
    if attack_kind == "regen_vae":
        if attack_factor is None:
            return "regen_vae"
        return f"regen_vae_q{fmt(attack_factor)}"
    if attack_kind in ADS_ATTACK_KINDS:
        epsilon = ADS_DEFAULT_EPSILON if attack_factor is None else attack_factor
        return f"{attack_kind}_eps{fmt(epsilon)}"
    raise ValueError(f"unsupported attack kind: {attack_kind}")


def apply_attack_pil(
    image: Image.Image,
    attack_kind: str,
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
    unmarker_stage: str = "high",
    unmarker_profile: str = "smoke",
    unmarker_iterations: int | None = None,
    unmarker_reference_dir: str | Path | None = None,
) -> Image.Image:
    if attack_kind == "identity":
        return image.convert("RGB")
    if attack_kind == "resize":
        return resize_roundtrip_pil(image, resize_factor)
    if attack_kind == "storage":
        return storage_roundtrip_pil(image)
    if attack_kind == "jpeg":
        if attack_factor is None:
            raise ValueError("jpeg attack requires attack_factor")
        return jpeg_roundtrip_pil(image, attack_factor)
    if attack_kind == "mblur":
        if attack_factor is None:
            raise ValueError("mblur attack requires attack_factor")
        return median_blur_pil(image, attack_factor)
    if attack_kind == "gblur":
        if attack_factor is None:
            raise ValueError("gblur attack requires attack_factor")
        return gaussian_blur_pil(image, attack_factor)
    if attack_kind == "regen_vae":
        from regen_attack import apply_regen_vae_pil

        quality = int(round(attack_factor)) if attack_factor is not None else 3
        return apply_regen_vae_pil(image, quality=quality)
    if attack_kind == "unmarker":
        from unmarker_attack import apply_unmarker_core_pil

        kwargs = {
            "stage": unmarker_stage,
            "profile": unmarker_profile,
            "max_iterations": unmarker_iterations,
        }
        if unmarker_reference_dir is not None:
            kwargs["unmarker_root"] = Path(unmarker_reference_dir).resolve()
        return apply_unmarker_core_pil(image, **kwargs)
    if attack_kind in ADS_ATTACK_KINDS:
        from ads_attack import apply_ads_pil

        epsilon = ADS_DEFAULT_EPSILON if attack_factor is None else attack_factor
        return apply_ads_pil(image, variant=attack_kind.split("_", 1)[1], epsilon=epsilon)
    raise ValueError(f"unsupported attack kind: {attack_kind}")


def resize_roundtrip_array_rgb(image: np.ndarray, factor: float) -> np.ndarray:
    attacked = resize_roundtrip_pil(Image.fromarray(image.astype(np.uint8), "RGB"), factor)
    return np.asarray(attacked, dtype=np.uint8)


def storage_roundtrip_array_rgb(image: np.ndarray) -> np.ndarray:
    attacked = storage_roundtrip_pil(Image.fromarray(image.astype(np.uint8), "RGB"))
    return np.asarray(attacked, dtype=np.uint8)


def attack_roundtrip_array_rgb(
    image: np.ndarray,
    attack_kind: str,
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
    unmarker_stage: str = "high",
    unmarker_profile: str = "smoke",
    unmarker_iterations: int | None = None,
    unmarker_reference_dir: str | Path | None = None,
) -> np.ndarray:
    attacked = apply_attack_pil(
        Image.fromarray(image.astype(np.uint8), "RGB"),
        attack_kind,
        resize_factor=resize_factor,
        attack_factor=attack_factor,
        unmarker_stage=unmarker_stage,
        unmarker_profile=unmarker_profile,
        unmarker_iterations=unmarker_iterations,
        unmarker_reference_dir=unmarker_reference_dir,
    )
    return np.asarray(attacked, dtype=np.uint8)


def resize_roundtrip_file(input_path: Path, output_path: Path, factor: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attacked = resize_roundtrip_pil(Image.open(input_path), factor)
    attacked.save(output_path)


def storage_roundtrip_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attacked = storage_roundtrip_pil(Image.open(input_path))
    attacked.save(output_path)


def attack_roundtrip_file(
    input_path: Path,
    output_path: Path,
    attack_kind: str,
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
    unmarker_stage: str = "high",
    unmarker_profile: str = "smoke",
    unmarker_iterations: int | None = None,
    unmarker_reference_dir: str | Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attacked = apply_attack_pil(
        Image.open(input_path),
        attack_kind,
        resize_factor=resize_factor,
        attack_factor=attack_factor,
        unmarker_stage=unmarker_stage,
        unmarker_profile=unmarker_profile,
        unmarker_iterations=unmarker_iterations,
        unmarker_reference_dir=unmarker_reference_dir,
    )
    attacked.save(output_path)


def tensor_minus1_1_to_pil(tensor_chw: torch.Tensor) -> Image.Image:
    array = (
        ((tensor_chw.detach().float().cpu() / 2.0 + 0.5).clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, "RGB")


def pil_to_tensor_minus1_1(image: Image.Image, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=dtype)


def resize_roundtrip_tensor_minus1_1(tensor_bchw: torch.Tensor, factor: float) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = resize_roundtrip_pil(tensor_minus1_1_to_pil(tensor_bchw[0]), factor)
    return pil_to_tensor_minus1_1(attacked, tensor_bchw.device, tensor_bchw.dtype)


def storage_roundtrip_tensor_minus1_1(tensor_bchw: torch.Tensor) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = storage_roundtrip_pil(tensor_minus1_1_to_pil(tensor_bchw[0]))
    return pil_to_tensor_minus1_1(attacked, tensor_bchw.device, tensor_bchw.dtype)


def attack_roundtrip_tensor_minus1_1(
    tensor_bchw: torch.Tensor,
    attack_kind: str,
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
    unmarker_stage: str = "high",
    unmarker_profile: str = "smoke",
    unmarker_iterations: int | None = None,
    unmarker_reference_dir: str | Path | None = None,
) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = apply_attack_pil(
        tensor_minus1_1_to_pil(tensor_bchw[0]),
        attack_kind,
        resize_factor=resize_factor,
        attack_factor=attack_factor,
        unmarker_stage=unmarker_stage,
        unmarker_profile=unmarker_profile,
        unmarker_iterations=unmarker_iterations,
        unmarker_reference_dir=unmarker_reference_dir,
    )
    return pil_to_tensor_minus1_1(attacked, tensor_bchw.device, tensor_bchw.dtype)


def tensor_0_1_to_pil(tensor_chw: torch.Tensor) -> Image.Image:
    array = (
        (tensor_chw.detach().float().cpu().clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, "RGB")


def pil_to_tensor_0_1(image: Image.Image, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def resize_roundtrip_tensor_0_1(tensor_bchw: torch.Tensor, factor: float) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = resize_roundtrip_pil(tensor_0_1_to_pil(tensor_bchw[0]), factor)
    return pil_to_tensor_0_1(attacked, tensor_bchw.device, tensor_bchw.dtype)


def storage_roundtrip_tensor_0_1(tensor_bchw: torch.Tensor) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = storage_roundtrip_pil(tensor_0_1_to_pil(tensor_bchw[0]))
    return pil_to_tensor_0_1(attacked, tensor_bchw.device, tensor_bchw.dtype)


def attack_roundtrip_tensor_0_1(
    tensor_bchw: torch.Tensor,
    attack_kind: str,
    resize_factor: float = 1.0,
    attack_factor: float | None = None,
    unmarker_stage: str = "high",
    unmarker_profile: str = "smoke",
    unmarker_iterations: int | None = None,
    unmarker_reference_dir: str | Path | None = None,
) -> torch.Tensor:
    if tensor_bchw.ndim != 4 or tensor_bchw.shape[0] != 1 or tensor_bchw.shape[1] != 3:
        raise ValueError(f"expected tensor shape [1,3,H,W], got {tuple(tensor_bchw.shape)}")
    attacked = apply_attack_pil(
        tensor_0_1_to_pil(tensor_bchw[0]),
        attack_kind,
        resize_factor=resize_factor,
        attack_factor=attack_factor,
        unmarker_stage=unmarker_stage,
        unmarker_profile=unmarker_profile,
        unmarker_iterations=unmarker_iterations,
        unmarker_reference_dir=unmarker_reference_dir,
    )
    return pil_to_tensor_0_1(attacked, tensor_bchw.device, tensor_bchw.dtype)
