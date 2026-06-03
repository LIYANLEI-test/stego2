#!/usr/bin/env python3
"""Selected quality-budget attack matrix for formal experiments.

These parameters were calibrated on 2026-05-27 under:
  stego-vs-attacked PSNR >= 30 dB and LPIPS <= 0.10

Keep this file as the single source of truth for formal selected-attack queues
and paper-table aggregation. Exploratory grids should use separate scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FORMAL_COUNTS = {
    "cross": 100,
    "gsd_cifar10": 500,
    "mas_grdh": 500,
    "mddm_128_pilot": 50,
    "pulsar": 500,
}

BASELINE_PROVENANCE = {
    "cross": "native_official",
    "gsd_cifar10": "native_official",
    "mas_grdh": "native_official",
    "mddm_128_pilot": "native_third_party",
    "pulsar": "native_official",
}

# Attack parameters were selected on these sample IDs in the calibration grid.
# Formal paper summaries exclude them to avoid selection/test leakage.
CALIBRATION_SAMPLE_COUNT = 10


@dataclass(frozen=True)
class SelectedAttack:
    method: str
    attack: str
    factor: str
    label: str
    metric: str
    provenance: str
    note: str = ""

    @property
    def safe_factor(self) -> str:
        return self.factor.replace(".", "_").replace("-", "_")

    @property
    def name_part(self) -> str:
        return f"{self.attack}_{self.safe_factor}"


SELECTED_ATTACKS: tuple[SelectedAttack, ...] = (
    SelectedAttack("cross", "resize", "1.5", "resize_1_5", "recovery_psnr", "native_official"),
    SelectedAttack("cross", "jpeg", "50", "jpeg_q50", "recovery_psnr", "native_official"),
    SelectedAttack("cross", "mblur", "3", "median_blur_k3", "recovery_psnr", "native_official"),
    SelectedAttack("cross", "gblur", "3", "gaussian_blur_k3", "recovery_psnr", "native_official"),
    SelectedAttack("cross", "regen_vae", "5", "regen_vae_q5", "recovery_psnr", "adapted_attack"),
    SelectedAttack("gsd_cifar10", "resize", "1.25", "resize_1_25", "bit_accuracy", "native_official"),
    SelectedAttack("gsd_cifar10", "jpeg", "80", "jpeg_q80", "bit_accuracy", "native_official"),
    SelectedAttack("gsd_cifar10", "mblur", "0.5", "median_blur_soft_0_5", "bit_accuracy", "native_official"),
    SelectedAttack("gsd_cifar10", "gblur", "0.5", "gaussian_blur_radius_0_5", "bit_accuracy", "native_official"),
    SelectedAttack("gsd_cifar10", "regen_vae", "6", "regen_vae_q6", "bit_accuracy", "adapted_attack"),
    SelectedAttack(
        "gsd_cifar10",
        "unmarker",
        "high_smoke_25",
        "unmarker_high_smoke_25",
        "bit_accuracy",
        "adapted_attack",
        "GSD-only smoke-profile UnMarker candidate.",
    ),
    SelectedAttack("mas_grdh", "resize", "1.5", "resize_1_5", "bit_accuracy", "native_official"),
    SelectedAttack("mas_grdh", "jpeg", "50", "jpeg_q50", "bit_accuracy", "native_official"),
    SelectedAttack("mas_grdh", "mblur", "0.5", "median_blur_soft_0_5", "bit_accuracy", "native_official"),
    SelectedAttack("mas_grdh", "gblur", "0.5", "gaussian_blur_radius_0_5", "bit_accuracy", "native_official"),
    SelectedAttack("mas_grdh", "regen_vae", "6", "regen_vae_q6", "bit_accuracy", "adapted_attack"),
    SelectedAttack(
        "mddm_128_pilot",
        "jpeg",
        "70",
        "jpeg_q70",
        "bit_accuracy",
        "native_third_party",
        "Pilot only; not official author code.",
    ),
    SelectedAttack(
        "mddm_128_pilot",
        "mblur",
        "0.5",
        "median_blur_soft_0_5",
        "bit_accuracy",
        "native_third_party",
        "Pilot only; not official author code.",
    ),
    SelectedAttack(
        "mddm_128_pilot",
        "gblur",
        "0.25",
        "gaussian_blur_radius_0_25",
        "bit_accuracy",
        "native_third_party",
        "Pilot only; not official author code.",
    ),
    SelectedAttack(
        "pulsar",
        "resize",
        "1.25",
        "resize_1_25",
        "bit_accuracy",
        "native_official",
        "Calibration produced native reveal failures within quality budget.",
    ),
    SelectedAttack(
        "pulsar",
        "jpeg",
        "95",
        "jpeg_q95",
        "bit_accuracy",
        "native_official",
        "Calibration produced native reveal failures within quality budget.",
    ),
    SelectedAttack(
        "pulsar",
        "ads",
        "resize224",
        "ads_resize224",
        "bit_accuracy",
        "adapted_attack",
        "ADS/Pulsar-paper resize-to-224 primitive adapted into the workspace Pulsar runner.",
    ),
    SelectedAttack(
        "pulsar",
        "ads",
        "jpeg90",
        "ads_jpeg90",
        "bit_accuracy",
        "adapted_attack",
        "ADS/Pulsar-paper JPEG Q90 primitive adapted into the workspace Pulsar runner.",
    ),
    SelectedAttack(
        "pulsar",
        "ads",
        "jpeg70",
        "ads_jpeg70",
        "bit_accuracy",
        "adapted_attack",
        "ADS/Pulsar-paper JPEG Q70 primitive adapted into the workspace Pulsar runner.",
    ),
    SelectedAttack(
        "pulsar",
        "mblur",
        "3",
        "median_blur_k3",
        "bit_accuracy",
        "native_official",
        "Calibration produced native reveal failures within quality budget.",
    ),
    SelectedAttack(
        "pulsar",
        "gblur",
        "3",
        "gaussian_blur_k3",
        "bit_accuracy",
        "native_official",
        "Calibration produced native reveal failures within quality budget.",
    ),
    SelectedAttack(
        "pulsar",
        "regen_vae",
        "6",
        "regen_vae_q6",
        "bit_accuracy",
        "adapted_attack",
        "Calibration produced native reveal failures within quality budget.",
    ),
)


def selected_for_methods(methods: set[str] | None = None) -> list[SelectedAttack]:
    if methods is None:
        return list(SELECTED_ATTACKS)
    return [item for item in SELECTED_ATTACKS if item.method in methods]


def default_count_for(method: str) -> int:
    try:
        return DEFAULT_FORMAL_COUNTS[method]
    except KeyError as exc:
        raise ValueError(f"unsupported method for selected attack matrix: {method}") from exc


def heldout_count_for(method: str) -> int:
    count = default_count_for(method) - CALIBRATION_SAMPLE_COUNT
    if count <= 0:
        raise ValueError(f"no held-out samples remain for method: {method}")
    return count


def baseline_provenance_for(method: str) -> str:
    try:
        return BASELINE_PROVENANCE[method]
    except KeyError as exc:
        raise ValueError(f"unsupported method for selected attack matrix: {method}") from exc


def attack_provenance_for(spec: SelectedAttack) -> str:
    if spec.attack == "regen_vae":
        return "adapted_watermarkattacker_regen_vae"
    if spec.attack == "unmarker":
        return "adapted_unmarker_smoke"
    if spec.attack == "ads":
        return "adapted_ads_pulsar_paper"
    return "common_image_transform"
