#!/usr/bin/env python3
"""Select attack parameters under fixed stego-vs-attacked quality budgets."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from identity_common import image_lpips, image_metrics  # noqa: E402


KNOWN_METHODS = ["mddm_128_pilot", "gsd_cifar10", "mas_grdh", "pulsar", "cross"]
KNOWN_ATTACKS = ["regen_vae", "ads_fgsm", "ads_qdir", "ecrs_fast", "ecrs_diff", "unmarker", "resize", "jpeg", "mblur", "gblur"]
BIT_PAYLOAD_METHODS = {"gsd_cifar10", "mas_grdh", "mddm_128_pilot", "pulsar"}
IMAGE_PAYLOAD_METHODS = {"cross", "rgs"}


@dataclass(frozen=True)
class ParsedName:
    method: str
    attack: str
    factor: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--psnr-min", type=float, default=30.0)
    parser.add_argument("--lpips-max", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_dir_name(name: str) -> ParsedName | None:
    for method in KNOWN_METHODS:
        prefix = f"{method}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        for attack in KNOWN_ATTACKS:
            attack_prefix = f"{attack}_"
            if rest.startswith(attack_prefix):
                factor_count = rest[len(attack_prefix) :]
                parts = factor_count.split("_")
                factor = "_".join(parts[:-1]) if parts and parts[-1].isdigit() else factor_count
                return ParsedName(method=method, attack=attack, factor=factor.replace("_", "."))
        return None
    return None


def as_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def exact_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if str(row.get("exact_match", "")).lower() == "true")


def score_metric_name(method: str, rows: list[dict[str, str]]) -> str:
    if any(row.get("bit_accuracy") not in (None, "") for row in rows):
        return "bit_accuracy"
    if any(row.get("recovery_psnr") not in (None, "") for row in rows):
        return "recovery_psnr"
    if method in BIT_PAYLOAD_METHODS:
        return "bit_accuracy"
    if method in IMAGE_PAYLOAD_METHODS:
        return "recovery_psnr"
    return "result_metric"


def row_pair_paths(row: dict[str, str]) -> tuple[Path, Path] | None:
    stego = row.get("stego_path") or row.get("image_path")
    attacked = row.get("attacked_path")
    if not stego or not attacked:
        return None
    stego_path = Path(stego)
    attacked_path = Path(attacked)
    if not stego_path.exists() or not attacked_path.exists():
        return None
    return stego_path, attacked_path


def summarize_dir(path: Path, device: str) -> dict[str, object] | None:
    parsed = parse_dir_name(path.name)
    if parsed is None:
        return None
    rows = read_rows(path / "identity_results.csv")
    failures = read_rows(path / "identity_failures.csv")
    if not rows and not failures:
        return None
    quality_psnr: list[float] = []
    quality_lpips: list[float] = []
    quality_ssim: list[float] = []
    quality_mae: list[float] = []
    scorable_failures = [row for row in failures if row_pair_paths(row) is not None]
    unscorable_failures = len(failures) - len(scorable_failures)
    for row in rows + scorable_failures:
        pair = row_pair_paths(row)
        if pair is None:
            psnr = as_float(row.get("attack_psnr"))
            if psnr is not None:
                quality_psnr.append(psnr)
            ssim = as_float(row.get("attack_ssim"))
            if ssim is not None:
                quality_ssim.append(ssim)
            continue
        metrics = image_metrics(pair[0], pair[1])
        psnr = as_float(metrics.get("psnr"))
        if psnr is not None:
            quality_psnr.append(psnr)
        ssim = as_float(metrics.get("ssim"))
        if ssim is not None:
            quality_ssim.append(ssim)
        mae = as_float(metrics.get("mae"))
        if mae is not None:
            quality_mae.append(mae)
        lp = image_lpips(pair[0], pair[1], device=device)
        lp_float = as_float(lp)
        if lp_float is not None:
            quality_lpips.append(lp_float)

    metric_name = score_metric_name(parsed.method, rows)
    metric_values = [value for row in rows if (value := as_float(row.get(metric_name))) is not None]
    if metric_name in {"bit_accuracy", "recovery_psnr"}:
        # A native reveal/decode failure after an attacked image was produced is
        # a complete payload-recovery failure for attack selection. Quality is
        # still computed independently from image_path/attacked_path above.
        metric_values.extend(0.0 for _ in scorable_failures)
    runtime_values = [
        value for row in rows + scorable_failures if (value := as_float(row.get("runtime_s"))) is not None
    ]
    return {
        "name": path.name,
        "method": parsed.method,
        "attack": parsed.attack,
        "factor": parsed.factor,
        "rows": len(rows),
        "failures": len(scorable_failures),
        "unscorable_failures": unscorable_failures,
        "scored_total": len(rows) + len(scorable_failures),
        "metric": metric_name,
        "metric_mean": mean(metric_values),
        "quality_psnr_mean": mean(quality_psnr),
        "quality_psnr_n": len(quality_psnr),
        "quality_lpips_mean": mean(quality_lpips),
        "quality_lpips_n": len(quality_lpips),
        "quality_ssim_mean": mean(quality_ssim),
        "quality_mae_mean": mean(quality_mae),
        "exact": exact_count(rows),
        "runtime_s_mean": mean(runtime_values),
    }


def is_within_budget(row: dict[str, object], psnr_min: float, lpips_max: float) -> bool:
    psnr = as_float(row.get("quality_psnr_mean"))
    lpips_value = as_float(row.get("quality_lpips_mean"))
    scored_total = int(row.get("scored_total", 0) or 0)
    psnr_n = int(row.get("quality_psnr_n", 0) or 0)
    lpips_n = int(row.get("quality_lpips_n", 0) or 0)
    if scored_total == 0 or int(row.get("unscorable_failures", 0) or 0):
        return False
    if psnr_n != scored_total or lpips_n != scored_total:
        return False
    if psnr is None or psnr < psnr_min:
        return False
    if lpips_value is None or lpips_value > lpips_max:
        return False
    return True


def is_better(candidate: dict[str, object], incumbent: dict[str, object] | None) -> bool:
    if incumbent is None:
        return True
    cand_metric = as_float(candidate.get("metric_mean"))
    inc_metric = as_float(incumbent.get("metric_mean"))
    if cand_metric is None:
        return False
    if inc_metric is None:
        return True
    # Lower bit accuracy/recovery PSNR means stronger attack.
    if cand_metric != inc_metric:
        return cand_metric < inc_metric
    cand_psnr = as_float(candidate.get("quality_psnr_mean")) or -float("inf")
    inc_psnr = as_float(incumbent.get("quality_psnr_mean")) or -float("inf")
    return cand_psnr > inc_psnr


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    summaries = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name == "logs" or path.name.endswith(".running"):
            continue
        summary = summarize_dir(path, args.device)
        if summary is not None:
            summaries.append(summary)

    selected: dict[tuple[str, str], dict[str, object]] = {}
    for row in summaries:
        if not is_within_budget(row, args.psnr_min, args.lpips_max):
            continue
        key = (str(row["method"]), str(row["attack"]))
        if is_better(row, selected.get(key)):
            selected[key] = row

    fields = [
        "method",
        "attack",
        "factor",
        "within_budget",
        "rows",
        "failures",
        "unscorable_failures",
        "scored_total",
        "metric",
        "metric_mean",
        "quality_psnr_mean",
        "quality_psnr_n",
        "quality_lpips_mean",
        "quality_lpips_n",
        "quality_ssim_mean",
        "quality_mae_mean",
        "exact",
        "runtime_s_mean",
        "name",
    ]
    output_path = Path(args.output).resolve() if args.output else root / "quality_budget_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            out = dict(row)
            out["within_budget"] = is_within_budget(row, args.psnr_min, args.lpips_max)
            writer.writerow({field: out.get(field, "") for field in fields})

    selected_path = output_path.with_name(output_path.stem + "_selected.csv")
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(selected.values(), key=lambda item: (str(item["method"]), str(item["attack"]))):
            out = dict(row)
            out["within_budget"] = True
            writer.writerow({field: out.get(field, "") for field in fields})

    print(f"root: {root}")
    print(f"budget: PSNR >= {args.psnr_min:g} dB, LPIPS <= {args.lpips_max:g} (both required)")
    print(f"summaries: {len(summaries)}")
    print(f"summary_csv: {output_path}")
    print(f"selected_csv: {selected_path}")
    print(f"{'method':16s} {'attack':10s} {'factor':12s} {'metric':14s} {'mean':>10s} {'psnr':>10s} {'lpips':>10s}")
    for row in sorted(selected.values(), key=lambda item: (str(item["method"]), str(item["attack"]))):
        print(
            f"{str(row['method']):16s} {str(row['attack']):10s} {str(row['factor']):12s} "
            f"{str(row['metric']):14s} {fmt(row['metric_mean']):>10s} "
            f"{fmt(row['quality_psnr_mean']):>10s} {fmt(row['quality_lpips_mean']):>10s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
