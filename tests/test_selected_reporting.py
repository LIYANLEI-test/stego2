from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import attack_common
import render_paper_tables as render
import select_quality_budget_attacks as selector
import summarize_attack_deltas as deltas
import summarize_selected_attack_runs as summary
import audit_selected_attack_results as audit
from selected_attack_matrix import SELECTED_ATTACKS, attack_provenance_for


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SelectedSummaryTests(unittest.TestCase):
    def test_formal_summary_excludes_calibration_and_scores_saved_reveal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = SELECTED_ATTACKS[0]
            out = root / "cross_resize_1_5_100"
            stego = out / "stego.png"
            attacked = out / "attacked.png"
            stego.parent.mkdir(parents=True)
            stego.touch()
            attacked.touch()
            write_rows(
                out / "identity_results.csv",
                [
                    {"sample_index": 0, "recovery_psnr": 99.0, "exact_match": "False", "runtime_s": 1.0},
                    {"sample_index": 10, "recovery_psnr": 20.0, "exact_match": "False", "runtime_s": 1.0},
                ],
            )
            write_rows(
                out / "identity_failures.csv",
                [{"sample_index": 11, "stego_path": stego, "attacked_path": attacked, "runtime_s": 1.0}],
            )
            with patch.object(summary, "quality_values", return_value=([40.0, 40.0], [1.0, 1.0], [])):
                row = summary.summarize_one(root, spec, "cpu", False, 2, False)
            self.assertEqual(row["rows"], 1)
            self.assertEqual(row["failures"], 1)
            self.assertEqual(row["unscorable_failures"], 0)
            self.assertEqual(row["recovery_mean"], 10.0)
            self.assertTrue(row["complete"])

    def test_unscorable_failure_does_not_complete_formal_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = SELECTED_ATTACKS[0]
            out = root / "cross_resize_1_5_100"
            write_rows(
                out / "identity_results.csv",
                [{"sample_index": 10, "recovery_psnr": 20.0, "exact_match": "False", "runtime_s": 1.0}],
            )
            write_rows(out / "identity_failures.csv", [{"sample_index": 11, "runtime_s": 1.0}])
            with patch.object(summary, "quality_values", return_value=([40.0], [1.0], [])):
                row = summary.summarize_one(root, spec, "cpu", False, 2, False)
            self.assertEqual(row["total"], 1)
            self.assertEqual(row["unscorable_failures"], 1)
            self.assertFalse(row["complete"])


class QualityAndTableTests(unittest.TestCase):
    def test_soft_blur_factors_create_milder_non_identity_attacks(self) -> None:
        array = np.zeros((7, 7, 3), dtype=np.uint8)
        array[:, :3] = 255
        array[3, 3] = 255
        image = Image.fromarray(array, "RGB")

        median_soft = np.asarray(attack_common.median_blur_pil(image, 0.5))
        median_full = np.asarray(attack_common.median_blur_pil(image, 3))
        self.assertGreater(np.abs(median_soft.astype(int) - array.astype(int)).sum(), 0)
        self.assertLess(
            np.abs(median_soft.astype(int) - array.astype(int)).sum(),
            np.abs(median_full.astype(int) - array.astype(int)).sum(),
        )

        gaussian_soft = np.asarray(attack_common.gaussian_blur_pil(image, 0.5))
        self.assertGreater(np.abs(gaussian_soft.astype(int) - array.astype(int)).sum(), 0)

    def test_ads_attack_kinds_are_paper_attack_variants(self) -> None:
        array = np.zeros((16, 16, 3), dtype=np.uint8)
        array[:, :8] = 255
        image = Image.fromarray(array, "RGB")

        self.assertEqual(attack_common.ADS_ATTACK_KINDS, ("ads_fgsm", "ads_qdir"))
        self.assertEqual(attack_common.attack_suffix("ads_qdir", attack_factor=0.01), "ads_qdir_eps0_01")
        with self.assertRaises(ValueError):
            attack_common.apply_attack_pil(image, "ads_resize224")

    def test_selected_matrix_includes_ads_as_adapted_attack_candidates(self) -> None:
        ads_rows = [item for item in SELECTED_ATTACKS if item.attack in attack_common.ADS_ATTACK_KINDS]
        self.assertEqual(
            {item.method for item in ads_rows},
            {"cross", "gsd_cifar10", "mas_grdh", "mddm_128_pilot", "pulsar"},
        )
        self.assertEqual({item.attack for item in ads_rows}, {"ads_fgsm", "ads_qdir"})
        self.assertEqual({item.factor for item in ads_rows}, {"0.01"})
        self.assertTrue(all(item.provenance == "adapted_attack" for item in ads_rows))
        self.assertTrue(all(attack_provenance_for(item) == "adapted_ads" for item in ads_rows))

    def test_selector_requires_complete_psnr_and_lpips_coverage(self) -> None:
        good = {
            "scored_total": 2,
            "unscorable_failures": 0,
            "quality_psnr_n": 2,
            "quality_lpips_n": 2,
            "quality_psnr_mean": 31.0,
            "quality_lpips_mean": 0.09,
        }
        self.assertTrue(selector.is_within_budget(good, 30.0, 0.10))
        self.assertFalse(selector.is_within_budget({**good, "quality_lpips_n": 1}, 30.0, 0.10))
        self.assertFalse(selector.is_within_budget({**good, "quality_lpips_mean": ""}, 30.0, 0.10))

    def test_table_separates_appendix_and_shows_conditional_delta(self) -> None:
        base = {
            "attack": "jpeg",
            "factor": "70",
            "label": "jpeg_q70",
            "metric": "bit_accuracy",
            "total": "40",
            "target": "40",
            "recovery_mean": "0.5",
            "delta_delta_mean": "0.2",
            "delta_delta_on_identity_success_mean": "0.3",
            "quality_psnr_mean": "31.0",
            "quality_psnr_n": "40",
            "quality_lpips_mean": "0.05",
            "quality_lpips_n": "40",
            "failure_rate": "0",
        }
        text = render.render_markdown(
            [
                {**base, "method": "gsd_cifar10", "provenance": "native_official"},
                {**base, "method": "mddm_128_pilot", "provenance": "native_third_party"},
            ]
        )
        main, appendix = text.split("## Appendix Pilot Table")
        self.assertIn("gsd_cifar10", main)
        self.assertNotIn("mddm_128_pilot", main)
        self.assertIn("mddm_128_pilot", appendix)
        self.assertIn("Delta (ID-ok)", text)

    def test_render_does_not_hide_incomplete_row_in_final_mode(self) -> None:
        rows = render.merged_rows(
            [{"method": "pulsar", "attack": "jpeg", "factor": "95", "complete": "False"}],
            [],
            include_incomplete=False,
        )
        self.assertEqual(len(rows), 1)

    def test_final_audit_does_not_hide_incomplete_unscorable_row(self) -> None:
        summary_rows = [
            {
                "method": "pulsar",
                "attack": "jpeg",
                "factor": "95",
                "provenance": "native_official",
                "complete": "False",
                "total": "0",
                "unscorable_failures": "1",
            }
        ]
        rows = audit.audit_rows(summary_rows, [], include_incomplete=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["budget_status"], "fail")
        self.assertIn("unscorable_failures=1", rows[0]["paper_caveats"])


class DeltaFailureTests(unittest.TestCase):
    def test_attack_failure_requires_saved_pair_for_zero_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            stego = directory / "stego.png"
            attacked = directory / "attacked.png"
            stego.touch()
            attacked.touch()
            write_rows(
                directory / "identity_failures.csv",
                [
                    {"sample_index": 10, "stego_path": stego, "attacked_path": attacked},
                    {"sample_index": 11, "stego_path": stego, "attacked_path": ""},
                ],
            )
            values, _, scored, unscored, _ = deltas.load_metric_map(directory, "pulsar", False, True)
            self.assertEqual(values, {10: 0.0})
            self.assertEqual(scored, 1)
            self.assertEqual(unscored, 1)

    def test_old_gsd_failure_row_can_recover_saved_attack_pair_from_run_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            images = directory / "images"
            images.mkdir()
            (images / "stego_000010.png").touch()
            (images / "stego_000010_resize_1_25.png").touch()
            write_rows(directory / "identity_failures.csv", [{"sample_index": 10}])
            values, _, scored, unscored, _ = deltas.load_metric_map(directory, "gsd_cifar10", False, True)
            self.assertEqual(values, {10: 0.0})
            self.assertEqual(scored, 1)
            self.assertEqual(unscored, 0)


if __name__ == "__main__":
    unittest.main()
