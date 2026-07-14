from pathlib import Path
import sys
import warnings

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "video_cluster_stats.py"


def test_analysis_module_exists() -> None:
    assert MODULE_PATH.is_file(), "video-clustered analysis module is not implemented"


sys.path.insert(0, str(MODULE_PATH.parent))

import video_cluster_stats as stats  # noqa: E402


def _table(sources: tuple[str, ...] = ("C1", "C3"), constant_delta: float | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cases = (("a1", "video_a", 0.3), ("a2", "video_a", 0.3), ("b1", "video_b", -0.3))
    for source in sources:
        for seed in (0, 1, 2):
            for case_id, video_id, delta in cases:
                if constant_delta is not None:
                    delta = constant_delta
                common = {
                    "source_center": source,
                    "model": "polyp_pvt",
                    "seed": seed,
                    "case_id": case_id,
                    "video_id": video_id,
                    "morphology_group": "hard_flat_IIa",
                    "shape": "IIa",
                    "size_bin": "mid",
                }
                rows.append({**common, "method": "reference", "dice_mean": 0.5})
                rows.append({**common, "method": "candidate", "dice_mean": 0.5 + delta})
    return pd.DataFrame(rows)


def _cross_cell_canceling_table() -> pd.DataFrame:
    """同一视频在两个 source 上的效应逐视频抵消。"""
    rows: list[dict[str, object]] = []
    for source, signs in (("C1", {"video_a": 1.0, "video_b": -1.0}),
                          ("C3", {"video_a": -1.0, "video_b": 1.0})):
        for seed in (0, 1, 2):
            for video_id, delta in signs.items():
                common = {
                    "source_center": source,
                    "model": "polyp_pvt",
                    "seed": seed,
                    "case_id": f"case_{video_id}",
                    "video_id": video_id,
                    "morphology_group": "hard_flat_IIa",
                    "shape": "IIa",
                    "size_bin": "mid",
                }
                rows.append({**common, "method": "reference", "dice_mean": 0.0})
                rows.append({**common, "method": "candidate", "dice_mean": delta})
    return pd.DataFrame(rows)


def test_case_equal_point_estimate_is_not_replaced_by_video_equal_mean() -> None:
    paired = stats.paired_case_deltas(
        _table(),
        candidate="candidate",
        reference="reference",
        metric="dice_mean",
        model="polyp_pvt",
        filters={"morphology_group": "hard_flat_IIa"},
    )

    cells = stats.case_equal_cell_deltas(paired)

    assert len(cells) == 6
    assert cells["delta"].tolist() == pytest.approx([0.1] * 6)
    assert cells["n_cases"].tolist() == [3] * 6
    assert cells["n_videos"].tolist() == [2] * 6
    assert cells["delta"].mean() != pytest.approx(0.0)


def test_cluster_bootstrap_is_degenerate_for_constant_case_deltas() -> None:
    paired = stats.paired_case_deltas(
        _table(constant_delta=0.2),
        candidate="candidate",
        reference="reference",
        metric="dice_mean",
        model="polyp_pvt",
        filters={"morphology_group": "hard_flat_IIa"},
    )

    low, high = stats.cluster_bootstrap_ci(paired, n_boot=200, random_seed=7)

    assert low == pytest.approx(0.2)
    assert high == pytest.approx(0.2)


def test_video_resample_is_aligned_across_all_source_seed_cells() -> None:
    paired = stats.paired_case_deltas(
        _cross_cell_canceling_table(),
        candidate="candidate",
        reference="reference",
        metric="dice_mean",
        model="polyp_pvt",
        filters={"morphology_group": "hard_flat_IIa"},
    )

    low, high = stats.cluster_bootstrap_ci(paired, n_boot=300, random_seed=9)

    # 同一批视频若跨 cell 同步重采样，两个 source 在每次 draw 都逐视频抵消。
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_pairing_rejects_a_missing_candidate_case() -> None:
    table = _table()
    missing = table[
        ~(
            (table["method"] == "candidate")
            & (table["source_center"] == "C1")
            & (table["seed"] == 0)
            & (table["case_id"] == "a1")
        )
    ]

    with pytest.raises(ValueError, match="case keys differ"):
        stats.paired_case_deltas(
            missing,
            candidate="candidate",
            reference="reference",
            metric="dice_mean",
            model="polyp_pvt",
            filters={"morphology_group": "hard_flat_IIa"},
        )


def test_summary_rejects_partial_source_seed_cells() -> None:
    with pytest.raises(ValueError, match="expected 6 source-seed cells"):
        stats.summarize_comparison(
            _table(sources=("C1",)),
            candidate="candidate",
            reference="reference",
            metric="dice_mean",
            model="polyp_pvt",
            filters={"morphology_group": "hard_flat_IIa"},
            n_boot=100,
        )


def test_holm_adjustment_matches_step_down_definition() -> None:
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cell_t_interval_is_distinct_from_supporting_video_interval() -> None:
    low, high = stats.cell_t_confidence_interval([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    assert low == pytest.approx(0.536685, abs=1e-6)
    assert high == pytest.approx(4.463315, abs=1e-6)


def test_paired_cell_contrast_preserves_the_six_training_units() -> None:
    left = pd.DataFrame(
        {
            "source_center": ["C1", "C1", "C1", "C3", "C3", "C3"],
            "seed": [0, 1, 2, 0, 1, 2],
            "delta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    right = left.assign(delta=[0.0, 0.1, 0.1, 0.2, 0.3, 0.4])

    contrast = stats.paired_cell_contrast(left, right)

    assert len(contrast) == 6
    assert contrast["contrast_delta"].tolist() == pytest.approx(
        [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    )


def test_paired_cell_contrast_rejects_different_cell_universes() -> None:
    left = pd.DataFrame(
        {"source_center": ["C1", "C3"], "seed": [0, 0], "delta": [0.1, 0.2]}
    )
    right = pd.DataFrame(
        {"source_center": ["C1"], "seed": [0], "delta": [0.0]}
    )

    with pytest.raises(ValueError, match="cell keys differ"):
        stats.paired_cell_contrast(left, right)


def test_video_cluster_contrast_bootstrap_preserves_stratum_difference() -> None:
    table = _table(constant_delta=0.2).copy()
    left = table.copy()
    left["shape"] = "IIa"
    left["case_id"] = "left_" + left["case_id"].astype(str)
    left["video_id"] = "left_" + left["video_id"].astype(str)
    right = table.copy()
    right["shape"] = "Ip"
    right["case_id"] = "right_" + right["case_id"].astype(str)
    right["video_id"] = "right_" + right["video_id"].astype(str)
    right.loc[right["method"] == "candidate", "dice_mean"] -= 0.4
    combined = pd.concat([left, right], ignore_index=True)

    low, high = stats.cluster_bootstrap_contrast_ci(
        combined,
        candidate="candidate",
        reference="reference",
        metric="dice_mean",
        model="polyp_pvt",
        left_filters={"shape": "IIa"},
        right_filters={"shape": "Ip"},
        n_boot=100,
        random_seed=11,
    )

    assert low == pytest.approx(0.4)
    assert high == pytest.approx(0.4)


def test_summary_names_primary_and_supporting_intervals_explicitly() -> None:
    result = stats.summarize_comparison(
        _table(),
        candidate="candidate",
        reference="reference",
        metric="dice_mean",
        model="polyp_pvt",
        filters={"morphology_group": "hard_flat_IIa"},
        n_boot=100,
    )

    assert result["cell_t_ci_low"] == pytest.approx(0.1)
    assert result["cell_t_ci_high"] == pytest.approx(0.1)
    assert result["supporting_video_cluster_ci_low"] <= result["mean_delta"]
    assert result["supporting_video_cluster_ci_high"] >= result["mean_delta"]
    assert "delta_ci_low" not in result
    assert "delta_ci_high" not in result


def test_cli_writes_a_traceable_analysis_bundle(tmp_path: Path) -> None:
    case_csv = tmp_path / "cases.csv"
    output_dir = tmp_path / "bundle"
    _table().to_csv(case_csv, index=False)

    return_code = stats.main(
        [
            "--case-csv",
            str(case_csv),
            "--output-dir",
            str(output_dir),
            "--candidate",
            "candidate",
            "--reference",
            "reference",
            "--n-boot",
            "100",
            "--stratum",
            "hard_flat_IIa:morphology_group:hard_flat_IIa",
        ]
    )

    assert return_code == 0
    assert (output_dir / "video_cluster_summary.csv").is_file()
    assert (output_dir / "video_cluster_cells.csv").is_file()
    assert (output_dir / "analysis_manifest.json").is_file()
    summary = pd.read_csv(output_dir / "video_cluster_summary.csv")
    assert summary.loc[0, "mean_delta"] == pytest.approx(0.1)


def test_zero_variance_cell_deltas_do_not_emit_ttest_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stats.summarize_comparison(
            _table(constant_delta=0.2),
            candidate="candidate",
            reference="reference",
            metric="dice_mean",
            model="polyp_pvt",
            filters={"morphology_group": "hard_flat_IIa"},
            n_boot=100,
        )

    assert result["paired_t_p"] == 0.0
    assert not caught
