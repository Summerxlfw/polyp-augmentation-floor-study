from pathlib import Path
import sys
import warnings

import pandas as pd
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(MODULE_DIR))

import f4_family_stats as family  # noqa: E402
import video_cluster_stats as stats  # noqa: E402


def _phenotype_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    effects = {"IIa": 0.20, "Is": 0.05, "Ip": -0.10, "Isp": 0.00}
    for source in ("C1", "C3"):
        for seed in (0, 1, 2):
            for index, (shape, effect) in enumerate(effects.items()):
                common = {
                    "source_center": source,
                    "model": "polyp_pvt",
                    "seed": seed,
                    "case_id": f"case_{shape}",
                    "video_id": f"video_{index}",
                    "shape": shape,
                    "morphology_group": "hard_flat_IIa" if shape == "IIa" else "non_flat",
                    "size_bin": "large" if shape == "Ip" else "small",
                }
                rows.append({**common, "method": "floor", "dice_mean": 0.5})
                rows.append({**common, "method": "candidate", "dice_mean": 0.5 + effect})
    return pd.DataFrame(rows)


def test_f4_shape_includes_registered_did_and_holm_family() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary, cells = family.summarize_family_with_contrast(
            _phenotype_table(),
            candidate="candidate",
            reference="floor",
            model="polyp_pvt",
            metric="dice_mean",
            family_label="F4-shape",
            members=(
                ("IIa", {"shape": "IIa"}),
                ("Is", {"shape": "Is"}),
                ("Ip", {"shape": "Ip"}),
                ("Isp", {"shape": "Isp"}),
            ),
            contrast=("IIa_minus_Ip", "IIa", "Ip"),
            n_boot=40,
            random_seed=7,
        )

    assert summary["comparison_id"].tolist() == [
        "IIa",
        "Is",
        "Ip",
        "Isp",
        "IIa_minus_Ip",
    ]
    did = summary[summary["comparison_id"] == "IIa_minus_Ip"].iloc[0]
    assert did["mean_delta"] == pytest.approx(0.30)
    assert did["supporting_video_cluster_ci_low"] == pytest.approx(0.30)
    assert did["supporting_video_cluster_ci_high"] == pytest.approx(0.30)
    assert did["inference_role"] == "multiplicity_controlled_exploratory_contrast"
    assert summary["holm_p"].tolist() == pytest.approx(
        stats.holm_adjust(summary["paired_t_p"].tolist())
    )
    assert len(cells) == 30
    assert not caught
