#!/usr/bin/env python3
"""重算 multiplicity-controlled shape/size families 与分层响应对比。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

import video_cluster_stats as paired_stats


F4_SHAPE_MEMBERS = (
    ("IIa", {"shape": "IIa"}),
    ("Is", {"shape": "Is"}),
    ("Ip", {"shape": "Ip"}),
    ("Isp", {"shape": "Isp"}),
)
F4_SIZE_MEMBERS = (
    ("small", {"size_bin": "small"}),
    ("mid", {"size_bin": "mid"}),
    ("large", {"size_bin": "large"}),
)


def _cell_p_values(values: np.ndarray) -> tuple[float, float, float]:
    if np.allclose(values, values[0], rtol=0.0, atol=1e-15):
        paired_t_p = 1.0 if abs(float(values[0])) <= 1e-15 else 0.0
        sign_flip_p = paired_stats.exact_sign_flip_p(values)
        wilcoxon_p = 1.0 if abs(float(values[0])) <= 1e-15 else sign_flip_p
    else:
        paired_t_p = float(ttest_1samp(values, popmean=0.0).pvalue)
        try:
            wilcoxon_p = float(
                wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
        sign_flip_p = paired_stats.exact_sign_flip_p(values)
    return paired_t_p, wilcoxon_p, sign_flip_p


def summarize_family_with_contrast(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    model: str,
    metric: str,
    family_label: str,
    members: Sequence[tuple[str, Mapping[str, object]]],
    contrast: tuple[str, str, str],
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总分层主效应，并把锁定的两个分层效应之差纳入同一 Holm 族。"""
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    member_cells: dict[str, pd.DataFrame] = {}
    member_filters: dict[str, Mapping[str, object]] = {}
    for member_label, filters in members:
        result = paired_stats.summarize_comparison(
            table,
            candidate=candidate,
            reference=reference,
            metric=metric,
            model=model,
            filters=filters,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        cells = pd.DataFrame(result.pop("cell_rows"))
        member_cells[member_label] = cells
        member_filters[member_label] = filters
        result.pop("filters")
        result["comparison_id"] = member_label
        result["multiplicity_family"] = family_label
        result["inference_role"] = "stratum_effect"
        summary_rows.append(result)
        for row in cells.to_dict(orient="records"):
            cell_rows.append(
                {
                    "multiplicity_family": family_label,
                    "comparison_id": member_label,
                    "candidate": candidate,
                    "reference": reference,
                    "model": model,
                    "metric": metric,
                    **row,
                }
            )

    contrast_label, left_label, right_label = contrast
    if left_label not in member_cells or right_label not in member_cells:
        raise ValueError("contrast members must be present in the family")
    contrasted = paired_stats.paired_cell_contrast(
        member_cells[left_label],
        member_cells[right_label],
    )
    values = contrasted["contrast_delta"].to_numpy(dtype=np.float64)
    cell_t_ci_low, cell_t_ci_high = paired_stats.cell_t_confidence_interval(values)
    paired_t_p, wilcoxon_p, sign_flip_p = _cell_p_values(values)
    video_ci_low, video_ci_high = paired_stats.cluster_bootstrap_contrast_ci(
        table,
        candidate=candidate,
        reference=reference,
        metric=metric,
        model=model,
        left_filters=member_filters[left_label],
        right_filters=member_filters[right_label],
        n_boot=n_boot,
        random_seed=random_seed,
    )
    summary_rows.append(
        {
            "candidate": candidate,
            "reference": reference,
            "model": model,
            "metric": metric,
            "n_sources": int(contrasted["source_center"].nunique()),
            "n_source_seed_cells": int(len(contrasted)),
            "n_paired_cases": np.nan,
            "n_paired_videos": np.nan,
            "mean_delta": float(values.mean()),
            "cell_t_ci_low": cell_t_ci_low,
            "cell_t_ci_high": cell_t_ci_high,
            "supporting_video_cluster_ci_low": video_ci_low,
            "supporting_video_cluster_ci_high": video_ci_high,
            "positive_cells": int((values > 0).sum()),
            "paired_t_p": paired_t_p,
            "wilcoxon_p": wilcoxon_p,
            "exact_sign_flip_p": sign_flip_p,
            "comparison_id": contrast_label,
            "multiplicity_family": family_label,
            "inference_role": "multiplicity_controlled_exploratory_contrast",
        }
    )
    for row in contrasted.to_dict(orient="records"):
        cell_rows.append(
            {
                "multiplicity_family": family_label,
                "comparison_id": contrast_label,
                "candidate": candidate,
                "reference": reference,
                "model": model,
                "metric": metric,
                "source_center": row["source_center"],
                "seed": int(row["seed"]),
                "delta": float(row["contrast_delta"]),
                "delta_left": float(row["delta_left"]),
                "delta_right": float(row["delta_right"]),
            }
        )

    holm_values = paired_stats.holm_adjust(
        [float(row["paired_t_p"]) for row in summary_rows]
    )
    for row, holm_p in zip(summary_rows, holm_values, strict=True):
        row["holm_p"] = holm_p
    return pd.DataFrame(summary_rows), pd.DataFrame(cell_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_family_bundle(
    case_csv: Path,
    output_dir: Path,
    candidate: str = "SLAug",
    reference: str = "strong_aug_floor",
    model: str = "polyp_pvt",
    metric: str = "dice_mean",
    n_boot: int = 10000,
    random_seed: int = 20260713,
) -> dict[str, Path]:
    table = pd.read_csv(case_csv)
    shape_summary, shape_cells = summarize_family_with_contrast(
        table,
        candidate=candidate,
        reference=reference,
        model=model,
        metric=metric,
        family_label="F4-shape",
        members=F4_SHAPE_MEMBERS,
        contrast=("IIa_minus_Ip", "IIa", "Ip"),
        n_boot=n_boot,
        random_seed=random_seed,
    )
    size_summary, size_cells = summarize_family_with_contrast(
        table,
        candidate=candidate,
        reference=reference,
        model=model,
        metric=metric,
        family_label="F4-size",
        members=F4_SIZE_MEMBERS,
        contrast=("large_minus_small", "large", "small"),
        n_boot=n_boot,
        random_seed=random_seed,
    )
    summary = pd.concat([shape_summary, size_summary], ignore_index=True, sort=False)
    cells = pd.concat([shape_cells, size_cells], ignore_index=True, sort=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "f4_family_summary.csv"
    cells_path = output_dir / "f4_family_cells.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    summary.to_csv(summary_path, index=False)
    cells.to_csv(cells_path, index=False)
    manifest = {
        "state": "done",
        "protocol": "Outcome-informed exploratory families: shape contains four Paris-stratum effects plus IIa-minus-Ip contrast; size contains three area-tertile effects plus large-minus-small contrast. Six repeated source-seed training units summarize fixed C1/C3 settings; Holm is applied within family.",
        "case_csv": str(case_csv.resolve()),
        "case_csv_sha256": _sha256(case_csv),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "paired_stats_script": str(Path(paired_stats.__file__).resolve()),
        "paired_stats_script_sha256": _sha256(Path(paired_stats.__file__).resolve()),
        "candidate": candidate,
        "reference": reference,
        "model": model,
        "metric": metric,
        "n_boot_for_stratum_supporting_intervals": n_boot,
        "random_seed": random_seed,
        "contrast_supporting_interval": "computed by synchronized video-cluster bootstrap with seed resampling within fixed source settings",
        "n_summary_rows": int(len(summary)),
        "n_cell_rows": int(len(cells)),
        "summary_csv": str(summary_path.resolve()),
        "cells_csv": str(cells_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary_path, "cells": cells_path, "manifest": manifest_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    paths = write_family_bundle(
        case_csv=args.case_csv,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
