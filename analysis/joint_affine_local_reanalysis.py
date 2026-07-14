#!/usr/bin/env python3
"""按 6 个 source×seed cell 重算 joint-affine 配对统计。"""

from __future__ import annotations

from collections.abc import Mapping
import json

import pandas as pd
from scipy import stats


COMPARISONS = {
    "JA1": ("joint_affine_floor", "strong_aug_floor"),
    "JA2": ("warp_alpha100", "joint_affine_floor"),
    "JA3": ("SLAug", "warp_alpha100"),
    "JA4": ("SLAug", "joint_affine_floor"),
    "TOTAL": ("SLAug", "strong_aug_floor"),
}

METRICS = (
    "dice_mean",
    "hd95_px_mean",
    "weighted_fbeta_mean",
    "boundary_iou_mean",
)


def six_cell_comparison(
    frame: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    metric: str,
    morphology_group: str,
) -> dict[str, object]:
    """先在 case 内聚合到 source×seed cell，再计算配对 t 区间。"""
    subset = frame
    if morphology_group != "all":
        subset = subset[subset["morphology_group"] == morphology_group]

    cell = (
        subset.groupby(["method", "source_center", "seed"], as_index=False)[metric]
        .mean()
    )
    index = ["source_center", "seed"]
    candidate_values = cell[cell["method"] == candidate].set_index(index)[metric]
    reference_values = cell[cell["method"] == reference].set_index(index)[metric]
    delta = (candidate_values - reference_values).sort_index()
    if len(delta) != 6 or delta.isna().any():
        raise ValueError(
            f"expected six complete cells for {candidate}-{reference}, got {len(delta)}"
        )

    mean_delta = float(delta.mean())
    standard_error = float(stats.sem(delta))
    critical_value = float(stats.t.ppf(0.975, len(delta) - 1))
    _, paired_t_p = stats.ttest_1samp(delta, 0.0)
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "morphology_group": morphology_group,
        "n_cells": len(delta),
        "mean_delta": mean_delta,
        "positive_cells": int((delta > 0).sum()),
        "ci_low": mean_delta - critical_value * standard_error,
        "ci_high": mean_delta + critical_value * standard_error,
        "paired_t_p": float(paired_t_p),
        "cell_deltas": [float(value) for value in delta],
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """对具名比较做 Holm step-down 校正并保持单调性。"""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min(1.0, float(value) * (total - rank))
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def build_results(frame: pd.DataFrame) -> pd.DataFrame:
    """生成 40 行本地审计表，并保持预注册 family 互斥。"""
    rows: list[dict[str, object]] = []
    for morphology_group in ("hard_flat_IIa", "all"):
        for metric in METRICS:
            group_rows: list[dict[str, object]] = []
            for comparison_id, (candidate, reference) in COMPARISONS.items():
                result = six_cell_comparison(
                    frame,
                    candidate=candidate,
                    reference=reference,
                    metric=metric,
                    morphology_group=morphology_group,
                )
                result["comparison_id"] = comparison_id
                result["ci_source"] = "paired_t_6_source_seed_cells"
                result["cell_deltas_json"] = json.dumps(result.pop("cell_deltas"))
                if comparison_id in {"JA1", "JA2", "JA4"}:
                    result["holm_family"] = "F3-new:JA1,JA2,JA4"
                elif comparison_id == "JA3":
                    result["holm_family"] = "F2/F5-existing:not-recomputed"
                else:
                    result["holm_family"] = "F1-existing:not-recomputed"
                result["paired_t_p_holm"] = float("nan")
                group_rows.append(result)

            new_family = {
                row["comparison_id"]: float(row["paired_t_p"])
                for row in group_rows
                if row["comparison_id"] in {"JA1", "JA2", "JA4"}
            }
            adjusted = holm_adjust(new_family)
            for row in group_rows:
                comparison_id = str(row["comparison_id"])
                if comparison_id in adjusted:
                    row["paired_t_p_holm"] = adjusted[comparison_id]
            rows.extend(group_rows)
    return pd.DataFrame(rows)
