#!/usr/bin/env python3
"""按 6 个 source×seed cell 重算 2×2 formulation contrasts。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from itertools import product
import json
from pathlib import Path

import pandas as pd
from scipy import stats


METHODS = {
    "U0B0": "warp_alpha100",
    "U1B0": "joint_affine_floor",
    "U0B1": "SLAug",
    "U1B1": "paired_affine_softmix",
}


def _exact_sign_flip_p(values: pd.Series) -> float:
    """枚举 2^n 个符号翻转，返回双侧精确 p 值。"""
    observed = abs(float(values.sum()))
    count = sum(
        abs(sum(sign * value for sign, value in zip(signs, values)))
        >= observed - 1e-12
        for signs in product((1, -1), repeat=len(values))
    )
    return count / (2 ** len(values))


def _summarize_delta(values: pd.Series) -> dict[str, object]:
    """用 6 个配对 cell 计算均值、t 区间和敏感性统计。"""
    values = values.astype(float).sort_index()
    if len(values) != 6 or values.isna().any():
        raise ValueError(f"expected six complete source×seed cells, got {len(values)}")

    mean_delta = float(values.mean())
    standard_error = float(stats.sem(values))
    critical_value = float(stats.t.ppf(0.975, len(values) - 1))
    _, paired_t_p = stats.ttest_1samp(values, 0.0)
    return {
        "n_cells": len(values),
        "mean_delta": mean_delta,
        "positive_cells": int((values > 0).sum()),
        "ci_low": mean_delta - critical_value * standard_error,
        "ci_high": mean_delta + critical_value * standard_error,
        "paired_t_p": float(paired_t_p),
        "exact_sign_flip_p": _exact_sign_flip_p(values),
        "cell_deltas": [float(value) for value in values],
    }


def factorial_contrasts(
    frame: pd.DataFrame,
    *,
    metric: str,
    morphology_group: str,
) -> dict[str, dict[str, object]]:
    """构造四个简单 contrast 与 difference-in-differences interaction。"""
    subset = frame
    if morphology_group != "all":
        subset = subset[subset["morphology_group"] == morphology_group]

    cell = subset.groupby(
        ["method", "source_center", "seed"], as_index=False
    )[metric].mean()
    matrix = cell.pivot(
        index=["source_center", "seed"], columns="method", values=metric
    ).sort_index()
    missing = set(METHODS.values()) - set(matrix.columns)
    if missing:
        raise ValueError(f"missing factorial methods: {sorted(missing)}")

    u0b0 = matrix[METHODS["U0B0"]]
    u1b0 = matrix[METHODS["U1B0"]]
    u0b1 = matrix[METHODS["U0B1"]]
    u1b1 = matrix[METHODS["U1B1"]]
    deltas = {
        "sync_no_blend": u1b0 - u0b0,
        "sync_blend": u1b1 - u0b1,
        "blend_unpaired": u0b1 - u0b0,
        "blend_paired": u1b1 - u1b0,
        "interaction": (u1b1 - u1b0) - (u0b1 - u0b0),
    }
    return {name: _summarize_delta(values) for name, values in deltas.items()}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down 校正，并强制校正值随 rank 单调。"""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min(1.0, float(value) * (total - rank))
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def f6_holm(rows: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    """仅校正预先声明的三个 post-hoc F6 检验。"""
    family = ("sync_blend", "blend_paired", "interaction")
    return holm_adjust(
        {name: float(rows[name]["paired_t_p"]) for name in family}
    )


def _result_rows(
    frame: pd.DataFrame,
    *,
    evaluation_source: str,
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    """把具名 contrast 展开为可审计长表。"""
    output: list[dict[str, object]] = []
    for morphology_group in ("hard_flat_IIa", "all"):
        for metric in metrics:
            contrasts = factorial_contrasts(
                frame,
                metric=metric,
                morphology_group=morphology_group,
            )
            adjusted = f6_holm(contrasts)
            for contrast_id, values in contrasts.items():
                row = dict(values)
                row["cell_deltas_json"] = json.dumps(row.pop("cell_deltas"))
                row.update(
                    {
                        "contrast_id": contrast_id,
                        "evaluation_source": evaluation_source,
                        "metric": metric,
                        "higher_is_better": metric != "hd95_px_mean",
                        "morphology_group": morphology_group,
                        "ci_source": "paired_t_6_source_seed_cells",
                        "holm_family": (
                            "F6-posthoc:sync_blend,blend_paired,interaction"
                            if contrast_id in adjusted
                            else "not_recomputed_here"
                        ),
                        "paired_t_p_holm_f6": adjusted.get(contrast_id),
                    }
                )
                output.append(row)
    return output


def build_results(
    flat_frame: pd.DataFrame,
    boundary_frame: pd.DataFrame,
) -> pd.DataFrame:
    """生成 flat Dice 与 boundary 四指标的 6-cell 统计表。"""
    rows = _result_rows(
        flat_frame,
        evaluation_source="flat_case",
        metrics=("dice_mean",),
    )
    rows.extend(
        _result_rows(
            boundary_frame,
            evaluation_source="boundary_case",
            metrics=(
                "dice_mean",
                "hd95_px_mean",
                "weighted_fbeta_mean",
                "boundary_iou_mean",
            ),
        )
    )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_server_mean_overwrite(raw_dir: Path) -> bool:
    path = raw_dir / "F6_interaction_stats.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    rows = frame[
        (frame["name"] == "interaction")
        & frame["morphology_group"].notna()
    ]
    return len(rows) == 1 and float(rows.iloc[0]["mean_delta"]) == 3.0


def _detect_case_cell_intervals(raw_dir: Path) -> bool:
    path = raw_dir / "flat_paired" / "paired_comparisons.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    primary = frame[
        (frame["candidate"] == "paired_affine_softmix")
        & (frame["reference"] == "SLAug")
        & (frame["metric"] == "dice_mean")
        & (frame["morphology_group"] == "hard_flat_IIa")
    ]
    return len(primary) == 1 and int(primary.iloc[0]["n_paired_clusters"]) != 6


def write_bundle(
    *,
    flat_path: Path,
    boundary_path: Path,
    output_dir: Path,
) -> None:
    """从服务器 raw 输入写出本地重算表与 provenance manifest。"""
    flat_path = Path(flat_path)
    boundary_path = Path(boundary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = build_results(
        pd.read_csv(flat_path),
        pd.read_csv(boundary_path),
    )
    results.to_csv(output_dir / "factorial_6cell_stats.csv", index=False)

    raw_dir = flat_path.parent
    manifest = {
        "inputs": {
            "flat": {"path": str(flat_path), "sha256": _sha256(flat_path)},
            "boundary": {
                "path": str(boundary_path),
                "sha256": _sha256(boundary_path),
            },
        },
        "output_rows": len(results),
        "primary_unit": "2 source centers x 3 seeds = 6 paired cells",
        "ci": "two-sided 95% paired-t interval across six cells",
        "f6_family": ["sync_blend", "blend_paired", "interaction"],
        "server_f6_mean_overwrite_detected": _detect_server_mean_overwrite(raw_dir),
        "server_case_cell_ci_replaced": _detect_case_cell_intervals(raw_dir),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(
        flat_path=args.flat,
        boundary_path=args.boundary,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
