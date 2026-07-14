#!/usr/bin/env python3
"""重建 10 臂 boundary 矩阵并生成描述性 6-cell 汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy import stats


EXPECTED_METHODS = {
    "strong_aug_floor",
    "SLAug",
    "CCSDG",
    "CSDG",
    "DSU",
    "MixStyle",
    "fourier_amp_aug",
    "ibn_whitening",
    "spectral_consistency",
    "spectral_ibn_combo",
}
METRICS = (
    "dice_mean",
    "hd95_px_mean",
    "weighted_fbeta_mean",
    "boundary_iou_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_zoo(
    *,
    eight_arm_path: Path,
    a1_source_path: Path,
    floor_dir: Path,
) -> pd.DataFrame:
    """合并 8 个补评臂、canonical A1 和 strong floor。"""
    eight_arm_path = Path(eight_arm_path)
    a1_source_path = Path(a1_source_path)
    floor_dir = Path(floor_dir)

    eight = pd.read_csv(eight_arm_path)
    a1_all = pd.read_csv(a1_source_path)
    a1 = a1_all[a1_all["method"] == "SLAug"].copy()
    floor_paths = sorted(floor_dir.glob("boundary_case_*.csv"))
    if len(floor_paths) != 6:
        raise ValueError(f"expected six strong-floor files, got {len(floor_paths)}")
    floor = pd.concat([pd.read_csv(path) for path in floor_paths], ignore_index=True)

    frame = pd.concat([eight, a1, floor], ignore_index=True)
    methods = set(frame["method"])
    if methods != EXPECTED_METHODS:
        raise ValueError(
            f"method set mismatch: missing={sorted(EXPECTED_METHODS - methods)}, "
            f"extra={sorted(methods - EXPECTED_METHODS)}"
        )
    if frame.duplicated(["run_id", "case_id"]).any():
        raise ValueError("duplicate run_id/case_id rows in merged boundary matrix")
    if set(frame.groupby("run_id").size()) != {285}:
        raise ValueError("every boundary run must contain exactly 285 cases")
    if set(frame.groupby("method")["run_id"].nunique()) != {6}:
        raise ValueError("every boundary method must contain exactly six runs")
    return frame


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """按 source×seed cell 汇总，不为扩展 boundary 矩阵新增推断 family。"""
    output: list[dict[str, object]] = []
    for morphology_group in ("hard_flat_IIa", "all"):
        subset = frame
        if morphology_group != "all":
            subset = subset[subset["morphology_group"] == morphology_group]
        cell = subset.groupby(
            ["method", "source_center", "seed"], as_index=False
        )[list(METRICS)].mean()
        floor = cell[cell["method"] == "strong_aug_floor"].set_index(
            ["source_center", "seed"]
        )
        for method in sorted(EXPECTED_METHODS):
            method_cell = cell[cell["method"] == method].set_index(
                ["source_center", "seed"]
            )
            if len(method_cell) != 6:
                raise ValueError(f"expected six cells for {method}, got {len(method_cell)}")
            for metric in METRICS:
                values = method_cell[metric].sort_index()
                delta = values - floor[metric].sort_index()
                mean_delta = float(delta.mean())
                if method == "strong_aug_floor":
                    ci_low = ci_high = 0.0
                else:
                    standard_error = float(stats.sem(delta))
                    critical_value = float(stats.t.ppf(0.975, len(delta) - 1))
                    ci_low = mean_delta - critical_value * standard_error
                    ci_high = mean_delta + critical_value * standard_error
                output.append(
                    {
                        "method": method,
                        "morphology_group": morphology_group,
                        "metric": metric,
                        "higher_is_better": metric != "hd95_px_mean",
                        "n_cells": len(values),
                        "cell_mean": float(values.mean()),
                        "cell_sd": float(values.std(ddof=1)),
                        "delta_vs_floor": mean_delta,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "inference_status": "descriptive_expanded_boundary",
                    }
                )
    return pd.DataFrame(output)


def write_bundle(
    *,
    eight_arm_path: Path,
    a1_source_path: Path,
    floor_dir: Path,
    output_dir: Path,
) -> None:
    """写描述性汇总与完整性 manifest。"""
    eight_arm_path = Path(eight_arm_path)
    a1_source_path = Path(a1_source_path)
    floor_dir = Path(floor_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_full_zoo(
        eight_arm_path=eight_arm_path,
        a1_source_path=a1_source_path,
        floor_dir=floor_dir,
    )
    summary = build_summary(frame)
    summary.to_csv(output_dir / "full_zoo_boundary_6cell_summary.csv", index=False)

    floor_paths = sorted(floor_dir.glob("boundary_case_*.csv"))
    manifest = {
        "inputs": {
            "eight_arm": {"path": str(eight_arm_path), "sha256": _sha256(eight_arm_path)},
            "a1_source": {"path": str(a1_source_path), "sha256": _sha256(a1_source_path)},
            "strong_floor": [
                {"path": str(path), "sha256": _sha256(path)} for path in floor_paths
            ],
        },
        "n_rows": len(frame),
        "n_distinct_runs": int(frame["run_id"].nunique()),
        "n_methods": int(frame["method"].nunique()),
        "cases_per_run": sorted(frame.groupby("run_id").size().unique().tolist()),
        "runs_per_method": sorted(
            frame.groupby("method")["run_id"].nunique().unique().tolist()
        ),
        "inference_status": "descriptive_expanded_boundary",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eight-arm", type=Path, required=True)
    parser.add_argument("--a1-source", type=Path, required=True)
    parser.add_argument("--floor-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(
        eight_arm_path=args.eight_arm,
        a1_source_path=args.a1_source,
        floor_dir=args.floor_dir,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
