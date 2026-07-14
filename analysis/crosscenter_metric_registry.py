#!/usr/bin/env python3
"""从锁定 raw 表生成跨中心与外部评估指标注册表。"""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_ci(values: pd.Series) -> tuple[float, float]:
    values = values.astype(float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean - half_width, mean + half_width


def build_registry(
    *,
    main_table_path: Path,
    boundary_summary_path: Path,
) -> pd.DataFrame:
    """生成每臂一行的描述性指标表，主推断仍由既有 family 表承担。"""
    main = pd.read_csv(main_table_path)
    boundary = pd.read_csv(boundary_summary_path)

    if len(main) != 300 or main["run_id"].nunique() != 60:
        raise ValueError("main table must contain 60 runs x 5 held-out centers")
    if set(main["method"]) != EXPECTED_METHODS:
        raise ValueError("main-table method set does not match the locked 10-arm matrix")
    if set(main.groupby("run_id")["center"].nunique()) != {5}:
        raise ValueError("every run must contain five frozen held-out centers")
    if set(main.groupby("method")["run_id"].nunique()) != {6}:
        raise ValueError("every method must contain six source-by-seed runs")

    run_macro = (
        main.groupby(
            ["method", "source_center", "seed", "run_id"], as_index=False
        )["dice_mean"]
        .mean()
        .rename(columns={"dice_mean": "polypgen_center_macro_dice"})
    )
    floor = run_macro[run_macro["method"] == "strong_aug_floor"].set_index(
        ["source_center", "seed"]
    )["polypgen_center_macro_dice"]

    expected_boundary_rows = 10 * 2 * 4
    if len(boundary) != expected_boundary_rows:
        raise ValueError(
            f"boundary summary must contain {expected_boundary_rows} rows"
        )
    if set(boundary["method"]) != EXPECTED_METHODS:
        raise ValueError("boundary-summary method set does not match the main table")

    rows: list[dict[str, object]] = []
    for method in sorted(EXPECTED_METHODS):
        method_runs = run_macro[run_macro["method"] == method].set_index(
            ["source_center", "seed"]
        )
        deltas = method_runs["polypgen_center_macro_dice"] - floor
        ci_low, ci_high = _paired_ci(deltas)

        values: dict[tuple[str, str], float] = {}
        for row in boundary[boundary["method"] == method].itertuples(index=False):
            values[(row.morphology_group, row.metric)] = float(row.cell_mean)

        rows.append(
            {
                "method": method,
                "n_runs": len(method_runs),
                "n_heldout_centers_per_run": 5,
                "polypgen_center_macro_dice": float(
                    method_runs["polypgen_center_macro_dice"].mean()
                ),
                "polypgen_delta_vs_floor": float(deltas.mean()),
                "polypgen_delta_ci_low": ci_low,
                "polypgen_delta_ci_high": ci_high,
                "polypgen_positive_cells_of_6": int((deltas > 0).sum()),
                "sunseg_all_dice": values[("all", "dice_mean")],
                "sunseg_hard_flat_dice": values[("hard_flat_IIa", "dice_mean")],
                "sunseg_hard_flat_hd95_px": values[
                    ("hard_flat_IIa", "hd95_px_mean")
                ],
                "sunseg_hard_flat_weighted_fbeta": values[
                    ("hard_flat_IIa", "weighted_fbeta_mean")
                ],
                "sunseg_hard_flat_boundary_iou": values[
                    ("hard_flat_IIa", "boundary_iou_mean")
                ],
                "inference_status": "descriptive_registry; inferential claims use F1-F6 tables",
            }
        )
    return pd.DataFrame(rows)


def write_registry(
    *,
    main_table_path: Path,
    boundary_summary_path: Path,
    output_csv: Path,
    manifest_path: Path,
) -> None:
    """写注册表与输入哈希。"""
    main_table_path = Path(main_table_path)
    boundary_summary_path = Path(boundary_summary_path)
    output_csv = Path(output_csv)
    manifest_path = Path(manifest_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    registry = build_registry(
        main_table_path=main_table_path,
        boundary_summary_path=boundary_summary_path,
    )
    registry.to_csv(output_csv, index=False)
    manifest = {
        "inputs": {
            str(main_table_path): _sha256(main_table_path),
            str(boundary_summary_path): _sha256(boundary_summary_path),
        },
        "output": str(output_csv),
        "output_sha256": _sha256(output_csv),
        "n_methods": int(registry["method"].nunique()),
        "n_runs": int(registry["n_runs"].sum()),
        "status": "descriptive_registry",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-table", type=Path, required=True)
    parser.add_argument("--boundary-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    write_registry(
        main_table_path=args.main_table,
        boundary_summary_path=args.boundary_summary,
        output_csv=args.output_csv,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
