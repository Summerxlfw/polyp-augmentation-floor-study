#!/usr/bin/env python3
"""重算预注册 F5 边界指标族，主推断使用 6 个 source×seed cell。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

import video_cluster_stats as paired_stats


F5_COMPARISONS = (
    ("SLAug", "warp_alpha015"),
    ("SLAug", "warp_alpha100"),
    ("SLAug", "warp_shift_only"),
    ("slaug_official_plus_warp", "SLAug_official"),
)
F5_METRICS = (
    ("F5-HD95", "hd95_px_mean", False),
    ("F5-Fbw", "weighted_fbeta_mean", True),
    ("F5-BIoU", "boundary_iou_mean", True),
)


def summarize_boundary_families(
    table: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[tuple[str, str, bool]],
    model: str,
    filters: Mapping[str, object],
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个 metric 独立做四成员 Holm，低值指标统一给出 improvement 方向。"""
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for family_label, metric, higher_is_better in metrics:
        family_results: list[dict[str, object]] = []
        for candidate, reference in comparisons:
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
            cells = result.pop("cell_rows")
            result.pop("filters")
            direction = 1.0 if higher_is_better else -1.0
            result["multiplicity_family"] = family_label
            result["higher_is_better"] = higher_is_better
            result["mean_improvement"] = direction * float(result["mean_delta"])
            result["improved_cells"] = int(
                sum(direction * float(row["delta"]) > 0.0 for row in cells)
            )
            if higher_is_better:
                result["improvement_cell_t_ci_low"] = result["cell_t_ci_low"]
                result["improvement_cell_t_ci_high"] = result["cell_t_ci_high"]
                result["improvement_supporting_ci_low"] = result[
                    "supporting_video_cluster_ci_low"
                ]
                result["improvement_supporting_ci_high"] = result[
                    "supporting_video_cluster_ci_high"
                ]
            else:
                result["improvement_cell_t_ci_low"] = -float(result["cell_t_ci_high"])
                result["improvement_cell_t_ci_high"] = -float(result["cell_t_ci_low"])
                result["improvement_supporting_ci_low"] = -float(
                    result["supporting_video_cluster_ci_high"]
                )
                result["improvement_supporting_ci_high"] = -float(
                    result["supporting_video_cluster_ci_low"]
                )
            family_results.append(result)
            for row in cells:
                cell_rows.append(
                    {
                        "multiplicity_family": family_label,
                        "candidate": candidate,
                        "reference": reference,
                        "model": model,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        **row,
                        "improvement": direction * float(row["delta"]),
                    }
                )
        holm_values = paired_stats.holm_adjust(
            [float(row["paired_t_p"]) for row in family_results]
        )
        for row, holm_p in zip(family_results, holm_values, strict=True):
            row["holm_p"] = holm_p
            summary_rows.append(row)
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
    metric_code: Path,
    model: str = "polyp_pvt",
    n_boot: int = 10000,
    random_seed: int = 20260713,
) -> dict[str, Path]:
    table = pd.read_csv(case_csv)
    summary, cells = summarize_boundary_families(
        table,
        comparisons=F5_COMPARISONS,
        metrics=F5_METRICS,
        model=model,
        filters={"morphology_group": "hard_flat_IIa"},
        n_boot=n_boot,
        random_seed=random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "f5_boundary_summary.csv"
    cells_path = output_dir / "f5_boundary_cells.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    summary.to_csv(summary_path, index=False)
    cells.to_csv(cells_path, index=False)
    manifest = {
        "state": "done",
        "protocol": "three separate four-comparison Holm families on hard-flat IIa cases; primary inference uses six source-seed cells; aligned video-cluster bootstrap is supporting",
        "case_csv": str(case_csv.resolve()),
        "case_csv_sha256": _sha256(case_csv),
        "metric_code": str(metric_code.resolve()),
        "metric_code_sha256": _sha256(metric_code),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "paired_stats_script": str(Path(paired_stats.__file__).resolve()),
        "paired_stats_script_sha256": _sha256(Path(paired_stats.__file__).resolve()),
        "comparisons": [f"{candidate}:{reference}" for candidate, reference in F5_COMPARISONS],
        "metrics": [
            {"family": family, "metric": metric, "higher_is_better": higher}
            for family, metric, higher in F5_METRICS
        ],
        "model": model,
        "stratum": "hard_flat_IIa",
        "n_boot": n_boot,
        "random_seed": random_seed,
        "empty_prediction_policy": "HD95=image diagonal; weighted F-beta=0; Boundary IoU=0; empty count retained",
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
    parser.add_argument("--metric-code", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    paths = write_family_bundle(
        case_csv=args.case_csv,
        output_dir=args.output_dir,
        metric_code=args.metric_code,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
