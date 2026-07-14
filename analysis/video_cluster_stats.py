#!/usr/bin/env python3
"""SUN-SEG 配对统计：保留病灶等权点估计，按视频做 cluster bootstrap。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon


PAIR_KEYS = ["source_center", "model", "seed", "case_id", "video_id"]
DEFAULT_STRATA = (
    ("hard_flat_IIa", {"morphology_group": "hard_flat_IIa"}),
    ("all", {}),
    ("shape_Is", {"shape": "Is"}),
    ("shape_Isp", {"shape": "Isp"}),
    ("shape_Ip", {"shape": "Ip"}),
    ("size_small", {"size_bin": "small"}),
    ("size_mid", {"size_bin": "mid"}),
    ("size_large", {"size_bin": "large"}),
)


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def paired_case_deltas(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    model: str,
    filters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """严格配对两个方法的 case 行，缺 case 或重复键都直接失败。"""
    filters = dict(filters or {})
    _require_columns(table, [*PAIR_KEYS, "method", metric, *filters])
    selected = table[table["model"] == model].copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    candidate_df = selected[selected["method"] == candidate][PAIR_KEYS + [metric]].copy()
    reference_df = selected[selected["method"] == reference][PAIR_KEYS + [metric]].copy()
    if candidate_df.empty or reference_df.empty:
        raise ValueError(f"empty method slice: candidate={len(candidate_df)}, reference={len(reference_df)}")
    for name, frame in (("candidate", candidate_df), ("reference", reference_df)):
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"duplicate {name} case keys")
    candidate_keys = set(candidate_df[PAIR_KEYS].itertuples(index=False, name=None))
    reference_keys = set(reference_df[PAIR_KEYS].itertuples(index=False, name=None))
    if candidate_keys != reference_keys:
        missing_candidate = len(reference_keys - candidate_keys)
        missing_reference = len(candidate_keys - reference_keys)
        raise ValueError(
            "case keys differ: "
            f"missing_candidate={missing_candidate}, missing_reference={missing_reference}"
        )
    merged = candidate_df.merge(
        reference_df,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    merged["delta"] = merged[f"{metric}_candidate"].astype(float) - merged[f"{metric}_reference"].astype(float)
    return merged.sort_values(["source_center", "seed", "video_id", "case_id"]).reset_index(drop=True)


def case_equal_cell_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    """每个 source×seed 内按 case 等权求点估计；视频只用于不确定性。"""
    _require_columns(paired, ["source_center", "seed", "case_id", "video_id", "delta"])
    if paired.empty:
        raise ValueError("paired case table is empty")
    grouped = paired.groupby(["source_center", "seed"], sort=True)
    rows = []
    for (source, seed), frame in grouped:
        rows.append(
            {
                "source_center": source,
                "seed": int(seed),
                "delta": float(frame["delta"].mean()),
                "n_cases": int(frame["case_id"].nunique()),
                "n_videos": int(frame["video_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_center", "seed"]).reset_index(drop=True)


def paired_cell_contrast(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """在相同 source×seed cell 上计算两个分层效应的差异中差。"""
    keys = ["source_center", "seed"]
    for label, frame in (("left", left), ("right", right)):
        _require_columns(frame, [*keys, "delta"])
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate {label} cell keys")
    left_keys = set(left[keys].itertuples(index=False, name=None))
    right_keys = set(right[keys].itertuples(index=False, name=None))
    if left_keys != right_keys:
        raise ValueError("cell keys differ between contrast members")
    merged = left[keys + ["delta"]].merge(
        right[keys + ["delta"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    merged["contrast_delta"] = merged["delta_left"] - merged["delta_right"]
    return merged.sort_values(keys).reset_index(drop=True)


def cluster_bootstrap_ci(
    paired: pd.DataFrame,
    n_boot: int = 10000,
    random_seed: int = 20260713,
) -> tuple[float, float]:
    """在 source 内重采 seed，并让同一批 video 跨全部 cell 同步重采样。"""
    _require_columns(paired, ["source_center", "seed", "case_id", "video_id", "delta"])
    if paired.empty:
        raise ValueError("paired case table is empty")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(random_seed)
    payload: dict[object, dict[object, dict[str, np.ndarray]]] = {}
    expected_case_universe: set[tuple[str, str]] | None = None
    video_keys: list[str] | None = None
    for source, source_df in paired.groupby("source_center", sort=True):
        payload[source] = {}
        for seed, seed_df in source_df.groupby("seed", sort=True):
            normalized = seed_df.assign(
                _video_key=seed_df["video_id"].astype(str),
                _case_key=seed_df["case_id"].astype(str),
            )
            case_universe = set(
                normalized[["_video_key", "_case_key"]].itertuples(index=False, name=None)
            )
            if expected_case_universe is None:
                expected_case_universe = case_universe
            elif case_universe != expected_case_universe:
                raise ValueError("case/video universe differs across source-seed cells")
            cell_payload = {
                str(video): video_df["delta"].to_numpy(dtype=np.float64)
                for video, video_df in normalized.groupby("_video_key", sort=True)
            }
            current_video_keys = sorted(cell_payload)
            if video_keys is None:
                video_keys = current_video_keys
            elif current_video_keys != video_keys:
                raise ValueError("video universe differs across source-seed cells")
            payload[source][seed] = cell_payload
    if not video_keys:
        raise ValueError("no video clusters found")
    draws = np.empty(n_boot, dtype=np.float64)
    for boot_idx in range(n_boot):
        sampled_video_indices = rng.integers(0, len(video_keys), size=len(video_keys))
        sampled_videos = [video_keys[index] for index in sampled_video_indices]
        source_means = []
        for seed_payload in payload.values():
            seeds = np.asarray(list(seed_payload))
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_means = []
            for seed in sampled_seeds:
                video_arrays = seed_payload[seed]
                case_draws = [video_arrays[video] for video in sampled_videos]
                seed_means.append(float(np.concatenate(case_draws).mean()))
            source_means.append(float(np.mean(seed_means)))
        draws[boot_idx] = float(np.mean(source_means))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def cell_t_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """以 source×seed cell 为独立单位计算双侧 t 置信区间。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("at least two one-dimensional cell values are required")
    if not np.isfinite(arr).all():
        raise ValueError("cell values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    mean = float(arr.mean())
    if np.allclose(arr, arr[0], rtol=0.0, atol=1e-15):
        return mean, mean
    standard_error = float(arr.std(ddof=1) / np.sqrt(arr.size))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=arr.size - 1))
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def exact_sign_flip_p(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    observed = abs(float(arr.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=arr.size):
        means.append(abs(float(np.mean(arr * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, index in enumerate(order):
        running = max(running, float((n - rank) * p[index]))
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def summarize_comparison(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    model: str,
    filters: Mapping[str, object] | None = None,
    n_boot: int = 10000,
    random_seed: int = 20260713,
    expected_cells: int = 6,
) -> dict[str, object]:
    paired = paired_case_deltas(table, candidate, reference, metric, model, filters)
    cells = case_equal_cell_deltas(paired)
    if len(cells) != expected_cells:
        raise ValueError(f"expected {expected_cells} source-seed cells, got {len(cells)}")
    deltas = cells["delta"].to_numpy(dtype=np.float64)
    cell_t_ci_low, cell_t_ci_high = cell_t_confidence_interval(deltas)
    video_ci_low, video_ci_high = cluster_bootstrap_ci(
        paired,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    if np.allclose(deltas, deltas[0], rtol=0.0, atol=1e-15):
        paired_t_p = 1.0 if abs(float(deltas[0])) <= 1e-15 else 0.0
        wilcoxon_p = (
            1.0
            if abs(float(deltas[0])) <= 1e-15
            else exact_sign_flip_p(deltas)
        )
    else:
        paired_t_p = float(ttest_1samp(deltas, popmean=0.0).pvalue)
        try:
            wilcoxon_p = float(
                wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
    return {
        "candidate": candidate,
        "reference": reference,
        "model": model,
        "metric": metric,
        "filters": dict(filters or {}),
        "n_sources": int(cells["source_center"].nunique()),
        "n_source_seed_cells": int(len(cells)),
        "n_paired_cases": int(paired["case_id"].nunique()),
        "n_paired_videos": int(paired["video_id"].nunique()),
        "mean_delta": float(deltas.mean()),
        "cell_t_ci_low": cell_t_ci_low,
        "cell_t_ci_high": cell_t_ci_high,
        "supporting_video_cluster_ci_low": video_ci_low,
        "supporting_video_cluster_ci_high": video_ci_high,
        "positive_cells": int((deltas > 0).sum()),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_flip_p": exact_sign_flip_p(deltas),
        "cell_rows": cells.to_dict(orient="records"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_stratum(value: str) -> tuple[str, dict[str, str]]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("stratum must be label:column:value")
    return parts[0], {parts[1]: parts[2]}


def write_analysis_bundle(
    case_csv: Path,
    output_dir: Path,
    candidate: str,
    reference: str,
    model: str,
    metric: str,
    strata: Sequence[tuple[str, Mapping[str, object]]],
    n_boot: int,
    random_seed: int,
) -> dict[str, Path]:
    table = pd.read_csv(case_csv)
    summary_rows = []
    cell_rows = []
    for label, filters in strata:
        result = summarize_comparison(
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
        result["stratum"] = label
        result["filters_json"] = json.dumps(result.pop("filters"), sort_keys=True)
        result["multiplicity_status"] = "not_applied_supportive_strata"
        summary_rows.append(result)
        for row in cells:
            cell_rows.append(
                {
                    "stratum": label,
                    "candidate": candidate,
                    "reference": reference,
                    "model": model,
                    "metric": metric,
                    **row,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "video_cluster_summary.csv"
    cells_path = output_dir / "video_cluster_cells.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_path, index=False)
    manifest = {
        "state": "done",
        "protocol": "primary paired-t interval on six source-seed cells; supporting case-equal interval from seed-within-source bootstrap plus one aligned video-cluster resample shared across all source-seed cells",
        "case_csv": str(case_csv.resolve()),
        "case_csv_sha256": _sha256(case_csv),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "candidate": candidate,
        "reference": reference,
        "model": model,
        "metric": metric,
        "n_boot": n_boot,
        "random_seed": random_seed,
        "strata": [{"label": label, "filters": dict(filters)} for label, filters in strata],
        "n_summary_rows": len(summary_rows),
        "n_cell_rows": len(cell_rows),
        "summary_csv": str(summary_path.resolve()),
        "cells_csv": str(cells_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary_path, "cells": cells_path, "manifest": manifest_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="SLAug")
    parser.add_argument("--reference", default="strong_aug_floor")
    parser.add_argument("--model", default="polyp_pvt")
    parser.add_argument("--metric", default="dice_mean")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--stratum", action="append", type=_parse_stratum)
    args = parser.parse_args(argv)
    strata = tuple(args.stratum) if args.stratum else DEFAULT_STRATA
    paths = write_analysis_bundle(
        case_csv=args.case_csv,
        output_dir=args.output_dir,
        candidate=args.candidate,
        reference=args.reference,
        model=args.model,
        metric=args.metric,
        strata=strata,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
