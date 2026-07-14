"""L1 source×seed 配对统计。

正文友好口径：6 个 source×seed 单元的 paired t-test、配对均值差和
95% CI。Wilcoxon、精确 sign-flip 与 Holm 校正作为审计列保留。
"""

from __future__ import annotations

import itertools
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    """对配对差均值做双侧精确 sign-flip 检验。"""
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[values != 0.0]
    if values.size == 0:
        return 1.0
    observed = abs(float(values.mean()))
    exceed = 0
    total = 2 ** int(values.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            exceed += 1
    return float(exceed / total)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm step-down 校正，返回原顺序。"""
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (values.size - rank) * values[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(adjusted_sorted)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def _filtered(
    table: pd.DataFrame,
    method: str,
    metric: str,
    morphology_group: str,
    model: str,
) -> pd.DataFrame:
    required = {
        "source_center",
        "model",
        "method",
        "seed",
        "morphology_group",
        "case_id",
        "cluster_id",
        metric,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"输入 CSV 缺列: {missing}")
    mask = (table["method"] == method) & (table["model"] == model)
    if morphology_group != "all":
        mask &= table["morphology_group"] == morphology_group
    selected = table[mask].copy()
    if selected.empty:
        raise ValueError(f"没有匹配数据: method={method}, model={model}, group={morphology_group}")
    selected[metric] = pd.to_numeric(selected[metric], errors="raise")
    return selected


def paired_case_deltas(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    morphology_group: str,
    model: str,
) -> pd.DataFrame:
    """严格一对一匹配 case；不允许 inner join 静默丢样本。"""
    candidate_df = _filtered(table, candidate, metric, morphology_group, model)
    reference_df = _filtered(table, reference, metric, morphology_group, model)
    keys = ["source_center", "model", "seed", "case_id", "cluster_id"]
    for label, frame in ((candidate, candidate_df), (reference, reference_df)):
        duplicated = frame.duplicated(keys, keep=False)
        if duplicated.any():
            raise ValueError(f"{label} 存在重复配对键: {int(duplicated.sum())} rows")
    candidate_keys = set(map(tuple, candidate_df[keys].itertuples(index=False, name=None)))
    reference_keys = set(map(tuple, reference_df[keys].itertuples(index=False, name=None)))
    if candidate_keys != reference_keys:
        raise ValueError(
            "配对键不完整: "
            f"candidate_only={len(candidate_keys - reference_keys)}, "
            f"reference_only={len(reference_keys - candidate_keys)}"
        )
    merged = candidate_df[keys + [metric]].merge(
        reference_df[keys + [metric]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    merged["delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return merged


def _cluster_table(case_values: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        case_values.groupby(["source_center", "seed", "cluster_id"], as_index=False)[value_column]
        .mean()
        .sort_values(["source_center", "seed", "cluster_id"])
    )


def _stratified_seed_cluster_bootstrap(
    cluster_values: pd.DataFrame,
    value_column: str,
    n_boot: int,
    random_seed: int,
) -> tuple[float, float]:
    if n_boot <= 0:
        raise ValueError("n_boot 必须为正")
    rng = np.random.default_rng(random_seed)
    sources = sorted(cluster_values["source_center"].unique())
    bootstrap_means = np.empty(n_boot, dtype=np.float64)
    for boot_idx in range(n_boot):
        source_means = []
        for source in sources:
            source_df = cluster_values[cluster_values["source_center"] == source]
            seeds = np.asarray(sorted(source_df["seed"].unique()))
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_means = []
            for seed in sampled_seeds:
                values = source_df[source_df["seed"] == seed][value_column].to_numpy(dtype=np.float64)
                sampled_values = rng.choice(values, size=values.size, replace=True)
                seed_means.append(float(sampled_values.mean()))
            source_means.append(float(np.mean(seed_means)))
        bootstrap_means[boot_idx] = float(np.mean(source_means))
    return tuple(np.percentile(bootstrap_means, [2.5, 97.5]).astype(float))


def paired_comparison(
    table: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    morphology_group: str,
    model: str,
    higher_is_better: bool,
    n_boot: int = 5000,
    random_seed: int = 20260713,
    expected_source_seed_cells: int = 6,
) -> dict[str, Any]:
    paired = paired_case_deltas(table, candidate, reference, metric, morphology_group, model)
    clusters = _cluster_table(paired, "delta")
    cells = clusters.groupby(["source_center", "seed"], as_index=False)["delta"].mean()
    if len(cells) != expected_source_seed_cells:
        raise ValueError(
            f"source×seed 单元数应为 {expected_source_seed_cells}，实际 {len(cells)}；拒绝 partial 统计"
        )
    deltas = cells["delta"].to_numpy(dtype=np.float64)
    ci_low, ci_high = _stratified_seed_cluster_bootstrap(clusters, "delta", n_boot, random_seed)
    try:
        wilcoxon_p = float(wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    paired_t_p = float(ttest_1samp(deltas, popmean=0.0).pvalue) if deltas.size > 1 else float("nan")
    direction = 1.0 if higher_is_better else -1.0
    improvement_low = ci_low * direction
    improvement_high = ci_high * direction
    if direction < 0:
        improvement_low, improvement_high = improvement_high, improvement_low
    return {
        "candidate": candidate,
        "reference": reference,
        "model": model,
        "morphology_group": morphology_group,
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "n_sources": int(cells["source_center"].nunique()),
        "n_source_seed_cells": int(len(cells)),
        "n_paired_cases": int(len(paired)),
        "n_paired_clusters": int(len(clusters)),
        "candidate_mean": float(paired[f"{metric}_candidate"].mean()),
        "reference_mean": float(paired[f"{metric}_reference"].mean()),
        "mean_delta": float(deltas.mean()),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "mean_improvement": float(deltas.mean() * direction),
        "improvement_ci_low": float(improvement_low),
        "improvement_ci_high": float(improvement_high),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_flip_p": exact_sign_flip_p(deltas),
    }


def multi_seed_summary(
    table: pd.DataFrame,
    method: str,
    metric: str,
    morphology_group: str,
    model: str,
    n_boot: int = 5000,
    random_seed: int = 20260713,
    expected_source_seed_cells: int = 6,
) -> dict[str, Any]:
    selected = _filtered(table, method, metric, morphology_group, model)
    clusters = _cluster_table(selected.rename(columns={metric: "value"}), "value")
    cells = clusters.groupby(["source_center", "seed"], as_index=False)["value"].mean()
    if len(cells) != expected_source_seed_cells:
        raise ValueError(
            f"source×seed 单元数应为 {expected_source_seed_cells}，实际 {len(cells)}；拒绝 partial CI"
        )
    ci_low, ci_high = _stratified_seed_cluster_bootstrap(clusters, "value", n_boot, random_seed)
    return {
        "method": method,
        "model": model,
        "morphology_group": morphology_group,
        "metric": metric,
        "n_sources": int(cells["source_center"].nunique()),
        "n_source_seed_cells": int(len(cells)),
        "n_cases": int(len(selected)),
        "n_clusters": int(len(clusters)),
        "mean": float(cells["value"].mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def analyze_case_table(
    table: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    methods: list[str],
    metrics: dict[str, bool],
    morphology_groups: list[str],
    model: str,
    n_boot: int = 5000,
    random_seed: int = 20260713,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    for candidate, reference in comparisons:
        for metric, higher_is_better in metrics.items():
            for group in morphology_groups:
                paired_rows.append(
                    paired_comparison(
                        table,
                        candidate,
                        reference,
                        metric,
                        group,
                        model,
                        higher_is_better,
                        n_boot=n_boot,
                        random_seed=random_seed,
                    )
                )
    paired_table = pd.DataFrame(paired_rows)
    if not paired_table.empty:
        for p_column in ("paired_t_p", "wilcoxon_p", "exact_sign_flip_p"):
            adjusted = pd.Series(index=paired_table.index, dtype=float)
            for _, family in paired_table.groupby(["model", "morphology_group", "metric"], sort=True):
                adjusted.loc[family.index] = holm_adjust(family[p_column].tolist())
            paired_table[f"{p_column}_holm"] = adjusted

    summary_rows = []
    for method in methods:
        for metric in metrics:
            for group in morphology_groups:
                summary_rows.append(
                    multi_seed_summary(
                        table,
                        method,
                        metric,
                        group,
                        model,
                        n_boot=n_boot,
                        random_seed=random_seed,
                    )
                )
    return paired_table, pd.DataFrame(summary_rows)


DEFAULT_METRICS = {
    "dice_mean": True,
    "hd95_px_mean": False,
    "weighted_fbeta_mean": True,
    "boundary_iou_mean": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L1 6-cell paired t-test + multi-seed CI")
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--comparison", action="append", required=True, help="candidate:reference，可重复")
    parser.add_argument("--model", default="polyp_pvt")
    parser.add_argument("--groups", nargs="+", default=["hard_flat_IIa", "all"])
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    unknown = sorted(set(args.metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise SystemExit(f"未知 metric: {unknown}; 可选={list(DEFAULT_METRICS)}")
    comparisons = []
    for value in args.comparison:
        if value.count(":") != 1:
            raise SystemExit(f"comparison 格式应为 candidate:reference，实际={value}")
        comparisons.append(tuple(value.split(":", maxsplit=1)))
    methods = sorted({method for pair in comparisons for method in pair})
    table = pd.read_csv(args.case_csv)
    paired, summaries = analyze_case_table(
        table,
        comparisons,
        methods,
        {metric: DEFAULT_METRICS[metric] for metric in args.metrics},
        args.groups,
        args.model,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_comparisons.csv"
    summary_path = args.output_dir / "multi_seed_ci.csv"
    manifest_path = args.output_dir / "stats_manifest.json"
    paired.to_csv(paired_path, index=False)
    summaries.to_csv(summary_path, index=False)
    manifest = {
        "state": "done",
        "input_case_csv": str(args.case_csv),
        "model": args.model,
        "comparisons": [f"{candidate}:{reference}" for candidate, reference in comparisons],
        "metrics": args.metrics,
        "morphology_groups": args.groups,
        "expected_source_seed_cells": 6,
        "body_columns": ["mean_delta", "delta_ci_low", "delta_ci_high", "paired_t_p"],
        "audit_columns": [
            "wilcoxon_p",
            "exact_sign_flip_p",
            "paired_t_p_holm",
            "wilcoxon_p_holm",
            "exact_sign_flip_p_holm",
        ],
        "n_boot": args.n_boot,
        "random_seed": args.random_seed,
        "paired_csv": str(paired_path),
        "multi_seed_ci_csv": str(summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
