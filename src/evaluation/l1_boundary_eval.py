#!/usr/bin/env python
"""L1 SUN-SEG 扁平子群的 Dice / HD95 / Fβw / Boundary IoU 评估。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from l1_boundary_metrics import (
    DEFAULT_BOUNDARY_DILATION_RATIO,
    DEFAULT_THRESHOLD,
    METRIC_PROTOCOL_VERSION,
    metric_bundle,
)


PROJECT_ROOT = Path(
    os.environ.get("POLYP_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser()
S1_ROOT = PROJECT_ROOT / "03_experiments" / "S1_loco_gate_smoke_20260710"
FORMAL_ROOT = PROJECT_ROOT / "03_experiments" / "L1_formal_20260711"
OUTPUT_ROOT = FORMAL_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "flat_boundary_eval_20260713"
METHOD_DISPLAY = {"SLAug": "spatial_warp_aug"}


METRIC_DIRECTIONS = {
    "dice_mean": "higher",
    "hd95_px_mean": "lower",
    "weighted_fbeta_mean": "higher",
    "boundary_iou_mean": "higher",
}
MORPHOLOGY_GROUPS = ("hard_flat_IIa", "non_flat", "all")


def _bootstrap_ci(values: np.ndarray, n_boot: int, random_seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(array, size=(n_boot, array.size), replace=True)
    return tuple(np.percentile(samples.mean(axis=1), [2.5, 97.5]).astype(float))


def summarize_case_table(
    case_table: pd.DataFrame,
    n_boot: int = 2000,
    random_seed: int = 20260713,
) -> pd.DataFrame:
    """按 run 和形态组做 case-equal、video/lesion-cluster CI 汇总。"""
    required = {
        "run_id",
        "source_center",
        "model",
        "method",
        "seed",
        "case_id",
        "cluster_id",
        "morphology_group",
        *METRIC_DIRECTIONS,
    }
    missing = sorted(required - set(case_table.columns))
    if missing:
        raise ValueError(f"case table 缺列: {missing}")
    rows: list[dict[str, Any]] = []
    run_keys = ["run_id", "source_center", "model", "method", "seed"]
    for key, run_df in case_table.groupby(run_keys, sort=True):
        for group in MORPHOLOGY_GROUPS:
            group_df = run_df if group == "all" else run_df[run_df["morphology_group"] == group]
            if group_df.empty:
                continue
            for metric, direction in METRIC_DIRECTIONS.items():
                values = pd.to_numeric(group_df[metric], errors="raise")
                cluster_values = (
                    group_df.assign(_metric=values)
                    .groupby("cluster_id")["_metric"]
                    .mean()
                    .to_numpy(dtype=np.float64)
                )
                ci_low, ci_high = _bootstrap_ci(cluster_values, n_boot, random_seed)
                rows.append(
                    {
                        **dict(zip(run_keys, key)),
                        "morphology_group": group,
                        "metric": metric,
                        "direction": direction,
                        "n_cases": int(group_df["case_id"].nunique()),
                        "n_clusters": int(group_df["cluster_id"].nunique()),
                        "mean": float(values.mean()),
                        "cluster_ci_low": ci_low,
                        "cluster_ci_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def runs_from_summaries(
    summary_root: Path,
    run_suffixes: tuple[str, ...],
    methods: tuple[str, ...],
    expected_runs: int,
    expected_runs_per_method: int | None = None,
) -> list[dict[str, Any]]:
    """从多个 suffix 收集 run，并严格核对 checkpoint、重复和数量。"""
    selected: dict[str, dict[str, Any]] = {}
    for suffix in run_suffixes:
        for run_dir in sorted(summary_root.glob(f"*{suffix}")):
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("method") not in methods:
                continue
            required = ("run_id", "source_center", "model", "method", "seed", "checkpoint_best")
            missing = [key for key in required if key not in summary]
            if missing:
                raise ValueError(f"{summary_path} 缺字段: {missing}")
            checkpoint = Path(summary["checkpoint_best"])
            if not checkpoint.is_file():
                raise ValueError(f"{summary['run_id']} checkpoint_best 缺失: {checkpoint}")
            run = {
                "run_id": str(summary["run_id"]),
                "source_center": str(summary["source_center"]),
                "model": str(summary["model"]),
                "method": str(summary["method"]),
                "seed": int(summary["seed"]),
                "checkpoint_best": str(checkpoint),
                "summary_path": str(summary_path),
            }
            previous = selected.get(run["run_id"])
            if previous is not None and previous != run:
                raise ValueError(f"run_id 重复且内容冲突: {run['run_id']}")
            selected[run["run_id"]] = run
    runs = sorted(selected.values(), key=lambda row: (row["method"], row["source_center"], row["seed"]))
    if len(runs) != expected_runs:
        raise ValueError(f"run 数应为 {expected_runs}，实际 {len(runs)}；拒绝 partial 评估")
    if expected_runs_per_method is not None:
        counts = pd.Series([run["method"] for run in runs]).value_counts().to_dict()
        bad = {method: int(counts.get(method, 0)) for method in methods if counts.get(method, 0) != expected_runs_per_method}
        if bad:
            raise ValueError(
                f"每方法 run 数应为 {expected_runs_per_method}，不完整方法={bad}；拒绝 partial 评估"
            )
    return runs


def verify_dice_parity(
    boundary_cases: pd.DataFrame,
    old_flat_cases: pd.DataFrame,
    expected_run_ids: list[str],
    tolerance: float = 5e-6,
) -> dict[str, Any]:
    """新评估 Dice 与旧 flat evaluator 做逐 run/case 对账。"""
    keys = ["run_id", "case_id"]
    expected = set(expected_run_ids)
    new = boundary_cases[boundary_cases["run_id"].isin(expected)][keys + ["dice_mean"]].copy()
    old = old_flat_cases[old_flat_cases["run_id"].isin(expected)][keys + ["dice_mean"]].copy()
    if set(new["run_id"].unique()) != expected or set(old["run_id"].unique()) != expected:
        raise ValueError(
            "Dice parity run_id 不完整: "
            f"new={new['run_id'].nunique()}/{len(expected)}, old={old['run_id'].nunique()}/{len(expected)}"
        )
    for label, frame in (("new", new), ("old", old)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Dice parity {label} 存在重复 run/case 键")
    new_keys = set(map(tuple, new[keys].itertuples(index=False, name=None)))
    old_keys = set(map(tuple, old[keys].itertuples(index=False, name=None)))
    if new_keys != old_keys:
        raise ValueError(
            "Dice parity 配对键不完整: "
            f"new_only={len(new_keys - old_keys)}, old_only={len(old_keys - new_keys)}"
        )
    merged = new.merge(old, on=keys, how="inner", validate="one_to_one", suffixes=("_new", "_old"))
    absolute_delta = (merged["dice_mean_new"].astype(float) - merged["dice_mean_old"].astype(float)).abs()
    maximum = float(absolute_delta.max()) if len(absolute_delta) else float("nan")
    if maximum > tolerance:
        raise ValueError(f"Dice parity 超容差: max_abs_delta={maximum:.8g}, tolerance={tolerance}")
    return {
        "n_runs": int(merged["run_id"].nunique()),
        "n_paired_cases": int(len(merged)),
        "max_abs_delta": maximum,
        "tolerance": tolerance,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evaluate_one_run(
    run: dict[str, Any],
    sunseg_rows: list[dict[str, str]],
    output_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    write_frame_csv: bool,
) -> dict[str, Any]:
    """单 run 推理；结果写独立版本目录，不复用旧 Dice done.json。"""
    import torch
    import torch.nn.functional as torch_functional
    from torch.utils.data import DataLoader

    if str(S1_ROOT) not in sys.path:
        sys.path.insert(0, str(S1_ROOT))
    import l1_flat_eval_orchestrator as flat
    import method_dev_batch2 as method_dev
    from preflight_gate_tools import SunSegMorphologyDataset
    from s1_loco_common import fuse_outputs

    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_id = run["run_id"]
    frame_path = run_root / f"boundary_frame_{run_id}.csv"
    case_path = run_root / f"boundary_case_{run_id}.csv"
    done_path = run_root / f"boundary_done_{run_id}.json"
    if done_path.is_file() and case_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("state") == "done" and done.get("metric_protocol_version") == METRIC_PROTOCOL_VERSION:
            return done

    start = time.time()
    method_dev.ARM_MODEL_KEY[run["method"]] = run["model"]
    model, model_key = flat._load_method_model(run["method"], Path(run["checkpoint_best"]), device)
    model.eval()
    loader = DataLoader(
        SunSegMorphologyDataset(sunseg_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    metadata_fields = [
        "run_id",
        "source_center",
        "model",
        "method",
        "method_display",
        "seed",
        "checkpoint",
        "case_id",
        "video_id",
        "lesion_id",
        "shape",
        "morphology_group",
        "cluster_id",
    ]
    frame_fields = metadata_fields + [
        "frame_stem",
        "dice",
        "hd95_px",
        "weighted_fbeta",
        "boundary_iou",
        "empty_prediction",
        "boundary_width_px",
    ]
    frame_file = frame_path.open("w", encoding="utf-8", newline="") if write_frame_csv else None
    frame_writer = csv.DictWriter(frame_file, fieldnames=frame_fields) if frame_file is not None else None
    if frame_writer is not None:
        frame_writer.writeheader()
    accumulators: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    empty_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    frame_count = 0
    try:
        with torch.no_grad():
            for image, mask, batch_indices in loader:
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logits = fuse_outputs(model_key, model(image))
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = torch_functional.interpolate(
                        logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
                    )
                probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                targets = mask.detach().cpu().numpy()
                for batch_position, local_idx in enumerate(batch_indices.tolist()):
                    source_row = sunseg_rows[local_idx]
                    probability = probabilities[batch_position, 0]
                    target = targets[batch_position, 0] > 0.5
                    metrics = metric_bundle(probability, target)
                    group = "hard_flat_IIa" if source_row["shape"] == "IIa" else "non_flat"
                    case_key = (
                        source_row["case_id"],
                        source_row["video_id"],
                        source_row["lesion_id"],
                        source_row["shape"],
                        group,
                    )
                    for metric in ("dice", "hd95_px", "weighted_fbeta", "boundary_iou"):
                        accumulators[case_key][metric].append(float(metrics[metric]))
                    empty_counts[case_key] += int(metrics["empty_prediction"])
                    frame_count += 1
                    if frame_writer is not None:
                        frame_writer.writerow(
                            {
                                "run_id": run_id,
                                "source_center": run["source_center"],
                                "model": model_key,
                                "method": run["method"],
                                "method_display": METHOD_DISPLAY.get(run["method"], run["method"]),
                                "seed": run["seed"],
                                "checkpoint": run["checkpoint_best"],
                                "case_id": source_row["case_id"],
                                "video_id": source_row["video_id"],
                                "lesion_id": source_row["lesion_id"],
                                "shape": source_row["shape"],
                                "morphology_group": group,
                                "cluster_id": f"{source_row['case_id']}:{source_row['video_id']}",
                                "frame_stem": source_row["frame_stem"],
                                **metrics,
                            }
                        )
    finally:
        if frame_file is not None:
            frame_file.close()

    case_rows: list[dict[str, Any]] = []
    for case_key, metric_values in sorted(accumulators.items()):
        case_id, video_id, lesion_id, shape, group = case_key
        n_frames = len(metric_values["dice"])
        case_rows.append(
            {
                "run_id": run_id,
                "source_center": run["source_center"],
                "model": model_key,
                "method": run["method"],
                "method_display": METHOD_DISPLAY.get(run["method"], run["method"]),
                "seed": run["seed"],
                "checkpoint": run["checkpoint_best"],
                "case_id": case_id,
                "video_id": video_id,
                "lesion_id": lesion_id,
                "shape": shape,
                "morphology_group": group,
                "cluster_id": f"{case_id}:{video_id}",
                "n_frames": n_frames,
                "dice_mean": float(np.mean(metric_values["dice"])),
                "hd95_px_mean": float(np.mean(metric_values["hd95_px"])),
                "weighted_fbeta_mean": float(np.mean(metric_values["weighted_fbeta"])),
                "boundary_iou_mean": float(np.mean(metric_values["boundary_iou"])),
                "n_empty_predictions": int(empty_counts[case_key]),
                "empty_prediction_rate": float(empty_counts[case_key] / n_frames),
            }
        )
    case_fields = metadata_fields + [
        "n_frames",
        "dice_mean",
        "hd95_px_mean",
        "weighted_fbeta_mean",
        "boundary_iou_mean",
        "n_empty_predictions",
        "empty_prediction_rate",
    ]
    _write_csv(case_path, case_rows, case_fields)
    done = {
        "state": "done",
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "run_id": run_id,
        "device": device,
        "batch_size": batch_size,
        "n_frames": frame_count,
        "n_cases": len(case_rows),
        "n_empty_predictions": int(sum(empty_counts.values())),
        "frame_csv": str(frame_path) if write_frame_csv else "",
        "case_csv": str(case_path),
        "seconds": round(time.time() - start, 2),
    }
    _write_json(done_path, done)
    return done


def _load_case_tables(output_root: Path, run_ids: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for run_id in run_ids:
        path = output_root / "runs" / f"boundary_case_{run_id}.csv"
        if not path.is_file():
            missing.append(run_id)
        else:
            frames.append(pd.read_csv(path))
    if missing:
        raise ValueError(f"boundary case CSV 缺 {len(missing)} 个 run: {missing}")
    table = pd.concat(frames, ignore_index=True)
    if table["run_id"].nunique() != len(run_ids):
        raise ValueError(f"case table run_id 数不符: {table['run_id'].nunique()} vs {len(run_ids)}")
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L1 flat 的 HD95/Fβw/Boundary IoU 严格评估")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--run-suffixes", nargs="+", required=True)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-runs", type=int, default=0)
    parser.add_argument("--runs-per-method", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--skip-frame-csv", action="store_true")
    parser.add_argument("--skip-dice-parity", action="store_true")
    parser.add_argument("--flat-case-csv", type=Path)
    parser.add_argument("--evaluate-only", action="store_true", help="并行 worker：只写各 run 结果，不汇总")
    parser.add_argument("--summarize-only", action="store_true", help="不推理，严格汇总既有 worker 结果")
    return parser


def execution_mode(evaluate_only: bool, summarize_only: bool) -> str:
    if evaluate_only and summarize_only:
        raise ValueError("--evaluate-only 与 --summarize-only 不能同时使用")
    if evaluate_only:
        return "evaluate_only"
    if summarize_only:
        return "summarize_only"
    return "run_all"


def main() -> int:
    args = build_parser().parse_args()
    mode = execution_mode(args.evaluate_only, args.summarize_only)
    if str(S1_ROOT) not in sys.path:
        sys.path.insert(0, str(S1_ROOT))
    import l1_flat_eval_orchestrator as flat
    from s1_loco_common import SMOKE_ROOT

    methods = tuple(args.methods)
    expected_runs = args.expected_runs or len(methods) * args.runs_per_method
    summary_root = args.summary_root or SMOKE_ROOT
    runs = runs_from_summaries(
        summary_root,
        tuple(args.run_suffixes),
        methods,
        expected_runs,
        expected_runs_per_method=args.runs_per_method,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] boundary eval runs={len(runs)}")
    for run in runs:
        print(f"  {run['run_id']}")
    sunseg = flat.sunseg_rows()
    worker_slug = "__".join(sorted(methods)).replace("/", "_")
    if mode != "summarize_only":
        failures = []
        for index, run in enumerate(runs, start=1):
            try:
                done = _evaluate_one_run(
                    run,
                    sunseg,
                    args.output_root,
                    args.device,
                    args.batch_size,
                    args.num_workers,
                    not args.skip_frame_csv,
                )
                print(f"[{index}/{len(runs)}] {run['run_id']}: {done['state']}", flush=True)
            except Exception as error:
                failures.append({"run_id": run["run_id"], "error": repr(error)})
                print(f"[{index}/{len(runs)}] {run['run_id']}: FAILED {error!r}", file=sys.stderr, flush=True)
        if failures:
            _write_json(
                args.output_root / f"boundary_eval_failed_{worker_slug}.json",
                {"state": "failed", "methods": list(methods), "failures": failures},
            )
            return 1
        if mode == "evaluate_only":
            worker_done = {
                "state": "done",
                "mode": mode,
                "methods": list(methods),
                "n_runs": len(runs),
                "run_ids": [run["run_id"] for run in runs],
                "device": args.device,
                "batch_size": args.batch_size,
                "metric_protocol_version": METRIC_PROTOCOL_VERSION,
            }
            _write_json(args.output_root / f"worker_done_{worker_slug}.json", worker_done)
            print(json.dumps(worker_done, ensure_ascii=False, indent=2))
            return 0

    run_ids = [run["run_id"] for run in runs]
    case_table = _load_case_tables(args.output_root, run_ids)
    case_all_path = args.output_root / "sunseg_boundary_case_all.csv"
    summary_path = args.output_root / "sunseg_boundary_summary_long.csv"
    case_table.to_csv(case_all_path, index=False)
    summary = summarize_case_table(case_table, n_boot=args.n_boot, random_seed=args.random_seed)
    summary.to_csv(summary_path, index=False)

    parity: dict[str, Any] = {"state": "skipped"}
    if not args.skip_dice_parity:
        flat_case_csv = args.flat_case_csv or (flat.FLAT_ROOT / "sunseg_case_all.csv")
        if not flat_case_csv.is_file():
            raise FileNotFoundError(f"Dice parity 输入缺失: {flat_case_csv}")
        parity = {
            "state": "pass",
            "source_csv": str(flat_case_csv),
            **verify_dice_parity(case_table, pd.read_csv(flat_case_csv), run_ids),
        }
    manifest = {
        "state": "done",
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "threshold": DEFAULT_THRESHOLD,
        "boundary_dilation_ratio": DEFAULT_BOUNDARY_DILATION_RATIO,
        "hd95": {
            "definition": "symmetric surface-distance 95th percentile",
            "unit": "pixel on fixed SUN-SEG cache resolution",
            "connectivity": 1,
            "empty_prediction_policy": "image diagonal penalty and explicit count",
        },
        "weighted_fbeta": "Margolin et al. continuous probability-map implementation; no per-image min-max",
        "n_expected_runs": expected_runs,
        "n_actual_runs": int(case_table["run_id"].nunique()),
        "run_ids": run_ids,
        "methods": list(methods),
        "method_display_aliases": METHOD_DISPLAY,
        "n_unique_cases": int(case_table["case_id"].nunique()),
        "n_case_rows": int(len(case_table)),
        "batch_size": args.batch_size,
        "dice_parity": parity,
        "case_csv": str(case_all_path),
        "summary_csv": str(summary_path),
    }
    _write_json(args.output_root / "metric_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
