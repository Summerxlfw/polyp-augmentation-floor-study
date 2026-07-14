#!/usr/bin/env python3
"""重算预注册 F1/F1-all：10 个候选臂相对 strong_aug_floor。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

import video_cluster_stats as paired_stats


F1_CANDIDATES = (
    "SLAug",
    "CCSDG",
    "CSDG",
    "DSU",
    "MixStyle",
    "fourier_amp_aug",
    "ibn_whitening",
    "spectral_consistency",
    "spectral_ibn_combo",
    "SLAug_official",
)

F1_FAMILIES = (
    ("F1", "hard_flat_IIa", {"morphology_group": "hard_flat_IIa"}),
    ("F1-all", "all", {}),
)


def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def resolve_family_rows(
    primary: pd.DataFrame,
    external: pd.DataFrame,
    candidates: Sequence[str],
    reference: str,
    metric: str,
) -> pd.DataFrame:
    """从主 60-run 表取九臂与地板，从外部 case 表补官方 SLAug。"""
    required = [*paired_stats.PAIR_KEYS, "method", metric, "morphology_group"]
    _require_columns(primary, required, "primary table")
    _require_columns(external, required, "external table")
    if reference not in set(primary["method"]):
        raise ValueError(f"reference method is absent from primary table: {reference}")

    frames = [primary[primary["method"] == reference].copy()]
    for candidate in candidates:
        primary_rows = primary[primary["method"] == candidate]
        external_rows = external[external["method"] == candidate]
        if not primary_rows.empty:
            frames.append(primary_rows.copy())
        elif not external_rows.empty:
            frames.append(external_rows.copy())
        else:
            raise ValueError(f"candidate method is absent from both tables: {candidate}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_keys = ["method", *paired_stats.PAIR_KEYS]
    if combined.duplicated(duplicate_keys).any():
        raise ValueError("duplicate method/case keys after family row resolution")
    expected_methods = {reference, *candidates}
    observed_methods = set(combined["method"])
    if observed_methods != expected_methods:
        raise ValueError(
            f"resolved methods differ: expected={sorted(expected_methods)}, "
            f"observed={sorted(observed_methods)}"
        )
    return combined


def summarize_prespecified_families(
    table: pd.DataFrame,
    candidates: Sequence[str],
    reference: str,
    model: str,
    metric: str,
    families: Sequence[tuple[str, str, Mapping[str, object]]],
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐候选计算 6-cell 主推断，并在每个预注册 family 内做 Holm。"""
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for family_label, stratum, filters in families:
        family_results: list[dict[str, object]] = []
        for candidate in candidates:
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
            result["stratum"] = stratum
            result["multiplicity_family"] = family_label
            family_results.append(result)
            for row in cells:
                cell_rows.append(
                    {
                        "multiplicity_family": family_label,
                        "stratum": stratum,
                        "candidate": candidate,
                        "reference": reference,
                        "model": model,
                        "metric": metric,
                        **row,
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
    primary_case_csv: Path,
    external_case_csv: Path,
    output_dir: Path,
    candidates: Sequence[str] = F1_CANDIDATES,
    reference: str = "strong_aug_floor",
    model: str = "polyp_pvt",
    metric: str = "dice_mean",
    n_boot: int = 10000,
    random_seed: int = 20260713,
) -> dict[str, Path]:
    primary = pd.read_csv(primary_case_csv)
    external = pd.read_csv(external_case_csv)
    combined = resolve_family_rows(
        primary,
        external,
        candidates=candidates,
        reference=reference,
        metric=metric,
    )
    summary, cells = summarize_prespecified_families(
        combined,
        candidates=candidates,
        reference=reference,
        model=model,
        metric=metric,
        families=F1_FAMILIES,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "f1_family_summary.csv"
    cells_path = output_dir / "f1_family_cells.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    summary.to_csv(summary_path, index=False)
    cells.to_csv(cells_path, index=False)
    manifest = {
        "state": "done",
        "protocol": "F1 and F1-all each contain ten predefined configuration-minus-floor comparisons; primary inference uses six repeated training units and Holm adjustment within family; video-cluster bootstrap is supporting",
        "primary_case_csv": str(primary_case_csv.resolve()),
        "primary_case_csv_sha256": _sha256(primary_case_csv),
        "external_case_csv": str(external_case_csv.resolve()),
        "external_case_csv_sha256": _sha256(external_case_csv),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "paired_stats_script": str(Path(paired_stats.__file__).resolve()),
        "paired_stats_script_sha256": _sha256(Path(paired_stats.__file__).resolve()),
        "candidate_order": list(candidates),
        "reference": reference,
        "model": model,
        "metric": metric,
        "families": [
            {"label": label, "stratum": stratum, "filters": dict(filters)}
            for label, stratum, filters in F1_FAMILIES
        ],
        "n_boot": n_boot,
        "random_seed": random_seed,
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
    parser.add_argument("--primary-case-csv", type=Path, required=True)
    parser.add_argument("--external-case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    paths = write_family_bundle(
        primary_case_csv=args.primary_case_csv,
        external_case_csv=args.external_case_csv,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
