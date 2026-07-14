#!/usr/bin/env python3
"""从正式 case CSV 重算并生成 BSPC 主文 Fig. 2/3。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

FIGURE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = FIGURE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats


MANUSCRIPT_DIR = FIGURE_DIR.parent
REPOSITORY_DIR = FIGURE_DIR.parent
OUTPUT_DIR = FIGURE_DIR / "generated"

FORMAL_CASE_CSV = Path(
    os.environ.get(
        "L1_FORMAL_CASE_CSV",
        REPOSITORY_DIR / "data" / "case_level" / "main_flat_cases.csv",
    )
)
BOUNDARY_CASE_CSV = Path(
    os.environ.get(
        "L1_BOUNDARY_CASE_CSV",
        REPOSITORY_DIR / "data" / "case_level" / "external_boundary_cases.csv",
    )
)
T2_CSV = REPOSITORY_DIR / "data" / "derived" / "T2a_f1_family_results.csv"
T3_CSV = REPOSITORY_DIR / "data" / "derived" / "T3_phenotype_effect_shell.csv"

FLOOR = "strong_aug_floor"
SPATIAL_WARP_KEY = "SLAug"
OFFICIAL_KEY = "SLAug_official"
SHAPE_ORDER = ["IIa", "Is", "Isp", "Ip"]
SIZE_ORDER = ["small", "mid", "large"]
METHOD_ORDER = [
    OFFICIAL_KEY,
    SPATIAL_WARP_KEY,
    "fourier_amp_aug",
    "CCSDG",
    "spectral_consistency",
    "MixStyle",
    "DSU",
    "CSDG",
    "spectral_ibn_combo",
    "ibn_whitening",
]
DISPLAY = {
    OFFICIAL_KEY: "Official SLAug",
    SPATIAL_WARP_KEY: "Spatial warp (ours)",
    "fourier_amp_aug": "Fourier amplitude†",
    "CCSDG": "CCSDG†",
    "spectral_consistency": "Spectral consistency†",
    "MixStyle": "MixStyle†",
    "DSU": "DSU†",
    "CSDG": "CSDG†",
    "spectral_ibn_combo": "Spectral + IBN†",
    "ibn_whitening": "IBN whitening†",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )


def _validate_formal(df: pd.DataFrame) -> None:
    assert len(df) == 17_100
    assert df["run_id"].nunique() == 60
    assert df["case_id"].nunique() == 285
    assert df["video_id"].nunique() == 100
    assert set(df["source_center"]) == {"C1", "C3"}
    assert set(df["seed"].astype(int)) == {0, 1, 2}
    assert df["method"].nunique() == 10
    assert FLOOR in set(df["method"])
    assert SPATIAL_WARP_KEY in set(df["method"])


def _cell_differences(
    candidate: pd.DataFrame,
    floor: pd.DataFrame,
    mask_column: str,
    mask_value: str,
) -> np.ndarray:
    cand = candidate[candidate[mask_column] == mask_value]
    ref = floor[floor[mask_column] == mask_value]
    keys = ["source_center", "seed"]
    cand_cells = cand.groupby(keys, observed=True)["dice_mean"].mean()
    ref_cells = ref.groupby(keys, observed=True)["dice_mean"].mean()
    paired = pd.concat([cand_cells.rename("candidate"), ref_cells.rename("floor")], axis=1)
    paired = paired.dropna()
    assert len(paired) == 6, (mask_column, mask_value, paired)
    return (paired["candidate"] - paired["floor"]).to_numpy(dtype=float)


def _paired_summary(deltas: np.ndarray) -> tuple[float, float, float, int]:
    assert len(deltas) == 6
    mean = float(np.mean(deltas))
    sem = float(stats.sem(deltas))
    half = float(stats.t.ppf(0.975, len(deltas) - 1) * sem)
    return mean, mean - half, mean + half, int(np.sum(deltas > 0))


def _build_heatmap_data(formal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["source_center", "seed", "shape", "size_bin"]
    cell_means = (
        formal.groupby(keys + ["method"], observed=True)["dice_mean"]
        .mean()
        .reset_index()
    )
    floor = cell_means[cell_means["method"] == FLOOR].drop(columns="method")
    candidates = cell_means[cell_means["method"] != FLOOR]
    paired = candidates.merge(
        floor,
        on=keys,
        how="left",
        suffixes=("_candidate", "_floor"),
        validate="many_to_one",
    )
    paired["delta"] = paired["dice_mean_candidate"] - paired["dice_mean_floor"]
    summary = (
        paired.groupby(["method", "shape", "size_bin"], observed=True)["delta"]
        .agg(mean_delta="mean", n_source_seed_cells="count")
        .reset_index()
    )
    cases = formal.drop_duplicates("case_id")
    counts = (
        cases.groupby(["shape", "size_bin"], observed=True)["case_id"]
        .nunique()
        .rename("n_cases")
        .reset_index()
    )
    return summary, counts


def _plot_heatmap(formal: pd.DataFrame) -> dict[str, Path]:
    summary, counts = _build_heatmap_data(formal)
    candidates = [method for method in METHOD_ORDER if method not in {OFFICIAL_KEY}]
    columns = [(shape, size) for shape in SHAPE_ORDER for size in SIZE_ORDER]
    values = np.full((len(candidates), len(columns)), np.nan, dtype=float)
    for row_idx, method in enumerate(candidates):
        for col_idx, (shape, size) in enumerate(columns):
            match = summary[
                (summary["method"] == method)
                & (summary["shape"] == shape)
                & (summary["size_bin"] == size)
            ]
            if not match.empty:
                assert int(match.iloc[0]["n_source_seed_cells"]) == 6
                values[row_idx, col_idx] = float(match.iloc[0]["mean_delta"])

    count_matrix = np.zeros((len(SHAPE_ORDER), len(SIZE_ORDER)), dtype=int)
    for i, shape in enumerate(SHAPE_ORDER):
        for j, size in enumerate(SIZE_ORDER):
            match = counts[(counts["shape"] == shape) & (counts["size_bin"] == size)]
            count_matrix[i, j] = 0 if match.empty else int(match.iloc[0]["n_cases"])

    assert count_matrix[SHAPE_ORDER.index("Ip")].tolist() == [0, 2, 26]
    plot_values = 100.0 * values
    limit = max(4.0, float(math.ceil(np.nanmax(np.abs(plot_values)))))
    cmap = plt.get_cmap("PuOr").copy()
    cmap.set_bad("#f2f2f2")

    fig = plt.figure(figsize=(11.2, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.1, 1.35], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(plot_values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(candidates)), [DISPLAY[m] for m in candidates])
    ax.set_xticks(
        np.arange(len(columns)),
        [f"{shape}\n{size.capitalize()}" for shape, size in columns],
        rotation=45,
        ha="right",
    )
    ax.set_title("A  Phenotype-by-area response across harmonized configurations", loc="left", weight="bold")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = plot_values[i, j]
            if np.isnan(value):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#9a9a9a",
                        hatch="///",
                        linewidth=0.0,
                    )
                )
            else:
                text_color = "white" if abs(value) > 0.62 * limit else "#222222"
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.7, color=text_color)
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color="white", lw=1.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("Mean paired Dice difference (points)")

    ax_counts = fig.add_subplot(grid[0, 1])
    count_cmap = plt.get_cmap("Greys").copy()
    ax_counts.imshow(count_matrix, cmap=count_cmap, vmin=0, vmax=count_matrix.max(), aspect="auto")
    ax_counts.set_xticks(np.arange(len(SIZE_ORDER)), [s.capitalize() for s in SIZE_ORDER], rotation=35, ha="right")
    ax_counts.set_yticks(np.arange(len(SHAPE_ORDER)), SHAPE_ORDER)
    ax_counts.set_title("B  Unique-case counts", loc="left", weight="bold")
    for i in range(count_matrix.shape[0]):
        for j in range(count_matrix.shape[1]):
            count = count_matrix[i, j]
            ax_counts.text(j, i, str(count), ha="center", va="center", color="white" if count > 30 else "#111111")
            if count == 0:
                ax_counts.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#777777", hatch="///", linewidth=0.0)
                )
    fig.text(
        0.01,
        0.01,
        "Values are candidate − strong augmentation floor; each tile equally averages six source×seed cells. "
        "† Harmonized coverage implementation. Hatched cells contain no cases; no tile is a separate significance test.",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.22)

    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUTPUT_DIR / f"fig2_phenotype_area_heatmap.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs[suffix] = path
    plt.close(fig)

    summary.to_csv(OUTPUT_DIR / "fig2_heatmap_values.csv", index=False)
    counts.to_csv(OUTPUT_DIR / "fig2_case_counts.csv", index=False)
    return outputs


def _forest_data(formal: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    case_sizes = formal[["case_id", "size_bin"]].drop_duplicates()
    assert case_sizes["case_id"].nunique() == 285
    official = boundary[boundary["method"] == OFFICIAL_KEY].merge(
        case_sizes,
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    assert len(official) == 1710
    floor = formal[formal["method"] == FLOOR]
    records: list[dict[str, object]] = []
    strata = [
        ("Hard-flat IIa", "morphology_group", "hard_flat_IIa"),
        ("Ip", "shape", "Ip"),
        ("Large", "size_bin", "large"),
    ]
    for method in METHOD_ORDER:
        candidate = official if method == OFFICIAL_KEY else formal[formal["method"] == method]
        for stratum_label, column, value in strata:
            deltas = _cell_differences(candidate, floor, column, value)
            mean, low, high, positive = _paired_summary(deltas)
            records.append(
                {
                    "method": method,
                    "display": DISPLAY[method],
                    "fidelity": "official" if method == OFFICIAL_KEY else ("ours" if method == SPATIAL_WARP_KEY else "coverage"),
                    "stratum": stratum_label,
                    "mean_delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_cells_of_6": positive,
                }
            )
    result = pd.DataFrame.from_records(records)

    t2 = pd.read_csv(T2_CSV)
    t3 = pd.read_csv(T3_CSV)
    expected = {
        (SPATIAL_WARP_KEY, "Hard-flat IIa"): float(
            t2[(t2["multiplicity_family"] == "F1") & (t2["candidate"] == "spatial_warp_aug")].iloc[0]["mean_delta"]
        ),
        (OFFICIAL_KEY, "Hard-flat IIa"): float(
            t2[(t2["multiplicity_family"] == "F1") & (t2["candidate"] == "official_SLAug")].iloc[0]["mean_delta"]
        ),
        (SPATIAL_WARP_KEY, "Ip"): float(t3[t3["stratum"] == "Ip"].iloc[0]["mean_delta"]),
        (SPATIAL_WARP_KEY, "Large"): float(t3[t3["stratum"] == "large"].iloc[0]["mean_delta"]),
    }
    for key, expected_value in expected.items():
        method, stratum = key
        actual = float(result[(result["method"] == method) & (result["stratum"] == stratum)].iloc[0]["mean_delta"])
        assert abs(actual - expected_value) < 1e-10, (key, actual, expected_value)
    return result


def _plot_forest(formal: pd.DataFrame, boundary: pd.DataFrame) -> dict[str, Path]:
    data = _forest_data(formal, boundary)
    data.to_csv(OUTPUT_DIR / "fig3_forest_values.csv", index=False)
    max_abs = float(np.nanmax(np.abs(data[["ci_low", "ci_high"]].to_numpy())))
    limit = max(0.05, math.ceil(max_abs / 0.05) * 0.05)
    y = np.arange(len(METHOD_ORDER))
    styles = {
        "official": {"marker": "D", "color": "#111111", "label": "Official-code port"},
        "ours": {"marker": "o", "color": "#0072B2", "label": "Ours"},
        "coverage": {"marker": "s", "color": "#777777", "label": "Harmonized coverage port†"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.5), sharey=True)
    for panel_idx, (ax, stratum) in enumerate(zip(axes, ["Hard-flat IIa", "Ip", "Large"], strict=True)):
        subset = data[data["stratum"] == stratum].set_index("method").loc[METHOD_ORDER]
        ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0)
        for row_idx, method in enumerate(METHOD_ORDER):
            row = subset.loc[method]
            style = styles[str(row["fidelity"])]
            ax.errorbar(
                float(row["mean_delta"]),
                row_idx,
                xerr=np.array(
                    [
                        [float(row["mean_delta"] - row["ci_low"])],
                        [float(row["ci_high"] - row["mean_delta"])],
                    ]
                ),
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markersize=4.8,
                elinewidth=1.1,
                capsize=2.2,
                zorder=2,
            )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
        ax.grid(axis="x", color="#e4e4e4", lw=0.6)
        ax.set_title(f"{chr(65 + panel_idx)}  {stratum}", loc="left", weight="bold")
        ax.set_xlabel("Paired Dice difference")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [DISPLAY[m] for m in METHOD_ORDER])
    legend = [
        Line2D([0], [0], marker=s["marker"], color="none", markerfacecolor=s["color"], markeredgecolor=s["color"], label=s["label"], markersize=5)
        for s in styles.values()
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.58, 0.995))
    fig.text(
        0.01,
        0.012,
        "Candidate − strong augmentation floor; points and 95% CIs use six paired source×seed differences. "
        "Ip and large are overlapping, not independent, strata. † Coverage rows are not native-recipe rankings.",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.88, bottom=0.16, wspace=0.16)

    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUTPUT_DIR / f"fig3_benefit_harm_forest.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs[suffix] = path
    plt.close(fig)
    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    formal = pd.read_csv(FORMAL_CASE_CSV)
    boundary = pd.read_csv(BOUNDARY_CASE_CSV)
    _validate_formal(formal)
    assert set(boundary["method"].unique()) >= {SPATIAL_WARP_KEY, OFFICIAL_KEY}
    heatmap_outputs = _plot_heatmap(formal)
    forest_outputs = _plot_forest(formal, boundary)

    manifest = {
        "formal_case_csv": str(FORMAL_CASE_CSV),
        "formal_case_csv_sha256": _sha256(FORMAL_CASE_CSV),
        "boundary_case_csv": str(BOUNDARY_CASE_CSV),
        "boundary_case_csv_sha256": _sha256(BOUNDARY_CASE_CSV),
        "t2_csv": str(T2_CSV),
        "t2_csv_sha256": _sha256(T2_CSV),
        "t3_csv": str(T3_CSV),
        "t3_csv_sha256": _sha256(T3_CSV),
        "outputs": {
            "fig2": {key: str(path) for key, path in heatmap_outputs.items()},
            "fig3": {key: str(path) for key, path in forest_outputs.items()},
        },
    }
    (OUTPUT_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Fig. 2/3 generated from formal evidence CSVs")


if __name__ == "__main__":
    main()
