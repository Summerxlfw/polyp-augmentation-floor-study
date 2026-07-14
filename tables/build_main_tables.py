#!/usr/bin/env python3
"""从锁定 CSV 生成 BSPC 主文 Table 1–6 的 LaTeX。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


LATEX_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = LATEX_DIR.parent
TABLE_DIR = MANUSCRIPT_DIR / "data" / "derived"
OUTPUT_DIR = LATEX_DIR / "generated_tables"

T0 = TABLE_DIR / "T0_training_completeness.csv"
T1 = TABLE_DIR / "T1_protocol_shell.csv"
T2A = TABLE_DIR / "T2a_f1_family_results.csv"
T2B = TABLE_DIR / "T2b_a1_vs_official.csv"
T2C = TABLE_DIR / "T2c_crosscenter_metric_registry.csv"
T3 = TABLE_DIR / "T3_phenotype_effect_shell.csv"
T5 = TABLE_DIR / "T5_boundary_f5.csv"
T6 = TABLE_DIR / "T6_factorial_formulation_contrasts.csv"

DISPLAY = {
    "official_SLAug": "SLAug",
    "spatial_warp_aug": "Spatial warp",
    "fourier_amp_aug": "Fourier amplitude",
    "CCSDG": "Channel-style perturbation",
    "spectral_consistency": "Spectral consistency",
    "MixStyle": "MixStyle-style perturbation",
    "DSU": "DSU-style perturbation",
    "CSDG": "Bias-field perturbation",
    "spectral_ibn_combo": "Spectral + whitening",
    "ibn_whitening": "Feature whitening",
}
ORDER = list(DISPLAY)
IMPLEMENTATION = {
    "official_SLAug": "Authors' code adapted",
    "spatial_warp_aug": "Study-defined configuration",
    "fourier_amp_aug": r"Study implementation$^{\dagger}$",
    "CCSDG": r"Study implementation$^{\dagger}$",
    "spectral_consistency": r"Study implementation$^{\dagger}$",
    "MixStyle": r"Study implementation$^{\dagger}$",
    "DSU": r"Study implementation$^{\dagger}$",
    "CSDG": r"Study implementation$^{\dagger}$",
    "spectral_ibn_combo": r"Study implementation$^{\dagger}$",
    "ibn_whitening": r"Study implementation$^{\dagger}$",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta(value: float) -> str:
    return f"{value:+.3f}"


def _ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _p(value: float) -> str:
    if value < 0.001:
        return r"$<$0.001"
    return f"{value:.3f}"


def _write(name: str, content: str) -> Path:
    path = OUTPUT_DIR / name
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def _table1() -> str:
    protocol = pd.read_csv(T1, dtype=str).fillna("")
    training = pd.read_csv(T0)
    assert len(training) == 10 and int(training["n_runs"].sum()) == 60
    source = protocol[protocol["dimension"] == "source_center"].set_index("level_or_arm")
    held = protocol[protocol["dimension"] == "held_out_centers"].set_index("level_or_arm")
    inventory = protocol[protocol["dimension"].isin(
        ["formal_matrix", "named_comparator", "secondary_controls"]
    )]
    total = protocol[protocol["dimension"] == "total_training_runs"].iloc[0]
    assert source.loc["C1", "n"] == "206 train / 45 validation"
    assert source.loc["C3", "n"] == "359 train / 96 validation"
    assert inventory["n"].astype(int).sum() == int(total["n"]) == 102
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Cohort composition, analysis units, and experimental summary. PolypGen counts are positive-mask frames; SUN-SEG cases are nested in videos.}}
\label{{tab:cohort}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.19\textwidth}}>{{\raggedright\arraybackslash}}p{{0.19\textwidth}}rX@{{}}}}
\toprule
Component & Setting & Count & Unit or composition \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{PolypGen source and frozen tests}}}} \\
Source center & C1 & 206 / 45 & training / validation frames \\
Frozen tests & C1-source setting & {held.loc['C1 source setting', 'n']} & frames from the remaining five centers \\
Source center & C3 & 359 / 96 & training / validation frames \\
Frozen tests & C3-source setting & {held.loc['C3 source setting', 'n']} & frames from the remaining five centers \\
\addlinespace
\multicolumn{{4}}{{l}}{{\textit{{SUN-SEG external evaluation}}}} \\
Dataset & SUN-SEG & 49,136 & frames \\
Analysis unit & Lesion case & 285 & cases nested in 100 videos \\
Paris morphology & Is / IIa / Ip / Isp & 134 / 101 / 28 / 22 & cases \\
Area proxy & Small / middle / large & 95 / 95 / 95 & reference-mask tertiles \\
\addlinespace
\multicolumn{{4}}{{l}}{{\textit{{Experimental units and training configurations}}}} \\
Repeated training unit & Source center $\times$ seed & 6 & 3 seeds nested within each of 2 fixed sources \\
Matched configuration set & 10 training configurations & 60 & completed runs \\
Named comparator & SLAug & 6 & completed runs \\
Secondary controls & 6 training configurations & 36 & completed runs \\
Complete study & 17 training configurations & 102 & completed runs \\
Training summary & Best validation Dice & {training['best_val_dice_min'].min():.4f}--{training['best_val_dice_max'].max():.4f} & 60 runs; batch size 4; epochs {int(training['epochs_min'].min())}--{int(training['epochs_max'].max())} \\
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""


def _table2() -> str:
    table = pd.read_csv(T2A)
    direct = pd.read_csv(T2B).iloc[0]
    hard = table[table["multiplicity_family"] == "F1"].set_index("candidate")
    all_case = table[table["multiplicity_family"] == "F1-all"].set_index("candidate")
    assert set(ORDER) == set(hard.index) == set(all_case.index)
    rows = []
    for candidate in ORDER:
        h = hard.loc[candidate]
        a = all_case.loc[candidate]
        rows.append(
            f"{DISPLAY[candidate]} & {IMPLEMENTATION[candidate]} & {_delta(h.mean_delta)} & "
            f"{_ci(h.cell_t_ci_low, h.cell_t_ci_high)} & {int(h.positive_cells_of_6)}/6 & "
            f"{_p(h.holm_p)} & {_delta(a.mean_delta)} & {_ci(a.cell_t_ci_low, a.cell_t_ci_high)} \\\\"
        )
    rows_text = "\n".join(rows)
    direct_row = (
        f"Spatial warp $-$ SLAug & Direct matched comparison & {_delta(direct.mean_delta)} & "
        f"{_ci(direct.cell_t_ci_low, direct.cell_t_ci_high)} & {int(direct.positive_cells_of_6)}/6 & "
        f"{_p(direct.paired_t_p)}$^{{\\ddagger}}$ & -- & -- \\\\"
    )
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Paired Dice differences under the shared strong-augmentation protocol. Positive values favor the configuration.}}
\label{{tab:strong-floor}}
\scriptsize
\setlength{{\tabcolsep}}{{3.7pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
& & \multicolumn{{4}}{{c}}{{Paris IIa}} & \multicolumn{{2}}{{c}}{{All cases}} \\
\cmidrule(lr){{3-6}}\cmidrule(l){{7-8}}
Configuration & Implementation & $\Delta$ & 95\% CI & Positive/6 & $p$ & $\Delta$ & 95\% CI \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{A. Configuration minus strong augmentation floor}}}} \\
{rows_text}
\addlinespace
\multicolumn{{8}}{{l}}{{\textit{{B. Direct matched comparison}}}} \\
{direct_row}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize $\Delta$ and confidence intervals use six paired differences from repeated training units, with three seeds nested within each of two fixed source settings. For panel A, $p$ is Holm-adjusted within the 10-comparison Paris IIa or all-case configuration family. $^{{\ddagger}}$Panel B reports the paired-t $p$ value for the direct head-to-head comparison. $^{{\dagger}}$Study implementations were evaluated under the shared protocol and are not reproductions of the original recipes of similarly named methods.
\end{{minipage}}
\end{{table*}}
"""


def _table3() -> str:
    table = pd.read_csv(T3)
    rows = []
    panels = [
        ("A. Paris morphology", ["IIa", "Is", "Isp", "Ip", "IIa_minus_Ip"]),
        ("B. Reference-mask area", ["small", "mid", "large", "large_minus_small"]),
    ]
    labels = {
        "IIa": "IIa",
        "Is": "Is",
        "Isp": "Isp",
        "Ip": "Ip",
        "IIa_minus_Ip": r"$\Delta_{IIa}-\Delta_{Ip}$",
        "small": "Small",
        "mid": "Middle",
        "large": "Large",
        "large_minus_small": r"$\Delta_{large}-\Delta_{small}$",
    }
    for panel, strata in panels:
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{panel}}}}} \\")
        for stratum in strata:
            row = table[table["stratum"] == stratum].iloc[0]
            n_cases = "--" if pd.isna(row["n_cases"]) else str(int(row["n_cases"]))
            n_videos = "--" if pd.isna(row["n_videos"]) else str(int(row["n_videos"]))
            rows.append(
                f"{labels[stratum]} & {n_cases} & {n_videos} & {_delta(row.mean_delta)} & "
                f"{_ci(row.cell_t_ci_low, row.cell_t_ci_high)} & "
                f"{_ci(row.supporting_video_cluster_ci_low, row.supporting_video_cluster_ci_high)} & "
                f"{int(row.positive_cells_of_6)}/6 & "
                f"{_p(row.holm_p)} \\\\"
            )
        rows.append(r"\addlinespace")
    rows_text = "\n".join(rows[:-1])
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Phenotype-conditioned Dice response of spatial warp augmentation relative to the strong augmentation floor.}}
\label{{tab:phenotype}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
Stratum or contrast & Cases & Videos & $\Delta$ & Six-unit 95\% CI & Video-cluster 95\% CI & Positive/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.92\textwidth}}
\footnotesize Six-unit intervals summarize repeated trainings under two fixed source settings; synchronized video-cluster intervals separately address test-video sampling. Morphology and area are outcome-informed exploratory correction families, each including the displayed difference-in-differences contrast. Morphology and area overlap: 26 of 28 Ip cases were large and none was small; the two panels are not independent confirmation.
\end{{minipage}}
\end{{table*}}
"""


def _table4() -> str:
    table = pd.read_csv(T2C).set_index("method")
    order = [
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
    ]
    labels = {
        "strong_aug_floor": "Strong augmentation floor",
        "SLAug": "Spatial warp",
        "CCSDG": "Channel-style perturbation",
        "CSDG": "Bias-field perturbation",
        "DSU": "DSU-style perturbation",
        "MixStyle": "MixStyle-style perturbation",
        "fourier_amp_aug": "Fourier amplitude",
        "ibn_whitening": "Feature whitening",
        "spectral_consistency": "Spectral consistency",
        "spectral_ibn_combo": "Spectral + whitening",
    }
    assert set(order) == set(table.index)
    rows = []
    for method in order:
        row = table.loc[method]
        delta = (
            "--"
            if method == "strong_aug_floor"
            else _delta(row.polypgen_delta_vs_floor)
        )
        direction = (
            "--"
            if method == "strong_aug_floor"
            else f"{int(row.polypgen_positive_cells_of_6)}/6"
        )
        rows.append(
            f"{labels[method]} & {row.polypgen_center_macro_dice:.3f} & "
            f"{delta} & {direction} & {row.sunseg_all_dice:.3f} & "
            f"{row.sunseg_hard_flat_dice:.3f} & "
            f"{row.sunseg_hard_flat_hd95_px:.2f} & "
            f"{row.sunseg_hard_flat_weighted_fbeta:.3f} & "
            f"{row.sunseg_hard_flat_boundary_iou:.3f} \\\\"
        )
    rows_text = "\n".join(rows)
    return rf"""
\begin{{table}}[H]
\centering
\caption{{Descriptive absolute performance across the shared-protocol 10-configuration matrix.}}
\label{{tab:absolute-metrics}}
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{PolypGen frozen centers}} & \multicolumn{{5}}{{c}}{{SUN-SEG}} \\
\cmidrule(lr){{2-4}}\cmidrule(l){{5-9}}
Configuration & Macro Dice & $\Delta$ & Positive/6 & All Dice & IIa Dice & IIa HD95 & IIa F$_\beta^w$ & IIa BIoU \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize PolypGen macro Dice gives equal weight to the five frozen centers within each run; $\Delta$ is configuration minus floor across the six repeated training units nested within two fixed sources. SUN-SEG values are absolute six-unit means. This table is descriptive; adjusted inference is reported in main-text Tables 2 and 3. HD95 is in pixels and lower is better.
\end{{minipage}}
\end{{table}}
"""


def _table5() -> str:
    table = pd.read_csv(T6)
    labels = {
        "sync_no_blend": "Paired target effect, no blend",
        "sync_blend": "Paired target effect, blend present",
        "blend_unpaired": "Blend effect, unpaired target",
        "blend_paired": "Blend effect, paired target",
        "interaction": "Difference-in-differences interaction",
    }
    rows = []
    for row in table.itertuples(index=False):
        rows.append(
            f"{labels[row.contrast_id]} & {_delta(row.mean_delta_dice)} & "
            f"{_ci(row.cell_t_ci_low, row.cell_t_ci_high)} & "
            f"{int(row.positive_cells_of_6)}/6 & {_p(row.holm_p)} \\\\"
        )
    rows_text = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Paris IIa Dice contrasts in the completed four-formulation diagnostic.}}
\label{{tab:factorial}}
\scriptsize
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Contrast & $\Delta$ & 95\% CI & Positive/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.92\textwidth}}
\footnotesize The paired-softmix cell and its three-comparison correction family were specified after inspection of the earlier controls. Previously defined control contrasts retain their original correction families. Because this cell was specified post hoc, these contrasts are interpreted as formulation diagnostics rather than causal component effects.
\end{{minipage}}
\end{{table*}}
"""


def _table6() -> str:
    table = pd.read_csv(T5)
    metric_labels = {
        "HD95_px": "HD95",
        "weighted_Fbeta": r"F$_\beta^w$",
        "Boundary_IoU": "BIoU",
    }
    comparison_labels = {
        "spatial_warp_aug - warp_alpha015": "Spatial warp $-$ weak warp",
        "spatial_warp_aug - warp_alpha100": "Spatial warp $-$ severe mismatch",
        "spatial_warp_aug - warp_shift_only": "Spatial warp $-$ shift only",
        "slaug_official_plus_warp - official_SLAug": "Combination $-$ SLAug",
    }
    rows = []
    for row in table.itertuples(index=False):
        rows.append(
            f"{metric_labels[row.metric]} & {comparison_labels[row.comparison]} & "
            f"{_delta(row.mean_improvement)} & "
            f"{_ci(row.improvement_cell_t_ci_low, row.improvement_cell_t_ci_high)} & "
            f"{int(row.improved_cells_of_6)}/6 & {_p(row.holm_p)} \\\\"
        )
    rows_text = "\n".join(rows)
    return rf"""
\begin{{table}}[H]
\centering
\caption{{Boundary-sensitive improvement in the four prespecified isolation comparisons.}}
\label{{tab:boundary-isolation}}
\scriptsize
\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\toprule
Metric & Configuration comparison & Improvement & 95\% CI & Positive/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.94\textwidth}}
\footnotesize Improvement is oriented so that positive values favor the first-named configuration; HD95 was sign-reversed because lower is better. Holm correction was applied separately within the four-comparison HD95, weighted-F, and Boundary-IoU families.
\end{{minipage}}
\end{{table}}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table1": _write("table1_cohort_run_structure.tex", _table1()),
        "table2": _write("table2_strong_floor_comparisons.tex", _table2()),
        "table3": _write("table3_phenotype_response.tex", _table3()),
        "table4": _write("table4_absolute_metric_registry.tex", _table4()),
        "table5": _write("table5_factorial_contrasts.tex", _table5()),
        "table6": _write("table6_boundary_isolation.tex", _table6()),
    }
    manifest = {
        "inputs": {
            str(path): _sha256(path)
            for path in (T0, T1, T2A, T2B, T2C, T3, T5, T6)
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (OUTPUT_DIR / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Table 1–6 LaTeX generated from locked CSVs")


if __name__ == "__main__":
    main()
