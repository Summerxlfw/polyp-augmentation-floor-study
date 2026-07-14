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
    "official_SLAug": "Official SLAug",
    "spatial_warp_aug": "Spatial warp",
    "fourier_amp_aug": "Fourier amplitude",
    "CCSDG": "CCSDG",
    "spectral_consistency": "Spectral consistency",
    "MixStyle": "MixStyle",
    "DSU": "DSU",
    "CSDG": "CSDG",
    "spectral_ibn_combo": "Spectral + IBN",
    "ibn_whitening": "IBN whitening",
}
ORDER = list(DISPLAY)
FIDELITY = {
    "official_SLAug": "official-code",
    "spatial_warp_aug": "ours",
    "fourier_amp_aug": r"coverage$^{\dagger}$",
    "CCSDG": r"coverage$^{\dagger}$",
    "spectral_consistency": r"coverage$^{\dagger}$",
    "MixStyle": r"coverage$^{\dagger}$",
    "DSU": r"coverage$^{\dagger}$",
    "CSDG": r"coverage$^{\dagger}$",
    "spectral_ibn_combo": r"coverage$^{\dagger}$",
    "ibn_whitening": r"coverage$^{\dagger}$",
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
    assert source.loc["C1", "n"] == "206 train / 45 validation"
    assert source.loc["C3", "n"] == "359 train / 96 validation"
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Cohort composition, analysis units, and run completeness. PolypGen counts are positive-mask frames; SUN-SEG cases are nested in videos.}}
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
\multicolumn{{4}}{{l}}{{\textit{{Experimental units and run matrix}}}} \\
Primary paired unit & Source center $\times$ seed & 6 & 2 sources $\times$ 3 seeds \\
Harmonized formal matrix & 10 training configurations & 60 & completed runs \\
Official comparator & Official SLAug & 6 & completed runs \\
Structural control & Joint affine & 6 & completed runs \\
Optimization audit & Best validation Dice & {training['best_val_dice_min'].min():.4f}--{training['best_val_dice_max'].max():.4f} & batch size 4; epochs {int(training['epochs_min'].min())}--{int(training['epochs_max'].max())} \\
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
            f"{DISPLAY[candidate]} & {FIDELITY[candidate]} & {_delta(h.mean_delta)} & "
            f"{_ci(h.cell_t_ci_low, h.cell_t_ci_high)} & {int(h.positive_cells_of_6)}/6 & "
            f"{_p(h.holm_p)} & {_delta(a.mean_delta)} & {_ci(a.cell_t_ci_low, a.cell_t_ci_high)} \\\\"
        )
    rows_text = "\n".join(rows)
    direct_row = (
        f"Spatial warp $-$ official SLAug & ours vs official-code & {_delta(direct.mean_delta)} & "
        f"{_ci(direct.cell_t_ci_low, direct.cell_t_ci_high)} & {int(direct.positive_cells_of_6)}/6 & "
        f"{_p(direct.paired_t_p)}$^{{\\ddagger}}$ & -- & -- \\\\"
    )
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Paired Dice differences under the harmonized strong-augmentation protocol. Positive values favor the candidate.}}
\label{{tab:strong-floor}}
\scriptsize
\setlength{{\tabcolsep}}{{3.7pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
& & \multicolumn{{4}}{{c}}{{Hard-flat IIa}} & \multicolumn{{2}}{{c}}{{All cases}} \\
\cmidrule(lr){{3-6}}\cmidrule(l){{7-8}}
Intervention & Fidelity & $\Delta$ & 95\% CI & $+$/6 & $p$ & $\Delta$ & 95\% CI \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{A. Candidate minus strong augmentation floor}}}} \\
{rows_text}
\addlinespace
\multicolumn{{8}}{{l}}{{\textit{{B. Direct matched comparison}}}} \\
{direct_row}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize $\Delta$ and confidence intervals use six paired source-by-seed differences. For panel A, $p$ is Holm-adjusted within the 10-comparison F1 or F1-all family. $^{{\ddagger}}$Panel B reports the paired-t $p$ value for the registered size-one head-to-head comparison. $^{{\dagger}}$Coverage implementations broaden the intervention families examined but are not native-recipe rankings.
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
                f"{_ci(row.cell_t_ci_low, row.cell_t_ci_high)} & {int(row.positive_cells_of_6)}/6 & "
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
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
Stratum or contrast & Cases & Videos & $\Delta$ & 95\% CI & $+$/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.92\textwidth}}
\footnotesize Estimates and confidence intervals use six paired source-by-seed differences. F4-shape and F4-size are separate correction families and include the displayed difference-in-differences contrast. Morphology and area overlap: 26 of 28 Ip cases were large and none was small; the two panels are not independent confirmation.
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
        "CCSDG": "CCSDG",
        "CSDG": "CSDG",
        "DSU": "DSU",
        "MixStyle": "MixStyle",
        "fourier_amp_aug": "Fourier amplitude",
        "ibn_whitening": "IBN whitening",
        "spectral_consistency": "Spectral consistency",
        "spectral_ibn_combo": "Spectral + IBN",
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
\begin{{table*}}[t]
\centering
\caption{{Descriptive absolute performance across the harmonized 10-configuration matrix.}}
\label{{tab:absolute-metrics}}
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{PolypGen frozen centers}} & \multicolumn{{5}}{{c}}{{SUN-SEG}} \\
\cmidrule(lr){{2-4}}\cmidrule(l){{5-9}}
Intervention & Macro Dice & $\Delta$ & $+$/6 & All Dice & IIa Dice & IIa HD95 & IIa F$_\beta^w$ & IIa BIoU \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize PolypGen macro Dice gives equal weight to the five frozen centers within each run; $\Delta$ is candidate minus floor across the six source-by-seed cells. SUN-SEG values are absolute six-cell means. This table is descriptive; adjusted inference is reported in Tables~\ref{{tab:strong-floor}} and \ref{{tab:phenotype}}. HD95 is in pixels and lower is better.
\end{{minipage}}
\end{{table*}}
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
\caption{{Hard-flat Dice contrasts in the completed four-formulation diagnostic.}}
\label{{tab:factorial}}
\small
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Contrast & $\Delta$ & 95\% CI & $+$/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.92\textwidth}}
\footnotesize The paired-softmix cell and F6 family were specified after inspection of the earlier controls. F3-new and F2 values retain their frozen families; the remaining contrasts and interaction use F6. The table characterizes implemented formulations and is not an orthogonal causal decomposition.
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
        "slaug_official_plus_warp - official_SLAug": "Combination $-$ official SLAug",
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
\begin{{table*}}[t]
\centering
\caption{{Boundary-sensitive improvement in the four prespecified isolation comparisons.}}
\label{{tab:boundary-isolation}}
\scriptsize
\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\toprule
Metric & Candidate comparison & Improvement & 95\% CI & $+$/6 & Holm $p$ \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.94\textwidth}}
\footnotesize Improvement is oriented so that positive values favor the candidate; HD95 was sign-reversed because lower is better. Holm correction was applied separately within the four-comparison HD95, weighted-F, and Boundary-IoU families.
\end{{minipage}}
\end{{table*}}
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
