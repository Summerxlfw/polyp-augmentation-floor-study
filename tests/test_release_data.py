from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_main_case_table_has_locked_shape() -> None:
    table = pd.read_csv(DATA / "case_level" / "main_flat_cases.csv")
    assert len(table) == 17_100
    assert table["run_id"].nunique() == 60
    assert table["method"].nunique() == 10
    assert set(table.groupby("run_id").size()) == {285}


def test_factorial_tables_have_four_complete_configurations() -> None:
    for name in ("factorial_4cell_flat_case.csv", "factorial_4cell_boundary_case.csv"):
        table = pd.read_csv(DATA / "case_level" / name)
        assert set(table["method"]) == {
            "warp_alpha100",
            "joint_affine_floor",
            "SLAug",
            "paired_affine_softmix",
        }
        assert table["run_id"].nunique() == 24
        assert set(table.groupby("run_id").size()) == {285}


def test_derived_primary_tables_are_present() -> None:
    required = {
        "T0_training_completeness.csv",
        "T2a_f1_family_results.csv",
        "T3_phenotype_effect_shell.csv",
        "T5_boundary_f5.csv",
        "T6_factorial_formulation_contrasts.csv",
    }
    assert required <= {path.name for path in (DATA / "derived").glob("*.csv")}


def test_complete_training_inventory_sums_to_102_runs() -> None:
    table = pd.read_csv(DATA / "derived" / "T1_protocol_shell.csv")
    total = table[table["dimension"] == "total_training_runs"].iloc[0]
    components = table[
        table["dimension"].isin(
            ["formal_matrix", "named_comparator", "secondary_controls"]
        )
    ]

    assert int(total["n"]) == 102
    assert components["n"].astype(int).sum() == 102


def test_public_figure_labels_are_neutral_and_provenance_is_explicit() -> None:
    table = pd.read_csv(DATA / "derived" / "fig3_forest_values.csv")
    assert "Official SLAug" not in set(table["display"])
    assert "Spatial warp (ours)" not in set(table["display"])
    assert set(table["implementation_class"]) == {
        "author_code_adaptation",
        "study_configuration",
        "protocol_aligned",
    }
    assert set(table["stratum"]) == {"Paris IIa", "Ip", "Large"}


def test_main_phenotype_contrast_has_video_cluster_interval() -> None:
    table = pd.read_csv(DATA / "derived" / "f4_family_summary.csv")
    row = table[table["comparison_id"] == "IIa_minus_Ip"].iloc[0]

    assert row["inference_role"] == "multiplicity_controlled_exploratory_contrast"
    assert row["supporting_video_cluster_ci_low"] > 0
    assert row["supporting_video_cluster_ci_high"] > row["supporting_video_cluster_ci_low"]
