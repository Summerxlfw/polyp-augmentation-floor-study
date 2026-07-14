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
