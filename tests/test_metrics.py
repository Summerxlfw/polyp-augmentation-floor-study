from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from l1_boundary_metrics import boundary_iou, hd95, metric_bundle  # noqa: E402
from l1_paired_stats import exact_sign_flip_p, holm_adjust  # noqa: E402


def test_identical_masks_have_perfect_boundary_metrics() -> None:
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True
    probability = mask.astype(float)
    bundle = metric_bundle(probability, mask)
    assert bundle["dice"] == 1.0
    assert bundle["hd95_px"] == 0.0
    assert bundle["boundary_iou"] == 1.0
    assert bundle["weighted_fbeta"] == 1.0


def test_empty_prediction_hd95_uses_image_diagonal() -> None:
    target = np.zeros((30, 40), dtype=bool)
    target[5:10, 5:10] = True
    prediction = np.zeros_like(target)
    assert hd95(prediction, target) == 50.0
    assert boundary_iou(prediction, target) == 0.0


def test_sign_flip_and_holm_are_deterministic() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert exact_sign_flip_p(values) == 0.03125
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
