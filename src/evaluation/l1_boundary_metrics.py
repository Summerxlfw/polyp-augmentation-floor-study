"""L1 扁平息肉评估使用的边界指标。

协议：固定阈值 0.5；HD95 使用对称表面距离 95 分位；Boundary IoU
使用官方 0.02×图像对角线边界宽度；加权 F-beta 使用 Margolin 等人的
连续前景图公式。SUN-SEG flat 队列只含阳性 mask，空 GT 会显式报错。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    convolve,
    distance_transform_edt,
    generate_binary_structure,
)


METRIC_PROTOCOL_VERSION = "l1-boundary-v1-20260713"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOUNDARY_DILATION_RATIO = 0.02


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask 必须是二维，实际 shape={array.shape}")
    return array.astype(bool, copy=False)


def _validate_probability(probability: np.ndarray, gt: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != gt.shape:
        raise ValueError(f"probability/gt shape 不一致: {probability.shape} vs {gt.shape}")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability 含非有限值")
    if probability.size and (float(probability.min()) < 0.0 or float(probability.max()) > 1.0):
        raise ValueError("probability 必须位于 [0, 1]")
    return probability


def boundary_width(shape: tuple[int, int], dilation_ratio: float = DEFAULT_BOUNDARY_DILATION_RATIO) -> int:
    """按 Boundary IoU 官方实现计算边界宽度。"""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"无效图像 shape: {shape}")
    if dilation_ratio <= 0:
        raise ValueError("dilation_ratio 必须为正")
    image_diagonal = math.sqrt(float(shape[0] ** 2 + shape[1] ** 2))
    return max(1, int(round(dilation_ratio * image_diagonal)))


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = DEFAULT_BOUNDARY_DILATION_RATIO) -> np.ndarray:
    """将二值 mask 转成内侧边界带，等价于官方 3×3 erosion 实现。"""
    mask_bool = _as_bool_mask(mask)
    height, width = mask_bool.shape
    iterations = boundary_width(mask_bool.shape, dilation_ratio)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    eroded = binary_erosion(
        padded,
        structure=np.ones((3, 3), dtype=bool),
        iterations=iterations,
        border_value=0,
    )
    eroded = eroded[1 : height + 1, 1 : width + 1]
    return mask_bool & ~eroded


def boundary_iou(
    prediction: np.ndarray,
    gt: np.ndarray,
    dilation_ratio: float = DEFAULT_BOUNDARY_DILATION_RATIO,
) -> float:
    pred_boundary = mask_to_boundary(prediction, dilation_ratio)
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    union = np.count_nonzero(pred_boundary | gt_boundary)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(pred_boundary & gt_boundary)
    return float(intersection / union)


def _surface_distances(
    result: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, float],
    connectivity: int,
) -> np.ndarray:
    footprint = generate_binary_structure(2, connectivity)
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1, border_value=0)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1, border_value=0)
    distance_map = distance_transform_edt(~reference_border, sampling=spacing)
    return distance_map[result_border]


def hd95(
    prediction: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    connectivity: int = 1,
) -> float:
    """对称表面距离的 95 分位；单侧空 mask 用图像对角线惩罚。"""
    pred = _as_bool_mask(prediction)
    target = _as_bool_mask(gt)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/gt shape 不一致: {pred.shape} vs {target.shape}")
    if len(spacing) != 2 or min(spacing) <= 0:
        raise ValueError(f"无效 spacing: {spacing}")
    pred_empty = not bool(pred.any())
    gt_empty = not bool(target.any())
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float(math.sqrt((pred.shape[0] * spacing[0]) ** 2 + (pred.shape[1] * spacing[1]) ** 2))
    distances = np.hstack(
        (
            _surface_distances(pred, target, spacing, connectivity),
            _surface_distances(target, pred, spacing, connectivity),
        )
    )
    return float(np.percentile(distances, 95))


def _gaussian_kernel(size: int = 7, sigma: float = 5.0) -> np.ndarray:
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return kernel / kernel.sum()


def weighted_fbeta(probability: np.ndarray, gt: np.ndarray) -> float:
    """Margolin et al. 的加权 F-beta（beta=1），输入为连续概率图。"""
    target = _as_bool_mask(gt)
    probability = _validate_probability(probability, target)
    if not target.any():
        raise ValueError("L1 positive-only 协议不允许空 GT")

    target_float = target.astype(np.float64)
    error = np.abs(probability - target_float)
    distance, nearest = distance_transform_edt(~target, return_indices=True)
    propagated_error = error.copy()
    propagated_error[~target] = propagated_error[nearest[0][~target], nearest[1][~target]]
    smoothed_error = convolve(propagated_error, _gaussian_kernel(), mode="nearest")
    minimum_error = error.copy()
    replace = target & (smoothed_error < error)
    minimum_error[replace] = smoothed_error[replace]

    importance = np.ones_like(target_float)
    importance[~target] = 2.0 - np.exp(np.log(0.5) / 5.0 * distance[~target])
    weighted_error = minimum_error * importance
    true_positive_weighted = float(target.sum() - weighted_error[target].sum())
    false_positive_weighted = float(weighted_error[~target].sum())
    recall = 1.0 - float(weighted_error[target].mean())
    precision = true_positive_weighted / (true_positive_weighted + false_positive_weighted + np.finfo(float).eps)
    score = 2.0 * recall * precision / (recall + precision + np.finfo(float).eps)
    return float(np.clip(score, 0.0, 1.0))


def metric_bundle(
    probability: np.ndarray,
    gt: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    dilation_ratio: float = DEFAULT_BOUNDARY_DILATION_RATIO,
) -> dict[str, Any]:
    target = _as_bool_mask(gt)
    probability = _validate_probability(probability, target)
    if not target.any():
        raise ValueError("L1 positive-only 协议不允许空 GT")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须位于 [0, 1]")
    prediction = probability >= threshold
    intersection = float(np.count_nonzero(prediction & target))
    denominator = float(np.count_nonzero(prediction) + np.count_nonzero(target))
    dice = 2.0 * intersection / denominator if denominator else 1.0
    return {
        "dice": dice,
        "hd95_px": hd95(prediction, target),
        "weighted_fbeta": weighted_fbeta(probability, target),
        "boundary_iou": boundary_iou(prediction, target, dilation_ratio),
        "empty_prediction": not bool(prediction.any()),
        "boundary_width_px": boundary_width(target.shape, dilation_ratio),
    }
