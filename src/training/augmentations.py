"""Standalone implementations of the four spatial-formulation cells.

The sampling ranges and interpolation/padding choices match the final formal
training snapshot. This module avoids project-specific model adapters so the
augmentation controls can be unit-tested independently.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image


def sample_spatial_warp_params(
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    """Sample scale, x/y translation, and mixing coefficient."""
    scale = random.uniform(0.84, 1.16)
    dx = random.uniform(-0.08, 0.08) * width
    dy = random.uniform(-0.08, 0.08) * height
    alpha = random.uniform(0.35, 0.65)
    return scale, dx, dy, alpha


def _affine_matrix(
    height: int,
    width: int,
    scale: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    return np.array(
        [
            [scale, 0, (1 - scale) * width / 2 + dx],
            [0, scale, (1 - scale) * height / 2 + dy],
        ],
        dtype=np.float32,
    )


def image_only_warp_mix(
    image: Image.Image,
    params: tuple[float, float, float, float],
) -> Image.Image:
    """U0B1: warp and mix the image while leaving the target unchanged."""
    scale, dx, dy, alpha = params
    array = np.asarray(image).astype(np.float32)
    height, width = array.shape[:2]
    matrix = _affine_matrix(height, width, scale, dx, dy)
    warped = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    output = alpha * warped + (1.0 - alpha) * array
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def image_only_affine(
    image: Image.Image,
    params: tuple[float, float, float],
) -> Image.Image:
    """U0B0: affine-warp only the image, with no alpha mixing."""
    scale, dx, dy = params
    return image_only_warp_mix(image, (scale, dx, dy, 1.0))


def joint_affine(
    image: Image.Image,
    mask: Image.Image,
    params: tuple[float, float, float],
) -> tuple[Image.Image, Image.Image]:
    """U1B0: apply one affine matrix to image and binary target."""
    scale, dx, dy = params
    array = np.asarray(image).astype(np.float32)
    height, width = array.shape[:2]
    matrix = _affine_matrix(height, width, scale, dx, dy)
    warped_image = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_array = np.asarray(mask).astype(np.float32)
    warped_mask = cv2.warpAffine(
        mask_array,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (
        Image.fromarray(np.clip(warped_image, 0, 255).astype(np.uint8)),
        Image.fromarray(warped_mask.astype(np.uint8)),
    )


def paired_affine_softmix(
    image: Image.Image,
    mask: Image.Image,
    params: tuple[float, float, float, float],
) -> tuple[Image.Image, Image.Image]:
    """U1B1: use one matrix and one alpha for image and soft target."""
    scale, dx, dy, alpha = params
    array = np.asarray(image).astype(np.float32)
    height, width = array.shape[:2]
    matrix = _affine_matrix(height, width, scale, dx, dy)
    warped_image = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    image_output = alpha * warped_image + (1.0 - alpha) * array

    mask_array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    warped_mask = cv2.warpAffine(
        mask_array,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_output = alpha * warped_mask + (1.0 - alpha) * mask_array
    return (
        Image.fromarray(np.clip(image_output, 0, 255).astype(np.uint8)),
        Image.fromarray(mask_output.astype(np.float32), mode="F"),
    )
