from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "training"))

from augmentations import (  # noqa: E402
    image_only_affine,
    image_only_warp_mix,
    joint_affine,
    paired_affine_softmix,
)


def _example() -> tuple[Image.Image, Image.Image]:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (80, 160, 240)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[9:23, 10:22] = 255
    return Image.fromarray(image), Image.fromarray(mask)


def test_identity_parameters_preserve_all_inputs() -> None:
    image, mask = _example()
    fixed = (1.0, 0.0, 0.0)
    assert np.array_equal(np.asarray(image_only_affine(image, fixed)), np.asarray(image))
    joint_image, joint_mask = joint_affine(image, mask, fixed)
    assert np.array_equal(np.asarray(joint_image), np.asarray(image))
    assert np.array_equal(np.asarray(joint_mask), np.asarray(mask))


def test_alpha_one_matches_joint_affine_image_and_target() -> None:
    image, mask = _example()
    affine = (1.10, 2.0, -1.0)
    paired_image, paired_mask = paired_affine_softmix(image, mask, (*affine, 1.0))
    joint_image, joint_mask = joint_affine(image, mask, affine)
    assert np.array_equal(np.asarray(paired_image), np.asarray(joint_image))
    assert np.array_equal(
        np.asarray(paired_mask),
        np.asarray(joint_mask, dtype=np.float32) / 255.0,
    )


def test_alpha_zero_is_original_image_and_float_target() -> None:
    image, mask = _example()
    output_image, output_mask = paired_affine_softmix(
        image,
        mask,
        (1.10, 2.0, -1.0, 0.0),
    )
    assert np.array_equal(np.asarray(output_image), np.asarray(image))
    assert output_mask.mode == "F"
    assert np.array_equal(
        np.asarray(output_mask),
        np.asarray(mask, dtype=np.float32) / 255.0,
    )


def test_intermediate_alpha_creates_bounded_soft_target() -> None:
    image, mask = _example()
    params = (1.0, 4.0, 0.0, 0.5)
    output_image, output_mask = paired_affine_softmix(image, mask, params)
    image_only = image_only_warp_mix(image, params)
    values = np.asarray(output_mask)
    assert np.array_equal(np.asarray(output_image), np.asarray(image_only))
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0
    assert np.any((values > 0.0) & (values < 1.0))
