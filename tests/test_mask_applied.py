"""Guard: the mask must be applied to train, val AND test batches.

The bug this catches: `image = image * FACE_MASK` was once indented inside
`if training:`, so validation and test batches went through unmasked -- the model
was evaluated on different inputs than it trained on.

Uses tiny generated all-white JPEGs, so masked pixels are the only dark ones.
No dataset or GPU needed.
"""

import numpy as np
import pytest
import tensorflow as tf

from model import Config, build_dataset

IMG_SIZE = 32
CORNER = (2, 2)                     # well outside the face ellipse
CENTRE = (IMG_SIZE // 2, IMG_SIZE // 2)


@pytest.fixture(scope="module")
def white_images(tmp_path_factory):
    """Four all-white JPEGs; every non-white pixel later on comes from the mask."""
    d = tmp_path_factory.mktemp("imgs")
    white = tf.fill((IMG_SIZE, IMG_SIZE, 3), tf.constant(255, tf.uint8))
    paths = []
    for i in range(4):
        p = d / f"{i}.jpg"
        tf.io.write_file(str(p), tf.io.encode_jpeg(white, quality=100))
        paths.append(str(p))
    return paths


def _first_batch(paths, tmp_path, mask_mode, split, training):
    cfg = Config(
        name="unit",
        real_dir="unused", fake_dir="unused",
        mask_mode=mask_mode,
        img_size=IMG_SIZE, native_size=IMG_SIZE, batch_size=len(paths),
        cache_dir=str(tmp_path / f"cache_{mask_mode}_{split}_{training}"),
    )
    ds = build_dataset(paths, [0] * len(paths), cfg, split,
                       shuffle=False, training=training)
    images, _ = next(iter(ds))
    return images.numpy()


@pytest.mark.parametrize("split,training", [("train", True), ("val", False), ("test", False)])
def test_face_only_zeroes_background_in_every_split(white_images, tmp_path, split, training):
    batch = _first_batch(white_images, tmp_path, "face_only", split, training)

    corner = batch[:, CORNER[0], CORNER[1], :]
    centre = batch[:, CENTRE[0], CENTRE[1], :]

    assert np.all(corner < 0.05), (
        f"{split} (training={training}): background pixel not masked -- "
        "is the mask line indented inside `if training:` again?")
    assert np.all(centre > 0.5), f"{split}: face region should survive face_only"


@pytest.mark.parametrize("split,training", [("train", True), ("val", False), ("test", False)])
def test_background_only_zeroes_face_in_every_split(white_images, tmp_path, split, training):
    batch = _first_batch(white_images, tmp_path, "background_only", split, training)

    corner = batch[:, CORNER[0], CORNER[1], :]
    centre = batch[:, CENTRE[0], CENTRE[1], :]

    assert np.all(centre < 0.05), (
        f"{split} (training={training}): face pixel not masked -- "
        "is the mask line indented inside `if training:` again?")
    assert np.all(corner > 0.5), f"{split}: background should survive background_only"


def test_mask_mode_none_leaves_image_untouched(white_images, tmp_path):
    batch = _first_batch(white_images, tmp_path, "none", "val", False)
    assert np.all(batch > 0.5)


def test_bad_mask_mode_is_rejected():
    with pytest.raises(ValueError):
        Config(name="x", real_dir="a", fake_dir="b", mask_mode="face-only")
