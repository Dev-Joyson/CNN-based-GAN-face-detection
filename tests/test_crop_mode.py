"""Guard: crop mode must never resample.

The whole point of input_mode=crop is that the model sees native pixels. A
resize sneaking back in -- via native_size, random_crop_resize, or a
resize_with_crop_or_pad on a wrong size -- would silently turn the experiment
back into the resize pipeline. So: the cached image must be an exact slice of
the decoded file, and the evaluation view an exact slice of that.
"""

import numpy as np
import tensorflow as tf

from model import Config, eval_view, load_and_resize

SRC, CACHE, IMG = 64, 48, 32


def _png(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (SRC, SRC, 3), dtype=np.uint8)   # PNG: lossless, so exact
    p = tmp_path / "x.png"
    tf.io.write_file(str(p), tf.io.encode_png(img))
    return str(p), img


def test_cached_image_is_an_exact_centre_slice(tmp_path):
    path, img = _png(tmp_path)
    cfg = Config(name="u", real_dir="-", fake_dir="-", input_mode="crop",
                 cache_size=CACHE, img_size=IMG, native_size=SRC)
    cached, _ = load_and_resize(tf.constant(path), tf.constant(0), cfg)
    off = (SRC - CACHE) // 2
    np.testing.assert_array_equal(cached.numpy(), img[off:off + CACHE, off:off + CACHE])


def test_eval_view_is_an_exact_centre_slice_of_the_cache(tmp_path):
    path, img = _png(tmp_path)
    cfg = Config(name="u", real_dir="-", fake_dir="-", input_mode="crop",
                 cache_size=CACHE, img_size=IMG, native_size=SRC)
    cached, _ = load_and_resize(tf.constant(path), tf.constant(0), cfg)
    view = eval_view(cached, cfg).numpy()
    off = (SRC - IMG) // 2
    assert view.shape == (IMG, IMG, 3)
    np.testing.assert_array_equal(view, img[off:off + IMG, off:off + IMG])


def test_crop_mode_refuses_a_mask():
    import pytest
    with pytest.raises(ValueError):
        Config(name="u", real_dir="-", fake_dir="-", input_mode="crop", mask_mode="face_only")
