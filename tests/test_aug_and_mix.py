"""Guards for the two generalisation knobs: the `wang` augmentation recipe and
multi-folder fakes with a stated mix. Both are one wrong line away from silently
training on the wrong thing (a blur that is not a blur; a 60/40 mix that is 50/50)."""
import os

import numpy as np
import pytest
import tensorflow as tf

from model import Config, load_paths, random_gaussian_blur


def _cfg(tmp_path, **kw):
    return Config(name="u", real_dir=str(tmp_path / "real"), input_mode="crop", mask_mode="none",
                  cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "out"), **kw)


def _folder(tmp_path, name, n):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"{i:05d}.png").write_bytes(b"")
    return str(d)


def test_gaussian_blur_keeps_shape_and_mean():
    tf.random.set_seed(0)
    img = tf.random.uniform([64, 64, 3])
    out = random_gaussian_blur(img)
    assert out.shape == img.shape
    # a normalised kernel preserves the mean (up to SAME-padding edge effects)
    assert abs(float(tf.reduce_mean(out)) - float(tf.reduce_mean(img))) < 0.02


def test_gaussian_blur_actually_blurs():
    img = tf.cast(tf.random.uniform([64, 64, 3]) > 0.5, tf.float32)   # hard edges
    # force a wide sigma by sampling until we get one (sigma is internal; use variance drop)
    outs = [random_gaussian_blur(img) for _ in range(8)]
    assert min(float(tf.math.reduce_std(o)) for o in outs) < float(tf.math.reduce_std(img)) * 0.7


def test_fake_mix_counts_are_exact(tmp_path):
    real = _folder(tmp_path, "real", 40)
    a = _folder(tmp_path, "sg2", 30)
    b = _folder(tmp_path, "sg1", 12)
    cfg = _cfg(tmp_path, fake_dir=[a, b], fake_mix=[18, 12], limit_per_class=30)
    cfg.real_dir = real
    splits = load_paths(cfg)
    fakes = [p for s in splits.values() for p, y in zip(*s) if y == 1]
    assert sum(p.startswith(a) for p in fakes) == 18
    assert sum(p.startswith(b) for p in fakes) == 12


def test_fake_mix_must_sum_to_limit(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, fake_dir=["x", "y"], fake_mix=[10, 10], limit_per_class=25)


def test_single_folder_cache_key_unchanged(tmp_path):
    # existing caches must stay valid: a plain string fake_dir hashes as before
    c1 = _cfg(tmp_path, fake_dir="/f")
    key = os.path.basename(os.path.dirname(c1.cache_path("train")))
    assert key.endswith("_crop512") and "mix" not in key


def test_unknown_aug_rejected(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, fake_dir="/f", aug="strong")
