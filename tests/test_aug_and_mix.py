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


def test_scale_window_shapes_and_validation(tmp_path):
    from model import random_scale_window
    img = tf.cast(tf.random.uniform([512, 512, 3]) * 255, tf.uint8)
    for _ in range(6):
        out = random_scale_window(img, 256, [1, 2])
        assert out.shape == (256, 256, 3) and out.dtype == tf.uint8
    with pytest.raises(ValueError):
        _cfg(tmp_path, fake_dir="/f", scale_aug=[1, 4])          # 256*4 > cache 512
    with pytest.raises(ValueError):
        Config(name="u", real_dir="-", fake_dir="/f", input_mode="resize", mask_mode="none", scale_aug=[1, 2])


def test_online_kd_trains_student_only():
    from distill import OnlineKD
    tf.random.set_seed(0)
    student = tf.keras.Sequential([tf.keras.layers.Input((8, 8, 3)), tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(1, activation="sigmoid")])
    teacher = tf.keras.Sequential([tf.keras.layers.Input((8, 8, 3)), tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(1, activation="sigmoid")])
    t_before = [w.numpy().copy() for w in teacher.weights]
    s_before = [w.numpy().copy() for w in student.weights]
    kd = OnlineKD(student, [teacher], temperature=2.0, alpha=0.7)
    kd.compile(optimizer=tf.keras.optimizers.Adam(1e-2))
    x = tf.random.uniform([16, 8, 8, 3]); y = tf.cast(tf.random.uniform([16]) > 0.5, tf.int32)
    hist = kd.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8), epochs=2, verbose=0)
    assert all(np.array_equal(a, b.numpy()) for a, b in zip(t_before, teacher.weights)), "teacher moved"
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(s_before, student.weights)), "student did not move"
    assert "auc" in hist.history and "loss" in hist.history
