"""Dual-branch (spatial + FFT) CNN for StyleGAN2-vs-FFHQ face detection.

Single source of truth: config, data pipeline, FFT layer, model, training loop.
Everything that varies between experiments lives in configs/*.yaml.

Refactored from Test13_FFT_More_dataset.ipynb.
"""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from glob import glob

import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

MASK_MODES = ("face_only", "background_only", "none")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    name: str
    real_dir: str
    fake_dir: str

    # experiment knobs
    mask_mode: str = "face_only"          # face_only | background_only | none
    limit_per_class: int = 20000

    # data / model geometry
    img_size: int = 256
    native_size: int = 512                # normalize native resolution first
    batch_size: int = 144

    # training
    epochs: int = 500
    lr: float = 3e-4
    patience: int = 6
    seed: int = 42

    # mask geometry (feathered ellipse over an aligned FFHQ/StyleGAN face)
    mask_rx: float = 0.38
    mask_ry: float = 0.48
    mask_feather: float = 0.05

    # io
    cache_dir: str = "/content/cache"     # keyed by limit_per_class, NOT mask_mode
    out_dir: str = "experiments"

    def __post_init__(self):
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {self.mask_mode!r}")

    @property
    def run_dir(self):
        return os.path.join(self.out_dir, self.name)

    def cache_path(self, split):
        """Cache is shared across mask modes: switching mask does NOT invalidate it."""
        return os.path.join(self.cache_dir, f"n{self.limit_per_class}", split)


def load_config(path):
    with open(path) as f:
        return Config(**yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# Mask
# --------------------------------------------------------------------------- #

def feathered_ellipse(size, rx_frac=0.38, ry_frac=0.48, feather=0.05):
    """(size, size, 1) soft mask: 1 inside the face ellipse -> 0 outside."""
    yy, xx = tf.meshgrid(tf.range(size, dtype=tf.float32),
                         tf.range(size, dtype=tf.float32), indexing='ij')
    c = (size - 1) / 2.0
    r = tf.sqrt(((xx - c) / (rx_frac * size)) ** 2 + ((yy - c) / (ry_frac * size)) ** 2)
    m = tf.clip_by_value((1.0 - r) / feather + 0.5, 0.0, 1.0)
    return m[..., tf.newaxis]


def apply_mask(image, face_mask, mask_mode):
    if mask_mode == "face_only":
        return image * face_mask
    if mask_mode == "background_only":
        return image * (1.0 - face_mask)
    return image


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #

def load_paths(cfg):
    """Stratified 70/15/15 split over real(0) / fake(1)."""
    real_images = sorted(glob(os.path.join(cfg.real_dir, '*')))[:cfg.limit_per_class]
    fake_images = sorted(glob(os.path.join(cfg.fake_dir, '*')))[:cfg.limit_per_class]
    if not real_images or not fake_images:
        raise FileNotFoundError(
            f"No images found. real_dir={cfg.real_dir!r} ({len(real_images)}), "
            f"fake_dir={cfg.fake_dir!r} ({len(fake_images)})")

    paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=cfg.seed)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=0.5, stratify=rest_labels,
        random_state=cfg.seed)

    return {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }


def random_jpeg(image):
    quality = tf.random.uniform([], 60, 100, dtype=tf.int32)
    image_uint8 = tf.cast(image * 255.0, tf.uint8)
    image_uint8 = tf.image.adjust_jpeg_quality(image_uint8, quality)
    return tf.cast(image_uint8, tf.float32) / 255.0


def random_blur(image):
    return tf.cond(
        tf.random.uniform([]) < 0.3,
        lambda: tf.nn.avg_pool2d(image[None], ksize=3, strides=1, padding="SAME")[0],
        lambda: image,
    )


def random_crop_resize(image, img_size):
    crop_frac = tf.random.uniform([], 0.95, 1.0)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    crop_h = tf.cast(crop_frac * tf.cast(h, tf.float32), tf.int32)
    crop_w = tf.cast(crop_frac * tf.cast(w, tf.float32), tf.int32)

    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    return tf.image.resize(image, [img_size, img_size],
                           method=tf.image.ResizeMethod.BICUBIC)


def load_and_resize(path, label, cfg):
    """The slow, deterministic part -- this is what gets cached."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Normalize native resolution (fix resolution bias), then resize to model input
    image = tf.image.resize(image, [cfg.native_size, cfg.native_size],
                            method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.resize(image, [cfg.img_size, cfg.img_size],
                            method=tf.image.ResizeMethod.BICUBIC)

    # clip THEN cast to uint8 -> cache is 1/4 the size (bicubic overshoots 0..255)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    return image, label


def augment_and_mask(image, label, cfg, face_mask, training):
    image = tf.cast(image, tf.float32) / 255.0     # float conversion AFTER cache

    if training:
        image = random_crop_resize(image, cfg.img_size)
        image = random_blur(image)
        image = tf.cond(tf.random.uniform([]) < 0.3,
                        lambda: random_jpeg(image),
                        lambda: image)

    # NOTE: outside the `if training` block on purpose. The mask must hit
    # train + val + test identically -- indenting this line inside `if training:`
    # was a real bug (see tests/test_mask_applied.py). Do not move it.
    image = apply_mask(image, face_mask, cfg.mask_mode)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_dataset(paths, labels, cfg, split, shuffle, training, face_mask=None):
    if face_mask is None:
        face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry,
                                      cfg.mask_feather)

    cache_path = cfg.cache_path(split)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: load_and_resize(p, y, cfg),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache(cache_path)                      # cache AFTER resize, BEFORE augment
    if shuffle:
        ds = ds.shuffle(1000)                      # shuffle AFTER cache
    ds = ds.map(lambda x, y: augment_and_mask(x, y, cfg, face_mask, training),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(cfg):
    splits = load_paths(cfg)
    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry,
                                  cfg.mask_feather)
    for name, (_, labels) in splits.items():
        print(f"{name:5s}: {dict(Counter(labels))}")

    return {
        name: build_dataset(paths, labels, cfg, name,
                            shuffle=(name == "train"),
                            training=(name == "train"),
                            face_mask=face_mask)
        for name, (paths, labels) in splits.items()
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@tf.keras.utils.register_keras_serializable(package="ganfd")
def fft_layer(x):
    # x: (batch, H, W, 3), float in [0,1]

    # 1) Collapse color so the inner two axes become (H, W) -- the real image grid
    g = tf.image.rgb_to_grayscale(x)          # (B, H, W, 1)
    g = tf.squeeze(g, axis=-1)                # (B, H, W)  <- last two are row, column

    # 2) 2D FFT over (H, W), now the correct axes
    fft = tf.signal.fft2d(tf.cast(g, tf.complex64))
    mag = tf.abs(fft)

    # 3) Center the zero-frequency
    mag = tf.signal.fftshift(mag, axes=[1, 2])

    # 4) Log-compress the huge dynamic range (once, not twice)
    mag = tf.math.log1p(mag)

    # 5) Per-sample normalize to [0,1] so scale is consistent across images
    mn = tf.reduce_min(mag, axis=[1, 2], keepdims=True)
    mx = tf.reduce_max(mag, axis=[1, 2], keepdims=True)
    mag = (mag - mn) / (mx - mn + 1e-6)

    return mag[..., tf.newaxis]                # (B, H, W, 1)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return layers.MaxPooling2D()(x)


def build_model(cfg):
    input_img = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))

    # Spatial branch
    x = conv_block(input_img, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)

    # FFT branch
    f = layers.Lambda(fft_layer, name="fft")(input_img)
    f = layers.Conv2D(16, 3, activation='relu', padding='same')(f)
    f = layers.MaxPooling2D()(f)
    f = layers.Conv2D(32, 3, activation='relu', padding='same')(f)
    f = layers.GlobalAveragePooling2D()(f)

    # Fusion
    combined = layers.Concatenate()([x, f])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=input_img, outputs=output)


def compile_model(model, cfg):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(cfg):
    tf.keras.utils.set_random_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    print(f"=== {cfg.name} | mask_mode={cfg.mask_mode} | "
          f"limit_per_class={cfg.limit_per_class} ===")
    print(f"cache: {cfg.cache_path('<split>')}  out: {cfg.run_dir}")

    ds = build_datasets(cfg)
    model = compile_model(build_model(cfg), cfg)
    model.summary()

    ckpt = os.path.join(cfg.run_dir, "model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_auc', mode='max',
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                        patience=cfg.patience,
                                        restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.run_dir, "history.csv")),
    ]

    model.fit(ds["train"], validation_data=ds["val"], epochs=cfg.epochs,
              callbacks=callbacks)

    metrics = {"config": asdict(cfg), "params": int(model.count_params())}
    for split in ("val", "test"):
        # return_dict keeps metric names ('accuracy', 'auc') -- Keras 3's
        # model.metrics_names collapses them to 'compile_metrics'.
        scores = model.evaluate(ds[split], verbose=0, return_dict=True)
        metrics[split] = {k: float(v) for k, v in scores.items()}
        print(f"{split}: {metrics[split]}")

    with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics
