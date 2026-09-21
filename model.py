"""Dual-branch (spatial + FFT) CNN for StyleGAN2-vs-FFHQ face detection.

Single source of truth: config, data pipeline, FFT layer, model, training loop.
Everything that varies between experiments lives in configs/*.yaml.

Pipeline follows Test16_FFT_fix_final.ipynb (the "fixed" version): decode_image
so PNG fakes load, crop_frac 0.85, seeded full-set shuffle, named spatial convs.
"""

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
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
    native_size: int = 512                # resize mode: intermediate resize before img_size
    batch_size: int = 144

    # input_mode -- how a 256^2 input is made from a 1024^2 image:
    #   resize: 1024 -> native_size -> img_size, bicubic. Two low-passes; the
    #           GAN fingerprint above img_size's Nyquist is gone before the
    #           model sees a pixel (post-pipeline highfreq AUC 0.52 on SG2).
    #   crop:   NO resampling. Cache the centre cache_size^2 at native pixels;
    #           train on random img_size^2 windows of it, evaluate on the
    #           centre window. Same input size, same model, same latency --
    #           only where the pixels come from changes. cache_size trades
    #           coverage (hair, background) against cache size and per-epoch
    #           disk read: 512 -> 27 GB, 768 -> 62 GB, 1024 -> 157 GB.
    input_mode: str = "resize"
    cache_size: int = 512

    # training
    epochs: int = 500
    lr: float = 3e-4
    patience: int = 6
    seed: int = 42

    # architecture
    fft_branch: bool = True               # False = spatial branch only (the ablation)

    # augmentation (train split only)
    crop_frac_min: float = 0.85           # Test16: random crop keeps 85-100% of the side
    shuffle_buffer: int = 4096            # images held for shuffling; Test16 used the whole train set

    # mask geometry (feathered ellipse over an aligned FFHQ/StyleGAN face)
    mask_rx: float = 0.38
    mask_ry: float = 0.48
    mask_feather: float = 0.05

    # io
    cache_dir: str = "/content/cache"     # keyed by limit_per_class, NOT mask_mode
    out_dir: str = "experiments"

    # knowledge distillation (train.py dispatches to distill.py when set):
    #   headline: run whose baselines/<teacher>/model.keras are the teachers
    #             and whose split this config must reproduce (same data fields)
    #   teachers: list of baseline names, averaged if several
    #   temperature, alpha, extra (use the unsampled pool), init (warm-start
    #   from the headline's checkpoint), calibrate (temperature-scale the
    #   teacher on val first -- Guo et al. 2017 -- because a teacher at
    #   0.003/0.999 mean probability is saturated and its raw logits are a
    #   worse target than hard labels; seen on test18)
    distill: dict = field(default=None)

    def __post_init__(self):
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {self.mask_mode!r}")
        if self.input_mode not in ("resize", "crop"):
            raise ValueError(f"input_mode must be resize or crop, got {self.input_mode!r}")
        if self.input_mode == "crop" and self.mask_mode != "none":
            raise ValueError("crop mode: a 256^2 patch of a 1024^2 face is not a whole "
                             "aligned face, so the ellipse mask does not apply; mask_mode must be none")
        if self.input_mode == "crop" and self.cache_size < self.img_size:
            raise ValueError("crop mode: cache_size must be >= img_size")
        if self.distill is not None:
            d = {"temperature": 2.0, "alpha": 0.7, "extra": True, "init": False,
                 "calibrate": False, **self.distill}
            missing = {"headline", "teachers"} - set(d)
            if missing:
                raise ValueError(f"distill block needs {sorted(missing)}")
            unknown = set(d) - {"headline", "teachers", "temperature", "alpha", "extra", "init", "calibrate"}
            if unknown:
                raise ValueError(f"distill block has unknown keys {sorted(unknown)}")
            self.distill = d

    @property
    def run_dir(self):
        return os.path.join(self.out_dir, self.name)

    def cache_path(self, split):
        """Cache is shared across mask modes: switching mask does NOT invalidate it.

        Everything that changes the cached bytes IS in the key: which folders, how
        many, the two resize sizes, and the seed (which picks the split members).
        Without the seed, a 3-seed run would read seed-42's images from disk and
        label them seed-43 -- silently, with plausible numbers.
        """
        ident = f"{self.real_dir}|{self.fake_dir}".encode()
        tag = hashlib.sha1(ident).hexdigest()[:8]
        key = f"n{self.limit_per_class}_i{self.img_size}_n{self.native_size}_s{self.seed}_{tag}"
        if self.input_mode == "crop":
            key += f"_crop{self.cache_size}"          # a different cache: native pixels, no resize
        return os.path.join(self.cache_dir, key, split)


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

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(folder):
    """Image files in a folder, sorted. Anything else -- a readme, .DS_Store, a
    zip left from the download -- would reach decode_image mid-epoch and kill
    the run. (The FakeMix folder had 25,001 entries for 25,000 images.)"""
    return sorted(p for p in glob(os.path.join(folder, '*'))
                  if p.lower().endswith(IMAGE_EXTS))


def load_paths(cfg):
    """Seeded random sample of limit_per_class from each folder, then a
    stratified 70/15/15 split over real(0) / fake(1).

    A random sample, not the first N by filename: the fake folder mixes
    StyleGAN1 and StyleGAN2, and FFHQ is numbered, so a sorted prefix could
    silently be one generator or one slice of FFHQ. Seeded, so the same
    config always picks the same images.
    """
    def sample(folder):
        files = list_images(folder)
        if len(files) <= cfg.limit_per_class:
            return files
        return sorted(random.Random(cfg.seed).sample(files, cfg.limit_per_class))

    real_images = sample(cfg.real_dir)
    fake_images = sample(cfg.fake_dir)
    if not real_images or not fake_images:
        raise FileNotFoundError(
            f"No images found. real_dir={cfg.real_dir!r} ({len(real_images)}), "
            f"fake_dir={cfg.fake_dir!r} ({len(fake_images)})")
    print(f"real: {len(real_images)} of pool   fake: {len(fake_images)} of pool")

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
    # clip BEFORE the uint8 cast: the bicubic resize in random_crop_resize
    # overshoots past 1.0, and 1.07 * 255 = 273 wraps around to 17 -- bright
    # edge pixels turn into black speckles.
    image_uint8 = tf.cast(tf.clip_by_value(image, 0.0, 1.0) * 255.0, tf.uint8)
    image_uint8 = tf.image.adjust_jpeg_quality(image_uint8, quality)
    return tf.cast(image_uint8, tf.float32) / 255.0


def random_blur(image):
    return tf.cond(
        tf.random.uniform([]) < 0.3,
        lambda: tf.nn.avg_pool2d(image[None], ksize=3, strides=1, padding="SAME")[0],
        lambda: image,
    )


def random_crop_resize(image, img_size, crop_frac_min):
    crop_frac = tf.random.uniform([], crop_frac_min, 1.0)
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
    # decode_image handles JPEG and PNG (the fakes may be PNG); expand_animations
    # keeps a static (H, W, 3) rather than a (frames, H, W, 3) for GIF-like input
    image = tf.io.decode_image(image, channels=3, expand_animations=False)

    if cfg.input_mode == "crop":
        # centre cache_size^2 at native pixels. resize_with_crop_or_pad crops;
        # it never interpolates. (Pads only if an image is smaller -- none are.)
        image = tf.image.resize_with_crop_or_pad(image, cfg.cache_size, cfg.cache_size)
        return tf.cast(image, tf.uint8), label

    # Normalize native resolution (fix resolution bias), then resize to model input
    image = tf.image.resize(image, [cfg.native_size, cfg.native_size],
                            method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.resize(image, [cfg.img_size, cfg.img_size],
                            method=tf.image.ResizeMethod.BICUBIC)

    # clip THEN cast to uint8 -> cache is 1/4 the size (bicubic overshoots 0..255)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    return image, label


def eval_view(image, cfg):
    """The evaluation-time input: in crop mode the centre img_size^2 window of
    the cached region; in resize mode the cached image is already it. Shared
    by the val/test pipeline and predict.py so the demo cannot drift."""
    if cfg.input_mode == "crop":
        return tf.image.resize_with_crop_or_pad(image, cfg.img_size, cfg.img_size)
    return image


def augment_and_mask(image, label, cfg, face_mask, training):
    if cfg.input_mode == "crop":
        # a random window in training, the centre one otherwise. Native pixels,
        # no resampling anywhere. random_crop_resize is skipped: it resizes.
        image = (tf.image.random_crop(image, [cfg.img_size, cfg.img_size, 3])
                 if training else eval_view(image, cfg))
    image = tf.cast(image, tf.float32) / 255.0     # float conversion AFTER cache

    if training:
        if cfg.input_mode == "resize":
            image = random_crop_resize(image, cfg.img_size, cfg.crop_frac_min)
            image = tf.clip_by_value(image, 0.0, 1.0)   # bicubic overshoots again
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


def cached_dataset(paths, labels, cfg, split):
    """(uint8 image, label) in path order, cached on disk. The slow part,
    shared by build_dataset and by distill.py (which needs the same order to
    line teacher logits up with images)."""
    cache_path = cfg.cache_path(split)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: load_and_resize(p, y, cfg),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache(cache_path)                    # cache AFTER resize, BEFORE augment


def build_dataset(paths, labels, cfg, split, shuffle, training, face_mask=None):
    if face_mask is None:
        face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry,
                                      cfg.mask_feather)

    ds = cached_dataset(paths, labels, cfg, split)
    if shuffle:
        # AFTER cache: shuffling before it would freeze the first epoch's order
        # into the cache file and every later epoch would replay it.
        ds = ds.shuffle(min(cfg.shuffle_buffer, len(paths)), seed=cfg.seed,
                        reshuffle_each_iteration=True)
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


def conv_block(x, filters, name=None):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=name)(x)
    return layers.MaxPooling2D()(x)


def build_model(cfg):
    input_img = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))

    # Spatial branch -- fixed names so Grad-CAM can ask for "spatial_conv_5"
    x = conv_block(input_img, 32, name="spatial_conv_1")
    x = conv_block(x, 64, name="spatial_conv_2")
    x = conv_block(x, 128, name="spatial_conv_3")
    x = conv_block(x, 256, name="spatial_conv_4")
    x = conv_block(x, 256, name="spatial_conv_5")
    x = layers.GlobalAveragePooling2D()(x)

    if cfg.fft_branch:
        # FFT branch
        f = layers.Lambda(fft_layer, name="fft")(input_img)
        f = layers.Conv2D(16, 3, activation='relu', padding='same')(f)
        f = layers.MaxPooling2D()(f)
        f = layers.Conv2D(32, 3, activation='relu', padding='same')(f)
        f = layers.GlobalAveragePooling2D()(f)
        combined = layers.Concatenate()([x, f])
    else:
        # ablation: what does the spectrum buy? The branch is ~0.9% of the
        # params (4.8k of the convs + 4.1k of the fusion width), so any AUC it
        # buys is bought cheaply.
        combined = x

    # Fusion
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

def train(cfg, resume=False, build_fn=None):
    """Train from scratch, or -- with resume=True -- pick up a run the VM lost.

    Colab deletes VMs on idle and on a max lifetime; nothing inside the VM can
    stop that. What makes it survivable is that model.keras (best val_auc so
    far, optimizer state included) and history.csv are on Drive after every
    epoch. Resume reloads that checkpoint and continues the epoch count from
    history.csv, so a dead VM costs the epochs since the last improvement, not
    the run. Caveat, stated plainly: the checkpoint is the BEST epoch, not the
    last one, so a resume re-trains from the best weights at the last epoch
    number -- and EarlyStopping's patience counter starts fresh.
    """
    tf.keras.utils.set_random_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    print(f"=== {cfg.name} | mask_mode={cfg.mask_mode} | "
          f"limit_per_class={cfg.limit_per_class} ===")
    print(f"cache: {cfg.cache_path('<split>')}  out: {cfg.run_dir}")

    ds = build_datasets(cfg)

    ckpt = os.path.join(cfg.run_dir, "model.keras")
    history_csv = os.path.join(cfg.run_dir, "history.csv")
    initial_epoch = 0
    if resume and os.path.exists(ckpt):
        model = tf.keras.models.load_model(ckpt, safe_mode=False)   # compiled + optimizer state
        if os.path.exists(history_csv):
            with open(history_csv) as f:
                initial_epoch = max(sum(1 for _ in f) - 1, 0)      # rows minus header
        print(f"RESUME: loaded {ckpt}, continuing from epoch {initial_epoch}")
    else:
        if resume:
            print(f"resume requested but no checkpoint at {ckpt} -- starting fresh")
        model = compile_model((build_fn or build_model)(cfg), cfg)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_auc', mode='max',
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                        patience=cfg.patience,
                                        restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(history_csv, append=initial_epoch > 0),
        # live curves: %tensorboard --logdir <out_dir> in a notebook cell
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.run_dir, "tb"),
                                       write_graph=False),
    ]

    model.fit(ds["train"], validation_data=ds["val"], epochs=cfg.epochs,
              initial_epoch=initial_epoch, callbacks=callbacks)

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
