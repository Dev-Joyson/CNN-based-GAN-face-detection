"""Guard: the FFT branch must transform over (height, width), not the channel axis.

The bug this catches: tf.signal.fft2d always acts on the LAST TWO axes. Handing it
an (B, H, W, 3) tensor makes it transform (W, 3) -- a 3-point transform over colour.
model.fft_layer squeezes to (B, H, W) first, so this test compares its output against
numpy's fft2 of the grayscale image and also uses a non-square input, which alone
would change the output shape if the axes were wrong.
"""

import numpy as np
import tensorflow as tf

from model import fft_layer

# H != W so a wrong-axis FFT can't even produce the right shape
H, W = 8, 16
RGB_TO_GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def _reference_spectrum(img):
    """What the FFT branch should produce, computed with numpy."""
    gray = img @ RGB_TO_GRAY                        # (B, H, W)
    mag = np.abs(np.fft.fft2(gray, axes=(1, 2)))    # over H, W
    mag = np.fft.fftshift(mag, axes=(1, 2))
    mag = np.log1p(mag)
    mn = mag.min(axis=(1, 2), keepdims=True)
    mx = mag.max(axis=(1, 2), keepdims=True)
    return (mag - mn) / (mx - mn + 1e-6)


def test_output_shape_keeps_spatial_grid():
    x = tf.random.uniform((2, H, W, 3))
    out = fft_layer(x)
    assert out.shape == (2, H, W, 1), (
        f"expected (2, {H}, {W}, 1); a channel-axis FFT would not keep the H,W grid")


def test_matches_numpy_fft_over_height_width():
    rng = np.random.default_rng(0)
    img = rng.random((2, H, W, 3), dtype=np.float32)

    got = fft_layer(tf.constant(img)).numpy()[..., 0]
    want = _reference_spectrum(img)

    np.testing.assert_allclose(got, want, atol=1e-4)


def test_differs_from_channel_axis_fft():
    """Sanity check that the test above is actually discriminating."""
    rng = np.random.default_rng(1)
    img = rng.random((1, H, W, 3), dtype=np.float32)

    correct = fft_layer(tf.constant(img)).numpy()[..., 0]
    # the buggy version: fft2d straight on (B, H, W, 3) -> transforms (W, 3)
    wrong = np.abs(np.fft.fft2(img, axes=(2, 3)))[..., 0]

    assert correct.shape != wrong.shape or not np.allclose(correct, wrong)


def test_row_frequency_lands_on_the_vertical_axis():
    """A pattern varying only along H must put its energy on the H axis of the spectrum."""
    size = 32
    rows = np.arange(size, dtype=np.float32)
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * rows / size)   # varies with row only
    img = np.repeat(stripes[:, None], size, axis=1)             # (H, W)
    img = np.repeat(img[..., None], 3, axis=-1)[None]           # (1, H, W, 3)

    spec = fft_layer(tf.constant(img.astype(np.float32))).numpy()[0, ..., 0]
    c = size // 2
    # energy sits in the centre column (varying row-frequency), not the centre row
    assert spec[:, c].sum() > spec[c, :].sum()
