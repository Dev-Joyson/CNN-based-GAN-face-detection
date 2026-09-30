"""The demonstration page: one local HTML page that lays out the research in the
order of the talk and runs the headline model live on the presenter's laptop.

    python demo/app.py                      # then open http://localhost:8000
    python demo/app.py --config configs/test27_stride2_halfwidth.yaml --model <model.keras> --port 8000
    python demo/app.py --compare-model <mobilenet model.keras>     # second model beside ours in the table

Why this file: the guidelines score a "presentation and demonstration". The
slides carry the argument; this page is the instrument. For one image it shows
everything the model did: the window it was cut (pipeline drawn on the original),
the downscaled alternative and both predictions, the high-frequency residual and
spectrum of each view, first- and last-layer feature maps, Grad-CAM, JPEG and
window-position probes, latency, and the architecture table. Same preprocessing
as training (predict.preprocess -> load_and_resize -> eval_view); nothing is
re-implemented. No internet; standard library only (no Flask).

--compare-model: one pretrained baseline (MobileNetV3-Small by default) scored
and timed on the identical native window, shown as two extra columns in the
results table. The thesis claim is a resource comparison, so the demo shows one
live. Latency on the laptop is for feel only; the page says so, and the
measured numbers are the one-session table in README.

Routes: /                                   the page
        /model.json                         layer table (from the loaded model)
        /samples.json, /samples/<class>/<f> bundled 1024^2 sample images
        POST /predict                       image bytes -> JSON, everything above
"""

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import tempfile
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from PIL import Image

from evaluate import last_spatial_conv, overlay
from model import build_model, eval_view, feathered_ellipse, load_and_resize, load_config
from predict import preprocess
from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "test27_stride2_halfwidth.yaml")
DEFAULT_MODEL = os.path.expanduser("~/Downloads/thesis-figures/test27_stride2_halfwidth/model.keras")
DEFAULT_COMPARE = os.path.expanduser("~/Downloads/mobile_net_pretrained.keras")

class Engine:
    def __init__(self, model_path, config_path, tflite_path=None, compare_path=None, compare_name=None):
        native = load_config(config_path)
        from dataclasses import replace
        self.cfg = {"native": native,
                    "downscaled": replace(native, input_mode="resize", scale_aug=None)}
        self.total_params = None
        try:
            self.model = tf.keras.models.load_model(model_path, safe_mode=False)
        except Exception as e:
            print(f"load_model failed ({type(e).__name__}); rebuilding from config and loading weights only")
            self.model = build_model(self.cfg["native"])
            self.model.load_weights(model_path)
        self.layer = last_spatial_conv(self.model)
        self.feat = tf.keras.Model(self.model.inputs, [self.model.get_layer("spatial_conv_1").output,
                                                       self.model.get_layer(self.layer).output])
        self.table = [{"layer": l.name, "type": type(l).__name__,
                       "shape": "×".join(str(d) for d in l.output.shape[1:]), "params": int(l.count_params())}
                      for l in self.model.layers]
        self.total_params = int(self.model.count_params())
        c = self.cfg["native"]
        self.mask = feathered_ellipse(c.img_size, c.mask_rx, c.mask_ry, c.mask_feather)
        self.infer = tf.function(lambda x: self.model(x, training=False))
        self.tflite = None
        if tflite_path and os.path.exists(tflite_path):
            self.tflite = tf.lite.Interpreter(model_path=tflite_path, num_threads=os.cpu_count())
            self.tflite.allocate_tensors()
            self.t_in = self.tflite.get_input_details()[0]["index"]
            self.t_out = self.tflite.get_output_details()[0]["index"]
        x = tf.zeros([1, c.img_size, c.img_size, 3])
        for _ in range(20):
            self.infer(x).numpy()
        if self.tflite:
            for _ in range(20):
                self._tflite(x.numpy())
        self.compare = None
        if compare_path and os.path.exists(compare_path):
            try:
                cm = tf.keras.models.load_model(compare_path, safe_mode=False)
            except Exception as e:
                print(f"compare load_model failed ({type(e).__name__}); rebuilding and loading weights only")
                from baselines import build_baseline
                cm = build_baseline("mobilenet_v3_small", c, pretrained=False)
                cm.load_weights(compare_path)
            infer = tf.function(lambda x: cm(x, training=False))
            for _ in range(20):
                infer(x).numpy()
            self.compare = {"name": compare_name or cm.name, "model": cm, "infer": infer,
                            "params": int(cm.count_params())}
        elif compare_path:
            print(f"compare model not found at {compare_path}; running ours only")

    def _tflite(self, x):
        self.tflite.set_tensor(self.t_in, x.astype(np.float32)); self.tflite.invoke()
        return float(self.tflite.get_tensor(self.t_out).ravel()[0])

    @staticmethod
    def _png(arr01):
        buf = io.BytesIO()
        Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _p(self, x):
        return float(self.infer(x).numpy()[0, 0])

    def _gradcam(self, x, for_fake):
        """Grad-CAM for the model's VERDICT on the LOGIT (pre-sigmoid): evidence for
        'fake' when it says fake, for 'real' (negated logit) when it says real.
        evaluate.py's version differentiates the sigmoid output for 'fake' only:
        flat on a confident real (no positive evidence) AND flat on a confident
        fake (sigmoid saturated at 0.9999, gradient ~0). The logit has neither problem."""
        dense = self.model.layers[-1]
        grad_model = tf.keras.models.Model(self.model.inputs,
                                           [self.model.get_layer(self.layer).output, dense.input])
        with tf.GradientTape() as tape:
            conv_out, h = grad_model(x)
            logit = tf.matmul(h, dense.kernel) + dense.bias
            target = logit[:, 0] if for_fake else -logit[:, 0]
        grads = tape.gradient(target, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heat = tf.maximum(heat, 0)
        return (heat / (tf.reduce_max(heat) + 1e-8)).numpy()

    @staticmethod
    def _gray(x):
        return tf.image.rgb_to_grayscale(x)[0, ..., 0].numpy()

    def _residual(self, x):
        """Image minus a 5x5 box blur, amplified: the high-frequency content the
        crop keeps and the resize deletes. Illustration only; the model reads pixels."""
        blur = tf.nn.avg_pool2d(x, ksize=5, strides=1, padding="SAME")
        r = (x - blur)[0].numpy()
        return self._png(np.clip(r * 4 + 0.5, 0, 1)), round(float(np.abs(r).mean()) * 100, 2)

    def _spectrum(self, x):
        g = self._gray(x)
        f = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())))
        f = np.log1p(f); f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        return self._png(np.repeat(f[..., None], 3, -1))

    def _feature_strip(self, fm, n=8):
        """First n channels of a feature map, each min-max normalised, tiled."""
        fm = fm[0].numpy(); tiles = []
        for c in range(min(n, fm.shape[-1])):
            t = fm[..., c]; t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            tiles.append(np.repeat(t[..., None], 3, -1))
        strip = np.concatenate(tiles, axis=1)
        return self._png(strip), list(fm.shape[:2])

    def predict(self, image_bytes):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes); path = f.name
        try:
            with Image.open(path) as im:
                w, h = im.size
                thumb = im.convert("RGB").copy(); thumb.thumbnail((512, 512))
            xn = preprocess(path, self.cfg["native"], self.mask)
            xd = preprocess(path, self.cfg["downscaled"], self.mask)
            region, _ = load_and_resize(tf.constant(path), tf.constant(0), self.cfg["native"])
        finally:
            os.unlink(path)

        sx = thumb.width / w
        d = ImageDraw.Draw(thumb)
        c = self.cfg["native"]
        for size, col in ((c.cache_size, "#f5a623"), (c.img_size, "#1f4e79")):
            x0 = (w - size) / 2 * sx; y0 = (h - size) / 2 * sx
            d.rectangle([x0, y0, x0 + size * sx, y0 + size * sx], outline=col, width=3)
        buf = io.BytesIO(); thumb.save(buf, format="JPEG", quality=90)
        original = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        times = []
        for _ in range(20):
            t0 = time.perf_counter(); p = self._p(xn); times.append((time.perf_counter() - t0) * 1e3)
        p_down = self._p(xd)

        heat = self._gradcam(xn, for_fake=p > 0.5)
        f1, f5 = self.feat(xn)
        strip1, shape1 = self._feature_strip(f1); strip5, shape5 = self._feature_strip(f5)
        res_n, hf_n = self._residual(xn); res_d, hf_d = self._residual(xd)

        u8 = tf.cast(tf.clip_by_value(xn[0], 0, 1) * 255, tf.uint8)
        jpeg = {}
        for q in (95, 75):
            xq = tf.cast(tf.image.adjust_jpeg_quality(u8, q), tf.float32)[None] / 255.0
            jpeg[str(q)] = round(self._p(xq), 4)
        S, s_ = int(region.shape[0]), c.img_size
        pos = {"centre": ((S - s_) // 2, (S - s_) // 2), "top-left": (0, 0), "top-right": (0, S - s_),
               "bottom-left": (S - s_, 0), "bottom-right": (S - s_, S - s_)}
        windows = {}
        for name, (r0, c0) in pos.items():
            xw = tf.cast(region[r0:r0 + s_, c0:c0 + s_], tf.float32)[None] / 255.0
            windows[name] = round(self._p(xw), 4)

        out = {"input_size": [w, h], "original": original,
               "native": {"p_fake": round(p, 4), "window": self._png(xn[0].numpy()),
                          "gradcam": self._png(overlay(xn[0].numpy(), heat)),
                          "gradcam_for": "fake" if p > 0.5 else "real",
                          "residual": res_n, "highfreq": hf_n, "spectrum": self._spectrum(xn)},
               "downscaled": {"p_fake": round(p_down, 4), "window": self._png(xd[0].numpy()),
                              "residual": res_d, "highfreq": hf_d, "spectrum": self._spectrum(xd)},
               "features": {"conv1": strip1, "conv1_shape": shape1, "conv5": strip5, "conv5_shape": shape5,
                            "last_layer": self.layer},
               "probes": {"jpeg": jpeg, "windows": windows},
               "latency_ms": round(float(np.median(times)), 2)}
        if self.tflite:
            tt = []
            for _ in range(20):
                t0 = time.perf_counter(); q = self._tflite(xn.numpy()); tt.append((time.perf_counter() - t0) * 1e3)
            out["int8"] = {"p_fake": round(q, 4), "latency_ms": round(float(np.median(tt)), 2)}
        if self.compare:
            tc = []
            for _ in range(20):
                t0 = time.perf_counter(); pc = float(self.compare["infer"](xn).numpy().ravel()[0])
                tc.append((time.perf_counter() - t0) * 1e3)
            out["compare"] = {"name": self.compare["name"], "p_fake": round(pc, 4),
                              "latency_ms": round(float(np.median(tc)), 2)}
        return out

def make_handler(engine, samples_dir):
    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _file(self, path):
            if not os.path.isfile(path):
                return self._send("not found", "text/plain", 404)
            with open(path, "rb") as f:
                self._send(f.read(), mimetypes.guess_type(path)[0] or "application/octet-stream")

        def do_GET(self):
            u = urlparse(self.path); p = u.path
            if p in ("/", "/index.html"):
                return self._file(os.path.join(HERE, "index.html"))
            if p == "/model.json":
                return self._send(json.dumps({"layers": engine.table, "total_params": engine.total_params,
                                              "last_layer": engine.layer,
                                              "compare": ({"name": engine.compare["name"],
                                                           "params": engine.compare["params"]}
                                                          if engine.compare else None)}))
            if p == "/samples.json":
                items = []
                for cls in ("real", "fake"):
                    d = os.path.join(samples_dir, cls)
                    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            items.append({"cls": cls, "file": f, "url": f"/samples/{cls}/{f}"})
                return self._send(json.dumps(items))
            if p.startswith("/samples/"):
                parts = p.split("/")
                if len(parts) == 4 and parts[2] in ("real", "fake"):
                    return self._file(os.path.join(samples_dir, parts[2], os.path.basename(parts[3])))
            return self._send("not found", "text/plain", 404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/predict":
                return self._send("not found", "text/plain", 404)
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                self._send(json.dumps(engine.predict(body)))
            except Exception as e:
                self._send(json.dumps({"error": str(e)}), code=500)
    return H

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="the model's config (architecture knobs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tflite", default=os.path.join(os.path.dirname(DEFAULT_MODEL), "model_int8.tflite"),
                    help="optional int8 .tflite of the same model; shown beside fp32 if present")
    ap.add_argument("--compare-model", default=DEFAULT_COMPARE,
                    help="a baseline .keras scored and timed beside ours; '' to disable")
    ap.add_argument("--compare-name", default="MobileNetV3-Small")
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"loading {args.model} with {os.path.basename(args.config)} ...", flush=True)
    engine = Engine(args.model, args.config, args.tflite, args.compare_model or None, args.compare_name)
    print(f"model ready ({'fp32 + int8' if engine.tflite else 'fp32'})"
          + (f"; comparing against {engine.compare['name']} ({engine.compare['params']:,} params)" if engine.compare else "")
          + f"; samples from {args.samples}")
    print(f"open  http://localhost:{args.port}   (Ctrl+C to stop)", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(engine, args.samples)).serve_forever()

if __name__ == "__main__":
    main()
