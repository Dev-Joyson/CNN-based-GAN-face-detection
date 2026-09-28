#!/usr/bin/env python3
"""The demonstration page: one local HTML page that lays out the research in the
order of the talk and runs the headline model live on the presenter's laptop.

    python demo/app.py                      # then open http://localhost:8000
    python demo/app.py --model <model.keras> --figures <dir> --port 8000

Why this file: the guidelines score a "presentation and demonstration". A page
served from the laptop needs no internet, uses the same preprocessing as
training (predict.preprocess -> load_and_resize -> eval_view, nothing re-
implemented), and reads every number from demo/data.json, which is copied from
the README tables with its source named. Standard library only (no Flask).

Routes: /            the page
        /data.json   the numbers
        /figures/<f> the rendered figures (from --figures)
        /samples.json, /samples/<class>/<f>   bundled 1024^2 sample images
        POST /predict?mode=native|downscaled  image bytes -> JSON (p_fake,
             latency, the 256^2 window, Grad-CAM), optionally int8 via --tflite
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
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf                                   # noqa: E402
from PIL import Image                                     # noqa: E402

from evaluate import compute_gradcam, last_spatial_conv, overlay   # noqa: E402
from model import build_model, feathered_ellipse, load_config   # noqa: E402
from predict import preprocess                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.expanduser("~/Downloads/thesis-figures/test19_sg2_crop_no_fft/model.keras")
DEFAULT_FIGURES = os.path.expanduser("~/Downloads/thesis-figures/figures")


class Engine:
    def __init__(self, model_path, tflite_path=None):
        self.cfg = {"native": load_config(os.path.join(ROOT, "configs", "test19_sg2_crop_no_fft.yaml")),
                    "downscaled": load_config(os.path.join(ROOT, "configs", "probe_test19_no_fft_on_resize.yaml"))}
        try:
            self.model = tf.keras.models.load_model(model_path, safe_mode=False)
        except Exception as e:
            # the checkpoint was written by a newer Keras than this laptop has
            # (e.g. a Dense config key it does not know). The architecture is
            # the config's; only the weights are needed from the file.
            print(f"load_model failed ({type(e).__name__}); rebuilding from config and loading weights only")
            self.model = build_model(self.cfg["native"])
            self.model.load_weights(model_path)
        self.layer = last_spatial_conv(self.model)
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

    def _tflite(self, x):
        self.tflite.set_tensor(self.t_in, x.astype(np.float32)); self.tflite.invoke()
        return float(self.tflite.get_tensor(self.t_out).ravel()[0])

    @staticmethod
    def _png(arr01):
        buf = io.BytesIO()
        Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def predict(self, image_bytes, mode):
        cfg = self.cfg[mode]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes); path = f.name
        try:
            with Image.open(path) as im:
                w, h = im.size
            x = preprocess(path, cfg, self.mask)                    # the training pipeline, unchanged
        finally:
            os.unlink(path)
        times = []
        for _ in range(20):
            t0 = time.perf_counter(); p = float(self.infer(x).numpy()[0, 0]); times.append((time.perf_counter() - t0) * 1e3)
        out = {"mode": mode, "input_size": [w, h], "p_fake": round(p, 4),
               "latency_ms": round(float(np.median(times)), 2),
               "window": self._png(x[0].numpy())}
        heat = compute_gradcam(self.model, x, self.layer)
        out["gradcam"] = self._png(overlay(x[0].numpy(), heat))
        if self.tflite:
            tt = []
            for _ in range(20):
                t0 = time.perf_counter(); q = self._tflite(x.numpy()); tt.append((time.perf_counter() - t0) * 1e3)
            out["int8"] = {"p_fake": round(q, 4), "latency_ms": round(float(np.median(tt)), 2)}
        return out


def make_handler(engine, figures_dir, samples_dir):
    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):            # quiet
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
            if p == "/data.json":
                return self._file(os.path.join(HERE, "data.json"))
            if p == "/samples.json":
                items = []
                for cls in ("real", "fake"):
                    d = os.path.join(samples_dir, cls)
                    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            items.append({"cls": cls, "file": f, "url": f"/samples/{cls}/{f}"})
                return self._send(json.dumps(items))
            if p.startswith("/figures/"):
                return self._file(os.path.join(figures_dir, os.path.basename(p)))
            if p.startswith("/samples/"):
                parts = p.split("/")
                if len(parts) == 4 and parts[2] in ("real", "fake"):
                    return self._file(os.path.join(samples_dir, parts[2], os.path.basename(parts[3])))
            return self._send("not found", "text/plain", 404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/predict":
                return self._send("not found", "text/plain", 404)
            mode = parse_qs(u.query).get("mode", ["native"])[0]
            if mode not in ("native", "downscaled"):
                return self._send('{"error":"mode"}', code=400)
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                self._send(json.dumps(engine.predict(body, mode)))
            except Exception as e:                       # a bad upload must not kill the demo
                self._send(json.dumps({"error": str(e)}), code=500)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tflite", default=os.path.join(os.path.dirname(DEFAULT_MODEL), "model_int8.tflite"),
                    help="optional int8 .tflite of the same model; shown beside fp32 if present")
    ap.add_argument("--figures", default=DEFAULT_FIGURES)
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"loading {args.model} ...", flush=True)
    engine = Engine(args.model, args.tflite)
    print(f"model ready ({'fp32 + int8' if engine.tflite else 'fp32'}); figures from {args.figures}; samples from {args.samples}")
    print(f"open  http://localhost:{args.port}   (Ctrl+C to stop)", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(engine, args.figures, args.samples)).serve_forever()


if __name__ == "__main__":
    main()
