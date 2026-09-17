"""Put the repo root on sys.path so `pytest tests -q` can import model.py.

Without this, pytest's default `prepend` import mode inserts tests/ -- not the
repo root -- so `from model import fft_layer` fails, and the guard tests error
out instead of running. `python -m pytest` happens to work because it adds the
cwd; this file makes the bare command work too, which is what the README and
the Colab notebook both use.

pytest collects a root conftest.py automatically and prepends its directory.
Nothing else belongs in here.
"""
