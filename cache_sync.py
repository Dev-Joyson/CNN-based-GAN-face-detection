#!/usr/bin/env python3
"""Keep the tf.data caches in Drive so a reclaimed VM costs minutes, not hours.

    python cache_sync.py --config configs/test17_sg2.yaml --to-drive     # after a build
    python cache_sync.py --config configs/test17_sg2.yaml --from-drive   # on a fresh VM

Rebuilding a cache reads 50-100k small PNGs off Drive at ~12/s: 1-2.5 hours
of idle GPU, paid again every time Colab reclaims the VM (four times so far).
The finished cache is a handful of large files; copying those back from Drive
is a sequential read and takes minutes. Same config -> same cache key -> same
folder name on both sides, so nothing can be mixed up between experiments.

Only complete cache files are copied: a *.lockfile or *.tempstate* means a
build was interrupted and that split is skipped.
"""

import argparse
import os
import shutil
import time

from model import load_config


def sync(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"no cache at {src_dir}")
    names = sorted(os.listdir(src_dir))
    partial = {n.split(".")[0].rsplit("_", 1)[0] for n in names if "lockfile" in n or "tempstate" in n}
    os.makedirs(dst_dir, exist_ok=True)
    t0 = time.time()
    for n in names:
        split = n.split(".")[0]
        if "lockfile" in n or "tempstate" in n or split in partial:
            print(f"  skip {n} (incomplete build)")
            continue
        src, dst = os.path.join(src_dir, n), os.path.join(dst_dir, n)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"  ok   {n} (already there)")
            continue
        mb = os.path.getsize(src) / 1e6
        print(f"  copy {n}  {mb:,.0f} MB ...", end="", flush=True)
        shutil.copyfile(src, dst)
        print(f" {(time.time() - t0) / 60:.1f} min")
    print(f"done: {dst_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--to-drive", action="store_true")
    g.add_argument("--from-drive", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    local = os.path.dirname(cfg.cache_path("x"))                 # /content/cache/<key>
    drive = os.path.join(cfg.out_dir, "cache", os.path.basename(local))   # <out_dir>/cache/<key>
    if args.to_drive:
        sync(local, drive)
    else:
        sync(drive, local)


if __name__ == "__main__":
    main()
