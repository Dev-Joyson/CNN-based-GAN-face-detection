#!/usr/bin/env python3
"""Entry point: python train.py --config configs/test13_face.yaml"""

import argparse

from model import load_config, train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <run_dir>/model.keras after a lost Colab VM")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.distill:
        from distill import distill
        distill(cfg)
    else:
        train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
