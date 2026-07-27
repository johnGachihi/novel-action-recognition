#!/usr/bin/env python3
"""CLI for Stage 1 novelty-detection benchmarks.

Examples:
  python3 run_stage1.py                              # all methods, all protocols
  python3 run_stage1.py --subset UCF101 --methods dear mahalanobis
  python3 run_stage1.py --methods dear --override with_avuloss=False   # ablation
"""
import argparse
import json

from nac.stage1 import ALL_METHODS, run_stage1


def parse_override(s):
    k, v = s.split('=', 1)
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            pass
    if v in ('True', 'False'):
        return k, v == 'True'
    return k, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subset', choices=['combined', 'UCF101', 'HMDB51', 'all'], default='all')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--epochs', type=int, default=75)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--override', nargs='*', default=[],
                    help='evidential loss kwargs, e.g. with_avuloss=False lambda_ceiling=0.05')
    ap.add_argument('--features', default='features_videomae.npz')
    ap.add_argument('--out', default=None, help='write results JSON here')
    args = ap.parse_args()

    overrides = dict(parse_override(s) for s in args.override)
    subsets = [None, 'UCF101', 'HMDB51'] if args.subset == 'all' else \
              [None if args.subset == 'combined' else args.subset]
    all_results = {}
    for sub in subsets:
        all_results[sub or 'combined'] = run_stage1(
            subset=sub, methods=args.methods, epochs=args.epochs, seed=args.seed,
            features_path=args.features, overrides=overrides or None)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
