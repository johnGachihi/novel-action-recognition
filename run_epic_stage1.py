#!/usr/bin/env python3
"""CLI for the EPIC-KITCHENS-100 DEAR novelty-detection benchmark (verb + noun
label spaces). Mirrors run_stage1.py's interface.

Examples:
  python3 run_epic_stage1.py                          # both label spaces, all methods
  python3 run_epic_stage1.py --label-space verb --methods dear mahalanobis
"""
import argparse
import json

from nac.epic_stage1 import ALL_METHODS, run_epic_stage1


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
    ap.add_argument('--label-space', choices=['verb', 'noun', 'all'], default='all')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--epochs', type=int, default=75)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--override', nargs='*', default=[],
                    help='evidential loss kwargs, e.g. with_avuloss=False lambda_ceiling=0.05')
    ap.add_argument('--features', default='features_epic.npz')
    ap.add_argument('--split', default='class_split_epic.json')
    ap.add_argument('--out', default=None, help='write results JSON here')
    args = ap.parse_args()

    overrides = dict(parse_override(s) for s in args.override)
    spaces = ['verb', 'noun'] if args.label_space == 'all' else [args.label_space]
    all_results = {}
    for sp in spaces:
        all_results[sp] = run_epic_stage1(
            label_space=sp, methods=args.methods, epochs=args.epochs, seed=args.seed,
            features_path=args.features, split_path=args.split, overrides=overrides or None)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
