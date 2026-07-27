#!/usr/bin/env python3
"""CLI for Stage 2 continual novel-category discovery.

Examples:
  python3 run_stage2.py                                   # all methods, all protocols
  python3 run_stage2.py --subset UCF101 --methods dear_reject
  python3 run_stage2.py --no-whiten                       # geometry ablation
  python3 run_stage2.py --calib members                   # self-fulfilling radii ablation
  python3 run_stage2.py --buffer-mode anchored            # anchor re-absorption ablation
  python3 run_stage2.py --merge none --calib-q 80         # merge rule / operating point
"""
import argparse
import json

from nac.stage2 import ALL_METHODS, run_continual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subset', choices=['combined', 'UCF101', 'HMDB51', 'all'], default='all')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-whiten', action='store_true')
    ap.add_argument('--calib', choices=['heldout', 'members'], default='heldout')
    ap.add_argument('--calib-q', type=float, default=90)
    ap.add_argument('--buffer-mode', choices=['alone', 'anchored'], default='alone')
    ap.add_argument('--merge', choices=['spacing', 'sample-radius', 'none'], default='spacing')
    ap.add_argument('--gate-epochs', type=int, default=50)
    ap.add_argument('--p1-assign', choices=['greedy', 'sinkhorn'], default='greedy',
                    help='phase-1 unlabeled assignment rule (sinkhorn = UNO-style balanced OT)')
    ap.add_argument('--sinkhorn-eps', type=float, default=0.05)
    ap.add_argument('--buffer-clusterer', choices=['kmeans', 'hdbscan'], default='kmeans',
                    help='hdbscan discovers the category count itself (no K oracle) with a noise bucket')
    ap.add_argument('--min-cluster-size', type=int, default=10, help='hdbscan only')
    ap.add_argument('--features', default='features_videomae.npz')
    ap.add_argument('--out', default=None, help='write results JSON here')
    args = ap.parse_args()

    subsets = [None, 'UCF101', 'HMDB51'] if args.subset == 'all' else \
              [None if args.subset == 'combined' else args.subset]
    all_results = {}
    for sub in subsets:
        all_results[sub or 'combined'] = run_continual(
            subset=sub, methods=args.methods, seed=args.seed, whiten=not args.no_whiten,
            calib=args.calib, calib_q=args.calib_q, buffer_mode=args.buffer_mode,
            merge=args.merge, gate_epochs=args.gate_epochs, p1_assignment=args.p1_assign,
            sinkhorn_eps=args.sinkhorn_eps, buffer_clusterer=args.buffer_clusterer,
            min_cluster_size=args.min_cluster_size, features_path=args.features)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
