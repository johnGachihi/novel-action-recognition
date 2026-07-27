#!/usr/bin/env python3
"""CLI for EPIC-KITCHENS-100 Stage 2 continual discovery (verb + noun).
Mirrors run_stage2.py's flags, plus --label-space. Defaults to the
already-validated best config from the UCF101/HMDB51 ablation work
(sinkhorn phase-1 assignment, eps=0.3) rather than run_stage2.py's
greedy/ablation-baseline defaults.

verb and noun are fully independent computations over the same read-only
feature file, so --label-space all runs them as two concurrent processes
instead of a sequential loop -- each is a small workload (whitening, k-means,
one linear DEAR head) that leaves the GPU mostly idle between kernel launches,
so running two at once is close to free wall-clock-wise and better uses the
machine than serializing them.

Examples:
  python3 run_epic_stage2.py                              # both label spaces, all methods
  python3 run_epic_stage2.py --label-space verb --methods dear_reject
  python3 run_epic_stage2.py --no-whiten                   # geometry ablation
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from nac.epic_stage2 import ALL_METHODS, run_epic_continual


def _run_one(kwargs):
    label_space = kwargs.pop('label_space')
    return label_space, run_epic_continual(label_space=label_space, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label-space', choices=['verb', 'noun', 'all'], default='all')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-whiten', action='store_true')
    ap.add_argument('--calib', choices=['heldout', 'members'], default='heldout')
    ap.add_argument('--calib-q', type=float, default=90)
    ap.add_argument('--buffer-mode', choices=['alone', 'anchored'], default='alone')
    ap.add_argument('--merge', choices=['spacing', 'sample-radius', 'none'], default='spacing')
    ap.add_argument('--gate-epochs', type=int, default=50)
    ap.add_argument('--p1-assign', choices=['greedy', 'sinkhorn'], default='sinkhorn')
    ap.add_argument('--sinkhorn-eps', type=float, default=0.05)
    ap.add_argument('--sinkhorn-col-weights', choices=['uniform', 'train_freq'], default='train_freq',
                    help='train_freq: needed for EPIC\'s long-tailed classes, see nac/epic_stage2.py docstring')
    ap.add_argument('--buffer-clusterer', choices=['kmeans', 'hdbscan'], default='kmeans')
    ap.add_argument('--min-cluster-size', type=int, default=10, help='hdbscan only')
    ap.add_argument('--features', default='features_epic.npz')
    ap.add_argument('--split', default='class_split_epic.json')
    ap.add_argument('--out', default=None, help='write results JSON here')
    ap.add_argument('--checkpoint-out', default=None,
                    help='save dear_reject checkpoint(s) here, e.g. epic_stage2_ckpt (writes _verb.pt/_noun.pt)')
    ap.add_argument('--no-parallel', action='store_true', help='run verb/noun sequentially instead of concurrently')
    args = ap.parse_args()

    spaces = ['verb', 'noun'] if args.label_space == 'all' else [args.label_space]
    common = dict(methods=args.methods, seed=args.seed, whiten=not args.no_whiten,
                  calib=args.calib, calib_q=args.calib_q, buffer_mode=args.buffer_mode,
                  merge=args.merge, gate_epochs=args.gate_epochs, p1_assignment=args.p1_assign,
                  sinkhorn_eps=args.sinkhorn_eps, sinkhorn_col_weights=args.sinkhorn_col_weights,
                  buffer_clusterer=args.buffer_clusterer,
                  min_cluster_size=args.min_cluster_size, features_path=args.features,
                  split_path=args.split)

    all_results = {}
    if len(spaces) > 1 and not args.no_parallel:
        jobs = [{'label_space': sp, **common} for sp in spaces]
        with ProcessPoolExecutor(max_workers=len(spaces)) as ex:
            for label_space, res in ex.map(_run_one, jobs):
                all_results[label_space] = res
    else:
        for sp in spaces:
            _, res = _run_one({'label_space': sp, **common})
            all_results[sp] = res

    if args.checkpoint_out:
        import torch
        for sp, res in all_results.items():
            if res['checkpoint'] is not None:
                torch.save(res['checkpoint'], f"{args.checkpoint_out}_{sp}.pt")
                print(f"saved {args.checkpoint_out}_{sp}.pt")

    if args.out:
        serializable = {sp: {'phase1': r['phase1'], 'phase2': r['phase2']} for sp, r in all_results.items()}
        with open(args.out, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
