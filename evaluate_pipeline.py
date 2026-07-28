#!/usr/bin/env python3
"""Unified Action Recognition Evaluation Pipeline.
Runs both Stage 1 (Novelty Detection) and Stage 2 (Continual Category Discovery)
benchmarks on EPIC-KITCHENS-100 and outputs a clear comparison of results.
"""
import argparse
import json
import os
import sys

from nac.epic_stage1 import run_epic_stage1, ALL_METHODS as STAGE1_METHODS
from nac.epic_stage2 import run_epic_continual, ALL_METHODS as STAGE2_METHODS

def format_stage1_table(results):
    lines = []
    lines.append("| Space | Method | AUROC | Closed-Set Acc |")
    lines.append("|---|---|---|---|")
    for space, res in results.items():
        for method, metrics in res.items():
            lines.append(f"| {space} | {method} | {metrics['auroc']:.4f} | {metrics['closed_set_acc']:.4f} |")
    return "\n".join(lines)

def format_stage2_table(results):
    lines = []
    lines.append("| Space | Method | Known Acc | Seen-Novel Acc | Unseen-Novel Recall | False New Rate | New Cat Hungarian Acc | New Cat NMI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for space, res in results.items():
        phase2 = res['phase2']
        for method, metrics in phase2.items():
            k_acc = metrics.get('known_acc', 0.0)
            sn_acc = metrics.get('seen_novel_acc', 0.0)
            un_rec = metrics.get('unseen_detect_recall', 0.0)
            fn_rate = metrics.get('false_new_rate', 0.0)
            new_hung = metrics.get('new_cat_hungarian_acc', float('nan'))
            new_nmi = metrics.get('new_cat_nmi', float('nan'))
            
            new_hung_str = f"{new_hung:.4f}" if not os.isnan(new_hung) else "N/A"
            new_nmi_str = f"{new_nmi:.4f}" if not os.path.isnan(new_nmi) else "N/A"
            
            lines.append(
                f"| {space} | {method} | {k_acc:.4f} | {sn_acc:.4f} | {un_rec:.4f} | {fn_rate:.4f} | {new_hung_str} | {new_nmi_str} |"
            )
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', default='features_epic.npz', help='Path to extracted features file')
    ap.add_argument('--split', default='class_split_epic.json', help='Path to class split json file')
    ap.add_argument('--out', default='evaluation_results.json', help='Write detailed results JSON here')
    ap.add_argument('--label-space', choices=['verb', 'noun', 'all'], default='all', help='Which label space to evaluate')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    args = ap.parse_args()

    if not os.path.exists(args.features):
        print(f"Error: Features file not found at {args.features}. Please run feature extraction first.")
        sys.exit(1)

    spaces = ['verb', 'noun'] if args.label_space == 'all' else [args.label_space]
    
    print("=" * 60)
    print("RUNNING ACTION RECOGNITION EVALUATION PIPELINE")
    print(f"Features: {args.features}")
    print(f"Split: {args.split}")
    print(f"Spaces: {', '.join(spaces)}")
    print("=" * 60)

    # 1. Run Stage 1: Novelty Detection
    print("\n--- [Running Stage 1: Novelty Detection] ---")
    stage1_results = {}
    for sp in spaces:
        print(f"\nEvaluating space: {sp}")
        stage1_results[sp] = run_epic_stage1(
            label_space=sp, methods=STAGE1_METHODS, epochs=30, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=True
        )

    # 2. Run Stage 2: Continual Category Discovery
    print("\n--- [Running Stage 2: Continual Category Discovery] ---")
    stage2_results = {}
    for sp in spaces:
        print(f"\nEvaluating space: {sp}")
        stage2_results[sp] = run_epic_continual(
            label_space=sp, methods=STAGE2_METHODS, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=True
        )

    # 3. Format and print tables
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n### Stage 1: Novelty Detection")
    s1_table = format_stage1_table(stage1_results)
    print(s1_table)
    
    print("\n### Stage 2: Continual Category Discovery")
    s2_table = format_stage2_table(stage2_results)
    print(s2_table)

    # Save to file
    out_dict = {
        'stage1': stage1_results,
        'stage2': {sp: {'phase1': r['phase1'], 'phase2': r['phase2']} for sp, r in stage2_results.items()}
    }
    with open(args.out, 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved detailed evaluation results to {args.out}")

if __name__ == '__main__':
    main()
