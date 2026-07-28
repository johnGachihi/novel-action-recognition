#!/usr/bin/env python3
"""Unified Action Recognition Evaluation Pipeline.
Runs both Stage 1 (Novelty Detection) and Stage 2 (Continual Category Discovery)
benchmarks on EPIC-KITCHENS-100 and outputs a clear comparison of results.
"""
import argparse
import json
import math
import os
import sys

from nac.epic_stage1 import run_epic_stage1, ALL_METHODS as STAGE1_METHODS
from nac.epic_stage2 import run_epic_continual, ALL_METHODS as STAGE2_METHODS

def format_stage1_table(results):
    lines = []
    lines.append("| Space | Method | AUROC | Closed-Set Acc |")
    lines.append("|---|---|---|---|")
    for space, res in results.items():
        for method, metrics in res['metrics'].items():
            lines.append(f"| {space} | {method} | {metrics['auroc']:.4f} | {metrics['closed_set_acc']:.4f} |")
    return "\n".join(lines)

def format_stage2_table(results):
    lines = []
    lines.append("| Space | Method | Known Acc | Seen-Novel Acc | Unseen-Novel Recall | False New Rate | New Cat Hungarian Acc | New Cat NMI | Overall Hungarian Acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for space, res in results.items():
        phase2 = res['phase2']
        for method, metrics in phase2.items():
            k_acc = metrics.get('known_acc', 0.0)
            sn_acc = metrics.get('seen_novel_acc', 0.0)
            un_rec = metrics.get('unseen_detect_recall', 0.0)
            fn_rate = metrics.get('false_new_rate', 0.0)
            new_hung = metrics.get('new_cat_hungarian_acc', float('nan'))
            new_nmi = metrics.get('new_cat_nmi', float('nan'))
            overall = metrics.get('overall_hungarian_acc', 0.0)
            
            new_hung_str = f"{new_hung:.4f}" if not math.isnan(new_hung) else "N/A"
            new_nmi_str = f"{new_nmi:.4f}" if not math.isnan(new_nmi) else "N/A"
            
            lines.append(
                f"| {space} | {method} | {k_acc:.4f} | {sn_acc:.4f} | {un_rec:.4f} | {fn_rate:.4f} | {new_hung_str} | {new_nmi_str} | {overall:.4f} |"
            )
    return "\n".join(lines)

def plot_diagnostics(stage1_results, stage2_results):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.decomposition import PCA
    import numpy as np

    os.makedirs("plots", exist_ok=True)

    # ----------------------------------------------------
    # 1. Consolidated Training Summary Plot
    # ----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Stage 1 Verb Training
    ax_loss_v, ax_auc_v = axes[0, 0], axes[0, 1]
    if 'verb' in stage1_results:
        histories = stage1_results['verb'].get('histories', {})
        for name, hist in histories.items():
            if hist:
                ax_loss_v.plot(hist['train_loss'], label=f'{name} (Train)')
                if hist['val_loss']:
                    ax_loss_v.plot(hist['val_loss'], linestyle='--', label=f'{name} (Val)')
                if hist['val_auroc']:
                    ax_auc_v.plot(hist['val_auroc'], label=name)
        ax_loss_v.set_title("Stage 1 Verb: Loss vs Epoch")
        ax_loss_v.set_xlabel("Epoch")
        ax_loss_v.set_ylabel("Loss")
        ax_loss_v.legend(fontsize=8, loc='upper right')
        ax_loss_v.grid(True)
        
        ax_auc_v.set_title("Stage 1 Verb: Val AUROC vs Epoch")
        ax_auc_v.set_xlabel("Epoch")
        ax_auc_v.set_ylabel("AUROC")
        ax_auc_v.legend(fontsize=8, loc='lower right')
        ax_auc_v.grid(True)
    else:
        ax_loss_v.axis('off')
        ax_auc_v.axis('off')

    # Stage 1 Noun Training
    ax_loss_n, ax_auc_n = axes[1, 0], axes[1, 1]
    if 'noun' in stage1_results:
        histories = stage1_results['noun'].get('histories', {})
        for name, hist in histories.items():
            if hist:
                ax_loss_n.plot(hist['train_loss'], label=f'{name} (Train)')
                if hist['val_loss']:
                    ax_loss_n.plot(hist['val_loss'], linestyle='--', label=f'{name} (Val)')
                if hist['val_auroc']:
                    ax_auc_n.plot(hist['val_auroc'], label=name)
        ax_loss_n.set_title("Stage 1 Noun: Loss vs Epoch")
        ax_loss_n.set_xlabel("Epoch")
        ax_loss_n.set_ylabel("Loss")
        ax_loss_n.legend(fontsize=8, loc='upper right')
        ax_loss_n.grid(True)
        
        ax_auc_n.set_title("Stage 1 Noun: Val AUROC vs Epoch")
        ax_auc_n.set_xlabel("Epoch")
        ax_auc_n.set_ylabel("AUROC")
        ax_auc_n.legend(fontsize=8, loc='lower right')
        ax_auc_n.grid(True)
    else:
        ax_loss_n.axis('off')
        ax_auc_n.axis('off')

    # Stage 2 Training Loss
    ax_s2_v, ax_s2_n = axes[2, 0], axes[2, 1]
    if 'verb' in stage2_results and stage2_results['verb'].get('history'):
        hist = stage2_results['verb']['history']
        ax_s2_v.plot(hist['train_loss'], label='Train Loss')
        if hist['val_loss']:
            ax_s2_v.plot(hist['val_loss'], label='Val Loss', linestyle='--')
        ax_s2_v.set_title("Stage 2 Verb (dear_reject): Loss vs Epoch")
        ax_s2_v.set_xlabel("Epoch")
        ax_s2_v.set_ylabel("Loss")
        ax_s2_v.legend(fontsize=8)
        ax_s2_v.grid(True)
    else:
        ax_s2_v.axis('off')

    if 'noun' in stage2_results and stage2_results['noun'].get('history'):
        hist = stage2_results['noun']['history']
        ax_s2_n.plot(hist['train_loss'], label='Train Loss')
        if hist['val_loss']:
            ax_s2_n.plot(hist['val_loss'], label='Val Loss', linestyle='--')
        ax_s2_n.set_title("Stage 2 Noun (dear_reject): Loss vs Epoch")
        ax_s2_n.set_xlabel("Epoch")
        ax_s2_n.set_ylabel("Loss")
        ax_s2_n.legend(fontsize=8)
        ax_s2_n.grid(True)
    else:
        ax_s2_n.axis('off')

    plt.tight_layout()
    plt.savefig("plots/training_summary.png")
    plt.close()

    # ----------------------------------------------------
    # 2. Stage 1 ROC Curves Summary Plot
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, sp in enumerate(['verb', 'noun']):
        ax = axes[i]
        if sp in stage1_results:
            s1_res = stage1_results[sp]
            is_novel_test = np.array(s1_res['is_novel_test'])
            novelty_scores = s1_res['novelty_scores']
            
            for name, scores in novelty_scores.items():
                scores_arr = np.array(scores)
                if len(is_novel_test) > 0 and np.isfinite(scores_arr).all() and len(np.unique(is_novel_test)) > 1:
                    fpr, tpr, _ = roc_curve(is_novel_test, scores_arr)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Stage 1 ROC Curves ({sp})')
            ax.legend(loc='lower right')
            ax.grid(True)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig("plots/stage1_roc_summary.png")
    plt.close()

    # ----------------------------------------------------
    # 3. Stage 1 Uncertainty Distribution Grid (2x5)
    # ----------------------------------------------------
    methods_to_plot = ['standard_classifier', 'sensoy_tuned', 'dear', 'dear_noreg', 'dear_kl']
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for row_idx, sp in enumerate(['verb', 'noun']):
        if sp in stage1_results:
            s1_res = stage1_results[sp]
            is_novel_test = np.array(s1_res['is_novel_test'])
            novelty_scores = s1_res['novelty_scores']
            
            for col_idx, name in enumerate(methods_to_plot):
                ax = axes[row_idx, col_idx]
                if name in novelty_scores:
                    scores_arr = np.array(novelty_scores[name])
                    if len(is_novel_test) > 0 and np.isfinite(scores_arr).all():
                        known_scores = scores_arr[is_novel_test == 0]
                        novel_scores = scores_arr[is_novel_test == 1]
                        
                        ax.hist(known_scores, bins=20, alpha=0.6, label='Known', color='blue', edgecolor='k')
                        ax.hist(novel_scores, bins=20, alpha=0.6, label='Novel', color='red', edgecolor='k')
                        ax.set_title(f'{sp.capitalize()}: {name}')
                        ax.set_xlabel('Anomaly Score')
                        ax.set_ylabel('Count')
                        ax.legend(fontsize=8)
                        ax.grid(True)
                    else:
                        ax.axis('off')
                else:
                    ax.axis('off')
        else:
            for col_idx in range(5):
                axes[row_idx, col_idx].axis('off')
                
    plt.tight_layout()
    plt.savefig("plots/stage1_uncertainty_summary.png")
    plt.close()

    # ----------------------------------------------------
    # 4. Stage 2 PCA & Discovery Summary Plot (2x3 Side-by-Side)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for row_idx, sp in enumerate(['verb', 'noun']):
        ax_gt = axes[row_idx, 0]
        ax_base = axes[row_idx, 1]
        ax_disc = axes[row_idx, 2]
        
        if sp in stage2_results:
            s2_res = stage2_results[sp]
            stream_feats = np.array(s2_res['stream_feats'])
            stream_type = np.array(s2_res['stream_type'])
            stream_predictions = s2_res['stream_predictions']

            if len(stream_feats) > 2:
                pca = PCA(n_components=2, random_state=0)
                feats_2d = pca.fit_transform(stream_feats)

                # 1. Ground Truth
                for stype in np.unique(stream_type):
                    mask = stream_type == stype
                    ax_gt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], label=stype, alpha=0.7, edgecolors='none')
                ax_gt.set_xlabel('PCA 1')
                ax_gt.set_ylabel('PCA 2')
                ax_gt.set_title(f'{sp.capitalize()} Space: Ground Truth Stream')
                ax_gt.legend()
                ax_gt.grid(True)

                # 2. Closed-Set Baseline (standard_classifier)
                if 'standard_classifier' in stream_predictions:
                    preds_base = np.array(stream_predictions['standard_classifier'])
                    scatter_b = ax_base.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_base, cmap='tab20', alpha=0.7, edgecolors='none')
                    ax_base.set_xlabel('PCA 1')
                    ax_base.set_ylabel('PCA 2')
                    ax_base.set_title(f'{sp.capitalize()} Space: Standard Classifier (Closed-Set)')
                    fig.colorbar(scatter_b, ax=ax_base, label='Predicted Class ID')
                    ax_base.grid(True)
                else:
                    ax_base.text(0.5, 0.5, "Standard Classifier not run", ha='center', va='center')

                # 3. Discovery Model (dear_reject)
                best_method = 'dear_reject' if 'dear_reject' in stream_predictions else list(stream_predictions.keys())[0]
                preds_disc = np.array(stream_predictions[best_method])
                scatter_d = ax_disc.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_disc, cmap='tab20', alpha=0.7, edgecolors='none')
                ax_disc.set_xlabel('PCA 1')
                ax_disc.set_ylabel('PCA 2')
                ax_disc.set_title(f'{sp.capitalize()} Space: Discovery Model ({best_method})')
                fig.colorbar(scatter_d, ax=ax_disc, label='Predicted Cluster/Class ID')
                ax_disc.grid(True)
            else:
                ax_gt.text(0.5, 0.5, "Not enough features to run PCA", ha='center', va='center')
                ax_base.text(0.5, 0.5, "Not enough features to run PCA", ha='center', va='center')
                ax_disc.text(0.5, 0.5, "Not enough features to run PCA", ha='center', va='center')
        else:
            ax_gt.axis('off')
            ax_base.axis('off')
            ax_disc.axis('off')
            
    plt.tight_layout()
    plt.savefig("plots/stage2_pca_summary.png")
    plt.close()

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

    # Plot and save diagnostic summary plots
    print("\nGenerating diagnostic summary plots...")
    plot_diagnostics(stage1_results, stage2_results)
    print("Consolidated summary plots saved under plots/")

    # Save to file
    out_dict = {
        'stage1': {sp: r['metrics'] for sp, r in stage1_results.items()},
        'stage2': {sp: {'phase1': r['phase1'], 'phase2': r['phase2']} for sp, r in stage2_results.items()}
    }
    with open(args.out, 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved detailed evaluation results to {args.out}")

if __name__ == '__main__':
    main()
