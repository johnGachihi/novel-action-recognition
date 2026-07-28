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
    lines.append("| Space | Model | AUROC | Closed-Set Acc |")
    lines.append("|---|---|---|---|")
    for space, res in results.items():
        # Model 1: standard_classifier (VideoMAE)
        m1 = res['videomae']['metrics']['standard_classifier']
        lines.append(f"| {space} | Model 1: standard_classifier (VideoMAE) | {m1['auroc']:.4f} | {m1['closed_set_acc']:.4f} |")
        
        # Model 2: dear (VideoMAE)
        m2 = res['videomae']['metrics']['dear']
        lines.append(f"| {space} | Model 2: dear (VideoMAE) | {m2['auroc']:.4f} | {m2['closed_set_acc']:.4f} |")
        
        # Model 3/4: dear (Multimodal)
        m3 = res['multimodal']['metrics']['dear']
        lines.append(f"| {space} | Model 3 & 4: dear (Multimodal) | {m3['auroc']:.4f} | {m3['closed_set_acc']:.4f} |")
    return "\n".join(lines)

def format_stage2_table(results):
    lines = []
    lines.append("| Space | Model | Known Acc | Seen-Novel Acc | Unseen-Novel Recall | False New Rate | New Cat Hungarian Acc | New Cat NMI | Overall Hungarian Acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for space, res in results.items():
        # Model 1: standard_classifier (VideoMAE)
        m1 = res['videomae']['phase2']['standard_classifier']
        lines.append(
            f"| {space} | Model 1: standard_classifier (VideoMAE) | {m1['known_acc']:.4f} | {m1['seen_novel_acc']:.4f} | {m1['unseen_detect_recall']:.4f} | {m1['false_new_rate']:.4f} | N/A | N/A | {m1['overall_hungarian_acc']:.4f} |"
        )
        
        # Model 2: dear_reject (VideoMAE)
        m2 = res['videomae']['phase2']['dear_reject']
        new_hung2 = m2.get('new_cat_hungarian_acc', float('nan'))
        new_nmi2 = m2.get('new_cat_nmi', float('nan'))
        new_hung_str2 = f"{new_hung2:.4f}" if not math.isnan(new_hung2) else "N/A"
        new_nmi_str2 = f"{new_nmi2:.4f}" if not math.isnan(new_nmi2) else "N/A"
        lines.append(
            f"| {space} | Model 2: dear_reject (VideoMAE) | {m2['known_acc']:.4f} | {m2['seen_novel_acc']:.4f} | {m2['unseen_detect_recall']:.4f} | {m2['false_new_rate']:.4f} | {new_hung_str2} | {new_nmi_str2} | {m2['overall_hungarian_acc']:.4f} |"
        )
        
        # Model 3: dear_reject (Multimodal)
        m3 = res['multimodal']['phase2']['dear_reject']
        new_hung3 = m3.get('new_cat_hungarian_acc', float('nan'))
        new_nmi3 = m3.get('new_cat_nmi', float('nan'))
        new_hung_str3 = f"{new_hung3:.4f}" if not math.isnan(new_hung3) else "N/A"
        new_nmi_str3 = f"{new_nmi3:.4f}" if not math.isnan(new_nmi3) else "N/A"
        lines.append(
            f"| {space} | Model 3: dear_reject (Multimodal) | {m3['known_acc']:.4f} | {m3['seen_novel_acc']:.4f} | {m3['unseen_detect_recall']:.4f} | {m3['false_new_rate']:.4f} | {new_hung_str3} | {new_nmi_str3} | {m3['overall_hungarian_acc']:.4f} |"
        )

        # Model 4: dear_weighted_reject (Multimodal)
        m4 = res['multimodal']['phase2']['dear_weighted_reject']
        new_hung4 = m4.get('new_cat_hungarian_acc', float('nan'))
        new_nmi4 = m4.get('new_cat_nmi', float('nan'))
        new_hung_str4 = f"{new_hung4:.4f}" if not math.isnan(new_hung4) else "N/A"
        new_nmi_str4 = f"{new_nmi4:.4f}" if not math.isnan(new_nmi4) else "N/A"
        lines.append(
            f"| {space} | Model 4: dear_weighted_reject (Multimodal) | {m4['known_acc']:.4f} | {m4['seen_novel_acc']:.4f} | {m4['unseen_detect_recall']:.4f} | {m4['false_new_rate']:.4f} | {new_hung_str4} | {new_nmi_str4} | {m4['overall_hungarian_acc']:.4f} |"
        )
    return "\n".join(lines)

def plot_diagnostics(stage1_results, stage2_results):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.decomposition import PCA
    import numpy as np

    os.makedirs("plots", exist_ok=True)

    # Helper to plot history if available
    def plot_history_line(ax, hist, label, is_loss=True):
        if hist is None:
            return
        if is_loss:
            ax.plot(hist['train_loss'], label=f'{label} (Train)')
            if hist.get('val_loss'):
                ax.plot(hist['val_loss'], linestyle='--', label=f'{label} (Val)')
        else:
            if hist.get('val_auroc'):
                ax.plot(hist['val_auroc'], label=label)

    # ----------------------------------------------------
    # 1. Consolidated Training Summary Plot
    # ----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Stage 1 Verb Training
    ax_loss_v, ax_auc_v = axes[0, 0], axes[0, 1]
    if 'verb' in stage1_results:
        res = stage1_results['verb']
        plot_history_line(ax_loss_v, res['videomae']['histories'].get('standard_classifier'), 'Model 1 (VideoMAE CE)')
        plot_history_line(ax_loss_v, res['videomae']['histories'].get('dear'), 'Model 2 (VideoMAE EDL)')
        plot_history_line(ax_loss_v, res['multimodal']['histories'].get('dear'), 'Model 3 & 4 (Multimodal EDL)')
        
        plot_history_line(ax_auc_v, res['videomae']['histories'].get('standard_classifier'), 'Model 1 (VideoMAE CE)', is_loss=False)
        plot_history_line(ax_auc_v, res['videomae']['histories'].get('dear'), 'Model 2 (VideoMAE EDL)', is_loss=False)
        plot_history_line(ax_auc_v, res['multimodal']['histories'].get('dear'), 'Model 3 & 4 (Multimodal EDL)', is_loss=False)
        
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
        res = stage1_results['noun']
        plot_history_line(ax_loss_n, res['videomae']['histories'].get('standard_classifier'), 'Model 1 (VideoMAE CE)')
        plot_history_line(ax_loss_n, res['videomae']['histories'].get('dear'), 'Model 2 (VideoMAE EDL)')
        plot_history_line(ax_loss_n, res['multimodal']['histories'].get('dear'), 'Model 3 & 4 (Multimodal EDL)')
        
        plot_history_line(ax_auc_n, res['videomae']['histories'].get('standard_classifier'), 'Model 1 (VideoMAE CE)', is_loss=False)
        plot_history_line(ax_auc_n, res['videomae']['histories'].get('dear'), 'Model 2 (VideoMAE EDL)', is_loss=False)
        plot_history_line(ax_auc_n, res['multimodal']['histories'].get('dear'), 'Model 3 & 4 (Multimodal EDL)', is_loss=False)
        
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
    if 'verb' in stage2_results:
        res = stage2_results['verb']
        plot_history_line(ax_s2_v, res['videomae'].get('history'), 'Model 2 (VideoMAE EDL)')
        plot_history_line(ax_s2_v, res['multimodal'].get('history'), 'Model 3 & 4 (Multimodal EDL)')
        ax_s2_v.set_title("Stage 2 Verb (dear_reject): Loss vs Epoch")
        ax_s2_v.set_xlabel("Epoch")
        ax_s2_v.set_ylabel("Loss")
        ax_s2_v.legend(fontsize=8)
        ax_s2_v.grid(True)
    else:
        ax_s2_v.axis('off')

    if 'noun' in stage2_results:
        res = stage2_results['noun']
        plot_history_line(ax_s2_n, res['videomae'].get('history'), 'Model 2 (VideoMAE EDL)')
        plot_history_line(ax_s2_n, res['multimodal'].get('history'), 'Model 3 & 4 (Multimodal EDL)')
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
    # 2. Stage 1 ROC Curves Summary Plot (1x2 Panel)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, sp in enumerate(['verb', 'noun']):
        ax = axes[i]
        if sp in stage1_results:
            res = stage1_results[sp]
            is_novel_v = np.array(res['videomae']['is_novel_test'])
            if len(is_novel_v) > 0 and len(np.unique(is_novel_v)) > 1:
                scores_m1 = np.array(res['videomae']['novelty_scores']['standard_classifier'])
                fpr1, tpr1, _ = roc_curve(is_novel_v, scores_m1)
                ax.plot(fpr1, tpr1, label=f'Model 1: CE (VideoMAE) (AUC = {auc(fpr1, tpr1):.3f})')
                
                scores_m2 = np.array(res['videomae']['novelty_scores']['dear'])
                fpr2, tpr2, _ = roc_curve(is_novel_v, scores_m2)
                ax.plot(fpr2, tpr2, label=f'Model 2: EDL (VideoMAE) (AUC = {auc(fpr2, tpr2):.3f})')
                
                scores_m3 = np.array(res['multimodal']['novelty_scores']['dear'])
                fpr3, tpr3, _ = roc_curve(is_novel_v, scores_m3)
                ax.plot(fpr3, tpr3, label=f'Model 3 & 4: EDL (Multimodal) (AUC = {auc(fpr3, tpr3):.3f})')
                
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
    # 3. Stage 1 Uncertainty Distribution Grid (2x3)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for row_idx, sp in enumerate(['verb', 'noun']):
        if sp in stage1_results:
            res = stage1_results[sp]
            is_novel_v = np.array(res['videomae']['is_novel_test'])
            
            # Model 1
            ax = axes[row_idx, 0]
            scores = np.array(res['videomae']['novelty_scores']['standard_classifier'])
            ax.hist(scores[is_novel_v == 0], bins=20, alpha=0.6, label='Known', color='blue', edgecolor='k')
            ax.hist(scores[is_novel_v == 1], bins=20, alpha=0.6, label='Novel', color='red', edgecolor='k')
            ax.set_title(f'{sp.capitalize()} Model 1 (VideoMAE CE)')
            ax.legend(fontsize=8)
            ax.grid(True)
            
            # Model 2
            ax = axes[row_idx, 1]
            scores = np.array(res['videomae']['novelty_scores']['dear'])
            ax.hist(scores[is_novel_v == 0], bins=20, alpha=0.6, label='Known', color='blue', edgecolor='k')
            ax.hist(scores[is_novel_v == 1], bins=20, alpha=0.6, label='Novel', color='red', edgecolor='k')
            ax.set_title(f'{sp.capitalize()} Model 2 (VideoMAE EDL)')
            ax.legend(fontsize=8)
            ax.grid(True)
            
            # Model 3 & 4
            ax = axes[row_idx, 2]
            scores = np.array(res['multimodal']['novelty_scores']['dear'])
            ax.hist(scores[is_novel_v == 0], bins=20, alpha=0.6, label='Known', color='blue', edgecolor='k')
            ax.hist(scores[is_novel_v == 1], bins=20, alpha=0.6, label='Novel', color='red', edgecolor='k')
            ax.set_title(f'{sp.capitalize()} Model 3 & 4 (Multimodal EDL)')
            ax.legend(fontsize=8)
            ax.grid(True)
        else:
            for col_idx in range(3):
                axes[row_idx, col_idx].axis('off')
                
    plt.tight_layout()
    plt.savefig("plots/stage1_uncertainty_summary.png")
    plt.close()

    # ----------------------------------------------------
    # 4. Stage 2 PCA & Discovery Summary Plot (2x5 Grid)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    
    for row_idx, sp in enumerate(['verb', 'noun']):
        ax_gt = axes[row_idx, 0]
        ax_m1 = axes[row_idx, 1]
        ax_m2 = axes[row_idx, 2]
        ax_m3 = axes[row_idx, 3]
        ax_m4 = axes[row_idx, 4]
        
        if sp in stage2_results:
            res_v = stage2_results[sp]['videomae']
            res_m = stage2_results[sp]['multimodal']
            
            stream_feats = np.array(res_m['stream_feats'])
            stream_type = np.array(res_m['stream_type'])

            if len(stream_feats) > 2:
                pca = PCA(n_components=2, random_state=0)
                feats_2d = pca.fit_transform(stream_feats)

                # Col 1: Ground Truth
                for stype in np.unique(stream_type):
                    mask = stream_type == stype
                    ax_gt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], label=stype, alpha=0.7, edgecolors='none')
                ax_gt.set_xlabel('PCA 1')
                ax_gt.set_ylabel('PCA 2')
                ax_gt.set_title(f'{sp.capitalize()} Space: Ground Truth')
                ax_gt.legend()
                ax_gt.grid(True)

                # Col 2: Model 1: Standard Classifier (VideoMAE)
                preds_m1 = np.array(res_v['stream_predictions']['standard_classifier'])
                scatter_m1 = ax_m1.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_m1, cmap='tab20', alpha=0.7, edgecolors='none')
                ax_m1.set_xlabel('PCA 1')
                ax_m1.set_title(f'Model 1: CE (VideoMAE)')
                fig.colorbar(scatter_m1, ax=ax_m1, label='Class ID')
                ax_m1.grid(True)

                # Col 3: Model 2: dear_reject (VideoMAE)
                preds_m2 = np.array(res_v['stream_predictions']['dear_reject'])
                scatter_m2 = ax_m2.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_m2, cmap='tab20', alpha=0.7, edgecolors='none')
                ax_m2.set_xlabel('PCA 1')
                ax_m2.set_title(f'Model 2: EDL (VideoMAE)')
                fig.colorbar(scatter_m2, ax=ax_m2, label='Cluster ID')
                ax_m2.grid(True)

                # Col 4: Model 3: dear_reject (Multimodal)
                preds_m3 = np.array(res_m['stream_predictions']['dear_reject'])
                scatter_m3 = ax_m3.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_m3, cmap='tab20', alpha=0.7, edgecolors='none')
                ax_m3.set_xlabel('PCA 1')
                ax_m3.set_title(f'Model 3: EDL (Multimodal)')
                fig.colorbar(scatter_m3, ax=ax_m3, label='Cluster ID')
                ax_m3.grid(True)

                # Col 5: Model 4: dear_weighted_reject (Multimodal)
                preds_m4 = np.array(res_m['stream_predictions']['dear_weighted_reject'])
                scatter_m4 = ax_m4.scatter(feats_2d[:, 0], feats_2d[:, 1], c=preds_m4, cmap='tab20', alpha=0.7, edgecolors='none')
                ax_m4.set_xlabel('PCA 1')
                ax_m4.set_title(f'Model 4: Weighted EDL (Multimodal)')
                fig.colorbar(scatter_m4, ax=ax_m4, label='Cluster ID')
                ax_m4.grid(True)
            else:
                for ax in (ax_gt, ax_m1, ax_m2, ax_m3, ax_m4):
                    ax.text(0.5, 0.5, "Not enough features to run PCA", ha='center', va='center')
        else:
            for ax in (ax_gt, ax_m1, ax_m2, ax_m3, ax_m4):
                ax.axis('off')
            
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
    print("RUNNING MULTIMODAL ACTION RECOGNITION EVALUATION PIPELINE")
    print(f"Features: {args.features}")
    print(f"Split: {args.split}")
    print(f"Spaces: {', '.join(spaces)}")
    print("=" * 60)

    # 1. Run Stage 1: Novelty Detection
    print("\n--- [Running Stage 1: Novelty Detection] ---")
    stage1_results = {}
    for sp in spaces:
        print(f"\nEvaluating space: {sp} (VideoMAE only)")
        res_v = run_epic_stage1(
            label_space=sp, methods=STAGE1_METHODS, epochs=30, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=False,
            feature_mode='videomae'
        )
        print(f"Evaluating space: {sp} (Multimodal)")
        res_m = run_epic_stage1(
            label_space=sp, methods=STAGE1_METHODS, epochs=30, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=False,
            feature_mode='multimodal'
        )
        stage1_results[sp] = {'videomae': res_v, 'multimodal': res_m}

    # 2. Run Stage 2: Continual Category Discovery
    print("\n--- [Running Stage 2: Continual Category Discovery] ---")
    stage2_results = {}
    for sp in spaces:
        print(f"\nEvaluating space: {sp} (VideoMAE only)")
        res_v = run_epic_continual(
            label_space=sp, methods=STAGE2_METHODS, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=False,
            feature_mode='videomae'
        )
        print(f"Evaluating space: {sp} (Multimodal)")
        res_m = run_epic_continual(
            label_space=sp, methods=STAGE2_METHODS, seed=args.seed,
            features_path=args.features, split_path=args.split, verbose=False,
            feature_mode='multimodal'
        )
        stage2_results[sp] = {'videomae': res_v, 'multimodal': res_m}

    # 3. Format and print tables
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY (4 TARGET MODELS)")
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
        'stage1': {sp: {'videomae': r['videomae']['metrics'], 'multimodal': r['multimodal']['metrics']} for sp, r in stage1_results.items()},
        'stage2': {sp: {'videomae': r['videomae']['phase2'], 'multimodal': r['multimodal']['phase2']} for sp, r in stage2_results.items()}
    }
    with open(args.out, 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved detailed evaluation results to {args.out}")

if __name__ == '__main__':
    main()
