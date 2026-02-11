#!/usr/bin/env python3
"""Load best hyperparameter configurations and their predictions."""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

#OurCode
import sys
sys.path.append("../../../src")
import layers
import likelihoods
import losses
import metrics
import priors
import utils

BASE_DIR = "../../../results/CIFAR-10/laplace_results"
# BASE_DIR = "../../../results/CIFAR-10/linear_results"
NS = [100, 1000, 10000, 50000]
RANDOM_STATES = [1001, 2001, 3001]
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
RANK = 1024
LENGTHSCALE = 20.0
OUTPUTSCALE = 1.0


def load_all_metrics(base_dir=BASE_DIR):
    """Load metrics for all (n, rs, lr) combinations."""
    all_metrics = {}
    
    for n in NS:
        for rs in RANDOM_STATES:
            for lr in LEARNING_RATES:
                run_dir = f"{base_dir}/n_{n}/rs_{rs}/rank_{RANK}-ls_{LENGTHSCALE}-os_{OUTPUTSCALE}-lr_{lr}"
                # run_dir = f"{base_dir}/n_{n}/rs_{rs}/lr_{lr}"
                metrics_path = os.path.join(run_dir, "metrics.json")
                
                if not os.path.exists(metrics_path):
                    continue
                
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                
                key = (n, rs, lr)
                all_metrics[key] = {
                    'metrics': metrics_data,
                    'run_dir': run_dir,
                }
    
    return all_metrics


def select_best_configs(all_metrics, selection_metric='val_balanced_accuracy'):
    """Select best learning rate for each (n, rs) based on selection metric."""
    best_configs = {}
    
    # Group by (n, rs)
    grouped = {}
    for (n, rs, lr), data in all_metrics.items():
        if (n, rs) not in grouped:
            grouped[(n, rs)] = []
        grouped[(n, rs)].append((lr, data))
    
    # Select best for each group
    for (n, rs), configs in grouped.items():
        best_lr, best_data = max(configs, key=lambda x: x[1]['metrics']['val'][selection_metric.replace('val_', '')])
        
        best_configs[(n, rs)] = {
            'learning_rate': best_lr,
            'metrics': best_data['metrics'],
            'run_dir': best_data['run_dir'],
        }
    
    return best_configs


def load_best_predictions(best_configs):
    """Load predictions.pth for all best configurations."""
    predictions = {}
    
    for (n, rs), config in best_configs.items():
        pred_path = os.path.join(config['run_dir'], 'predictions.pth')
        
        if os.path.exists(pred_path):
            predictions[(n, rs)] = torch.load(pred_path, map_location='cpu')
    
    return predictions


def get_ds_scores(logits):
    """Calculate Dempster-Shafer scores from logits."""
    return metrics.dempster_shafer_score(logits)


def get_accuracies_by_n(predictions, best_configs, split='test'):
    """Get accuracies grouped by dataset size n."""
    results = {n: [] for n in NS}
    
    for (n, rs), preds in predictions.items():
        accuracy = best_configs[(n, rs)]['metrics'][split]['accuracy']
        results[n].append(accuracy)
    
    return results


def apply_abstention_threshold(probs, logits, labels, ds_scores, percentile):
    """Apply abstention threshold and calculate accuracy on remaining samples."""
    threshold = np.percentile(ds_scores.numpy(), percentile)
    keep_mask = ds_scores <= threshold
    
    if keep_mask.sum() == 0:
        return 0.0
    
    pred_labels = probs[keep_mask].argmax(dim=-1)
    true_labels = labels[keep_mask]
    accuracy = (pred_labels == true_labels).float().mean().item()
    
    return accuracy


def plot_test_accuracies(predictions, best_configs):
    """Plot test accuracies vs dataset size with min/max bands and table."""
    results = get_accuracies_by_n(predictions, best_configs, split='test')
    
    ns = sorted(results.keys())
    means = [np.mean(results[n]) for n in ns]
    mins = [np.min(results[n]) for n in ns]
    maxs = [np.max(results[n]) for n in ns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot
    ax1.plot(ns, means, 'o-', linewidth=2, markersize=8)
    ax1.fill_between(ns, mins, maxs, alpha=0.3)
    ax1.set_xlabel('Dataset Size (n)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Dataset Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Table
    table_data = [[n, f"{np.mean(results[n]):.4f}", f"{np.min(results[n]):.4f}", f"{np.max(results[n]):.4f}"] 
                  for n in ns]
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=table_data, 
                     colLabels=['n', 'Mean', 'Min', 'Max'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    return fig


def plot_test_and_ood_abstention(predictions):
    """Plot histogram of test and ood abstention scores for each dataset size."""
    num_plots = len(NS)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    for idx, n in enumerate(NS):
        ax = axes[idx]
        
        test_ds_list = []
        ood_ds_list = []
        
        for (n_key, rs), preds in predictions.items():
            if n_key != n:
                continue
            
            test_ds = get_ds_scores(preds['test_logits'])
            ood_ds = get_ds_scores(preds['ood_logits'])
            test_ds_list.append(test_ds)
            ood_ds_list.append(ood_ds)
        
        if len(test_ds_list) > 0:
            test_ds_stacked = torch.stack(test_ds_list, dim=0)
            ood_ds_stacked = torch.stack(ood_ds_list, dim=0)
            
            test_ds_mean = test_ds_stacked.mean(dim=0).numpy()
            ood_ds_mean = ood_ds_stacked.mean(dim=0).numpy()
            
            ax.hist(test_ds_mean, bins=50, alpha=0.6, color='blue', label='Test (ID)', density=True)
            ax.hist(ood_ds_mean, bins=50, alpha=0.6, color='red', label='OOD', density=True)
            ax.set_xlabel('Dempster-Shafer Score', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'n={n}', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_test_accuracies_with_abstention(predictions, best_configs):
    """Plot test accuracy at different abstention thresholds for each dataset size."""
    percentiles = np.linspace(0, 100, 21)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(NS):
        ax = axes[idx]
        
        for (n_key, rs), preds in predictions.items():
            if n_key != n:
                continue
            
            probs = preds['test_probs']
            logits = preds['test_logits']
            labels = preds['test_labels']  # Use real labels!
            ds_scores = get_ds_scores(logits)
            
            accuracies = [apply_abstention_threshold(probs, logits, labels, ds_scores, p) 
                         for p in percentiles]
            
            ax.plot(percentiles, accuracies, alpha=0.6, label=f'rs={rs}')
        
        ax.set_xlabel('Percentile Threshold', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title(f'n={n}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_ood_accuracy(predictions, best_configs):
    """Plot OOD accuracies vs dataset size with min/max bands."""
    results = get_accuracies_by_n(predictions, best_configs, split='ood')
    
    ns = sorted(results.keys())
    means = [np.mean(results[n]) for n in ns]
    mins = [np.min(results[n]) for n in ns]
    maxs = [np.max(results[n]) for n in ns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, means, 'o-', linewidth=2, markersize=8)
    ax.fill_between(ns, mins, maxs, alpha=0.3)
    ax.set_xlabel('Dataset Size (n)', fontsize=12)
    ax.set_ylabel('OOD Accuracy', fontsize=12)
    ax.set_title('OOD Accuracy vs Dataset Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_ood_accuracies_with_abstention(predictions, best_configs):
    """Plot OOD accuracy at different abstention thresholds for each dataset size."""
    percentiles = np.linspace(0, 100, 21)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(NS):
        ax = axes[idx]
        
        for (n_key, rs), preds in predictions.items():
            if n_key != n:
                continue
            
            probs = preds['ood_probs']
            logits = preds['ood_logits']
            labels = preds['ood_labels']
            ds_scores = get_ds_scores(logits)
            
            accuracies = [apply_abstention_threshold(probs, logits, labels, ds_scores, p) 
                         for p in percentiles]
            
            ax.plot(percentiles, accuracies, alpha=0.6, label=f'rs={rs}')
        
        ax.set_xlabel('Percentile Threshold', fontsize=11)
        ax.set_ylabel('OOD Accuracy', fontsize=11)
        ax.set_title(f'n={n}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_curves_and_auroc(predictions):
    """Plot ROC curves for test (label 0) vs ood (label 1) using DS scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(NS):
        ax = axes[idx]
        
        for (n_key, rs), preds in predictions.items():
            if n_key != n:
                continue
            
            test_ds = get_ds_scores(preds['test_logits'])
            ood_ds = get_ds_scores(preds['ood_logits'])
            
            # Combine: test=0, ood=1
            ds_scores = torch.cat([test_ds, ood_ds]).numpy()
            labels = np.concatenate([np.zeros(len(test_ds)), np.ones(len(ood_ds))])
            
            fpr, tpr, _ = roc_curve(labels, ds_scores)
            auroc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, alpha=0.7, label=f'rs={rs} (AUROC={auroc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'n={n}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_all_plots(predictions, best_configs, save_dir=BASE_DIR):
    """Generate and save all plots."""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    figs = {
        'test_accuracies.png': plot_test_accuracies(predictions, best_configs),
        'abstention_distributions.png': plot_test_and_ood_abstention(predictions),
        'test_accuracies_with_abstention.png': plot_test_accuracies_with_abstention(predictions, best_configs),
        'ood_accuracy.png': plot_ood_accuracy(predictions, best_configs),
        'ood_accuracies_with_abstention.png': plot_ood_accuracies_with_abstention(predictions, best_configs),
        'roc_curves.png': plot_roc_curves_and_auroc(predictions),
    }
    
    for filename, fig in figs.items():
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return plots_dir


def main():
    all_metrics = load_all_metrics()
    best_configs = select_best_configs(all_metrics)
    predictions = load_best_predictions(best_configs)
    plots_dir = save_all_plots(predictions, best_configs)
    
    return all_metrics, best_configs, predictions


if __name__ == "__main__":
    all_metrics, best_configs, predictions = main()