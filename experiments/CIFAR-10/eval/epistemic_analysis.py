#!/usr/bin/env python3
"""Compute epistemic uncertainty vs dataset size across multiple methods."""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../../src")
import metrics


METHODS = [
    {
        "base_dir": "../../../results/CIFAR-10/linear_results",
        "label": "Linear",
        "color": "tab:blue",
        "load_style": "linear",
    },
    {
        "base_dir": "../../../results/CIFAR-10/laplace_results",
        "label": "RFF + Laplace",
        "color": "tab:red",
        "load_style": "laplace",
    },
]

NS = [100, 1000, 10000, 50000]
RANDOM_STATES = [1001, 2001, 3001]
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
RANK = 1024
LENGTHSCALE = 20.0
OUTPUTSCALE = 1.0
SAVE_DIR = "/cluster/tufts/hugheslab/swilli26/RFF_PAPER/random-Fourier-features/experiments/cifar10/combined_plots"

def get_run_dir(base_dir, n, rs, lr, load_style):
    if load_style == "linear":
        return f"{base_dir}/n_{n}/rs_{rs}/lr_{lr}"
    elif load_style == "laplace":
        return f"{base_dir}/n_{n}/rs_{rs}/rank_{RANK}-ls_{LENGTHSCALE}-os_{OUTPUTSCALE}-lr_{lr}"
    else:
        raise ValueError(f"Unknown load_style: {load_style}")


def load_best_predictions(base_dir, load_style):
    predictions = {}
    best_configs = {}

    for n in NS:
        for rs in RANDOM_STATES:
            best_lr, best_val_acc, best_dir, best_metrics = None, -1, None, None

            for lr in LEARNING_RATES:
                run_dir = get_run_dir(base_dir, n, rs, lr, load_style)
                metrics_path = os.path.join(run_dir, "metrics.json")
                if not os.path.exists(metrics_path):
                    continue
                with open(metrics_path) as f:
                    m = json.load(f)
                val_acc = m['val']['balanced_accuracy']
                if val_acc > best_val_acc:
                    best_lr, best_val_acc, best_dir, best_metrics = lr, val_acc, run_dir, m

            if best_dir is not None:
                pred_path = os.path.join(best_dir, "predictions.pth")
                if os.path.exists(pred_path):
                    predictions[(n, rs)] = torch.load(pred_path, map_location='cpu')
                    best_configs[(n, rs)] = best_metrics
                    print(f"  n={n}, rs={rs}: best_lr={best_lr}, val_bal_acc={best_val_acc:.4f}")

    return predictions, best_configs


def compute_epistemic_uncertainties(predictions, split='test'):
    results = {n: {'ds': [], 'mi': [], 'var': []} for n in NS}

    for (n, rs), preds in predictions.items():
        key = f'{split}_logits'
        if key not in preds:
            continue
        logits = preds[key]

        ds = metrics.subjective_logic_epistemic_uncertainty(logits).mean().item()
        mi = metrics.mutual_information_epistemic_uncertainty(logits).mean().item()
        var = metrics.variance_based_epistemic_uncertainty(logits).mean().item()

        results[n]['ds'].append(ds)
        results[n]['mi'].append(mi)
        results[n]['var'].append(var)

    return results


def get_test_accuracies(best_configs):
    results = {n: [] for n in NS}
    for (n, rs), m in best_configs.items():
        results[n].append(m['test']['balanced_accuracy'])
    return results


def plot_method_on_axes(axes, ns, vals_by_n, color, label, marker='o'):
    valid_ns = [n for n in ns if len(vals_by_n[n]) > 0]
    if not valid_ns:
        return
    means = [np.mean(vals_by_n[n]) for n in valid_ns]
    mins = [np.min(vals_by_n[n]) for n in valid_ns]
    maxs = [np.max(vals_by_n[n]) for n in valid_ns]

    for ax_target in (axes if isinstance(axes, list) else [axes]):
        ax_target.plot(valid_ns, means, f'{marker}-', linewidth=2, markersize=8, color=color, label=label)
        ax_target.fill_between(valid_ns, mins, maxs, alpha=0.15, color=color)
        for seed_vals in zip(*[vals_by_n[n] for n in valid_ns]):
            ax_target.plot(valid_ns, seed_vals, 'x', alpha=0.4, markersize=6, color=color)


def plot_all_methods(all_method_data, split='test'):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    measures = [
        ('acc', 'Test Balanced Accuracy', 'Test Accuracy'),
        ('ds', 'Epistemic Uncertainty', 'Dempster-Shafer (u = K/S)'),
        ('mi', 'Epistemic Uncertainty', 'Mutual Information'),
        ('var', 'Epistemic Uncertainty', 'Variance of Expected'),
    ]

    for method_info, (uq_results, acc_results) in all_method_data.items():
        label = method_info[0]
        color = method_info[1]

        # Accuracy
        plot_method_on_axes(axes[0], NS, acc_results, color, label)

        # UQ measures
        for idx, (key, _, _) in enumerate(measures[1:]):
            vals_by_n = {n: uq_results[n][key] for n in NS}
            plot_method_on_axes(axes[idx + 1], NS, vals_by_n, color, label)

    for idx, (_, ylabel, title) in enumerate(measures):
        axes[idx].set_xlabel('Dataset Size (n)', fontsize=12)
        axes[idx].set_ylabel(ylabel, fontsize=12)
        axes[idx].set_title(title, fontsize=13)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()

    fig.suptitle(f'Test Accuracy & Epistemic Uncertainty vs Dataset Size ({split})', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load all methods
    all_predictions = {}
    all_best_configs = {}
    for method in METHODS:
        print(f"\nLoading: {method['label']} from {method['base_dir']}")
        preds, configs = load_best_predictions(method['base_dir'], method['load_style'])
        all_predictions[method['label']] = preds
        all_best_configs[method['label']] = configs

    # Plot per split
    for split in ['test', 'ood']:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        # key = (label, color), value = (uq_results, acc_results)
        all_method_data = {}
        for method in METHODS:
            label = method['label']
            color = method['color']

            uq_results = compute_epistemic_uncertainties(all_predictions[label], split=split)
            acc_results = get_test_accuracies(all_best_configs[label])

            all_method_data[(label, color)] = (uq_results, acc_results)

        fig = plot_all_methods(all_method_data, split=split)
        save_path = os.path.join(SAVE_DIR, f'epistemic_vs_n_{split}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {save_path}")