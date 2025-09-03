#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot5.py

This script creates two plots from the results of main5.py:
1. Violin plot for each method showing MSE distribution
2. Plot showing the effect of robust parameter theta on averaged MSE

Usage:
    python plot5.py --dist normal
    python plot5.py --dist quadratic
    python plot5.py --dist laplace
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from matplotlib.patches import Rectangle

# Set larger font sizes for better readability
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 12,     # Legend font size
    'figure.titlesize': 18     # Figure title size
})

def load_data(file_path):
    """Load pickled data from file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_violin_plot(all_results, raw_experiments_data, optimal_results, dist, filters, filter_labels):
    """Create violin plot for each method showing MSE distribution using best parameters"""
    
    # Collect MSE data for each filter using their optimal robust parameters
    mse_data = {filt: [] for filt in filters}
    
    for filt in filters:
        if filt in ['finite', 'inf']:
            # Non-robust methods: use data from first robust parameter
            robust_val = list(raw_experiments_data.keys())[0]
            experiments = raw_experiments_data[robust_val]
            for exp in experiments:
                if filt in exp:
                    mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                    mse_data[filt].extend(mse_values)
        else:
            # Robust methods: use data from optimal robust parameter
            optimal_theta = optimal_results[filt]['robust_val']
            if optimal_theta in raw_experiments_data:
                experiments = raw_experiments_data[optimal_theta]
                for exp in experiments:
                    if filt in exp:
                        mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                        mse_data[filt].extend(mse_values)
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Prepare data for violin plot - only include filters with data
    active_filters = [filt for filt in filters if mse_data[filt]]
    violin_data = [mse_data[filt] for filt in active_filters]
    violin_labels = [filter_labels[filt] for filt in active_filters]
    
    # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(len(active_filters)), showmeans=True, showmedians=True)
    
    # Customize violin plot colors with specific colors for certain methods
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'  # Changed from red to orange
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    for i, (pc, filt) in enumerate(zip(parts['bodies'], active_filters)):
        color = get_color_for_filter(filt, i)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xticks(range(len(active_filters)))
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'violin_plot_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved as: {output_path}")

def create_regret_violin_plot(all_results, raw_experiments_data, optimal_regret_results, dist, filters, filter_labels):
    """Create violin plot for each method showing regret distribution using best regret parameters"""
    
    # First, we need to reconstruct regret values from MSE and baseline MSE for each raw experiment
    regret_data = {filt: [] for filt in filters}
    
    for filt in filters:
        if filt in ['finite', 'inf']:
            # Non-robust methods: use data from first robust parameter
            robust_val = list(raw_experiments_data.keys())[0]
            experiments = raw_experiments_data[robust_val]
            # Get baseline MSE from this parameter set
            baseline_mse_vals = []
            for exp in experiments:
                if 'mmse_baseline' in exp:
                    baseline_mse_vals.extend([np.mean(sim['mse']) for sim in exp['mmse_baseline']])
            
            for exp in experiments:
                if filt in exp and 'mmse_baseline' in exp:
                    mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                    baseline_values = [np.mean(sim['mse']) for sim in exp['mmse_baseline']]
                    regret_values = [mse - baseline for mse, baseline in zip(mse_values, baseline_values)]
                    regret_data[filt].extend(regret_values)
        else:
            # Robust methods: use data from optimal regret parameter
            optimal_theta = optimal_regret_results[filt]['robust_val']
            if optimal_theta in raw_experiments_data:
                experiments = raw_experiments_data[optimal_theta]
                for exp in experiments:
                    if filt in exp and 'mmse_baseline' in exp:
                        mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                        baseline_values = [np.mean(sim['mse']) for sim in exp['mmse_baseline']]
                        regret_values = [mse - baseline for mse, baseline in zip(mse_values, baseline_values)]
                        regret_data[filt].extend(regret_values)
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Prepare data for violin plot
    violin_data = [regret_data[filt] for filt in filters if regret_data[filt]]
    violin_labels = [filter_labels[filt] for filt in filters if regret_data[filt]]
    
    # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(len(violin_labels)), showmeans=True, showmedians=True)
    
    # Customize violin plot colors with specific colors for certain methods
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    active_filters = [filt for filt in filters if regret_data[filt]]
    for i, (pc, filt) in enumerate(zip(parts['bodies'], active_filters)):
        color = get_color_for_filter(filt, i)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add horizontal line at y=0 (perfect regret)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, label='MMSE Baseline (Regret = 0)')
    
    # Set labels and title
    ax.set_xticks(range(len(violin_labels)))
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    ax.set_ylabel('Regret (MSE - MMSE Baseline)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'regret_violin_plot_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regret violin plot saved as: {output_path}")

def create_theta_effect_plot(all_results, dist, filters, filter_labels):
    """Create plot showing effect of robust parameter theta on averaged MSE"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract robust values and sort them
    robust_vals = sorted(all_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    
    # Define color function for consistent coloring
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'  # Changed from red to orange
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    # Define letter labels (A) to (I)
    letter_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']
    
    # Plot each filter
    for i, filt in enumerate(filters):
        if filt in ['finite', 'inf']:
            # For non-robust methods, plot horizontal line
            mse_vals = [all_results[robust_vals[0]]['mse'][filt]] * len(robust_vals)
            mse_stds = [all_results[robust_vals[0]]['mse_std'][filt]] * len(robust_vals)
            label = f"{letter_labels[i]} {filter_labels[filt]}"  # Remove (Non-robust) and add letter
            linestyle = '-'
        else:
            # For robust methods, plot actual theta effect
            mse_vals = [all_results[rv]['mse'][filt] for rv in robust_vals]
            mse_stds = [all_results[rv]['mse_std'][filt] for rv in robust_vals]
            label = f"{letter_labels[i]} {filter_labels[filt]}"  # Add letter label
            linestyle = '-'
        
        # Plot without error bars
        # For non-robust methods, draw horizontal line without markers
        if filt in ['finite', 'inf']:
            ax.plot(robust_vals, mse_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    label=label)
        else:
            ax.plot(robust_vals, mse_vals, 
                    marker=markers[i % len(markers)], 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=8,
                    label=label)
    
    # Customize plot
    ax.set_xlabel('θ')
    ax.set_ylabel('Average Mean Squared Error (MSE)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'theta_effect_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Theta effect plot saved as: {output_path}")

def create_regret_theta_effect_plot(all_results, dist, filters, filter_labels):
    """Create plot showing effect of robust parameter theta on regret (MSE difference from MMSE baseline)"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract robust values and sort them
    robust_vals = sorted(all_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    
    # Define color function for consistent coloring
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    # Define letter labels (A) to (I)
    letter_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']
    
    # Plot each filter
    for i, filt in enumerate(filters):
        if filt in ['finite', 'inf']:
            # For non-robust methods, plot horizontal line
            regret_vals = [all_results[robust_vals[0]]['regret'][filt]] * len(robust_vals)
            regret_stds = [all_results[robust_vals[0]]['regret_std'][filt]] * len(robust_vals)
            label = f"{letter_labels[i]} {filter_labels[filt]}"
            linestyle = '-'
        else:
            # For robust methods, plot actual theta effect
            regret_vals = [all_results[rv]['regret'][filt] for rv in robust_vals]
            regret_stds = [all_results[rv]['regret_std'][filt] for rv in robust_vals]
            label = f"{letter_labels[i]} {filter_labels[filt]}"
            linestyle = '-'
        
        # Plot without error bars
        # For non-robust methods, draw horizontal line without markers
        if filt in ['finite', 'inf']:
            ax.plot(robust_vals, regret_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    label=label)
        else:
            ax.plot(robust_vals, regret_vals, 
                    marker=markers[i % len(markers)], 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=8,
                    label=label)
    
    # Add horizontal line at y=0 (perfect regret)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, label='MMSE Baseline (Regret = 0)')
    
    # Customize plot
    ax.set_xlabel('θ')
    ax.set_ylabel('Average Regret (MSE - MMSE Baseline)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'regret_theta_effect_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regret theta effect plot saved as: {output_path}")

def create_optimal_comparison_plot(optimal_results, dist, filter_labels):
    """Create bar plot comparing optimal MSE for each method"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    filters = list(optimal_results.keys())
    mse_vals = [optimal_results[filt]['mse'] for filt in filters]
    mse_stds = [optimal_results[filt]['mse_std'] for filt in filters]
    labels = [filter_labels[filt] for filt in filters]
    
    # Define color function for consistent coloring
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'  # Changed from red to orange
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    # Create bar plot with custom colors
    bar_colors = [get_color_for_filter(filt, i) for i, filt in enumerate(filters)]
    bars = ax.bar(range(len(filters)), mse_vals, yerr=mse_stds, 
                  color=bar_colors, alpha=0.7, capsize=4)
    
    # Add optimal theta values as text on bars
    for i, (bar, filt) in enumerate(zip(bars, filters)):
        height = bar.get_height()
        theta_val = optimal_results[filt]['robust_val']
        ax.text(bar.get_x() + bar.get_width()/2., height + mse_stds[i],
                f'θ={theta_val}',
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(filters)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optimal Mean Squared Error (MSE)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'optimal_comparison_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal comparison plot saved as: {output_path}")

def create_optimal_regret_comparison_plot(optimal_regret_results, dist, filter_labels):
    """Create bar plot comparing optimal regret for each method"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    filters = list(optimal_regret_results.keys())
    regret_vals = [optimal_regret_results[filt]['regret'] for filt in filters]
    regret_stds = [optimal_regret_results[filt]['regret_std'] for filt in filters]
    labels = [filter_labels[filt] for filt in filters]
    
    # Define color function for consistent coloring
    def get_color_for_filter(filt, i):
        if filt == 'drkf_inf':
            return 'green'
        elif filt == 'risk':
            return 'orange'
        else:
            # Use tab10 colormap for other methods
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            return colors[i % len(colors)]
    
    # Create bar plot with custom colors
    bar_colors = [get_color_for_filter(filt, i) for i, filt in enumerate(filters)]
    bars = ax.bar(range(len(filters)), regret_vals, yerr=regret_stds, 
                  color=bar_colors, alpha=0.7, capsize=4)
    
    # Add optimal theta values as text on bars
    for i, (bar, filt) in enumerate(zip(bars, filters)):
        height = bar.get_height()
        theta_val = optimal_regret_results[filt]['robust_val']
        # Position text above or below bar depending on regret sign
        text_y = height + regret_stds[i] if height >= 0 else height - regret_stds[i]
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
                f'θ={theta_val}',
                ha='center', va=va, fontsize=8)
    
    # Add horizontal line at y=0 (perfect regret)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, label='MMSE Baseline (Regret = 0)')
    
    ax.set_xticks(range(len(filters)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optimal Regret (MSE - MMSE Baseline)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'optimal_regret_comparison_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal regret comparison plot saved as: {output_path}")

def main(dist):
    """Main function to create all plots"""
    
    results_path = "./results/estimation/"
    
    # Load results
    try:
        all_results = load_data(os.path.join(results_path, f'overall_results_{dist}_estimation.pkl'))
        optimal_results = load_data(os.path.join(results_path, f'optimal_results_{dist}_estimation.pkl'))
        optimal_regret_results = load_data(os.path.join(results_path, f'optimal_regret_results_{dist}_estimation.pkl'))
        raw_experiments_data = load_data(os.path.join(results_path, f'raw_experiments_{dist}_estimation.pkl'))
    except FileNotFoundError as e:
        print(f"Error: Could not find results file. Make sure you've run main5.py first.")
        print(f"Missing file: {e}")
        return
    
    # Get filters from the loaded results to match main5.py execution list
    available_filters = ['finite', 'inf', 'risk', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    # Only use filters that have results in the data
    filters = [f for f in available_filters if f in optimal_results]
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'risk': "Risk-Sensitive Filter",
        'drkf_neurips': "DRKF (NeurIPS)",
        'bcot': "BCOT",
        'drkf_finite_cdc': "DRKF (ours, finite, CDC)",
        'drkf_inf_cdc': "DRKF (ours, inf, CDC)",
        'drkf_finite': "DRKF (ours, finite)",
        'drkf_inf': "DRKF (ours, inf)"
    }
    
    print(f"Creating plots for {dist} distribution...")
    
    # Create all plots (original MSE-based plots)
    create_violin_plot(all_results, raw_experiments_data, optimal_results, dist, filters, filter_labels)
    create_theta_effect_plot(all_results, dist, filters, filter_labels)
    create_optimal_comparison_plot(optimal_results, dist, filter_labels)
    
    # Create regret-based plots
    create_regret_violin_plot(all_results, raw_experiments_data, optimal_regret_results, dist, filters, filter_labels)
    create_regret_theta_effect_plot(all_results, dist, filters, filter_labels)
    create_optimal_regret_comparison_plot(optimal_regret_results, dist, filter_labels)
    
    print(f"All plots (MSE and Regret) created successfully for {dist} distribution!")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # MSE-based rankings
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['mse'])
    print(f"\nRanking by optimal MSE ({dist} distribution):")
    for i, (filt, info) in enumerate(sorted_optimal, 1):
        print(f"{i:2d}. {filter_labels[filt]:<30} MSE: {info['mse']:.4f} (±{info['mse_std']:.4f}) θ: {info['robust_val']}")
    
    # Show improvement over baseline
    baseline_mse = sorted_optimal[0][1]['mse']  # Best performing method
    print(f"\nMSE comparison to best method ({filter_labels[sorted_optimal[0][0]]}):")
    for filt, info in sorted_optimal[1:]:
        improvement = ((info['mse'] - baseline_mse) / baseline_mse) * 100
        print(f"  {filter_labels[filt]:<30} +{improvement:5.1f}% worse")
    
    # Regret-based rankings
    sorted_optimal_regret = sorted(optimal_regret_results.items(), key=lambda item: item[1]['regret'])
    print(f"\nRanking by optimal Regret ({dist} distribution):")
    for i, (filt, info) in enumerate(sorted_optimal_regret, 1):
        print(f"{i:2d}. {filter_labels[filt]:<30} Regret: {info['regret']:.4f} (±{info['regret_std']:.4f}) θ: {info['robust_val']}")
    
    # Show regret comparison - baseline is always 0 (MMSE baseline has perfect regret)
    print(f"\nRegret comparison to MMSE Baseline (Regret = 0):")
    for filt, info in sorted_optimal_regret:
        regret_value = info['regret']
        print(f"  {filter_labels[filt]:<30} Regret: {regret_value:+6.4f}")
    
    # Show differences between MSE-optimal and Regret-optimal theta values
    print(f"\nComparison of optimal θ values (MSE vs Regret optimization):")
    print(f"{'Method':<30} {'MSE-optimal θ':<15} {'Regret-optimal θ':<15} {'Same?':<10}")
    print("-" * 70)
    for filt in filters:
        if filt in optimal_results and filt in optimal_regret_results:
            mse_theta = optimal_results[filt]['robust_val']
            regret_theta = optimal_regret_results[filt]['robust_val']
            same = "Yes" if mse_theta == regret_theta else "No"
            print(f"{filter_labels[filt]:<30} {str(mse_theta):<15} {str(regret_theta):<15} {same:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create plots from main5.py results")
    parser.add_argument('--dist', default="normal", type=str,
                        help="Distribution type (normal, quadratic, or laplace)")
    
    args = parser.parse_args()
    main(args.dist)