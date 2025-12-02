#!/usr/bin/env python3
"""Create visualizations for readability scores."""
import os
from pathlib import Path
import logging
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data_processed"
READ = BASE / "readability"
LOG = BASE / "reports"
FIG = BASE / "figures"

READ.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
# Create subdirectories for different plot types
FIG_VIOLIN = FIG / 'violin'
FIG_DENSITY = FIG / 'density'
FIG_MEAN = FIG / 'mean_comparison'
FIG_HEATMAP = FIG / 'heatmap'
FIG_VIOLIN.mkdir(parents=True, exist_ok=True)
FIG_DENSITY.mkdir(parents=True, exist_ok=True)
FIG_MEAN.mkdir(parents=True, exist_ok=True)
FIG_HEATMAP.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log("[plot_readability] START")
    
    # Read readability scores
    read_csv = READ / 'readability_scores.csv'
    if not read_csv.exists():
        append_log(f"[plot_readability] {read_csv} not found")
        return
    
    scores_df = pd.read_csv(read_csv)
    append_log(f"[plot_readability] loaded {len(scores_df)} readability scores")
    
    # Load labels
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[plot_readability] source data not found for labels")
        return
    
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)
    
    # Merge with labels
    merged = scores_df.merge(df[['_id','conspiracy']], on='_id', how='left') if '_id' in df.columns else scores_df
    if 'conspiracy' not in merged.columns:
        append_log("[plot_readability] no conspiracy labels found, creating plots without labels")
        merged['conspiracy'] = 'unknown'
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        
        metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'smog', 'ari', 'coleman_liau']
        metric_labels = {
            'flesch_reading_ease': 'Flesch Reading Ease',
            'flesch_kincaid_grade': 'Flesch-Kincaid Grade',
            'gunning_fog': 'Gunning Fog Index',
            'smog': 'SMOG Index',
            'ari': 'Automated Readability Index',
            'coleman_liau': 'Coleman-Liau Index'
        }
        
        # 1. Violin plots comparing distributions by label
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                ax = axes[idx]
                data_to_plot = []
                labels_to_plot = []
                
                for label in ['yes', 'no', 'cant_tell']:
                    subset = merged[merged['conspiracy'] == label][metric].dropna()
                    if len(subset) > 0:
                        data_to_plot.append(subset)
                        labels_to_plot.append(label)
                
                if data_to_plot:
                    parts = ax.violinplot(data_to_plot, positions=range(len(labels_to_plot)), 
                                         showmeans=True, showmedians=True)
                    # Color the violins
                    colors = ['#e74c3c', '#3498db', '#95a5a6']  # red, blue, gray
                    for pc, color in zip(parts['bodies'], colors[:len(labels_to_plot)]):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    ax.set_xticks(range(len(labels_to_plot)))
                    ax.set_xticklabels(labels_to_plot)
                    ax.set_ylabel(metric_labels[metric], fontsize=11)
                    ax.set_title(f'{metric_labels[metric]} by Label', fontsize=12, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
        
        # Note: Old grid plot - individual violin plots should be created separately
        # This grid plot is kept for backward compatibility but consider removing
        plt.tight_layout()
        plt.savefig(FIG_VIOLIN / 'readability_violin_plots_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        append_log("[plot_readability] saved readability_violin_plots_grid.png")
        
        # 2. Density plots overlaying distributions - separate plot for each metric
        for metric in metrics:
            # Create a separate figure for each metric
            fig, ax = plt.subplots(figsize=(10, 7))
            
            for label, color in zip(['yes', 'no', 'cant_tell'], ['#e74c3c', '#3498db', '#95a5a6']):
                subset = merged[merged['conspiracy'] == label][metric].dropna()
                if len(subset) > 0:
                    sns.kdeplot(data=subset, ax=ax, label=label, color=color, fill=True, alpha=0.5, linewidth=2)
            
            ax.set_xlabel(metric_labels[metric], fontsize=13, fontweight='bold')
            ax.set_ylabel('Density', fontsize=13, fontweight='bold')
            ax.set_title(f'{metric_labels[metric]} Distribution by Label', fontsize=14, fontweight='bold', pad=15)
            ax.legend(title='Label', fontsize=11, title_fontsize=12)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save individual plot to density directory
            metric_safe = metric.replace('_', '_')
            plot_filename = FIG_DENSITY / f'readability_{metric}_density.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            append_log(f"[plot_readability] saved {plot_filename}")
        
        append_log("[plot_readability] saved all individual readability density plots")
        
        # 3. Correlation heatmap between readability metrics
        numeric_data = merged[metrics].dropna()
        if len(numeric_data) > 0:
            corr_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Readability Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            # Improve labels
            ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], rotation=45, ha='right')
            ax.set_yticklabels([metric_labels.get(m, m) for m in metrics], rotation=0)
            plt.tight_layout()
            plt.savefig(FIG_HEATMAP / 'readability_correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            append_log("[plot_readability] saved readability_correlation_heatmap.png")
        
        # 4. Summary statistics comparison by label
        summary_stats = []
        for label in ['yes', 'no', 'cant_tell']:
            subset = merged[merged['conspiracy'] == label][metrics].dropna()
            if len(subset) > 0:
                for metric in metrics:
                    summary_stats.append({
                        'label': label,
                        'metric': metric_labels[metric],
                        'mean': subset[metric].mean(),
                        'median': subset[metric].median(),
                        'std': subset[metric].std()
                    })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(READ / 'readability_summary_by_label.csv', index=False)
            append_log("[plot_readability] saved readability_summary_by_label.csv")
            
            # Create bar plot comparing means - separate plot for each metric
            for metric in metrics:
                metric_label = metric_labels[metric]
                metric_summary = summary_df[summary_df['metric'] == metric_label]
                
                if len(metric_summary) > 0:
                    # Create a separate figure for each metric
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    x_pos = np.arange(len(metric_summary))
                    colors_list = []
                    for label in metric_summary['label']:
                        if label == 'yes':
                            colors_list.append('#e74c3c')
                        elif label == 'no':
                            colors_list.append('#3498db')
                        else:
                            colors_list.append('#95a5a6')
                    
                    bars = ax.bar(x_pos, metric_summary['mean'], 
                                 yerr=metric_summary['std'],
                                 capsize=8, alpha=0.8, edgecolor='black', linewidth=1.2,
                                 color=colors_list)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(metric_summary['label'], fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'Mean {metric_label}', fontsize=13, fontweight='bold')
                    ax.set_title(f'{metric_label} - Mean ± Std by Label', fontsize=14, fontweight='bold', pad=15)
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save individual plot to mean_comparison directory
                    metric_safe = metric.replace('_', '_')
                    plot_filename = FIG_MEAN / f'readability_{metric}_mean_comparison.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    append_log(f"[plot_readability] saved {plot_filename}")
            
            append_log("[plot_readability] saved all individual readability mean comparison plots")
        
        append_log("[plot_readability] All plots generated successfully")
            
    except Exception as e:
        append_log(f"[plot_readability] plotting failed: {e}")
        import traceback
        append_log(traceback.format_exc())
    
    append_log("[plot_readability] END")

if __name__ == '__main__':
    main()

