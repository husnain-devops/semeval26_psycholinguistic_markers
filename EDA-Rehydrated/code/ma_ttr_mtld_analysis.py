#!/usr/bin/env python3
"""Compute MA-TTR (Moving Average Type-Token Ratio) and MTLD (Measure of Textual Lexical Diversity) metrics."""
import os
from pathlib import Path
import logging
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data_processed"
FEAT = BASE / "features"
LOG = BASE / "reports"
FIG = BASE / "figures"

FEAT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
# Create subdirectories for different plot types
FIG_VIOLIN = FIG / 'violin'
FIG_DENSITY = FIG / 'density'
FIG_MEAN = FIG / 'mean_comparison'
FIG_VIOLIN.mkdir(parents=True, exist_ok=True)
FIG_DENSITY.mkdir(parents=True, exist_ok=True)
FIG_MEAN.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log("[ma_ttr_mtld] START")
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        # fallback to csv
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[ma_ttr_mtld] data not found")
        return
    
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)
    
    # Check if lexicalrichness is available
    try:
        from lexicalrichness import LexicalRichness
        has_lexicalrichness = True
    except Exception as e:
        append_log(f"lexicalrichness not available: {e}")
        has_lexicalrichness = False
    
    rows = []
    for _, r in df.iterrows():
        text = str(r.get('text') or '')
        rec = {'_id': r.get('_id')}
        
        if has_lexicalrichness and text.strip():
            try:
                # Create LexicalRichness object
                lex = LexicalRichness(text)
                
                # Compute MA-TTR (Moving Average Type-Token Ratio)
                # MA-TTR uses a window-based approach to compute TTR
                # Use adaptive window size based on text length
                try:
                    word_count = lex.words
                    if word_count < 10:
                        # Text too short for MATTR
                        ma_ttr = None
                    else:
                        # Adaptive window size: use smaller of 50, or 20% of text length, minimum 10
                        adaptive_window = max(10, min(50, int(word_count * 0.2)))
                        ma_ttr = lex.mattr(window_size=adaptive_window)
                except Exception:
                    # If still fails, try with minimum window size
                    try:
                        word_count = lex.words
                        if word_count >= 10:
                            ma_ttr = lex.mattr(window_size=10)
                        else:
                            ma_ttr = None
                    except Exception:
                        ma_ttr = None
                
                # Compute MTLD (Measure of Textual Lexical Diversity)
                # MTLD is a measure that calculates lexical diversity by computing 
                # the mean length of sequential word strings that maintain a given TTR threshold
                try:
                    mtld = lex.mtld(threshold=0.72)
                except Exception:
                    # If threshold fails, try default
                    try:
                        mtld = lex.mtld()
                    except Exception:
                        mtld = None
                
                rec['ma_ttr'] = ma_ttr
                rec['mtld'] = mtld
                
            except Exception as e:
                append_log(f"Error processing {r.get('_id')}: {e}")
                rec['ma_ttr'] = None
                rec['mtld'] = None
        else:
            rec['ma_ttr'] = None
            rec['mtld'] = None
        
        rows.append(rec)
    
    out = pd.DataFrame(rows)
    outp = FEAT / 'ma_ttr_mtld_scores.csv'
    out.to_csv(outp, index=False)
    append_log(f"[ma_ttr_mtld] saved {outp}")
    
    # Merge with labels for analysis
    merged = out.merge(df[['_id', 'conspiracy']], on='_id', how='left') if '_id' in df.columns else out
    
    # Create summary statistics by label
    try:
        summary_csv = FEAT / 'ma_ttr_mtld_summary_by_label.csv'
        summary_json = FEAT / 'ma_ttr_mtld_summary_by_label.json'
        
        # Save descriptive statistics
        summary = merged.groupby('conspiracy').describe()
        summary.to_csv(summary_csv)
        
        # Save aggregated stats as JSON
        metrics = ['ma_ttr', 'mtld']
        numeric_cols = [col for col in metrics if col in merged.columns and merged[col].dtype in [float, int]]
        if numeric_cols:
            grp_df = merged.groupby('conspiracy')[numeric_cols].agg(['mean', 'std'])
            # Flatten MultiIndex columns for JSON serialization
            grp_df.columns = [f"{col}_{stat}" for col, stat in grp_df.columns]
            grp_dict = grp_df.to_dict(orient='index')
            with open(summary_json, 'w', encoding='utf-8') as f:
                json.dump(grp_dict, f, indent=2)
        
        append_log(f"[ma_ttr_mtld] saved summary csv/json")
        
        # Create visualizations
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 150
            
            metrics = ['ma_ttr', 'mtld']
            metric_labels = {
                'ma_ttr': 'MA-TTR (Moving Average Type-Token Ratio)',
                'mtld': 'MTLD (Measure of Textual Lexical Diversity)'
            }
            
            colors = {'yes': '#e74c3c', 'no': '#3498db', 'cant_tell': '#95a5a6'}
            
            # 1. Violin plots for each metric
            for metric in metrics:
                if metric not in merged.columns:
                    continue
                
                # Create a separate figure for each metric
                fig, ax = plt.subplots(figsize=(8, 6))
                
                data_for_plot = []
                labels_for_plot = []
                color_list = []
                
                for label in ['yes', 'no', 'cant_tell']:
                    subset = merged[merged['conspiracy'] == label][metric].dropna()
                    if len(subset) > 0:
                        data_for_plot.append(subset)
                        labels_for_plot.append(label)
                        color_list.append(colors[label])
                
                if data_for_plot:
                    # Create violin plot
                    parts = ax.violinplot(data_for_plot, positions=range(len(labels_for_plot)), 
                                         showmeans=True, showmedians=True)
                    
                    # Color the violins
                    for pc, color in zip(parts['bodies'], color_list):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    # Customize the plot
                    ax.set_xticks(range(len(labels_for_plot)))
                    ax.set_xticklabels(labels_for_plot, fontsize=12)
                    ax.set_ylabel(metric_labels[metric], fontsize=13, fontweight='bold')
                    ax.set_title(f'{metric_labels[metric]} by Label', fontsize=14, fontweight='bold', pad=15)
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save individual plot to violin directory
                    plot_filename = FIG_VIOLIN / f'{metric}_by_label.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    append_log(f"[ma_ttr_mtld] saved {plot_filename}")
            
            # 2. Density plots overlaying distributions
            for metric in metrics:
                if metric not in merged.columns:
                    continue
                
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
                plot_filename = FIG_DENSITY / f'{metric}_density.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                append_log(f"[ma_ttr_mtld] saved {plot_filename}")
            
            # 3. Mean comparison bar plots
            summary_stats = []
            for label in ['yes', 'no', 'cant_tell']:
                subset = merged[merged['conspiracy'] == label][metrics].dropna()
                if len(subset) > 0:
                    for metric in metrics:
                        if metric in subset.columns:
                            summary_stats.append({
                                'label': label,
                                'metric': metric_labels[metric],
                                'mean': subset[metric].mean(),
                                'median': subset[metric].median(),
                                'std': subset[metric].std()
                            })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                
                # Create bar plots for each metric
                for metric in metrics:
                    if metric not in merged.columns:
                        continue
                    
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
                        plot_filename = FIG_MEAN / f'{metric}_mean_comparison.png'
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                        append_log(f"[ma_ttr_mtld] saved {plot_filename}")
                
                append_log("[ma_ttr_mtld] saved all individual plots")
            
        except Exception as e:
            append_log(f"[ma_ttr_mtld] plotting failed: {e}")
            import traceback
            append_log(traceback.format_exc())
    
    except Exception as e:
        append_log(f"[ma_ttr_mtld] summary save failed: {e}")
        import traceback
        append_log(traceback.format_exc())
    
    append_log("[ma_ttr_mtld] END")

if __name__ == '__main__':
    main()

