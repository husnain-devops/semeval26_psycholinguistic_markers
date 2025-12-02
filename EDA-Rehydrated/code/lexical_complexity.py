#!/usr/bin/env python3
"""Compute lexical richness metrics per document."""
import os
from pathlib import Path
import logging
import json
import math

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
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def ttr(tokens):
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def hapax_ratio(tokens):
    if not tokens:
        return 0
    c = Counter(tokens)
    hapax = sum(1 for w,v in c.items() if v==1)
    return hapax / len(tokens)

def avg_word_len(tokens):
    if not tokens:
        return 0
    return sum(len(w) for w in tokens)/len(tokens)

def shannon_entropy(tokens):
    if not tokens:
        return 0
    c = Counter(tokens)
    total = sum(c.values())
    import math
    ent = -sum((v/total)*math.log2(v/total) for v in c.values())
    return ent

def main():
    append_log("[lexical_complexity] START")
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        # fallback to csv
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("data not found for lexical metrics")
        return
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)
    rows = []
    for _,r in df.iterrows():
        text = str(r.get('text') or '')
        tokens = [w.lower() for w in text.split() if w.isalpha()]
        rows.append({
            '_id': r.get('_id'),
            'ttr': ttr(tokens),
            'hapax_ratio': hapax_ratio(tokens),
            'avg_word_len': avg_word_len(tokens),
            'shannon_entropy': shannon_entropy(tokens),
        })
    out = pd.DataFrame(rows)
    outp = FEAT / 'lexical_complexity.csv'
    out.to_csv(outp, index=False)
    append_log(f"[lexical_complexity] saved {outp}")
    # summary by label
    merged = out.merge(df[['_id','conspiracy']], on='_id', how='left') if '_id' in df.columns else out
    try:
        summary_csv = FEAT / 'lexical_summary_by_label.csv'
        summary_json = FEAT / 'lexical_summary_by_label.json'
        summary = merged.groupby('conspiracy').describe().to_csv(summary_csv)
        # write basic json - only aggregate numeric columns
        numeric_cols = merged.select_dtypes(include=[float, int]).columns.tolist()
        if numeric_cols:
            grp_df = merged.groupby('conspiracy')[numeric_cols].agg(['mean','std'])
            # Flatten MultiIndex columns for JSON serialization
            grp_df.columns = [f"{col}_{stat}" for col, stat in grp_df.columns]
            grp_dict = grp_df.to_dict(orient='index')
            with open(summary_json,'w',encoding='utf-8') as f:
                json.dump(grp_dict,f,indent=2)
        else:
            append_log(f"[lexical_complexity] no numeric columns for summary")
        append_log(f"[lexical_complexity] saved summary csv/json")
        
        # Plot lexical metrics by label - separate violin plots for each metric
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 150
            
            metrics = ['ttr', 'hapax_ratio', 'avg_word_len', 'shannon_entropy']
            metric_labels = {
                'ttr': 'Type-Token Ratio (TTR)',
                'hapax_ratio': 'Hapax Ratio',
                'avg_word_len': 'Average Word Length',
                'shannon_entropy': 'Shannon Entropy'
            }
            
            colors = {'yes': '#e74c3c', 'no': '#3498db', 'cant_tell': '#95a5a6'}
            
            for metric in metrics:
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
                    metric_safe = metric.replace('_', '_')
                    plot_filename = FIG_VIOLIN / f'lexical_{metric}_by_label.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    append_log(f"[lexical_complexity] saved {plot_filename}")
            
            append_log(f"[lexical_complexity] saved all individual lexical plots")
        except Exception as e:
            append_log(f"[lexical_complexity] plotting failed: {e}")
            import traceback
            append_log(traceback.format_exc())
    except Exception as e:
        append_log(f"[lexical_complexity] summary save failed: {e}")
    append_log("[lexical_complexity] END")

if __name__ == '__main__':
    main()
