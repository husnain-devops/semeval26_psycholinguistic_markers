#!/usr/bin/env python3
"""Analyze Parts of Speech (POS) tags distribution for 3 classes."""
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
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def safe_spacy_load():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as e:
        append_log(f"spaCy model load failed: {e}")
        try:
            import spacy
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e2:
            append_log(f"spaCy download/load failed: {e2}")
            return None

def main():
    append_log("[pos_analysis] START")
    
    # Load data
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[pos_analysis] data not found")
        return
    
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)
    
    # Always compute POS tags from scratch using spaCy to get all tags
    nlp = safe_spacy_load()
    if not nlp:
        append_log("[pos_analysis] spaCy not available, cannot compute POS tags")
        return
    
    # Disable unnecessary pipeline components for speed (keep only tagger)
    try:
        nlp.disable_pipes(['ner', 'parser'])
        # Only enable tagger if available, otherwise use default
        if 'tagger' not in nlp.pipe_names:
            nlp.enable_pipe('tagger')
    except Exception:
        pass  # Continue with default pipeline
    
    # Define all POS tags to track
    pos_tags = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ', 
                'SCONJ', 'NUM', 'PART', 'INTJ', 'AUX', 'PUNCT', 'SYM', 'X']
    
    rows = []
    total_rows = len(df)
    append_log(f"[pos_analysis] Processing {total_rows} documents...")
    
    # Process in batches for better performance
    batch_size = 100
    texts = [str(r.get('text') or '') for _, r in df.iterrows()]
    ids = [r.get('_id') for _, r in df.iterrows()]
    
    processed = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Process batch with spaCy
        try:
            docs = list(nlp.pipe(batch_texts, batch_size=batch_size, n_process=1))
            
            for doc, text, _id in zip(docs, batch_texts, batch_ids):
                rec = {'_id': _id}
                
                # Initialize counts for all POS tags
                pos_counts = {tag: 0 for tag in pos_tags}
                n_tokens = 0
                
                # Count all tokens (including punctuation for comprehensive analysis)
                n_tokens = len([t for t in doc if not t.is_space])
                
                # Count each POS tag
                for token in doc:
                    if not token.is_space:
                        pos = token.pos_
                        if pos in pos_counts:
                            pos_counts[pos] += 1
                        else:
                            # Handle any unexpected tags
                            pos_counts['X'] += 1
                
                # Store counts
                rec['n_tokens'] = n_tokens
                for tag in pos_tags:
                    rec[f'n_{tag.lower()}'] = pos_counts[tag]
                
                # Compute ratios (proportions) - use total tokens including punctuation
                if n_tokens > 0:
                    for tag in pos_tags:
                        rec[f'{tag.lower()}_ratio'] = pos_counts[tag] / n_tokens
                else:
                    for tag in pos_tags:
                        rec[f'{tag.lower()}_ratio'] = 0.0
                
                rows.append(rec)
                
            processed += len(batch_texts)
            if processed % 500 == 0:
                append_log(f"[pos_analysis] Processed {processed}/{total_rows} documents ({100*processed/total_rows:.1f}%)")
                
        except Exception as e:
            append_log(f"Error processing batch starting at index {i}: {e}")
            # Fallback: process individually
            for text, _id in zip(batch_texts, batch_ids):
                rec = {'_id': _id}
                pos_counts = {tag: 0 for tag in pos_tags}
                n_tokens = 0
                
                if text.strip():
                    try:
                        doc = nlp(text)
                        n_tokens = len([t for t in doc if not t.is_space])
                        for token in doc:
                            if not token.is_space:
                                pos = token.pos_
                                if pos in pos_counts:
                                    pos_counts[pos] += 1
                                else:
                                    pos_counts['X'] += 1
                    except Exception as e2:
                        append_log(f"Error processing {_id}: {e2}")
                
                rec['n_tokens'] = n_tokens
                for tag in pos_tags:
                    rec[f'n_{tag.lower()}'] = pos_counts[tag]
                
                if n_tokens > 0:
                    for tag in pos_tags:
                        rec[f'{tag.lower()}_ratio'] = pos_counts[tag] / n_tokens
                else:
                    for tag in pos_tags:
                        rec[f'{tag.lower()}_ratio'] = 0.0
                
                rows.append(rec)
    
    append_log(f"[pos_analysis] Completed processing {len(rows)} documents")
    
    out = pd.DataFrame(rows)
    outp = FEAT / 'pos_analysis.csv'
    out.to_csv(outp, index=False)
    append_log(f"[pos_analysis] saved {outp}")
    
    # Merge with labels for analysis
    merged = out.merge(df[['_id', 'conspiracy']], on='_id', how='left') if '_id' in df.columns else out
    
    if 'conspiracy' not in merged.columns:
        append_log("[pos_analysis] no conspiracy labels found")
        return
    
    # Create summary statistics by label
    try:
        summary_csv = FEAT / 'pos_summary_by_label.csv'
        summary_json = FEAT / 'pos_summary_by_label.json'
        
        # Save descriptive statistics
        summary = merged.groupby('conspiracy').describe()
        summary.to_csv(summary_csv)
        
        # Save aggregated stats as JSON - all POS tag ratios
        pos_tags = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ', 
                    'SCONJ', 'NUM', 'PART', 'INTJ', 'AUX', 'PUNCT', 'SYM', 'X']
        ratio_cols = [f'{tag.lower()}_ratio' for tag in pos_tags]
        ratio_cols = [col for col in ratio_cols if col in merged.columns]
        
        if ratio_cols:
            grp_df = merged.groupby('conspiracy')[ratio_cols].agg(['mean', 'std'])
            # Flatten MultiIndex columns for JSON serialization
            grp_df.columns = [f"{col}_{stat}" for col, stat in grp_df.columns]
            grp_dict = grp_df.to_dict(orient='index')
            with open(summary_json, 'w', encoding='utf-8') as f:
                json.dump(grp_dict, f, indent=2)
        
        append_log(f"[pos_analysis] saved summary csv/json")
        
        # Create visualizations
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 150
            
            # Define all metrics and their labels
            metrics = [f'{tag.lower()}_ratio' for tag in pos_tags]
            metric_labels = {
                'noun_ratio': 'Noun Ratio',
                'propn_ratio': 'Proper Noun Ratio',
                'verb_ratio': 'Verb Ratio',
                'adj_ratio': 'Adjective Ratio',
                'adv_ratio': 'Adverb Ratio',
                'pron_ratio': 'Pronoun Ratio',
                'det_ratio': 'Determiner Ratio',
                'adp_ratio': 'Preposition Ratio',
                'cconj_ratio': 'Conjunction Ratio',
                'sconj_ratio': 'Subordinating Conjunction Ratio',
                'num_ratio': 'Number Ratio',
                'part_ratio': 'Particle Ratio',
                'intj_ratio': 'Interjection Ratio',
                'aux_ratio': 'Auxiliary Verb Ratio',
                'punct_ratio': 'Punctuation Ratio',
                'sym_ratio': 'Symbol Ratio',
                'x_ratio': 'Other/Unknown Ratio'
            }
            
            colors = {'yes': '#e74c3c', 'no': '#3498db', 'cant_tell': '#95a5a6'}
            
            # 1. Violin plots for each POS ratio
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
                    plot_filename = FIG_VIOLIN / f'pos_{metric}_by_label.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    append_log(f"[pos_analysis] saved {plot_filename}")
            
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
                plot_filename = FIG_DENSITY / f'pos_{metric}_density.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                append_log(f"[pos_analysis] saved {plot_filename}")
            
            # 3. Mean comparison bar plots
            summary_stats = []
            for label in ['yes', 'no', 'cant_tell']:
                subset = merged[merged['conspiracy'] == label]
                if len(subset) > 0:
                    for metric in metrics:
                        if metric in subset.columns:
                            metric_label = metric_labels.get(metric, metric.replace('_', ' ').title())
                            metric_data = subset[metric].dropna()
                            if len(metric_data) > 0:
                                summary_stats.append({
                                    'label': label,
                                    'metric': metric_label,
                                    'mean': metric_data.mean(),
                                    'median': metric_data.median(),
                                    'std': metric_data.std()
                                })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                
                # Create bar plots for each metric
                for metric in metrics:
                    if metric not in merged.columns:
                        continue
                    metric_label = metric_labels.get(metric, metric.replace('_', ' ').title())
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
                        plot_filename = FIG_MEAN / f'pos_{metric}_mean_comparison.png'
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                        append_log(f"[pos_analysis] saved {plot_filename}")
                
                append_log("[pos_analysis] saved all individual plots")
            
        except Exception as e:
            append_log(f"[pos_analysis] plotting failed: {e}")
            import traceback
            append_log(traceback.format_exc())
    
    except Exception as e:
        append_log(f"[pos_analysis] summary save failed: {e}")
        import traceback
        append_log(traceback.format_exc())
    
    append_log("[pos_analysis] END")

if __name__ == '__main__':
    main()

