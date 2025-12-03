#!/usr/bin/env python3
"""Compute sentiment polarity (VADER) and basic NRC-like emotion counts if possible."""
import os
from pathlib import Path
import logging
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "features"
PROC = BASE / "data_processed"
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
    append_log("[sentiment_emotion] START")
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[sentiment_emotion] source not found")
        return
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        use_vader = True
    except Exception as e:
        append_log(f"VADER not available: {e}")
        sid = None
        use_vader = False

    # Try to load emotion lexicon (NRC-like) if available
    use_emotions = False
    emotion_lexicon = None
    try:
        # Try NRCLex library first
        from nrclex import NRCLex
        use_emotions = True
        append_log("[sentiment_emotion] NRCLex available for emotion analysis")
    except Exception:
        try:
            # Try text2emotion as alternative
            import text2emotion as te
            use_emotions = True
            emotion_lexicon = 'text2emotion'
            append_log("[sentiment_emotion] text2emotion available for emotion analysis")
        except Exception:
            append_log("[sentiment_emotion] No emotion analysis library available, using VADER only")

    rows = []
    for _,r in df.iterrows():
        text = str(r.get('text') or '')
        rec = {'_id': r.get('_id')}
        
        # VADER sentiment scores
        if use_vader:
            sc = sid.polarity_scores(text)
            rec.update(sc)
        else:
            rec.update({'neg':None,'neu':None,'pos':None,'compound':None})
        
        # Emotion analysis
        if use_emotions:
            try:
                if emotion_lexicon == 'text2emotion':
                    emotions = te.get_emotion(text)
                    # text2emotion returns: Happy, Angry, Surprise, Sad, Fear
                    rec['emotion_happy'] = emotions.get('Happy', 0.0)
                    rec['emotion_angry'] = emotions.get('Angry', 0.0)
                    rec['emotion_surprise'] = emotions.get('Surprise', 0.0)
                    rec['emotion_sad'] = emotions.get('Sad', 0.0)
                    rec['emotion_fear'] = emotions.get('Fear', 0.0)
                else:
                    # NRCLex
                    emotion_obj = NRCLex(text)
                    emotion_counts = emotion_obj.affect_frequencies
                    # NRC emotions: fear, anger, anticipation, trust, surprise, sadness, joy, disgust
                    rec['emotion_fear'] = emotion_counts.get('fear', 0.0)
                    rec['emotion_anger'] = emotion_counts.get('anger', 0.0)
                    rec['emotion_anticipation'] = emotion_counts.get('anticipation', 0.0)
                    rec['emotion_trust'] = emotion_counts.get('trust', 0.0)
                    rec['emotion_surprise'] = emotion_counts.get('surprise', 0.0)
                    rec['emotion_sadness'] = emotion_counts.get('sadness', 0.0)
                    rec['emotion_joy'] = emotion_counts.get('joy', 0.0)
                    rec['emotion_disgust'] = emotion_counts.get('disgust', 0.0)
            except Exception as e:
                append_log(f"Emotion analysis failed for {r.get('_id')}: {e}")
                if emotion_lexicon == 'text2emotion':
                    rec.update({'emotion_happy':0.0,'emotion_angry':0.0,'emotion_surprise':0.0,'emotion_sad':0.0,'emotion_fear':0.0})
                else:
                    rec.update({'emotion_fear':0.0,'emotion_anger':0.0,'emotion_anticipation':0.0,'emotion_trust':0.0,
                               'emotion_surprise':0.0,'emotion_sadness':0.0,'emotion_joy':0.0,'emotion_disgust':0.0})
        else:
            # No emotion analysis available
            pass
        
        rows.append(rec)
    
    out = pd.DataFrame(rows)
    outp = FEAT / 'sentiment_emotion.csv'
    out.to_csv(outp, index=False)
    append_log(f"[sentiment_emotion] saved {outp}")
    
    # Merge with labels for analysis and visualizations
    try:
        merged = out.merge(df[['_id','conspiracy']], on='_id', how='left')
        
        if 'conspiracy' not in merged.columns:
            append_log("[sentiment_emotion] no conspiracy labels found")
        else:
            # Save aggregated statistics
            agg = merged.groupby('conspiracy')[['compound']].mean().reset_index()
            agg.to_csv(FEAT / 'sentiment_by_label.csv', index=False)
            append_log(f"[sentiment_emotion] saved sentiment_by_label.csv")
            
            # Create summary statistics
            summary_csv = FEAT / 'sentiment_emotion_summary_by_label.csv'
            summary_json = FEAT / 'sentiment_emotion_summary_by_label.json'
            
            summary = merged.groupby('conspiracy').describe()
            summary.to_csv(summary_csv)
            
            # Save aggregated stats as JSON
            sentiment_cols = ['neg', 'neu', 'pos', 'compound']
            emotion_cols = [col for col in merged.columns if col.startswith('emotion_')]
            all_metric_cols = sentiment_cols + emotion_cols
            all_metric_cols = [col for col in all_metric_cols if col in merged.columns]
            
            if all_metric_cols:
                grp_df = merged.groupby('conspiracy')[all_metric_cols].agg(['mean', 'std'])
                grp_df.columns = [f"{col}_{stat}" for col, stat in grp_df.columns]
                grp_dict = grp_df.to_dict(orient='index')
                with open(summary_json, 'w', encoding='utf-8') as f:
                    json.dump(grp_dict, f, indent=2)
            
            append_log(f"[sentiment_emotion] saved summary csv/json")
            
            # Create visualizations
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                sns.set_style("whitegrid")
                plt.rcParams['figure.dpi'] = 150
                
                colors = {'yes': '#e74c3c', 'no': '#3498db', 'cant_tell': '#95a5a6'}
                
                # Sentiment metrics
                sentiment_metrics = ['neg', 'neu', 'pos', 'compound']
                sentiment_labels = {
                    'neg': 'Negative Sentiment',
                    'neu': 'Neutral Sentiment',
                    'pos': 'Positive Sentiment',
                    'compound': 'Compound Sentiment Score'
                }
                
                # Emotion metrics (if available)
                emotion_metrics = [col for col in merged.columns if col.startswith('emotion_')]
                emotion_labels = {}
                for col in emotion_metrics:
                    # Convert emotion_fear -> Fear, etc.
                    emotion_name = col.replace('emotion_', '').replace('_', ' ').title()
                    emotion_labels[col] = f'Emotion: {emotion_name}'
                
                all_metrics = sentiment_metrics + emotion_metrics
                all_labels = {**sentiment_labels, **emotion_labels}
                
                # 1. Violin plots for each metric
                for metric in all_metrics:
                    if metric not in merged.columns:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    data_for_plot = []
                    labels_to_plot = []
                    color_list = []
                    
                    for label in ['yes', 'no', 'cant_tell']:
                        subset = merged[merged['conspiracy'] == label][metric].dropna()
                        if len(subset) > 0:
                            data_for_plot.append(subset)
                            labels_to_plot.append(label)
                            color_list.append(colors[label])
                    
                    if data_for_plot:
                        parts = ax.violinplot(data_for_plot, positions=range(len(labels_to_plot)), 
                                             showmeans=True, showmedians=True)
                        
                        for pc, color in zip(parts['bodies'], color_list):
                            pc.set_facecolor(color)
                            pc.set_alpha(0.7)
                        
                        ax.set_xticks(range(len(labels_to_plot)))
                        ax.set_xticklabels(labels_to_plot, fontsize=12)
                        ax.set_ylabel(all_labels.get(metric, metric), fontsize=13, fontweight='bold')
                        ax.set_title(f'{all_labels.get(metric, metric)} by Label', fontsize=14, fontweight='bold', pad=15)
                        ax.grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        
                        plot_filename = FIG_VIOLIN / f'sentiment_{metric}_by_label.png'
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                        append_log(f"[sentiment_emotion] saved {plot_filename}")
                
                # 2. Density plots
                for metric in all_metrics:
                    if metric not in merged.columns:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    for label, color in zip(['yes', 'no', 'cant_tell'], ['#e74c3c', '#3498db', '#95a5a6']):
                        subset = merged[merged['conspiracy'] == label][metric].dropna()
                        if len(subset) > 0:
                            sns.kdeplot(data=subset, ax=ax, label=label, color=color, fill=True, alpha=0.5, linewidth=2)
                    
                    ax.set_xlabel(all_labels.get(metric, metric), fontsize=13, fontweight='bold')
                    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
                    ax.set_title(f'{all_labels.get(metric, metric)} Distribution by Label', fontsize=14, fontweight='bold', pad=15)
                    ax.legend(title='Label', fontsize=11, title_fontsize=12)
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    plot_filename = FIG_DENSITY / f'sentiment_{metric}_density.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    append_log(f"[sentiment_emotion] saved {plot_filename}")
                
                # 3. Mean comparison bar plots
                summary_stats = []
                for label in ['yes', 'no', 'cant_tell']:
                    subset = merged[merged['conspiracy'] == label][all_metrics].dropna()
                    if len(subset) > 0:
                        for metric in all_metrics:
                            if metric in subset.columns:
                                metric_data = subset[metric].dropna()
                                if len(metric_data) > 0:
                                    summary_stats.append({
                                        'label': label,
                                        'metric': all_labels.get(metric, metric),
                                        'mean': metric_data.mean(),
                                        'median': metric_data.median(),
                                        'std': metric_data.std()
                                    })
                
                if summary_stats:
                    summary_df = pd.DataFrame(summary_stats)
                    
                    for metric in all_metrics:
                        if metric not in merged.columns:
                            continue
                        
                        metric_label = all_labels.get(metric, metric)
                        metric_summary = summary_df[summary_df['metric'] == metric_label]
                        
                        if len(metric_summary) > 0:
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
                            
                            plot_filename = FIG_MEAN / f'sentiment_{metric}_mean_comparison.png'
                            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                            plt.close()
                            append_log(f"[sentiment_emotion] saved {plot_filename}")
                    
                    append_log("[sentiment_emotion] saved all individual plots")
            
            except Exception as e:
                append_log(f"[sentiment_emotion] plotting failed: {e}")
                import traceback
                append_log(traceback.format_exc())
    
    except Exception as e:
        append_log(f"[sentiment_emotion] analysis failed: {e}")
        import traceback
        append_log(traceback.format_exc())
    
    append_log("[sentiment_emotion] END")

if __name__ == '__main__':
    main()
