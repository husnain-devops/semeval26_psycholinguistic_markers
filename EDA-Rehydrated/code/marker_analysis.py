#!/usr/bin/env python3
"""Analyze markers field: counts, cooccurrence, PMI and heatmap."""
import os
from pathlib import Path
import logging
import json
import itertools

SEED = 42

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "features"
FIG = BASE / "figures"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

FEAT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
# Create subdirectories for different plot types
FIG_MEAN = FIG / 'mean_comparison'
FIG_HEATMAP = FIG / 'heatmap'
FIG_MEAN.mkdir(parents=True, exist_ok=True)
FIG_HEATMAP.mkdir(parents=True, exist_ok=True)

import pandas as pd
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log("[marker_analysis] START")
    dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[marker_analysis] data_clean.csv not found")
        return
    df = pd.read_csv(dfp)
    
    # Check if markers column exists - if not, skip analysis
    if 'markers' not in df.columns:
        append_log("[marker_analysis] markers column not found in data. Skipping marker analysis.")
        append_log("[marker_analysis] END")
        return
    
    types = ['Actor','Action','Victim','Evidence','Effect']
    rows = []
    co = defaultdict(int)
    counts = Counter()
    N = 0
    for _,r in df.iterrows():
        markers = r.get('markers')
        N += 1
        present = []
        if isinstance(markers, str):
            try:
                markers_obj = json.loads(markers)
            except Exception:
                markers_obj = None
        else:
            markers_obj = markers
        if isinstance(markers_obj, dict):
            for t in types:
                if markers_obj.get(t):
                    counts[t] += 1
                    present.append(t)
        for a,b in itertools.combinations(sorted(present),2):
            co[(a,b)] += 1
    # save counts
    cnt_df = pd.DataFrame([{'marker':k,'count':v} for k,v in counts.items()])
    cnt_df.to_csv(FEAT / 'marker_counts.csv', index=False)
    append_log(f"[marker_analysis] saved marker_counts.csv")
    
    # Plot marker counts
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        if len(cnt_df) > 0:
            plt.figure(figsize=(8, 5))
            cnt_df_sorted = cnt_df.sort_values('count', ascending=False)
            plt.bar(cnt_df_sorted['marker'], cnt_df_sorted['count'], color='steelblue', edgecolor='black')
            plt.xlabel('Marker Type')
            plt.ylabel('Count')
            plt.title('Marker Type Counts')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_MEAN / 'marker_counts.png', dpi=150, bbox_inches='tight')
            plt.close()
            append_log("[marker_analysis] marker counts plot saved")
    except Exception as e:
        append_log(f"[marker_analysis] marker counts plot failed: {e}")
    
    # cooccurrence matrix
    mat = {t:{u:0 for u in types} for t in types}
    for (a,b),v in co.items():
        mat[a][b] = v
        mat[b][a] = v
    coc_df = pd.DataFrame(mat)
    coc_df.to_csv(FEAT / 'marker_cooccurrence.csv')
    append_log(f"[marker_analysis] saved marker_counts and cooccurrence")
    # try to plot heatmap
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        sns.heatmap(coc_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Marker cooccurrence')
        plt.tight_layout()
        plt.savefig(FIG_HEATMAP / 'marker_cooccurrence_heatmap.png')
        append_log("[marker_analysis] heatmap saved")
    except Exception as e:
        append_log(f"[marker_analysis] could not create heatmap: {e}")
    append_log("[marker_analysis] END")

if __name__ == '__main__':
    main()
