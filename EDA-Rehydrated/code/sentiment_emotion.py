#!/usr/bin/env python3
"""Compute sentiment polarity (VADER) and basic NRC-like emotion counts if possible."""
import os
from pathlib import Path
import logging

SEED = 42

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "features"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

FEAT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

import pandas as pd

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

    rows = []
    for _,r in df.iterrows():
        text = str(r.get('text') or '')
        rec = {'_id': r.get('_id')}
        if use_vader:
            sc = sid.polarity_scores(text)
            rec.update(sc)
        else:
            rec.update({'neg':None,'neu':None,'pos':None,'compound':None})
        rows.append(rec)
    out = pd.DataFrame(rows)
    outp = FEAT / 'sentiment_emotion.csv'
    out.to_csv(outp, index=False)
    append_log(f"[sentiment_emotion] saved {outp}")
    # simple aggregated per label
    try:
        merged = out.merge(df[['_id','conspiracy']], on='_id', how='left')
        agg = merged.groupby('conspiracy')[['compound']].mean().reset_index()
        agg.to_csv(FEAT / 'sentiment_by_label.csv', index=False)
    except Exception as e:
        append_log(f"[sentiment_emotion] aggregated save failed: {e}")
    append_log("[sentiment_emotion] END")

if __name__ == '__main__':
    main()
