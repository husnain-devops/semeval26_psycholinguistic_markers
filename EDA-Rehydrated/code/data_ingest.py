#!/usr/bin/env python3
"""Read jsonl, normalize labels, save cleaned CSV and dataset summary."""
import json
import os
import sys
import csv
import logging
from pathlib import Path
import pandas as pd

SEED = 42

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

LOG.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def normalize_label(lbl):
    if lbl is None:
        return "cant_tell"
    s = str(lbl).strip().lower()
    if s in ("yes","y","true","1","conspiracy"):
        return "yes"
    if s in ("no","n","false","0"):
        return "no"
    return "cant_tell"

def main():
    append_log("[data_ingest] START")
    seed = SEED
    append_log(f"seed={seed}")
    src = RAW / "train_rehydrated.jsonl"
    if not src.exists():
        # attempt relative to repo root
        alt = Path.cwd() / "train_rehydrated.jsonl"
        if alt.exists():
            src = alt
        else:
            msg = f"INPUT MISSING: {src} and {alt} not found"
            append_log(msg)
            return
    records = []
    marker_counts = {"Actor":0,"Victim":0,"Evidence":0,"Action":0,"Effect":0}
    total_words = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception as e:
                append_log(f"json parse error: {e}")
                continue
            _id = obj.get("_id") or obj.get("id")
            text = obj.get("text") or obj.get("post_text") or obj.get("content")
            subreddit = obj.get("subreddit") or obj.get("subreddit_name")
            conspiracy = normalize_label(obj.get("conspiracy" ) or obj.get("label"))
            markers = obj.get("markers") or obj.get("marker")
            rec = {"_id":_id, "text": text, "subreddit": subreddit, "conspiracy": conspiracy, "markers": markers}
            records.append(rec)
            if text:
                total_words.append(len(str(text).split()))
            if markers and isinstance(markers, dict):
                for k in marker_counts.keys():
                    if markers.get(k):
                        marker_counts[k] += 1
    df = pd.DataFrame(records)
    # ensure columns
    df = df[["_id","text","subreddit","conspiracy","markers"]]
    out_csv = PROC / "data_clean.csv"
    df.to_csv(out_csv, index=False)
    summary = {
        "total_rows": int(len(df)),
        "distribution": df['conspiracy'].value_counts().to_dict(),
        "marker_counts": marker_counts,
        "mean_words": float(pd.Series(total_words).mean()) if total_words else 0,
        "median_words": float(pd.Series(total_words).median()) if total_words else 0,
        "min_words": int(pd.Series(total_words).min()) if total_words else 0,
        "max_words": int(pd.Series(total_words).max()) if total_words else 0,
        "seed": seed
    }
    with open(PROC / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_log(f"[data_ingest] saved {out_csv} and dataset_summary.json")
    append_log("[data_ingest] END")

if __name__ == '__main__':
    main()
