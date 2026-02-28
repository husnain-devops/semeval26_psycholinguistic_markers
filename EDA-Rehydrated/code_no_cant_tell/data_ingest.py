#!/usr/bin/env python3
"""Read jsonl, normalize labels, save cleaned CSV and dataset summary.

This variant drops all 'cant_tell' examples so that downstream analyses
only operate on the binary yes/no subset.
"""
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
    append_log("[data_ingest_no_cant_tell] START")
    seed = SEED
    append_log(f"seed={seed}")
    src = RAW / "train_rehydrated.jsonl"
    if not src.exists():
        # try EDA-Rehydrated root (when running from that directory)
        alt = BASE / "train_rehydrated.jsonl"
        # also try the code directory where a copy may live
        code_path = BASE / "code" / "train_rehydrated.jsonl"
        # finally, try the repo root
        repo_root_path = BASE.parent / "train_rehydrated.jsonl"
        
        if alt.exists():
            src = alt
        elif code_path.exists():
            src = code_path
        elif repo_root_path.exists():
            src = repo_root_path
        else:
            msg = (
                f"INPUT MISSING: {RAW / 'train_rehydrated.jsonl'}, "
                f"{alt}, {code_path}, and {repo_root_path} not found"
            )
            append_log(msg)
            return
    records = []
    total_words = []
    # Track duplicate IDs to handle them
    id_to_records = {}
    
    with open(src, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception as e:
                append_log(f"json parse error at line {line_num}: {e}")
                continue
            _id = obj.get("_id") or obj.get("id")
            text = obj.get("text") or obj.get("post_text") or obj.get("content")
            conspiracy = normalize_label(obj.get("conspiracy") or obj.get("label"))
            
            # Only extract _id, text, and conspiracy
            rec = {"_id": _id, "text": text, "conspiracy": conspiracy, "line_num": line_num}
            records.append(rec)
            
            # Track duplicates
            if _id not in id_to_records:
                id_to_records[_id] = []
            id_to_records[_id].append(rec)
            
            if text:
                total_words.append(len(str(text).split()))
    
    # Handle duplicate IDs
    duplicate_differ_count = 0
    duplicate_same_count = 0
    records_to_remove = set()  # Use set of line numbers to track records to remove
    
    for _id, recs in id_to_records.items():
        if len(recs) > 1:
            # Check if text or conspiracy differs
            texts = [str(r["text"]) for r in recs]
            conspiracies = [r["conspiracy"] for r in recs]
            unique_texts = set(texts)
            unique_conspiracies = set(conspiracies)
            
            # If text or conspiracy differs, append suffix to all duplicates
            if len(unique_texts) > 1 or len(unique_conspiracies) > 1:
                duplicate_differ_count += 1
                for idx, rec in enumerate(recs):
                    original_id = rec["_id"]
                    rec["_id"] = f"{original_id}_{idx + 1}"
                append_log(f"Duplicate ID {_id} with differing text/conspiracy: renamed {len(recs)} records to {_id}_1, {_id}_2, etc.")
            else:
                # Identical duplicates - keep only the first one, mark others for removal
                duplicate_same_count += 1
                original_id = recs[0]["_id"]
                # Mark all but the first record for removal
                for rec_to_remove in recs[1:]:
                    records_to_remove.add(rec_to_remove["line_num"])
                append_log(f"Duplicate ID {_id} with identical text/conspiracy: kept first record, removing {len(recs)-1} duplicate(s)")
    
    # Filter out records marked for removal
    records = [rec for rec in records if rec.get("line_num") not in records_to_remove]
    
    # Rebuild total_words to match filtered records
    total_words = []
    for rec in records:
        text = rec.get("text")
        if text:
            total_words.append(len(str(text).split()))
    
    if duplicate_differ_count > 0:
        append_log(f"[data_ingest_no_cant_tell] Found {duplicate_differ_count} IDs with duplicates that have differing text/conspiracy values (renamed)")
    if duplicate_same_count > 0:
        append_log(f"[data_ingest_no_cant_tell] Found {duplicate_same_count} IDs with identical duplicates (removed duplicates, kept first)")
    
    # Remove line_num before creating DataFrame
    for rec in records:
        rec.pop("line_num", None)
    
    df = pd.DataFrame(records)
    # ensure columns - only _id, text, and conspiracy
    df = df[["_id", "text", "conspiracy"]]
    # Drop all 'cant_tell' rows to keep only binary labels
    before = len(df)
    df = df[df["conspiracy"].isin(["yes", "no"])].reset_index(drop=True)
    after = len(df)
    append_log(f"[data_ingest_no_cant_tell] filtered out 'cant_tell' rows: {before-after} removed, {after} remaining")
    out_csv = PROC / "data_clean.csv"
    df.to_csv(out_csv, index=False)
    
    summary = {
        "total_rows": int(len(df)),
        "distribution": df['conspiracy'].value_counts().to_dict(),
        "mean_words": float(pd.Series(total_words).mean()) if total_words else 0,
        "median_words": float(pd.Series(total_words).median()) if total_words else 0,
        "min_words": int(pd.Series(total_words).min()) if total_words else 0,
        "max_words": int(pd.Series(total_words).max()) if total_words else 0,
        "seed": seed,
        "note": "cant_tell examples have been removed in this variant"
    }
    with open(PROC / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_log(f"[data_ingest_no_cant_tell] saved {out_csv} and dataset_summary.json")
    append_log("[data_ingest_no_cant_tell] END")

if __name__ == '__main__':
    main()

