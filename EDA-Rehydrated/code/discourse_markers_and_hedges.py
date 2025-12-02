#!/usr/bin/env python3
"""Count hedges, certainty words, discourse markers and passive voice frequency."""
import os
from pathlib import Path
import logging
import re

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

HEDGES = ["maybe","perhaps","could be","might","seems","possibly","may be","may"]
CERTAINTY = ["definitely","certainly","proof","proven","prove","undeniable","obviously"]
DISCOURSE = ["actually","in fact","the truth","frankly","to be honest"]

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log("[discourse_markers] START")
    dfp = PROC / 'data_processed.parquet'
    if not dfp.exists():
        dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log("[discourse_markers] source not found")
        return
    try:
        df = pd.read_parquet(dfp)
    except Exception:
        df = pd.read_csv(dfp)
    rows = []
    for _,r in df.iterrows():
        text = str(r.get('text') or '').lower()
        rec = {'_id': r.get('_id')}
        for name, lst in [('hedge', HEDGES), ('certainty', CERTAINTY), ('discourse', DISCOURSE)]:
            cnt = sum(text.count(p) for p in lst)
            rec[name + '_count'] = cnt
        # passive detection: simple heuristic 'was *ed' or 'is *ed' or 'were *ed'
        passive = len(re.findall(r'\b(was|were|is|are|been|be)\s+\w+ed\b', text))
        rec['passive_count'] = passive
        rows.append(rec)
    out = pd.DataFrame(rows)
    outp = FEAT / 'discourse_markers.csv'
    out.to_csv(outp, index=False)
    append_log(f"[discourse_markers] saved {outp}")
    append_log("[discourse_markers] END")

if __name__ == '__main__':
    main()
