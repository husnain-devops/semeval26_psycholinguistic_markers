#!/usr/bin/env python3
"""Generate a short markdown report summarizing results."""
import os
from pathlib import Path
import logging
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
FEAT = BASE / "features"
TOP = BASE / "topics"
MODELS = BASE / "models"
LOG = REPORTS

REPORTS.mkdir(parents=True, exist_ok=True)

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = REPORTS / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log('[reporting] START')
    summary_path = BASE / 'data_processed' / 'dataset_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            ds = json.load(f)
    else:
        ds = {}
    report = []
    report.append('# EDA Report')
    report.append('')
    report.append('## Dataset summary')
    report.append('')
    report.append('```json')
    report.append(json.dumps(ds, indent=2))
    report.append('```')
    # top features if available
    shap_csv = BASE / 'shap_top20.csv'
    if shap_csv.exists():
        st = pd.read_csv(shap_csv)
        report.append('## Top features by SHAP')
        report.append(st.head(10).to_markdown(index=False))
    # topics
    bertop = TOP / 'bertopic_topics.csv'
    if bertop.exists():
        bt = pd.read_csv(bertop)
        report.append('## BERTopic top topics')
        report.append(bt.head(10).to_markdown(index=False))
    lda12 = TOP / 'lda_topics_12.csv'
    if lda12.exists():
        lt = pd.read_csv(lda12)
        report.append('## LDA (k=12) topics')
        report.append(lt.head(10).to_markdown(index=False))
    out = '\n'.join(report)
    outp = REPORTS / 'EDA_report.md'
    with open(outp, 'w', encoding='utf-8') as f:
        f.write(out)
    append_log(f'[reporting] saved {outp}')
    append_log('[reporting] END')

if __name__ == '__main__':
    main()
