#!/usr/bin/env python3
"""Basic cleaning, tokenization, POS counts, URL counts."""
import os
from pathlib import Path
import logging
import re
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data_processed"
FEAT = BASE / "features"
LOG = BASE / "reports"

PROC.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

import pandas as pd

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

def clean_text(s):
    if pd.isna(s):
        return ""
    import unicodedata
    s = unicodedata.normalize('NFKC', str(s))
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def main():
    append_log("[preprocessing] START")
    dfp = PROC / "data_clean.csv"
    if not dfp.exists():
        append_log(f"data_clean.csv not found at {dfp}")
        return
    df = pd.read_csv(dfp)
    df['text_raw'] = df['text']
    df['text'] = df['text'].apply(clean_text)
    # URL and email counts
    url_re = re.compile(r'https?://\S+|www\.\S+')
    email_re = re.compile(r'\S+@\S+')
    df['url_count'] = df['text_raw'].apply(lambda s: len(url_re.findall(str(s))))
    df['email_count'] = df['text_raw'].apply(lambda s: len(email_re.findall(str(s))))

    nlp = safe_spacy_load()
    cols = ['n_tokens','n_sentences','n_nouns','n_verbs','n_adjs','n_advs','n_pronouns','stopword_ratio','commas','qmarks','exclaims','ellipses','allcaps_ratio']
    for c in cols:
        if c in ['stopword_ratio', 'allcaps_ratio']:
            df[c] = 0.0  # Initialize as float
        else:
            df[c] = 0

    for i, row in df.iterrows():
        text = str(row['text'])
        commas = text.count(',')
        qmarks = text.count('?')
        exclaims = text.count('!')
        ellipses = text.count('...')
        words = re.findall(r"\w+", text)
        n_tokens = len(words)
        allcaps = sum(1 for w in words if w.isupper() and len(w) > 1)
        stopwords = 0
        n_nouns = n_verbs = n_adjs = n_advs = n_pronouns = 0
        n_sent = 0
        if nlp:
            doc = nlp(text)
            n_sent = len(list(doc.sents))
            for t in doc:
                pos = t.pos_
                if pos == 'NOUN' or pos == 'PROPN':
                    n_nouns += 1
                if pos == 'VERB':
                    n_verbs += 1
                if pos == 'ADJ':
                    n_adjs += 1
                if pos == 'ADV':
                    n_advs += 1
                if pos == 'PRON':
                    n_pronouns += 1
                if t.is_stop:
                    stopwords += 1
        else:
            # fallback simple sentence split
            n_sent = max(1, text.count('.') + text.count('!') + text.count('?'))
        stop_ratio = stopwords / n_tokens if n_tokens>0 else 0
        df.at[i,'n_tokens'] = n_tokens
        df.at[i,'n_sentences'] = n_sent
        df.at[i,'n_nouns'] = n_nouns
        df.at[i,'n_verbs'] = n_verbs
        df.at[i,'n_adjs'] = n_adjs
        df.at[i,'n_advs'] = n_advs
        df.at[i,'n_pronouns'] = n_pronouns
        df.at[i,'stopword_ratio'] = float(stop_ratio)  # Explicit float conversion
        df.at[i,'commas'] = commas
        df.at[i,'qmarks'] = qmarks
        df.at[i,'exclaims'] = exclaims
        df.at[i,'ellipses'] = ellipses
        df.at[i,'allcaps_ratio'] = float(allcaps / n_tokens if n_tokens>0 else 0)  # Explicit float conversion

    out_parquet = PROC / 'data_processed.parquet'
    try:
        df.to_parquet(out_parquet, index=False)
        append_log(f"[preprocessing] saved parquet to {out_parquet}")
    except Exception as e:
        csv_out = PROC / 'data_processed.csv'
        df.to_csv(csv_out, index=False)
        append_log(f"[preprocessing] parquet write failed: {e}. Wrote CSV to {csv_out}")

    # token_pos_counts
    tpc = df[['__id','_id']] if '_id' in df.columns else None
    cols_out = ['_id','n_tokens','n_sentences','n_nouns','n_verbs','n_adjs','n_advs','n_pronouns']
    df[cols_out].to_csv(FEAT / 'token_pos_counts.csv', index=False)
    append_log(f"[preprocessing] saved token_pos_counts.csv")
    append_log("[preprocessing] END")

if __name__ == '__main__':
    main()
