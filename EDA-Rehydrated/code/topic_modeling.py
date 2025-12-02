#!/usr/bin/env python3
"""Topic modeling: simple LDA (gensim) and placeholder for BERTopic.
This script will do LDA if gensim is available; otherwise it will log and skip.
"""
import os
from pathlib import Path
import logging
import json

SEED = 42

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data_processed"
TOP = BASE / "topics"
FIG = BASE / "figures"
LOG = BASE / "reports"

TOP.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def lda_flow(texts, k):
    try:
        from gensim import corpora, models
    except Exception as e:
        append_log(f"gensim not available: {e}")
        return None
    # simple tokenization and dictionary
    tokenized = [[w.lower() for w in t.split() if w.isalpha()] for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=SEED, passes=5)
    topics = lda.show_topics(num_topics=k, num_words=15, formatted=False)
    rows = []
    for tid, words in topics:
        rows.append({'topic_id': tid, 'words': ' '.join(w for w,c in words)})
    df = pd.DataFrame(rows)
    df.to_csv(TOP / f'lda_topics_{k}.csv', index=False)
    try:
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis
        vis = gensimvis.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis, str(TOP / f'lda_vis_{k}.html'))
    except Exception as e:
        append_log(f"pyLDAvis not available or failed: {e}")
    return lda

def main():
    append_log('[topic_modeling] START')
    dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log('[topic_modeling] data_clean.csv not found')
        return
    df = pd.read_csv(dfp)
    texts = df['text'].fillna('').tolist()
    for k in [8,12,20]:
        try:
            lda = lda_flow(texts, k)
            if lda:
                append_log(f"[topic_modeling] LDA k={k} done")
        except Exception as e:
            append_log(f"[topic_modeling] LDA k={k} failed: {e}")

    # BERTopic placeholder: try to run if bertopic installed
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        topic_model = BERTopic(verbose=False)
        topics, probs = topic_model.fit_transform(texts, embeddings)
        topic_model.get_topic_info().to_csv(TOP / 'bertopic_topics.csv', index=False)
        # save mapping
        import pandas as pd
        pd.DataFrame({'_id': df['_id'], 'topic': topics}).to_csv(TOP / 'bertopic_doc_topics.csv', index=False)
        append_log('[topic_modeling] BERTopic done')
    except Exception as e:
        append_log(f"BERTopic not run: {e}")

    append_log('[topic_modeling] END')

if __name__ == '__main__':
    main()
