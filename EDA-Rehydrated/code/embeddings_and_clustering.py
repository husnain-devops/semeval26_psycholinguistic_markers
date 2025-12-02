#!/usr/bin/env python3
"""Compute sentence embeddings and run UMAP + HDBSCAN clustering if possible. Save embeddings and cluster assignments."""
import os
from pathlib import Path
import logging

SEED = 42

BASE = Path(__file__).resolve().parents[1]
EMB = BASE / "embeddings"
FIG = BASE / "figures"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

EMB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
# Create subdirectories for different plot types
FIG_CLUSTERING = FIG / 'clustering'
FIG_CLUSTERING.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
run_log = LOG / "run_log.txt"

def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)

def main():
    append_log('[embeddings_and_clustering] START')
    dfp = PROC / 'data_clean.csv'
    if not dfp.exists():
        append_log('[embeddings_and_clustering] data_clean.csv not found')
        return
    df = pd.read_csv(dfp)
    texts = df['text'].fillna('').tolist()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        emb = model.encode(texts, show_progress_bar=True)
        import joblib
        joblib.dump(emb, EMB / 'embeddings.npy')
        append_log('[embeddings_and_clustering] embeddings saved')
    except Exception as e:
        append_log(f"embeddings failed: {e}")
        return
    try:
        import umap
        import hdbscan
        reducer = umap.UMAP(random_state=SEED)
        emb2 = reducer.fit_transform(emb)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        clusters = clusterer.fit_predict(emb2)
        pd.DataFrame({'_id': df['_id'], 'cluster': clusters}).to_csv(EMB / 'doc_clusters.csv', index=False)
        append_log('[embeddings_and_clustering] clustering saved')
        # save simple plot
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,5))
            sc = plt.scatter(emb2[:,0], emb2[:,1], c=clusters, cmap='tab20', s=10)
            plt.title('UMAP seabed clusters')
            plt.savefig(FIG_CLUSTERING / 'umap_clusters.png')
        except Exception as e:
            append_log(f"plot failed: {e}")
    except Exception as e:
        append_log(f"UMAP/HDBSCAN not run: {e}")
    append_log('[embeddings_and_clustering] END')

if __name__ == '__main__':
    main()
