#!/usr/bin/env python3
"""Extract the same 62 features as training pipeline for dev set (dev_rehydrated.jsonl).

Outputs to features_dev/ so the notebook can evaluate on the final 100 samples.
Run from EDA-Rehydrated directory: python code/extract_dev_features.py
"""
import json
import logging
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FEAT_DEV = BASE / "features_dev"
PROC = BASE / "data_processed"
LOG = BASE / "reports"

FEAT_DEV.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
run_log = LOG / "run_log.txt"


def append_log(msg):
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    logging.info(msg)


def load_dev_df():
    """Load dev_rehydrated.jsonl into DataFrame with _id, text."""
    path = DATA / "dev_rehydrated.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dev data not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            rows.append({"_id": obj["_id"], "text": obj.get("text") or ""})
    df = pd.DataFrame(rows)
    append_log(f"[extract_dev_features] Loaded {len(df)} rows from {path}")
    return df


# --- Lexical complexity (ttr, hapax_ratio, avg_word_len, shannon_entropy) ---
def lexical_complexity(df):
    from collections import Counter
    import math

    def ttr(tokens):
        return len(set(tokens)) / len(tokens) if tokens else 0

    def hapax_ratio(tokens):
        if not tokens:
            return 0
        c = Counter(tokens)
        return sum(1 for _, v in c.items() if v == 1) / len(tokens)

    def avg_word_len(tokens):
        return sum(len(w) for w in tokens) / len(tokens) if tokens else 0

    def shannon_entropy(tokens):
        if not tokens:
            return 0
        c = Counter(tokens)
        total = sum(c.values())
        return -sum((v / total) * math.log2(v / total) for v in c.values())

    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "")
        tokens = [w.lower() for w in text.split() if w.isalpha()]
        rows.append({
            "_id": r["_id"],
            "ttr": ttr(tokens),
            "hapax_ratio": hapax_ratio(tokens),
            "avg_word_len": avg_word_len(tokens),
            "shannon_entropy": shannon_entropy(tokens),
        })
    return pd.DataFrame(rows)


# --- Discourse markers ---
def discourse_markers(df):
    HEDGES = ["maybe", "perhaps", "could be", "might", "seems", "possibly", "may be", "may"]
    CERTAINTY = ["definitely", "certainly", "proof", "proven", "prove", "undeniable", "obviously"]
    DISCOURSE = ["actually", "in fact", "the truth", "frankly", "to be honest"]
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "").lower()
        rec = {"_id": r["_id"]}
        for name, lst in [("hedge", HEDGES), ("certainty", CERTAINTY), ("discourse", DISCOURSE)]:
            rec[name + "_count"] = sum(text.count(p) for p in lst)
        passive = len(re.findall(r"\b(was|were|is|are|been|be)\s+\w+ed\b", text))
        rec["passive_count"] = passive
        rows.append(rec)
    return pd.DataFrame(rows)


# --- Sentiment (VADER) + emotion if available ---
def sentiment_emotion(df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        use_vader = True
    except Exception as e:
        append_log(f"VADER not available: {e}")
        sid = None
        use_vader = False

    use_emotions = False
    emotion_lib = None
    try:
        from nrclex import NRCLex
        use_emotions = True
        emotion_lib = "nrclex"
    except Exception:
        try:
            import text2emotion as te
            use_emotions = True
            emotion_lib = "text2emotion"
        except Exception:
            pass

    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "")
        rec = {"_id": r["_id"]}
        if use_vader:
            sc = sid.polarity_scores(text)
            rec.update(sc)
        else:
            rec.update({"neg": 0, "neu": 0, "pos": 0, "compound": 0})
        if use_emotions:
            try:
                if emotion_lib == "text2emotion":
                    emotions = te.get_emotion(text)
                    rec["emotion_happy"] = emotions.get("Happy", 0.0)
                    rec["emotion_angry"] = emotions.get("Angry", 0.0)
                    rec["emotion_surprise"] = emotions.get("Surprise", 0.0)
                    rec["emotion_sad"] = emotions.get("Sad", 0.0)
                    rec["emotion_fear"] = emotions.get("Fear", 0.0)
                else:
                    emotion_obj = NRCLex(text)
                    emotion_counts = emotion_obj.affect_frequencies
                    rec["emotion_fear"] = emotion_counts.get("fear", 0.0)
                    rec["emotion_anger"] = emotion_counts.get("anger", 0.0)
                    rec["emotion_anticipation"] = emotion_counts.get("anticipation", 0.0)
                    rec["emotion_trust"] = emotion_counts.get("trust", 0.0)
                    rec["emotion_surprise"] = emotion_counts.get("surprise", 0.0)
                    rec["emotion_sadness"] = emotion_counts.get("sadness", 0.0)
                    rec["emotion_joy"] = emotion_counts.get("joy", 0.0)
                    rec["emotion_disgust"] = emotion_counts.get("disgust", 0.0)
            except Exception:
                if emotion_lib == "text2emotion":
                    rec.update({"emotion_happy": 0, "emotion_angry": 0, "emotion_surprise": 0, "emotion_sad": 0, "emotion_fear": 0})
                else:
                    rec.update({"emotion_fear": 0, "emotion_anger": 0, "emotion_anticipation": 0, "emotion_trust": 0,
                                "emotion_surprise": 0, "emotion_sadness": 0, "emotion_joy": 0, "emotion_disgust": 0})
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Ensure same columns as train: if text2emotion we have happy/angry/surprise/sad/fear; if nrclex we have fear/anger/...
    # Training script may produce one or the other. Notebook expects consistent columns - fill missing with 0
    return out


# --- Token/POS counts (preprocessing-style: n_tokens, n_sentences, n_nouns, ...) ---
def token_pos_counts(df):
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        pass

    cols_out = ["_id", "n_tokens", "n_sentences", "n_nouns", "n_verbs", "n_adjs", "n_advs", "n_pronouns"]
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "")
        words = re.findall(r"\w+", text)
        n_tokens = len(words)
        n_sent = 0
        n_nouns = n_verbs = n_adjs = n_advs = n_pronouns = 0
        if nlp and text.strip():
            doc = nlp(text)
            n_sent = len(list(doc.sents))
            for t in doc:
                pos = t.pos_
                if pos in ("NOUN", "PROPN"):
                    n_nouns += 1
                elif pos == "VERB":
                    n_verbs += 1
                elif pos == "ADJ":
                    n_adjs += 1
                elif pos == "ADV":
                    n_advs += 1
                elif pos == "PRON":
                    n_pronouns += 1
        else:
            n_sent = max(1, text.count(".") + text.count("!") + text.count("?"))
        rows.append({
            "_id": r["_id"],
            "n_tokens": n_tokens,
            "n_sentences": n_sent,
            "n_nouns": n_nouns,
            "n_verbs": n_verbs,
            "n_adjs": n_adjs,
            "n_advs": n_advs,
            "n_pronouns": n_pronouns,
        })
    return pd.DataFrame(rows)


# --- Readability (textstat) ---
def readability_scores(df):
    try:
        import textstat
        has_textstat = True
    except Exception:
        has_textstat = False
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "")
        rec = {"_id": r["_id"]}
        if has_textstat:
            rec["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            rec["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            rec["gunning_fog"] = textstat.gunning_fog(text)
            rec["smog"] = textstat.smog_index(text)
            rec["ari"] = textstat.automated_readability_index(text)
            rec["coleman_liau"] = textstat.coleman_liau_index(text)
        else:
            rec.update({"flesch_reading_ease": 0, "flesch_kincaid_grade": 0, "gunning_fog": 0, "smog": 0, "ari": 0, "coleman_liau": 0})
        rows.append(rec)
    return pd.DataFrame(rows)


# --- MA-TTR, MTLD ---
def ma_ttr_mtld(df):
    try:
        from lexicalrichness import LexicalRichness
        has_lr = True
    except Exception:
        has_lr = False
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "")
        rec = {"_id": r["_id"], "ma_ttr": None, "mtld": None}
        if has_lr and text.strip():
            try:
                lex = LexicalRichness(text)
                wc = lex.words
                if wc >= 10:
                    rec["ma_ttr"] = lex.mattr(window_size=max(10, min(50, int(wc * 0.2))))
                try:
                    rec["mtld"] = lex.mtld(threshold=0.72)
                except Exception:
                    rec["mtld"] = lex.mtld()
            except Exception:
                pass
        rows.append(rec)
    return pd.DataFrame(rows)


# --- POS analysis (full tag ratios) ---
def pos_analysis(df):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
    if nlp is None:
        append_log("[extract_dev_features] spaCy not available, writing placeholder pos_analysis")
        ids = df["_id"].tolist()
        pos_tags = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM", "PART", "INTJ", "AUX", "PUNCT", "SYM", "X"]
        rows = []
        for _id in ids:
            rec = {"_id": _id, "n_tokens": 0}
            for tag in pos_tags:
                rec[f"n_{tag.lower()}"] = 0
                rec[f"{tag.lower()}_ratio"] = 0.0
            rows.append(rec)
        return pd.DataFrame(rows)

    pos_tags = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM", "PART", "INTJ", "AUX", "PUNCT", "SYM", "X"]
    rows = []
    texts = [str(r.get("text") or "") for _, r in df.iterrows()]
    ids = [r["_id"] for _, r in df.iterrows()]
    try:
        nlp.disable_pipes(["ner", "parser"])
    except Exception:
        pass
    for doc, _id in zip(nlp.pipe(texts, batch_size=50), ids):
        rec = {"_id": _id}
        pos_counts = {tag: 0 for tag in pos_tags}
        n_tokens = len([t for t in doc if not t.is_space])
        for token in doc:
            if not token.is_space:
                pos = token.pos_
                if pos in pos_counts:
                    pos_counts[pos] += 1
                else:
                    pos_counts["X"] += 1
        rec["n_tokens"] = n_tokens
        for tag in pos_tags:
            rec[f"n_{tag.lower()}"] = pos_counts[tag]
            rec[f"{tag.lower()}_ratio"] = pos_counts[tag] / n_tokens if n_tokens else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    append_log("[extract_dev_features] START")
    df = load_dev_df()
    if df.empty:
        append_log("[extract_dev_features] No dev data, exit")
        return

    steps = [
        ("lexical_complexity.csv", lexical_complexity),
        ("discourse_markers.csv", discourse_markers),
        ("sentiment_emotion.csv", sentiment_emotion),
        ("token_pos_counts.csv", token_pos_counts),
        ("readability_scores.csv", readability_scores),
        ("ma_ttr_mtld_scores.csv", ma_ttr_mtld),
        ("pos_analysis.csv", pos_analysis),
    ]
    for name, fn in steps:
        try:
            out_df = fn(df)
            out_path = FEAT_DEV / name
            out_df.to_csv(out_path, index=False)
            append_log(f"[extract_dev_features] Wrote {out_path} ({len(out_df)} rows)")
        except Exception as e:
            append_log(f"[extract_dev_features] ERROR {name}: {e}")
            raise

    append_log("[extract_dev_features] END")


if __name__ == "__main__":
    main()
