"""Micro lexical / POS features. Does not emit marker spans."""

from __future__ import annotations

from collections import Counter

from layers.nlp import parse

INGROUP = {"we", "us", "our", "ours", "ourselves"}
OUTGROUP = {"they", "them", "their", "theirs", "themselves"}
FIRST = {"i", "me", "my", "mine", "myself"}


def extract_pos_features(text: str, doc=None) -> dict:
    doc = doc if doc is not None else parse(text)
    tokens = [t for t in doc if not t.is_space]
    words = [t for t in tokens if t.is_alpha]
    pos_counts = Counter(t.pos_ for t in tokens)
    pronouns = [t for t in tokens if t.pos_ == "PRON"]
    lower = [t.text.lower() for t in pronouns]
    n_words = max(len(words), 1)
    types = {t.text.lower() for t in words}
    return {
        "n_tokens": len(tokens),
        "n_words": len(words),
        "n_sents": sum(1 for _ in doc.sents),
        "ttr": len(types) / n_words,
        "pronoun_rate": len(pronouns) / n_words,
        "ingroup_pronoun_rate": sum(1 for w in lower if w in INGROUP) / n_words,
        "outgroup_pronoun_rate": sum(1 for w in lower if w in OUTGROUP) / n_words,
        "first_person_rate": sum(1 for w in lower if w in FIRST) / n_words,
        "noun_rate": pos_counts.get("NOUN", 0) / n_words,
        "verb_rate": pos_counts.get("VERB", 0) / n_words,
        "adj_rate": pos_counts.get("ADJ", 0) / n_words,
        "pos_counts": dict(pos_counts),
    }
