"""Macro hedge / certainty / discourse features. Does not emit marker spans."""

from __future__ import annotations

import re

from layers.nlp import load_lexicon


def _compiled() -> dict[str, list[tuple[str, re.Pattern]]]:
    grouped: dict[str, list[tuple[str, re.Pattern]]] = {
        "hedge": [],
        "certainty": [],
        "discourse": [],
    }
    for kind, phrase in load_lexicon("hedges.txt"):
        if kind not in grouped:
            grouped[kind] = []
        grouped[kind].append(
            (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE))
        )
    return grouped


_COMPILED = None


def extract_modality_features(text: str) -> dict:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = _compiled()
    hits: dict[str, list[str]] = {k: [] for k in _COMPILED}
    counts = {f"{k}_count": 0 for k in _COMPILED}
    for kind, pairs in _COMPILED.items():
        for phrase, pat in pairs:
            found = pat.findall(text)
            if found:
                counts[f"{kind}_count"] += len(found)
                hits[kind].append(phrase)
    n_tok = max(len(text.split()), 1)
    return {
        **counts,
        "hedge_rate": counts.get("hedge_count", 0) / n_tok,
        "certainty_rate": counts.get("certainty_count", 0) / n_tok,
        "certainty_minus_hedge": counts.get("certainty_count", 0) - counts.get("hedge_count", 0),
        "hedge_hits": hits.get("hedge", []),
        "certainty_hits": hits.get("certainty", []),
        "discourse_hits": hits.get("discourse", []),
    }
