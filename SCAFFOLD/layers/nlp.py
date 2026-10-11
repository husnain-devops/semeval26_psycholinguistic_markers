"""Shared spaCy pipeline and lexicon loaders."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

LEXICON_DIR = Path(__file__).resolve().parents[1] / "lexicons"

_NLP = None
_NLP_LOAD_ERROR = None


def get_nlp():
    """Load en_core_web_sm once. Raises if spaCy or the model is missing."""
    global _NLP, _NLP_LOAD_ERROR
    if _NLP is not None:
        return _NLP
    if _NLP_LOAD_ERROR is not None:
        raise _NLP_LOAD_ERROR
    try:
        import spacy
    except ImportError as exc:
        _NLP_LOAD_ERROR = RuntimeError(
            "spaCy is required. Install SCAFFOLD/requirements.txt in a venv "
            "that already has en_core_web_sm (see EDA-Rehydrated/requirements.txt)."
        )
        raise _NLP_LOAD_ERROR from exc
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError as exc:
        _NLP_LOAD_ERROR = RuntimeError(
            "spaCy model en_core_web_sm is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        )
        raise _NLP_LOAD_ERROR from exc
    return _NLP


def parse(text: str):
    return get_nlp()(text or " ")


@lru_cache(maxsize=None)
def load_lexicon(name: str) -> list[tuple[str, str]]:
    """Return (kind, phrase) pairs, longest phrase first, comments skipped."""
    path = LEXICON_DIR / name
    rows: list[tuple[str, str]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            kind, phrase = line.split("\t", 1)
        else:
            kind, phrase = "cue", line
        phrase = phrase.strip()
        if phrase:
            rows.append((kind.strip(), phrase))
    rows.sort(key=lambda kv: len(kv[1]), reverse=True)
    return rows


def mention_offsets(token, text: str, max_tokens: int = 4) -> tuple[int, int]:
    """Short NP around a head, otherwise the head token only.

    Gold Actor/Victim spans are typically 1-4 tokens. Emitting a full subtree
    would drop token-IoU below 0.5.
    """
    subtree = [t for t in token.subtree if t.i >= token.left_edge.i and t.i <= token.right_edge.i]
    content = [t for t in subtree if not t.is_space]
    if 0 < len(content) <= max_tokens:
        start = content[0].idx
        end = content[-1].idx + len(content[-1])
        return start, end
    return token.idx, token.idx + len(token)


def is_verbal(token) -> bool:
    return token.pos_ in {"VERB", "AUX"} and token.dep_ != "aux"


def clause_has_passive(verb) -> bool:
    return any(c.dep_ == "auxpass" for c in verb.children) or any(
        c.dep_ == "nsubjpass" for c in verb.children
    )
