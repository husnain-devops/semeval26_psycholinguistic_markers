"""PDTB-style explicit causal connectives → Effect (consequence clause)."""

from __future__ import annotations

import re

from layers.nlp import load_lexicon, parse
from layers.spans import Span, span_from_offsets

MAX_EFFECT_CHARS = 120


def _connectives() -> list[tuple[str, str, re.Pattern]]:
    rows = []
    for kind, phrase in load_lexicon("causal_connectives.txt"):
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        rows.append((kind, phrase, pat))
    return rows


def _trim_clause(text: str, start: int, end: int) -> tuple[int, int]:
    chunk = text[start:end]
    # Cut at sentence-ish punctuation if the clause ran long.
    cut = None
    for i, ch in enumerate(chunk):
        if ch in ".!?;" and i > 12:
            cut = i + 1
            break
    if cut is not None:
        end = start + cut
    if end - start > MAX_EFFECT_CHARS:
        end = start + MAX_EFFECT_CHARS
    return start, end


def _sentence_window(doc, char_index: int):
    for sent in doc.sents:
        if sent.start_char <= char_index < sent.end_char:
            return sent
    return None


def _matrix_clause(text: str, doc, match_start: int, match_end: int) -> tuple[int, int] | None:
    sent = _sentence_window(doc, match_start)
    if sent is None:
        return None
    # "X because Y" → X is the left material; "Because Y, X" → X is after the comma.
    left_start, left_end = sent.start_char, match_start
    right_start, right_end = match_end, sent.end_char
    left = text[left_start:left_end].strip()
    if len(left) >= 8:
        return left_start, left_end
    if (right_end - right_start) >= 8:
        return right_start, right_end
    return None


def _follow_clause(text: str, doc, match_end: int) -> tuple[int, int] | None:
    sent = _sentence_window(doc, max(match_end - 1, 0))
    if sent is None:
        return None
    start, end = match_end, sent.end_char
    if end - start < 4:
        # Consequence may be the next sentence ("Therefore, ...").
        sents = list(doc.sents)
        for i, s in enumerate(sents):
            if s.start_char <= match_end <= s.end_char and i + 1 < len(sents):
                start, end = sents[i + 1].start_char, sents[i + 1].end_char
                break
    if end - start < 4:
        return None
    return start, end


def extract_effects(text: str, doc=None) -> list[Span]:
    doc = doc if doc is not None else parse(text)
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for kind, phrase, pat in _connectives():
        for match in pat.finditer(text):
            # Avoid matching the "as" in "as well" etc. for the short matrix connective.
            if phrase.lower() == "as":
                after = text[match.end() : match.end() + 8].lower()
                if after.startswith(" well") or after.startswith(" if") or after.startswith(" to"):
                    continue
            if phrase.lower() == "so":
                after = text[match.end() : match.end() + 6].lower()
                if after.startswith(" much") or after.startswith(" many") or after.startswith(" far"):
                    continue
            if kind == "matrix":
                bounds = _matrix_clause(text, doc, match.start(), match.end())
            else:
                bounds = _follow_clause(text, doc, match.end())
            if bounds is None:
                continue
            start, end = _trim_clause(text, *bounds)
            if any(start < o1 and end > o0 for o0, o1 in occupied):
                continue
            span = span_from_offsets(text, start, end, "Effect", "meso", f"causal:{phrase}")
            if span and len(span.text.split()) >= 1:
                spans.append(span)
                occupied.append((span.startIndex, span.endIndex))
    return spans
