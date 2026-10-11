"""Lightweight argument miner: evidential cues and quotes → Evidence."""

from __future__ import annotations

import re

from layers.nlp import load_lexicon
from layers.spans import Span, span_from_offsets

QUOTE_RE = re.compile(r"[\"“]([^\"”]{3,80})[\"”]")


def _cues() -> list[tuple[str, str, re.Pattern]]:
    rows = []
    for kind, phrase in load_lexicon("evidence_cues.txt"):
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        rows.append((kind, phrase, pat))
    return rows


def _short_follow(text: str, start: int, limit: int = 40) -> int:
    end = min(len(text), start + limit)
    chunk = text[start:end]
    for i, ch in enumerate(chunk):
        if ch in ".,;:!?)" and i > 3:
            return start + i
    # stop after a few tokens
    tokens = list(re.finditer(r"\S+", chunk))
    if tokens:
        last = tokens[min(len(tokens), 4) - 1]
        return start + last.end()
    return end


def extract_evidence(text: str) -> list[Span]:
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []

    def _add(start: int, end: int, source: str) -> None:
        if any(start < o1 and end > o0 for o0, o1 in occupied):
            return
        span = span_from_offsets(text, start, end, "Evidence", "macro", source)
        if span:
            spans.append(span)
            occupied.append((span.startIndex, span.endIndex))

    for kind, phrase, pat in _cues():
        for match in pat.finditer(text):
            if kind == "cue":
                _add(match.start(), match.end(), f"arg_cue:{phrase}")
                # "according to X" → also mark a short following source NP
                if phrase.lower() in {"according to", "as reported", "as shown"}:
                    follow_end = _short_follow(text, match.end())
                    if follow_end > match.end() + 1:
                        _add(match.end(), follow_end, f"arg_source:{phrase}")
            else:
                # reporting verb: emit the verb plus a short complement
                follow_end = _short_follow(text, match.end(), limit=50)
                _add(match.start(), max(match.end(), follow_end), f"arg_report:{phrase}")

    for match in QUOTE_RE.finditer(text):
        _add(match.start(1), match.end(1), "arg_quote")

    return spans
