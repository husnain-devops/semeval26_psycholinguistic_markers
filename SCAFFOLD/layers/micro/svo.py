"""Micro SVO extraction via textacy, with a spaCy UD fallback and passive flip."""

from __future__ import annotations

import logging

from layers.nlp import clause_has_passive, mention_offsets, parse
from layers.spans import Span, span_from_offsets

logger = logging.getLogger(__name__)

_TEXTACY_TRIPLES = None
_TEXTACY_TRIED = False

SKIP_ACTORS = {"it", "there", "this", "that", "these", "those"}


def _load_textacy():
    global _TEXTACY_TRIPLES, _TEXTACY_TRIED
    if _TEXTACY_TRIED:
        return _TEXTACY_TRIPLES
    _TEXTACY_TRIED = True
    try:
        from textacy.extract import subject_verb_object_triples

        _TEXTACY_TRIPLES = subject_verb_object_triples
        logger.info("SVO backend: textacy")
    except Exception as exc:  # noqa: BLE001
        logger.warning("textacy SVO unavailable (%s); using spaCy UD fallback", exc)
        _TEXTACY_TRIPLES = None
    return _TEXTACY_TRIPLES


def _tokens_from_svoside(side) -> list:
    if side is None:
        return []
    if hasattr(side, "__iter__") and not hasattr(side, "idx"):
        tokens = []
        for item in side:
            if hasattr(item, "idx"):
                tokens.append(item)
            elif hasattr(item, "__iter__"):
                tokens.extend(t for t in item if hasattr(t, "idx"))
        return tokens
    if hasattr(side, "idx"):
        return [side]
    return []


def _head_of(tokens):
    if not tokens:
        return None
    # Prefer the token that is not a dependent of another token in the set.
    ids = {t.i for t in tokens}
    for tok in tokens:
        if tok.head.i not in ids or tok.head.i == tok.i:
            return tok
    return tokens[0]


def _emit_pair(text, actor_tok, action_tok, victim_tok, source: str) -> list[Span]:
    out: list[Span] = []
    if actor_tok is not None and actor_tok.text.lower() not in SKIP_ACTORS:
        start, end = mention_offsets(actor_tok, text)
        span = span_from_offsets(text, start, end, "Actor", "micro", source)
        if span:
            out.append(span)
    if action_tok is not None:
        start, end = action_tok.idx, action_tok.idx + len(action_tok)
        span = span_from_offsets(text, start, end, "Action", "micro", source)
        if span:
            out.append(span)
    if victim_tok is not None and victim_tok.text.lower() not in SKIP_ACTORS:
        start, end = mention_offsets(victim_tok, text)
        span = span_from_offsets(text, start, end, "Victim", "micro", source)
        if span:
            out.append(span)
    return out


def _extract_textacy(text: str, doc) -> list[Span]:
    fn = _load_textacy()
    if fn is None:
        return []
    spans: list[Span] = []
    try:
        triples = list(fn(doc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("textacy SVO failed on document: %s", exc)
        return []
    for triple in triples:
        subj_tokens = _tokens_from_svoside(getattr(triple, "subject", None))
        verb_tokens = _tokens_from_svoside(getattr(triple, "verb", None))
        obj_tokens = _tokens_from_svoside(getattr(triple, "object", None))
        subj = _head_of(subj_tokens)
        verb = _head_of(verb_tokens)
        obj = _head_of(obj_tokens)
        if verb is not None and clause_has_passive(verb):
            # Syntactic subject is the patient; agent may sit in a by-phrase.
            agent = None
            for child in verb.children:
                if child.dep_ == "agent":
                    agent = next((c for c in child.children if c.dep_ in {"pobj", "pcomp"}), child)
                    break
            actor, victim = agent, subj
        else:
            actor, victim = subj, obj
        spans.extend(_emit_pair(text, actor, verb, victim, "svo_textacy"))
    return spans


def _ud_predicates(doc):
    for token in doc:
        if not token.pos_ in {"VERB"}:
            continue
        yield token


def extract_svo_ud(text: str, doc=None) -> list[Span]:
    """Passive-aware SVO from Universal Dependencies (always available)."""
    doc = doc if doc is not None else parse(text)
    spans: list[Span] = []
    for verb in _ud_predicates(doc):
        nsubj = None
        nsubjpass = None
        obj = None
        agent = None
        for child in verb.children:
            if child.dep_ == "nsubj":
                nsubj = child
            elif child.dep_ == "nsubjpass":
                nsubjpass = child
            elif child.dep_ in {"dobj", "obj"}:
                obj = child
            elif child.dep_ == "agent":
                agent = next((c for c in child.children if c.dep_ in {"pobj", "pcomp"}), child)
        if clause_has_passive(verb) or nsubjpass is not None:
            actor = agent
            victim = nsubjpass or nsubj
        else:
            actor = nsubj
            victim = obj
        if actor is None and victim is None:
            # Still emit the predicate as Action so nominal-light clauses contribute.
            spans.extend(_emit_pair(text, None, verb, None, "svo_ud"))
            continue
        spans.extend(_emit_pair(text, actor, verb, victim, "svo_ud"))
    return spans


def extract_svo(text: str, doc=None) -> list[Span]:
    doc = doc if doc is not None else parse(text)
    spans = _extract_textacy(text, doc)
    # UD pass fills gaps (textacy sometimes misses passives / intransitives).
    spans.extend(extract_svo_ud(text, doc))
    return spans
