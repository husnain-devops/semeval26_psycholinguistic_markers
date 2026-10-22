"""PropBank-style SRL: Arg0=Actor, predicate=Action, Arg1=Victim.

Tries a HuggingFace token-classification SRL model when --srl hf is set.
Falls back to a documented UD role proxy (nsubj/nsubjpass/dobj/agent).
"""

from __future__ import annotations

import logging
import os
import re

from layers.nlp import clause_has_passive, mention_offsets, parse
from layers.spans import Span, span_from_offsets

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = os.environ.get("SCAFFOLD_SRL_MODEL", "J4ckB/bert-base-uncased-srl")

_BACKEND = None
_HF_PIPELINE = None
_HF_TRIED = False

SKIP = {"it", "there", "this", "that"}
ARG_RE = re.compile(r"^(B|I)-(ARG0|ARG1|V|PRED|REL)$", re.I)
BIO_RE = re.compile(r"^(B|I)-(.+)$")


def srl_backend_name() -> str:
    return _BACKEND or "uninitialized"


def _set_backend(name: str) -> None:
    global _BACKEND
    if _BACKEND != name:
        logger.info("SRL backend: %s", name)
    _BACKEND = name


def _try_hf_pipeline():
    global _HF_PIPELINE, _HF_TRIED
    if _HF_TRIED:
        return _HF_PIPELINE
    _HF_TRIED = True
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    except Exception as exc:  # noqa: BLE001
        logger.warning("transformers unavailable for HF SRL: %s", exc)
        return None
    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(DEFAULT_HF_MODEL)
        _HF_PIPELINE = pipeline(
            "token-classification",
            model=model,
            tokenizer=tok,
            aggregation_strategy="simple",
        )
        logger.info("loaded HF SRL model %s", DEFAULT_HF_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF SRL model %s failed to load (%s); using UD proxy", DEFAULT_HF_MODEL, exc)
        _HF_PIPELINE = None
    return _HF_PIPELINE


def _normalize_role(label: str) -> str | None:
    lab = label.upper().replace(" ", "")
    lab = lab.split("-")[-1] if "-" in lab else lab
    if lab in {"ARG0", "A0", "C-ARG0"}:
        return "Actor"
    if lab in {"ARG1", "A1", "C-ARG1"}:
        return "Victim"
    if lab in {"V", "PRED", "REL", "VERB"}:
        return "Action"
    return None


def _extract_hf(text: str) -> list[Span]:
    nlp_pipe = _try_hf_pipeline()
    if nlp_pipe is None:
        return []
    try:
        ents = nlp_pipe(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF SRL inference failed: %s", exc)
        return []
    spans: list[Span] = []
    for ent in ents:
        typ = _normalize_role(str(ent.get("entity_group") or ent.get("entity") or ""))
        if typ is None:
            continue
        start = int(ent.get("start", -1))
        end = int(ent.get("end", -1))
        span = span_from_offsets(text, start, end, typ, "micro", "srl_hf")
        if span:
            spans.append(span)
    return spans


def extract_srl_ud(text: str, doc=None) -> list[Span]:
    """UD proxy for PropBank Arg0 / predicate / Arg1."""
    doc = doc if doc is not None else parse(text)
    spans: list[Span] = []
    for verb in doc:
        if verb.pos_ != "VERB":
            continue
        arg0 = None
        arg1 = None
        for child in verb.children:
            if child.dep_ in {"nsubj", "csubj"}:
                arg0 = child
            elif child.dep_ in {"nsubjpass", "csubjpass"}:
                arg1 = child
            elif child.dep_ in {"dobj", "obj", "attr", "dative"}:
                arg1 = arg1 or child
            elif child.dep_ == "agent":
                arg0 = next((c for c in child.children if c.dep_ in {"pobj", "pcomp"}), child)
        if clause_has_passive(verb) and arg0 is None:
            # Patient already in arg1; leave Actor empty if no by-phrase.
            pass
        if arg0 is not None and arg0.text.lower() not in SKIP:
            start, end = mention_offsets(arg0, text)
            span = span_from_offsets(text, start, end, "Actor", "micro", "srl_ud")
            if span:
                spans.append(span)
        pred = span_from_offsets(
            text, verb.idx, verb.idx + len(verb), "Action", "micro", "srl_ud"
        )
        if pred:
            spans.append(pred)
        if arg1 is not None and arg1.text.lower() not in SKIP:
            start, end = mention_offsets(arg1, text)
            span = span_from_offsets(text, start, end, "Victim", "micro", "srl_ud")
            if span:
                spans.append(span)
    return spans


def extract_srl(text: str, doc=None, backend: str = "auto") -> list[Span]:
    """backend: auto | hf | ud. auto tries HF only when SCAFFOLD_SRL=hf."""
    prefer = backend
    if prefer == "auto":
        prefer = os.environ.get("SCAFFOLD_SRL", "ud")
    if prefer == "hf":
        spans = _extract_hf(text)
        if spans:
            _set_backend(f"hf:{DEFAULT_HF_MODEL}")
            return spans
        logger.warning("HF SRL produced no spans; falling back to UD proxy")
    _set_backend("ud_proxy")
    return extract_srl_ud(text, doc)
