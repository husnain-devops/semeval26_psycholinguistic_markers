"""Meso coreference: inherit Actor/Victim onto later mentions in the same chain."""

from __future__ import annotations

import logging
import re

from layers.spans import Span, span_from_offsets

logger = logging.getLogger(__name__)

_FASTCOREF = None
_FASTCOREF_TRIED = False
_COREF_BACKEND = "uninitialized"


def _load_fastcoref():
    global _FASTCOREF, _FASTCOREF_TRIED
    if _FASTCOREF_TRIED:
        return _FASTCOREF
    _FASTCOREF_TRIED = True
    try:
        from fastcoref import FCoref

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:  # noqa: BLE001
            device = "cpu"
        _FASTCOREF = FCoref(device=device)
        _set_coref_backend(f"fastcoref:{device}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fastcoref unavailable (%s); using lexical-repetition fallback", exc)
        _FASTCOREF = None
        _set_coref_backend("lexical")
    return _FASTCOREF


def _set_coref_backend(name: str) -> None:
    global _COREF_BACKEND
    if _COREF_BACKEND != name:
        logger.info("coref backend: %s", name)
    _COREF_BACKEND = name


def coref_backend_name() -> str:
    return _COREF_BACKEND


def _clusters_fastcoref(text: str) -> list[list[tuple[int, int]]]:
    model = _load_fastcoref()
    if model is None or not text.strip():
        return []
    try:
        preds = model.predict(texts=[text])
    except Exception as exc:  # noqa: BLE001
        logger.warning("fastcoref inference failed: %s", exc)
        return []
    if not preds:
        return []
    pred = preds[0]
    try:
        clusters = pred.get_clusters(as_strings=False)
    except Exception:
        clusters = getattr(pred, "clusters", []) or []
    out = []
    for cluster in clusters:
        mentions = []
        for item in cluster:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                mentions.append((int(item[0]), int(item[1])))
        if len(mentions) >= 2:
            out.append(mentions)
    return out


def _clusters_lexical(text: str, seed_spans: list[Span]) -> list[list[tuple[int, int]]]:
    """Repeat later exact mentions of Actor/Victim strings (lexical cohesion)."""
    clusters: list[list[tuple[int, int]]] = []
    seen_norm: set[str] = set()
    for seed in seed_spans:
        if seed.type not in {"Actor", "Victim"}:
            continue
        surface = seed.text.strip()
        if len(surface) < 2:
            continue
        norm = surface.lower()
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        pattern = re.compile(re.escape(surface), flags=re.IGNORECASE)
        mentions = [(m.start(), m.end()) for m in pattern.finditer(text)]
        if len(mentions) >= 2:
            clusters.append(mentions)
    return clusters


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and a1 > b0


def expand_coref(text: str, seed_spans: list[Span]) -> list[Span]:
    """Emit Actor/Victim spans for other mentions in a seed's coref cluster."""
    seeds = [s for s in seed_spans if s.type in {"Actor", "Victim"}]
    if not seeds:
        return []
    clusters = _clusters_fastcoref(text)
    source = "coref_fastcoref"
    if not clusters:
        clusters = _clusters_lexical(text, seeds)
        source = "coref_lexical"

    inherited: list[Span] = []
    for cluster in clusters:
        types_here: set[str] = set()
        for seed in seeds:
            if any(_overlaps(seed.startIndex, seed.endIndex, m0, m1) for m0, m1 in cluster):
                types_here.add(seed.type)
        if not types_here:
            continue
        for start, end in cluster:
            for typ in types_here:
                already = any(
                    s.type == typ and _overlaps(s.startIndex, s.endIndex, start, end)
                    for s in seeds
                )
                if already:
                    continue
                span = span_from_offsets(text, start, end, typ, "meso", source)
                if span:
                    inherited.append(span)
    return inherited
