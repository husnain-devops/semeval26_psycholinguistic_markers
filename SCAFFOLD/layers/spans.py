"""Typed character-offset spans matching the PsyCoMark schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

MARKER_TYPES = ("Action", "Actor", "Effect", "Evidence", "Victim")


@dataclass
class Span:
    startIndex: int
    endIndex: int
    type: str
    text: str = ""
    layer: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.startIndex < 0 or self.endIndex <= self.startIndex:
            raise ValueError(f"invalid span offsets: {self.startIndex}:{self.endIndex}")
        if self.type not in MARKER_TYPES:
            raise ValueError(f"unknown marker type: {self.type}")

    def to_pred(self) -> dict:
        return {
            "startIndex": int(self.startIndex),
            "endIndex": int(self.endIndex),
            "type": self.type,
        }

    def to_debug(self) -> dict:
        rec = self.to_pred()
        rec["text"] = self.text
        rec["layer"] = self.layer
        rec["source"] = self.source
        return rec

    @property
    def key(self) -> tuple:
        return (self.startIndex, self.endIndex, self.type)


def clip_span(start: int, end: int, text: str) -> tuple[int, int]:
    """Clamp offsets to the document and drop empty ranges."""
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def span_from_offsets(
    text: str,
    start: int,
    end: int,
    typ: str,
    layer: str,
    source: str,
) -> Span | None:
    start, end = clip_span(start, end, text)
    if end <= start:
        return None
    return Span(
        startIndex=start,
        endIndex=end,
        type=typ,
        text=text[start:end],
        layer=layer,
        source=source,
    )


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Union per type; drop exact duplicates; collapse nested same-type to longest."""
    by_type: dict[str, list[Span]] = {t: [] for t in MARKER_TYPES}
    seen: set[tuple] = set()
    for span in spans:
        if span.key in seen:
            continue
        seen.add(span.key)
        by_type[span.type].append(span)

    kept: list[Span] = []
    for typ in MARKER_TYPES:
        group = sorted(
            by_type[typ],
            key=lambda s: (s.endIndex - s.startIndex, -s.startIndex),
            reverse=True,
        )
        selected: list[Span] = []
        for cand in group:
            nested = False
            for other in selected:
                if cand.startIndex >= other.startIndex and cand.endIndex <= other.endIndex:
                    nested = True
                    break
            if not nested:
                selected.append(cand)
        kept.extend(selected)
    kept.sort(key=lambda s: (s.startIndex, s.endIndex, s.type))
    return kept


def to_pred_markers(spans: Iterable[Span]) -> list[dict]:
    return [s.to_pred() for s in merge_spans(spans)]


def filter_layers(spans: Iterable[Span], layers: set[str]) -> list[Span]:
    return [s for s in spans if s.layer in layers]
