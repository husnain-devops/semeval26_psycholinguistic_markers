"""Write alignment_report.md: official O-F1, coverage, ablations, qualitative examples."""

from __future__ import annotations

import json
from pathlib import Path

from eval.score import IOU_THRESHOLD, _load_eval_token
from layers.paths import OUTPUTS
from layers.spans import MARKER_TYPES

RB_EXT_5M = {
    "note": "Published NUST PsyAI RB-EXT-5M on dev (token IoU >= 0.5). "
    "Overall ~0.22 after post-submission refinement; Actor ~0.34, Victim ~0.25, Effect ~0.10.",
    "F1_overall_dev": 0.22,
    "F1_Actor": 0.34,
    "F1_Victim": 0.25,
    "F1_Effect": 0.10,
}


def _match_examples(gold: list[dict], preds: list[dict], per_bucket: int = 4) -> dict:
    mod = _load_eval_token()
    pred_by_id = {r["_id"]: r for r in preds}
    buckets = {
        typ: {"tp": [], "fp": [], "fn": []}
        for typ in MARKER_TYPES
    }
    for g in gold:
        text = g.get("text") or ""
        doc_id = g["_id"]
        token_spans = mod.tokenize_text(text)
        gold_ms = [m for m in (g.get("markers") or []) if m.get("type") in MARKER_TYPES]
        pred_ms = [
            m
            for m in ((pred_by_id.get(doc_id) or {}).get("markers") or [])
            if m.get("type") in MARKER_TYPES
        ]
        used_pred = set()
        for gi, gm in enumerate(gold_ms):
            gset = mod.char_span_to_token_set(gm["startIndex"], gm["endIndex"], token_spans)
            best_i, best_iou = None, -1.0
            for pi, pm in enumerate(pred_ms):
                if pi in used_pred or pm.get("type") != gm.get("type"):
                    continue
                pset = mod.char_span_to_token_set(pm["startIndex"], pm["endIndex"], token_spans)
                iou = mod.calculate_token_iou(gset, pset)
                if iou > best_iou:
                    best_iou, best_i = iou, pi
            rec = {
                "_id": doc_id,
                "gold": text[gm["startIndex"]:gm["endIndex"]],
                "gold_type": gm["type"],
                "gold_off": [gm["startIndex"], gm["endIndex"]],
            }
            if best_i is not None and best_iou >= IOU_THRESHOLD:
                used_pred.add(best_i)
                pm = pred_ms[best_i]
                rec.update(
                    {
                        "pred": text[pm["startIndex"]:pm["endIndex"]],
                        "pred_off": [pm["startIndex"], pm["endIndex"]],
                        "layer": pm.get("layer"),
                        "source": pm.get("source"),
                        "iou": round(best_iou, 3),
                    }
                )
                if len(buckets[gm["type"]]["tp"]) < per_bucket:
                    buckets[gm["type"]]["tp"].append(rec)
            else:
                if len(buckets[gm["type"]]["fn"]) < per_bucket:
                    buckets[gm["type"]]["fn"].append(rec)
        for pi, pm in enumerate(pred_ms):
            if pi in used_pred:
                continue
            typ = pm["type"]
            if len(buckets[typ]["fp"]) < per_bucket:
                buckets[typ]["fp"].append(
                    {
                        "_id": doc_id,
                        "pred": text[pm["startIndex"]:pm["endIndex"]],
                        "pred_off": [pm["startIndex"], pm["endIndex"]],
                        "layer": pm.get("layer"),
                        "source": pm.get("source"),
                    }
                )
    return buckets


def _fmt_f1(official: dict, typ: str) -> str:
    return f"{official.get(f'F1 ({typ})', 0):.3f}"


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        line = "| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(row)) + " |"
        lines.append(line)
        if i == 0:
            lines.append("| " + " | ".join("-" * widths[j] for j in range(len(row))) + " |")
    return "\n".join(lines)


def write_alignment_report(
    split: str,
    ablation: list[dict],
    gold: list[dict],
    preds: list[dict],
    meta: dict,
    dest: Path | None = None,
) -> Path:
    dest = dest or (OUTPUTS / "alignment_report.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    examples = _match_examples(gold, preds)
    by_tag = {r["tag"]: r for r in ablation}
    full = by_tag.get("all") or ablation[-1]

    lines = [
        "# SCAFFOLD layer alignment report",
        "",
        f"Split: **{split}** ({len(gold)} documents, {full['n_gold']} gold spans).",
        "",
        "Official metric is token-overlap F1 from `eval_token.py` (IoU ≥ 0.5, one-to-one, type-constrained).",
        "Coverage is the share of gold spans that overlap *any* same-type prediction at IoU ≥ 0.5 (prior usefulness).",
        "",
        "## Layer ablations",
        "",
    ]
    header = ["Setting", "Pred", "P_agg", "R_agg", "F1_agg", "F1_macro", "Coverage"]
    table = [header]
    for rec in ablation:
        off = rec["official"]
        table.append(
            [
                rec["tag"],
                str(rec["n_pred"]),
                f"{off.get('P (Agg)', 0):.3f}",
                f"{off.get('R (Agg)', 0):.3f}",
                f"{off.get('F1 (Agg)', 0):.3f}",
                f"{off.get('F1 (Macro)', 0):.3f}",
                f"{rec['coverage']['coverage_aggregate']:.3f}",
            ]
        )
    lines.append(_table(table))
    lines.extend(["", "## Per-type official F1 and coverage (all layers)", ""])
    header = ["Type", "Gold", "Pred", "P", "R", "F1", "Coverage", "RB-EXT-5M F1 (ref)"]
    table = [header]
    off = full["official"]
    for typ in MARKER_TYPES:
        ref = RB_EXT_5M.get(f"F1_{typ}")
        ref_s = f"{ref:.2f}" if isinstance(ref, float) else "—"
        table.append(
            [
                typ,
                str(full["gold_by_type"].get(typ, 0)),
                str(full["pred_by_type"].get(typ, 0)),
                f"{off.get(f'P ({typ})', 0):.3f}",
                f"{off.get(f'R ({typ})', 0):.3f}",
                f"{off.get(f'F1 ({typ})', 0):.3f}",
                f"{full['coverage']['per_type'][typ]['coverage']:.3f}",
                ref_s,
            ]
        )
    lines.append(_table(table))
    lines.extend(
        [
            "",
            f"RB-EXT-5M reference: {RB_EXT_5M['note']}",
            "",
            "## Extractor metadata",
            "",
            "```json",
            json.dumps(meta, indent=2),
            "```",
            "",
            "## Qualitative examples (all layers)",
            "",
            "Up to four true positives, false positives, and false negatives per type.",
            "",
        ]
    )
    for typ in MARKER_TYPES:
        lines.append(f"### {typ}")
        lines.append("")
        for bucket, title in (("tp", "True positives"), ("fp", "False positives"), ("fn", "False negatives")):
            lines.append(f"**{title}**")
            lines.append("")
            items = examples[typ][bucket]
            if not items:
                lines.append("_none_")
                lines.append("")
                continue
            for item in items:
                if bucket == "tp":
                    lines.append(
                        f"- `{item['_id']}` gold `{item['gold']}` ↔ pred `{item['pred']}` "
                        f"(IoU {item.get('iou')}, {item.get('layer')}/{item.get('source')})"
                    )
                elif bucket == "fp":
                    lines.append(
                        f"- `{item['_id']}` pred `{item['pred']}` "
                        f"({item.get('layer')}/{item.get('source')})"
                    )
                else:
                    lines.append(f"- `{item['_id']}` gold `{item['gold']}`")
            lines.append("")

    lines.extend(
        [
            "## How to read these numbers",
            "",
            "Gold Actor/Action/Victim are conspiratorial roles, not every subject-verb-object.",
            "Rule layers are meant as *structural priors*: high coverage with low precision is still useful if a later encoder can reject extras.",
            "Meso should add Effect; macro should add Evidence. If those ablations do not raise coverage for those types, the layer is not helping.",
            "",
        ]
    )
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
