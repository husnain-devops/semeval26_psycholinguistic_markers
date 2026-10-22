"""Wrap the official eval_token.py scorer and compute gold-side coverage."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from layers.paths import OUTPUTS, eval_token_path
from layers.spans import MARKER_TYPES

IOU_THRESHOLD = 0.5


def _load_eval_token():
    path = eval_token_path()
    spec = importlib.util.spec_from_file_location("semeval_eval_token", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official scorer at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["semeval_eval_token"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_pred_layers(preds: list[dict], layers: set[str] | None) -> list[dict]:
    if layers is None:
        return preds
    out = []
    for row in preds:
        markers = []
        for m in row.get("markers") or []:
            layer = m.get("layer")
            if layer is None or layer in layers:
                markers.append(m)
        rec = dict(row)
        rec["markers"] = markers
        out.append(rec)
    return out


def _token_iou_sets(mod, text: str, start: int, end: int):
    token_spans = mod.tokenize_text(text)
    return mod.char_span_to_token_set(start, end, token_spans), token_spans


def coverage_metrics(gold: list[dict], preds: list[dict], iou_threshold: float = IOU_THRESHOLD) -> dict:
    """Share of gold spans that overlap any same-type pred at IoU >= threshold.

    This is gold-side coverage (one gold can share a pred). Official O-F1 is one-to-one.
    """
    mod = _load_eval_token()
    pred_by_id = {r["_id"]: r for r in preds}
    counts = {t: {"gold": 0, "covered": 0} for t in MARKER_TYPES}
    total_gold = 0
    total_covered = 0
    for g in gold:
        text = g.get("text") or ""
        token_spans = mod.tokenize_text(text)
        pred_markers = (pred_by_id.get(g["_id"]) or {}).get("markers") or []
        for gm in g.get("markers") or []:
            typ = gm.get("type")
            if typ not in MARKER_TYPES:
                continue
            total_gold += 1
            counts[typ]["gold"] += 1
            gset = mod.char_span_to_token_set(gm["startIndex"], gm["endIndex"], token_spans)
            hit = False
            for pm in pred_markers:
                if pm.get("type") != typ:
                    continue
                pset = mod.char_span_to_token_set(pm["startIndex"], pm["endIndex"], token_spans)
                if mod.calculate_token_iou(gset, pset) >= iou_threshold:
                    hit = True
                    break
            if hit:
                total_covered += 1
                counts[typ]["covered"] += 1
    out = {
        "coverage_aggregate": (total_covered / total_gold) if total_gold else 0.0,
        "gold_spans": total_gold,
        "covered_spans": total_covered,
        "per_type": {},
    }
    for typ in MARKER_TYPES:
        g = counts[typ]["gold"]
        c = counts[typ]["covered"]
        out["per_type"][typ] = {
            "gold": g,
            "covered": c,
            "coverage": (c / g) if g else 0.0,
        }
    return out


def official_scores(gold: list[dict], preds: list[dict], scores_path: Path | None = None) -> dict:
    mod = _load_eval_token()
    raw, _formatted = mod.evaluate(gold, preds, iou_threshold=IOU_THRESHOLD)
    if scores_path is not None:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        mod.save_scores_to_codabench(raw, str(scores_path))
    return raw


def score_predictions(
    gold: list[dict],
    preds: list[dict],
    split: str,
    tag: str,
) -> dict:
    scores_path = OUTPUTS / f"scores_{split}_{tag}.json"
    official = official_scores(gold, preds, scores_path)
    cover = coverage_metrics(gold, preds)
    pred_n = 0
    pred_by_type = {t: 0 for t in MARKER_TYPES}
    gold_n = 0
    gold_by_type = {t: 0 for t in MARKER_TYPES}
    for row in preds:
        for m in row.get("markers") or []:
            if m.get("type") in MARKER_TYPES:
                pred_n += 1
                pred_by_type[m["type"]] += 1
    for row in gold:
        for m in row.get("markers") or []:
            if m.get("type") in MARKER_TYPES:
                gold_n += 1
                gold_by_type[m["type"]] += 1
    return {
        "tag": tag,
        "split": split,
        "official": official,
        "coverage": cover,
        "n_pred": pred_n,
        "n_gold": gold_n,
        "pred_by_type": pred_by_type,
        "gold_by_type": gold_by_type,
        "scores_path": str(scores_path),
    }


ABLATIONS = (
    ("micro", {"micro"}),
    ("micro_meso", {"micro", "meso"}),
    ("all", {"micro", "meso", "macro"}),
)


def score_ablations(gold: list[dict], preds: list[dict], split: str) -> list[dict]:
    results = []
    for tag, layers in ABLATIONS:
        filtered = filter_pred_layers(preds, layers)
        write_jsonl(OUTPUTS / f"preds_{split}_{tag}.jsonl", filtered)
        results.append(score_predictions(gold, filtered, split, tag))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score SCAFFOLD preds against gold.")
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--preds", type=Path, default=None)
    args = parser.parse_args()
    from eval.build_gold import write_gold

    gold_path = write_gold(args.split)
    pred_path = args.preds or (OUTPUTS / f"preds_{args.split}.jsonl")
    gold = load_jsonl(gold_path)
    preds = load_jsonl(pred_path)
    blob = score_ablations(gold, preds, args.split)
    out = OUTPUTS / f"ablation_{args.split}.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for rec in blob:
        off = rec["official"]
        print(
            f"{rec['tag']:12}  F1_agg={off.get('F1 (Agg)', 0):.4f}  "
            f"F1_macro={off.get('F1 (Macro)', 0):.4f}  "
            f"cov={rec['coverage']['coverage_aggregate']:.4f}  "
            f"pred={rec['n_pred']} gold={rec['n_gold']}"
        )


if __name__ == "__main__":
    main()
