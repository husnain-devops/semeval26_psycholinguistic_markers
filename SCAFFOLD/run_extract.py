#!/usr/bin/env python3
"""Run Micro / Meso / Macro extractors and score them against PsyCoMark gold."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.build_gold import write_gold
from eval.report import write_alignment_report
from eval.score import load_jsonl, score_ablations, write_jsonl
from layers.macro.argument import extract_evidence
from layers.macro.modality import extract_modality_features
from layers.meso.causal import extract_effects
from layers.meso.coref import coref_backend_name, expand_coref
from layers.micro.pos import extract_pos_features
from layers.micro.srl import extract_srl, srl_backend_name
from layers.micro.svo import extract_svo
from layers.nlp import parse
from layers.paths import OUTPUTS
from layers.spans import Span, merge_spans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scaffold")

LAYER_GROUPS = {
    "micro": {"micro"},
    "meso": {"micro", "meso"},
    "macro": {"micro", "meso", "macro"},
    "all": {"micro", "meso", "macro"},
}


def extract_document(text: str, wanted: set[str], srl_backend: str) -> tuple[list[Span], dict]:
    doc = parse(text)
    spans: list[Span] = []
    features = {
        "pos": extract_pos_features(text, doc),
        "modality": extract_modality_features(text),
    }
    if "micro" in wanted:
        spans.extend(extract_svo(text, doc))
        spans.extend(extract_srl(text, doc, backend=srl_backend))
    if "meso" in wanted:
        spans.extend(expand_coref(text, [s for s in spans if s.layer == "micro"]))
        spans.extend(extract_effects(text, doc))
    if "macro" in wanted:
        spans.extend(extract_evidence(text))
    return merge_spans(spans), features


def run(split: str, layers: str, srl_backend: str, limit: int | None, score: bool) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    wanted = LAYER_GROUPS[layers]
    gold_path = write_gold(split)
    gold = load_jsonl(gold_path)
    if limit:
        gold = gold[:limit]
    logger.info("extracting %s docs from %s (layers=%s srl=%s)", len(gold), split, layers, srl_backend)

    preds = []
    feat_rows = []
    t0 = time.time()
    for i, rec in enumerate(gold, 1):
        spans, features = extract_document(rec["text"], wanted, srl_backend)
        markers = []
        for span in spans:
            item = span.to_pred()
            item["text"] = span.text
            item["layer"] = span.layer
            item["source"] = span.source
            markers.append(item)
        preds.append({"_id": rec["_id"], "markers": markers})
        feat_rows.append({"_id": rec["_id"], **features})
        if i % 25 == 0 or i == len(gold):
            logger.info("  %s/%s  (%.1fs)", i, len(gold), time.time() - t0)

    pred_path = OUTPUTS / f"preds_{split}.jsonl"
    write_jsonl(pred_path, preds)
    write_jsonl(OUTPUTS / f"features_{split}.jsonl", feat_rows)
    write_jsonl(
        OUTPUTS / f"preds_{split}_debug.jsonl",
        preds,
    )
    meta = {
        "split": split,
        "n_docs": len(gold),
        "layers": sorted(wanted),
        "srl_backend": srl_backend_name(),
        "srl_requested": srl_backend,
        "coref_backend": coref_backend_name(),
        "seconds": round(time.time() - t0, 2),
        "pred_file": str(pred_path),
        "gold_file": str(gold_path),
    }
    (OUTPUTS / f"run_meta_{split}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %s (%s spans)", pred_path, sum(len(p["markers"]) for p in preds))

    if not score:
        return
    ablation = score_ablations(gold, preds, split)
    (OUTPUTS / f"ablation_{split}.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    report = write_alignment_report(split, ablation, gold, preds, meta)
    logger.info("wrote %s", report)
    for rec in ablation:
        off = rec["official"]
        logger.info(
            "%s  F1_agg=%.4f F1_macro=%.4f cov=%.4f pred=%s",
            rec["tag"],
            off.get("F1 (Agg)", 0),
            off.get("F1 (Macro)", 0),
            rec["coverage"]["coverage_aggregate"],
            rec["n_pred"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--layers", choices=("all", "micro", "meso", "macro"), default="all")
    parser.add_argument("--srl", choices=("auto", "ud", "hf"), default="ud")
    parser.add_argument("--limit", type=int, default=None, help="optional doc cap for smoke tests")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    run(args.split, args.layers, args.srl, args.limit, score=not args.no_score)


if __name__ == "__main__":
    main()
