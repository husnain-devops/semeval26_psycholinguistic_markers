#!/usr/bin/env python3
"""
1:1 gold vs predicted marker spans for the five extraction models.

Runs each marker model (Action, Actor, Effect, Evidence, Victim) on
train_rehydrated.jsonl and writes, for every sample:

  - gold spans
  - predicted spans
  - a paired comparison (exact / overlap / partial / missed / false_positive)

The train/val tag uses the same 90/10 split as train_5_models.py
(random_state=42) so held-out validation rows can be filtered later.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE.parent
DEFAULT_DATA = BASE / "train_rehydrated.jsonl"
DEFAULT_MODELS = BASE / "models"
DEFAULT_OUT_DIR = BASE / "error_analysis"
MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]
MODEL_NAME = "roberta-large"
MAX_LENGTH = 512
IOU_OVERLAP = 0.5


def tokenize_text(text: str):
    """Starter-pack whitespace/punctuation tokens as (start, end) char spans."""
    return [(m.start(), m.end()) for m in re.finditer(r"(\w+|[^\w\s])", text)]


def char_span_to_token_set(char_start, char_end, token_spans):
    covered = set()
    for token_idx, (t_start, t_end) in enumerate(token_spans):
        if char_start < t_end and char_end > t_start:
            covered.add(token_idx)
    return covered


def token_iou(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def span_text(text, start, end):
    if start is None or end is None:
        return None
    return text[start:end]


def decode_bio_spans(label_ids, offsets, id2l, marker_type):
    """Turn BIO token labels into character spans. Flushes a span left open at EOS."""
    spans = []
    curr_start = None
    prev_end = None
    window_end = 0
    for idx, label_id in enumerate(label_ids):
        label = id2l[int(label_id)]
        o_start, o_end = int(offsets[idx][0]), int(offsets[idx][1])
        if o_start == o_end == 0:
            continue
        window_end = max(window_end, o_end)
        if label.startswith("B-"):
            if curr_start is not None:
                spans.append(
                    {"startIndex": int(curr_start), "endIndex": int(prev_end), "type": marker_type}
                )
            curr_start = o_start
            prev_end = o_end
        elif label.startswith("I-") and curr_start is not None:
            prev_end = o_end
        elif curr_start is not None:
            spans.append(
                {"startIndex": int(curr_start), "endIndex": int(prev_end), "type": marker_type}
            )
            curr_start = None
    if curr_start is not None:
        spans.append(
            {"startIndex": int(curr_start), "endIndex": int(prev_end), "type": marker_type}
        )
    return spans, window_end


def pair_spans(gold_spans, pred_spans, token_spans, window_end):
    """Greedy 1:1 pairing within one marker type, highest token-IoU first."""
    gold = []
    for g in gold_spans:
        start, end = int(g["startIndex"]), int(g["endIndex"])
        gold.append(
            {
                "startIndex": start,
                "endIndex": end,
                "text": g.get("text"),
                "out_of_window": start >= window_end,
                "tokens": char_span_to_token_set(start, end, token_spans),
                "matched": False,
            }
        )
    pred = []
    for p in pred_spans:
        start, end = int(p["startIndex"]), int(p["endIndex"])
        pred.append(
            {
                "startIndex": start,
                "endIndex": end,
                "tokens": char_span_to_token_set(start, end, token_spans),
                "matched": False,
            }
        )

    candidates = []
    for gi, g in enumerate(gold):
        if g["out_of_window"]:
            continue
        for pi, p in enumerate(pred):
            iou = token_iou(g["tokens"], p["tokens"])
            if iou > 0:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda row: row[0], reverse=True)

    links = {}
    for iou, gi, pi in candidates:
        if gold[gi]["matched"] or pred[pi]["matched"]:
            continue
        gold[gi]["matched"] = True
        pred[pi]["matched"] = True
        links[gi] = (pi, iou)

    pairs = []
    for gi, g in enumerate(gold):
        if g["out_of_window"]:
            status, pred_span, iou = "out_of_window", None, 0.0
        elif gi in links:
            pi, iou = links[gi]
            pred_span = pred[pi]
            same_bounds = (
                g["startIndex"] == pred_span["startIndex"] and g["endIndex"] == pred_span["endIndex"]
            )
            if same_bounds:
                status = "exact"
            elif iou >= IOU_OVERLAP:
                status = "overlap"
            else:
                status = "partial"
        else:
            status, pred_span, iou = "missed", None, 0.0
        pairs.append({"gold": g, "pred": pred_span, "status": status, "token_iou": round(float(iou), 4)})

    for p in pred:
        if not p["matched"]:
            pairs.append({"gold": None, "pred": p, "status": "false_positive", "token_iou": 0.0})
    return pairs


def load_data(path: Path):
    with path.open() as f:
        data = [json.loads(line) for line in f if line.strip()]
    train_idx, val_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    split_of = {i: "train" for i in train_idx}
    split_of.update({i: "val" for i in val_idx})
    return data, split_of


def load_marker_model(model_dir: Path, device):
    with open(model_dir / "config_labels.json") as f:
        id2l = {int(k): v for k, v in json.load(f)["id2l"].items()}
    # Saved tokenizer.json is from an older tokenizers build and fails to load.
    # Training tokenized with the base roberta-large tokenizer.
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
    base = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(id2l))
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer, id2l


def predict_marker(model, tokenizer, id2l, texts, marker_type, device, batch_size):
    """Return (spans, window_end) per text."""
    all_spans = []
    window_ends = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping").numpy()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**encoded).logits
            else:
                logits = model(**encoded).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        for i in range(len(batch)):
            spans, window_end = decode_bio_spans(preds[i], offsets[i], id2l, marker_type)
            all_spans.append(spans)
            window_ends.append(window_end)
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"    {marker_type}: {min(start + batch_size, len(texts))}/{len(texts)}")
    return all_spans, window_ends


def comparison_record(item, split, text, by_type_pairs):
    by_type = {}
    counts = {k: 0 for k in ("exact", "overlap", "partial", "missed", "false_positive", "out_of_window")}
    for marker_type, pairs in by_type_pairs.items():
        rows = []
        for pair in pairs:
            g, p = pair["gold"], pair["pred"]
            counts[pair["status"]] += 1
            rows.append(
                {
                    "status": pair["status"],
                    "token_iou": pair["token_iou"],
                    "gold_start": None if g is None else g["startIndex"],
                    "gold_end": None if g is None else g["endIndex"],
                    "gold_text": None if g is None else (g["text"] if g["text"] is not None else span_text(text, g["startIndex"], g["endIndex"])),
                    "pred_start": None if p is None else p["startIndex"],
                    "pred_end": None if p is None else p["endIndex"],
                    "pred_text": None if p is None else span_text(text, p["startIndex"], p["endIndex"]),
                }
            )
        by_type[marker_type] = rows
    return {
        "_id": item["_id"],
        "split": split,
        "conspiracy": item.get("conspiracy"),
        "text": text,
        "markers": by_type,
        "counts": counts,
    }


def write_outputs(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train_marker_gold_vs_pred.jsonl"
    tsv_path = out_dir / "train_marker_gold_vs_pred.tsv"
    summary_path = out_dir / "train_marker_gold_vs_pred_summary.json"

    totals = {k: 0 for k in ("exact", "overlap", "partial", "missed", "false_positive", "out_of_window")}
    per_type = {
        t: {k: 0 for k in totals} for t in MARKER_TYPES
    }
    per_split = {"train": {k: 0 for k in totals}, "val": {k: 0 for k in totals}}

    with jsonl_path.open("w") as jf, tsv_path.open("w", newline="") as tf:
        writer = csv.writer(tf, delimiter="\t")
        writer.writerow(
            [
                "_id",
                "split",
                "marker_type",
                "status",
                "token_iou",
                "gold_start",
                "gold_end",
                "gold_text",
                "pred_start",
                "pred_end",
                "pred_text",
            ]
        )
        for rec in records:
            jf.write(json.dumps(rec) + "\n")
            for marker_type, rows in rec["markers"].items():
                for row in rows:
                    totals[row["status"]] += 1
                    per_type[marker_type][row["status"]] += 1
                    per_split[rec["split"]][row["status"]] += 1
                    writer.writerow(
                        [
                            rec["_id"],
                            rec["split"],
                            marker_type,
                            row["status"],
                            row["token_iou"],
                            row["gold_start"] if row["gold_start"] is not None else "",
                            row["gold_end"] if row["gold_end"] is not None else "",
                            row["gold_text"] or "",
                            row["pred_start"] if row["pred_start"] is not None else "",
                            row["pred_end"] if row["pred_end"] is not None else "",
                            row["pred_text"] or "",
                        ]
                    )

    summary = {
        "n_samples": len(records),
        "n_train": sum(1 for r in records if r["split"] == "train"),
        "n_val": sum(1 for r in records if r["split"] == "val"),
        "iou_overlap_threshold": IOU_OVERLAP,
        "status_counts": totals,
        "per_type": per_type,
        "per_split": per_split,
        "jsonl": str(jsonl_path),
        "tsv": str(tsv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare 5 extraction models to gold training markers")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only the first N samples")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="all = every row in the training file, tagged train/val with the training split",
    )
    args = parser.parse_args()

    data, split_of = load_data(args.data)
    indices = [i for i in range(len(data)) if args.split == "all" or split_of[i] == args.split]
    if args.limit > 0:
        indices = indices[: args.limit]
    samples = [data[i] for i in indices]
    texts = [item["text"] for item in samples]
    print(f"Loaded {len(data)} rows from {args.data.name}; comparing {len(samples)} ({args.split})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pred_by_type = {}
    window_by_type = {}
    for marker_type in MARKER_TYPES:
        model_dir = args.models_dir / f"roberta-{marker_type}" / "final"
        if not model_dir.exists():
            raise SystemExit(f"Model not found: {model_dir}")
        print(f"Loading {marker_type} from {model_dir}")
        model, tokenizer, id2l = load_marker_model(model_dir, device)
        spans, windows = predict_marker(
            model, tokenizer, id2l, texts, marker_type, device, args.batch_size
        )
        pred_by_type[marker_type] = spans
        window_by_type[marker_type] = windows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = []
    for i, item in enumerate(samples):
        text = item["text"]
        token_spans = tokenize_text(text)
        by_type_pairs = {}
        for marker_type in MARKER_TYPES:
            gold = [m for m in (item.get("markers") or []) if m.get("type") == marker_type]
            by_type_pairs[marker_type] = pair_spans(
                gold,
                pred_by_type[marker_type][i],
                token_spans,
                window_by_type[marker_type][i],
            )
        records.append(comparison_record(item, split_of[indices[i]], text, by_type_pairs))

    summary = write_outputs(records, args.out_dir)
    print("\nStatus counts (one row per gold span or unmatched prediction):")
    for status, n in summary["status_counts"].items():
        print(f"  {status:16} {n}")
    print(f"\nWrote {summary['jsonl']}")
    print(f"Wrote {summary['tsv']}")
    print(f"Wrote {args.out_dir / 'train_marker_gold_vs_pred_summary.json'}")


if __name__ == "__main__":
    main()
