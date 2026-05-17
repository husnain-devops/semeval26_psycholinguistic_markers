"""
Copy of infer_5_models.py that runs on dev_rehydrated.jsonl, uses dev_public.jsonl
as gold markers, and computes span-overlap F1 for extracted markers.
"""
import json
import re
import torch
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent.parent
DEV_REHYDRATED_FILE = PROJECT_ROOT / "dev_rehydrated.jsonl"
DEV_PUBLIC_FILE = PROJECT_ROOT / "dev_public.jsonl"
MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]
MAX_LENGTH = 512
IOU_THRESHOLD = 0.5


def load_dev_data():
    """Load dev_rehydrated (text by _id) and dev_public (gold markers by _id)."""
    with open(DEV_REHYDRATED_FILE) as f:
        dev_rehydrated = [json.loads(line) for line in f]
    gold_by_id = {}
    with open(DEV_PUBLIC_FILE) as f:
        for line in f:
            item = json.loads(line)
            gold_by_id[item["_id"]] = item.get("markers") or []
    return dev_rehydrated, gold_by_id


def tokenize_text(text):
    """
    Tokenization used by starter-pack eval_token.py.
    Returns list of (start_char, end_char) token spans.
    """
    token_spans = []
    for match in re.finditer(r"(\w+|[^\w\s])", text):
        token_spans.append((match.start(), match.end()))
    return token_spans


def char_span_to_token_set(char_start, char_end, token_spans):
    """Convert character span to covered token index set."""
    covered_token_indices = set()
    for token_idx, (t_start, t_end) in enumerate(token_spans):
        if char_start < t_end and char_end > t_start:
            covered_token_indices.add(token_idx)
    return covered_token_indices


def calculate_token_iou(set_a, set_b):
    """IoU over token index sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    intersection = set_a.intersection(set_b)
    return len(intersection) / len(union)


def count_matched_spans_token_iou(pred_spans, gold_spans, token_spans, iou_threshold=IOU_THRESHOLD):
    """
    Starter-pack-style matching:
    For each gold span, find best unmatched predicted span of same type by token IoU.
    Count TP if best IoU >= threshold. Then FP/FN from unmatched spans.
    """
    # Copy spans with mutable matched flags
    pred = [
        {"start": p["startIndex"], "end": p["endIndex"], "type": p["type"], "matched": False}
        for p in pred_spans
        if p.get("type") in MARKER_TYPES
    ]
    gold = [
        {"start": g["startIndex"], "end": g["endIndex"], "type": g["type"], "matched": False}
        for g in gold_spans
        if g.get("type") in MARKER_TYPES
    ]

    per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in MARKER_TYPES}

    for true_span in gold:
        true_token_set = char_span_to_token_set(true_span["start"], true_span["end"], token_spans)
        best_iou = -1.0
        best_pred_idx = -1

        for pred_idx, pred_span in enumerate(pred):
            if pred_span["matched"] or pred_span["type"] != true_span["type"]:
                continue
            pred_token_set = char_span_to_token_set(pred_span["start"], pred_span["end"], token_spans)
            iou = calculate_token_iou(true_token_set, pred_token_set)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx != -1 and best_iou >= iou_threshold:
            true_span["matched"] = True
            pred[best_pred_idx]["matched"] = True
            per_type[true_span["type"]]["tp"] += 1

    for true_span in gold:
        if not true_span["matched"]:
            per_type[true_span["type"]]["fn"] += 1

    for pred_span in pred:
        if not pred_span["matched"]:
            per_type[pred_span["type"]]["fp"] += 1

    return per_type


def run_inference(dev_data):
    """Run 5 models on dev_rehydrated items; return list of predicted markers per item (same order as dev_data)."""
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", add_prefix_space=True)
    all_results = [[] for _ in range(len(dev_data))]
    models_base = BASE / ".." / "models"

    for m_type in MARKER_TYPES:
        print(f"Extracting {m_type}...")
        model_path = models_base / f"roberta-{m_type}" / "final"
        with open(model_path / "config_labels.json") as f:
            id2l = {int(k): v for k, v in json.load(f)["id2l"].items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RobertaForTokenClassification.from_pretrained(
            model_path, num_labels=3, ignore_mismatched_sizes=True
        ).to(device)

        for i, item in enumerate(dev_data):
            inputs = tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
            ).to(device)
            offsets = inputs.pop("offset_mapping")[0].cpu().numpy()

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                preds = np.argmax(probs, axis=-1)

            curr_start = None
            for idx, label_id in enumerate(preds):
                label = id2l[label_id]
                o_start, o_end = offsets[idx]
                if o_start == o_end == 0:
                    continue
                if label.startswith("B-"):
                    if curr_start is not None:
                        all_results[i].append(
                            {"startIndex": int(curr_start), "endIndex": int(prev_end), "type": m_type}
                        )
                    curr_start = o_start
                    prev_end = o_end
                elif label.startswith("I-") and curr_start is not None:
                    prev_end = o_end
                else:
                    if curr_start is not None:
                        all_results[i].append(
                            {"startIndex": int(curr_start), "endIndex": int(prev_end), "type": m_type}
                        )
                        curr_start = None
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_results


def compute_f1(dev_data, gold_by_id, all_results):
    """Compute starter-pack-compatible token-IoU F1 per type + micro/macro."""
    per_type = {t: {"tp": 0, "pred": 0, "gold": 0} for t in MARKER_TYPES}
    micro = {"tp": 0, "pred": 0, "gold": 0}

    for i, item in enumerate(dev_data):
        _id = item["_id"]
        pred_spans = all_results[i]
        gold_spans = gold_by_id.get(_id, [])
        token_spans = tokenize_text(item["text"])

        # Normalize gold: keep only startIndex, endIndex, type
        gold_spans = [
            {"startIndex": m["startIndex"], "endIndex": m["endIndex"], "type": m["type"]}
            for m in gold_spans
            if m.get("type") in MARKER_TYPES
        ]

        matched = count_matched_spans_token_iou(
            pred_spans, gold_spans, token_spans, iou_threshold=IOU_THRESHOLD
        )
        for t in MARKER_TYPES:
            tp = matched[t]["tp"]
            fp = matched[t]["fp"]
            fn = matched[t]["fn"]
            per_type[t]["tp"] += tp
            per_type[t]["pred"] += (tp + fp)
            per_type[t]["gold"] += (tp + fn)
            micro["tp"] += tp
            micro["pred"] += (tp + fp)
            micro["gold"] += (tp + fn)

    def p_r_f1(tp, pred, gold):
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    print("\n" + "=" * 60)
    print(f"Token-IoU Overlap F1 @ IoU>={IOU_THRESHOLD} (dev_rehydrated predictions vs dev_public gold)")
    print("=" * 60)
    f1_scores = []
    for t in MARKER_TYPES:
        d = per_type[t]
        p, r, f1 = p_r_f1(d["tp"], d["pred"], d["gold"])
        f1_scores.append(f1)
        print(f"  {t:10}  P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}  (tp={d['tp']} pred={d['pred']} gold={d['gold']})")
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    p, r, f1 = p_r_f1(micro["tp"], micro["pred"], micro["gold"])
    print(f"  {'MICRO':10}  P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}  (tp={micro['tp']} pred={micro['pred']} gold={micro['gold']})")
    print(f"  {'MACRO':10}  F1: {macro_f1:.4f}")
    print("=" * 60)
    return {
        "micro": {"precision": p, "recall": r, "f1": f1},
        "macro_f1": macro_f1,
        "per_type": per_type,
        "iou_threshold": IOU_THRESHOLD,
    }


def main():
    dev_data, gold_by_id = load_dev_data()
    print(f"Loaded {len(dev_data)} dev samples from {DEV_REHYDRATED_FILE.name}")
    print(f"Gold markers from {DEV_PUBLIC_FILE.name} for {len(gold_by_id)} ids")

    all_results = run_inference(dev_data)

    metrics = compute_f1(dev_data, gold_by_id, all_results)

    # Optionally save dev predictions
    out_path = BASE / "dev_markers_predicted.jsonl"
    with open(out_path, "w") as f:
        for i, item in enumerate(dev_data):
            markers = sorted(all_results[i], key=lambda x: x["startIndex"])
            f.write(json.dumps({"_id": item["_id"], "markers": markers}) + "\n")
    print(f"\nDev predictions saved to {out_path}")

    metrics_path = BASE / "dev_markers_f1_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "micro_f1": metrics["micro"]["f1"],
                "micro_precision": metrics["micro"]["precision"],
                "micro_recall": metrics["micro"]["recall"],
                "macro_f1": metrics["macro_f1"],
                "iou_threshold": metrics["iou_threshold"],
                "per_type": metrics["per_type"],
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
