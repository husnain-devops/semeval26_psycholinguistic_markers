"""
Copy of infer_5_models.py that runs on dev_rehydrated.jsonl, uses dev_public.jsonl
as gold markers, and computes span-overlap F1 for extracted markers.
"""
import json
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


def spans_overlap(a_start, a_end, b_start, b_end):
    """True if character ranges [a_start, a_end) and [b_start, b_end) overlap."""
    return not (a_end <= b_start or b_end <= a_start)


def count_matched_spans(pred_spans, gold_spans, by_type=True):
    """
    Greedy 1-to-1 matching: count how many pred spans match a gold span (same type, overlapping).
    pred_spans / gold_spans: list of {"startIndex", "endIndex", "type"}.
    Returns (tp, n_pred, n_gold) per type if by_type else micro (single totals).
    """
    if by_type:
        types = set(s["type"] for s in pred_spans) | set(s["type"] for s in gold_spans)
        out = {}
        for t in types:
            p = [x for x in pred_spans if x["type"] == t]
            g = [x for x in gold_spans if x["type"] == t]
            tp = _match_count(p, g)
            out[t] = (tp, len(p), len(g))
        return out
    tp = _match_count(pred_spans, gold_spans)
    return tp, len(pred_spans), len(gold_spans)


def _match_count(pred_list, gold_list):
    """Greedy match: each pred matches at most one gold (same type, overlap)."""
    pred_list = list(pred_list)
    gold_list = list(gold_list)
    matched_gold = set()
    tp = 0
    for p in pred_list:
        ps, pe = p["startIndex"], p["endIndex"]
        for j, g in enumerate(gold_list):
            if j in matched_gold:
                continue
            gs, ge = g["startIndex"], g["endIndex"]
            if spans_overlap(ps, pe, gs, ge):
                matched_gold.add(j)
                tp += 1
                break
    return tp


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
    """Compute span-overlap F1 per type and micro-averaged."""
    per_type = {t: {"tp": 0, "pred": 0, "gold": 0} for t in MARKER_TYPES}
    micro = {"tp": 0, "pred": 0, "gold": 0}

    for i, item in enumerate(dev_data):
        _id = item["_id"]
        pred_spans = all_results[i]
        gold_spans = gold_by_id.get(_id, [])
        # Normalize gold: keep only startIndex, endIndex, type (same as pred)
        gold_spans = [
            {"startIndex": m["startIndex"], "endIndex": m["endIndex"], "type": m["type"]}
            for m in gold_spans
        ]
        by_type = count_matched_spans(pred_spans, gold_spans, by_type=True)
        for t, (tp, n_pred, n_gold) in by_type.items():
            per_type[t]["tp"] += tp
            per_type[t]["pred"] += n_pred
            per_type[t]["gold"] += n_gold
            micro["tp"] += tp
            micro["pred"] += n_pred
            micro["gold"] += n_gold

    def p_r_f1(tp, pred, gold):
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    print("\n" + "=" * 60)
    print("Span-overlap F1 (dev_rehydrated predictions vs dev_public gold)")
    print("=" * 60)
    for t in MARKER_TYPES:
        d = per_type[t]
        p, r, f1 = p_r_f1(d["tp"], d["pred"], d["gold"])
        print(f"  {t:10}  P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}  (tp={d['tp']} pred={d['pred']} gold={d['gold']})")
    p, r, f1 = p_r_f1(micro["tp"], micro["pred"], micro["gold"])
    print(f"  {'MICRO':10}  P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}  (tp={micro['tp']} pred={micro['pred']} gold={micro['gold']})")
    print("=" * 60)
    return {"micro": {"precision": p, "recall": r, "f1": f1}, "per_type": per_type}


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
                "per_type": metrics["per_type"],
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
