import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
)

# --- Configuration: use local Hugging Face model from hf_models ---
MODEL_PATH = "/home/husnain/semeval26_psycholinguistic_markers/roberta-large-binary-conspiracy-lora/checkpoint-120"
TEST_FILE = "dev_rehydrated.jsonl"
DEV_PUBLIC_FILE = "dev_public.jsonl"  # Ground truth: _id -> conspiracy (Yes/No/Can't tell)
SUBMISSION_FILE = "submission.jsonl"
MAX_LENGTH = 512
BATCH_SIZE = 8
# LABEL_MAP built from loaded config (see main)


def load_competition_test_data(file_path):
    """
    Loads all data from a JSONL file for inference, preserving order,
    and retaining the document's unique ID.
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                sample_id = item.get("_id", f"sample_{i}")
                data.append({
                    "unique_sample_id": sample_id,
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data (max_length=MAX_LENGTH) to match the training script."""
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, max_length=MAX_LENGTH
        ),
        batched=True,
    )


def load_dev_public_gold(file_path):
    """Load dev_public.jsonl ground truth: _id -> conspiracy (Yes/No/Can't tell)."""
    gold = {}
    path = Path(file_path)
    if not path.exists():
        return gold
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                _id = item.get("_id")
                if _id is not None:
                    gold[_id] = item.get("conspiracy", "")
            except json.JSONDecodeError:
                pass
    return gold


def evaluate_vs_dev_public(unique_ids, predicted_labels, gold_path, out_summary_path=None):
    """
    Compare predictions to dev_public.jsonl ground truth.
    Metrics are computed on Yes/No only; 'Can't tell' samples are excluded.
    """
    gold = load_dev_public_gold(gold_path)
    if not gold:
        print(f"  No ground truth found at {gold_path}. Skipping evaluation.")
        return

    y_true, y_pred = [], []
    n_cant_tell = 0
    for i, _id in enumerate(unique_ids):
        g = gold.get(_id)
        if g is None:
            continue
        if g not in ("Yes", "No"):
            n_cant_tell += 1
            continue
        y_true.append(g)
        y_pred.append(predicted_labels[i])

    if n_cant_tell:
        print(f"  (Excluded {n_cant_tell} dev_public samples with \"Can't tell\" from metrics)")

    if not y_true:
        print("  No Yes/No ground-truth samples to evaluate.")
        return

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, labels=["No", "Yes"], average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, labels=["No", "Yes"], average="macro", zero_division=0)
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["No", "Yes"], average=None, zero_division=0
    )

    print(f"\n{'='*60}")
    print("Evaluation vs dev_public.jsonl (ground truth)")
    print(f"{'='*60}")
    print(f"  Matched {len(y_true)} samples (Yes/No only).")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (weighted):  {f1w:.4f}")
    print(f"  F1 (macro):     {f1m:.4f}")
    print("\n  Classification report (rows=true, cols=pred):")
    print(classification_report(y_true, y_pred, labels=["No", "Yes"]))
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    {cm}")

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figures_dir / "eval_vs_dev_public_confusion_matrix.png"

    # Keep logical plot size reasonable so text remains readable,
    # and increase output density via DPI for a crisp PNG.
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        annot_kws={"size": 20},
    )
    #plt.title("Confusion Matrix - dev_public")
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix image to {cm_path}")

    summary = {
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "f1_No": float(f1_per[0]),
        "f1_Yes": float(f1_per[1]),
        "n_eval": len(y_true),
        "n_cant_tell_excluded": n_cant_tell,
    }
    if out_summary_path:
        with open(out_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved metrics to {out_summary_path}")


if __name__ == '__main__':

    # 1. Load Data
    raw_data = load_competition_test_data(TEST_FILE)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_list(raw_data)

    # Store the unique IDs for later submission file generation
    unique_ids = test_dataset["unique_sample_id"]

    # 2. Load Tokenizer and Model from hf_models (local Hugging Face model)
    model_path = Path(MODEL_PATH).resolve()
    if not model_path.is_dir():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(-1)
    print(f"Loading tokenizer and model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        config = model.config
        num_labels = getattr(config, "num_labels", len(getattr(config, "id2label", {})))
        # Build label map: id -> "No" / "Yes" (binary submission). If model has 3 classes, map third to "No".
        LABEL_MAP = {}
        for i in range(num_labels):
            if i == 0:
                LABEL_MAP[i] = "No"
            elif i == 1:
                LABEL_MAP[i] = "Yes"
            else:
                LABEL_MAP[i] = "No"  # for binary submission, map any extra class to No
        print(f"  Model has {num_labels} label(s). Using LABEL_MAP: {LABEL_MAP}")
    except Exception as e:
        print(f"Error loading model or tokenizer from '{model_path}'.")
        print(f"Details: {e}")
        sys.exit(-1)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to GPU.")

    # 3. Tokenize Data
    tokenized_test_dataset = tokenize_data(test_dataset, tokenizer)

    # Remove columns that the model doesn't expect ('unique_sample_id' and 'text')
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["unique_sample_id", "text"])

    # 4. Prepare for Inference using Trainer with explicit data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    prediction_args = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none"
        ),
        data_collator=data_collator  # Use the padding collator
    )

    # 5. Perform Inference
    print("Starting prediction...")
    predictions_output = prediction_args.predict(tokenized_test_dataset)

    # Get the class with the highest probability
    logits = predictions_output.predictions
    predicted_class_ids = np.argmax(logits, axis=-1)

    # 6. Map IDs to Labels
    predicted_labels = [LABEL_MAP[int(id)] for id in predicted_class_ids]

    # 7. Save Results in Codalab-ready JSONL format
    print(f"Saving {len(predicted_labels)} predictions to {SUBMISSION_FILE} (JSONL format)...")

    jsonl_lines = []
    for i, label in enumerate(predicted_labels):
        # Create a dictionary containing the ID and the prediction, using '_id' as the key
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": label
        }
        # Convert the dictionary to a JSON string and append to the list
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")

    # Compare predictions to dev_public.jsonl ground truth (by _id)
    evaluate_vs_dev_public(
        list(unique_ids),
        predicted_labels,
        DEV_PUBLIC_FILE,
        out_summary_path="eval_vs_dev_public.json",
    )
