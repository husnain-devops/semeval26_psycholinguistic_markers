#!/usr/bin/env python3
"""
Eval-only: load trained RoBERTa-Large + LoRA and report:
  - Training set metrics
  - Validation set metrics
  - Dev set metrics (text: dev_rehydrated.jsonl, labels: dev_public.jsonl)
No training; no submission generation.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import PeftModel
from datasets import Dataset
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

BASE = Path(".")
MODEL_NAME = "roberta-large"
MAX_LENGTH = 512
BATCH_SIZE = 16
OUTPUT_DIR = BASE / "roberta-large-binary-conspiracy-lora"
SPLITS_DIR = BASE / "data_splits"
label_to_id = {"No": 0, "Yes": 1}
id_to_label = {0: "No", 1: "Yes"}
num_labels = 2


def find_latest_checkpoint(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    checkpoints = list(base_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]), reverse=True)
    return checkpoints[0]


def load_and_filter_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "conspiracy" in item and item["conspiracy"] in ["Yes", "No"]:
                    data.append({
                        "_id": item.get("_id", ""),
                        "text": item.get("text", ""),
                        "conspiracy": item["conspiracy"],
                    })
            except json.JSONDecodeError:
                pass
    return data


def load_test_from_dev(dev_public_path, dev_rehydrated_path):
    with open(dev_rehydrated_path, "r") as f:
        id_to_text = {json.loads(line)["_id"]: json.loads(line).get("text", "") for line in f}
    data = []
    with open(dev_public_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                _id = item.get("_id")
                label = item.get("conspiracy", "")
                if _id is None or label not in ("Yes", "No"):
                    continue
                data.append({
                    "_id": _id,
                    "text": id_to_text.get(_id, ""),
                    "conspiracy": label,
                })
            except json.JSONDecodeError:
                pass
    return data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    print("=" * 60)
    print("Binary Conspiracy Detection - EVAL ONLY (no training)")
    print("=" * 60)

    checkpoint_dir = find_latest_checkpoint(OUTPUT_DIR)
    if checkpoint_dir is None:
        raise FileNotFoundError(
            f"No checkpoint found under {OUTPUT_DIR}. Train first with train_and_infer_binary.py"
        )
    print(f"\n✓ Loading model from: {checkpoint_dir}")

    tokenizer = RobertaTokenizerFast.from_pretrained(str(checkpoint_dir))
    config = RobertaConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels
    config.id2label = id_to_label
    config.label2id = label_to_id

    base_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    if torch.cuda.is_available():
        model = model.to("cuda")
    print("✓ Model loaded")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    # Train/val from train_rehydrated
    train_file = BASE / "train_rehydrated.jsonl"
    train_data = load_and_filter_data(train_file)
    df = pd.DataFrame(train_data)

    train_ids_file = SPLITS_DIR / "binary_train_ids.txt"
    val_ids_file = SPLITS_DIR / "binary_val_ids.txt"
    if not train_ids_file.exists() or not val_ids_file.exists():
        raise FileNotFoundError(
            f"Split files not found in {SPLITS_DIR}. Run train_and_infer_binary.py once to create them."
        )
    with open(train_ids_file, "r") as f:
        train_ids = set(line.strip() for line in f)
    with open(val_ids_file, "r") as f:
        val_ids = set(line.strip() for line in f)
    train_df = df[df["_id"].isin(train_ids)].copy()
    val_df = df[df["_id"].isin(val_ids)].copy()

    dev_public_path = BASE / "dev_public.jsonl"
    dev_rehydrated_path = BASE / "dev_rehydrated.jsonl"
    test_data = load_test_from_dev(dev_public_path, dev_rehydrated_path)
    test_df = pd.DataFrame(test_data)

    train_texts = train_df["text"].tolist()
    train_labels = [label_to_id[l] for l in train_df["conspiracy"].tolist()]
    val_texts = val_df["text"].tolist()
    val_labels = [label_to_id[l] for l in val_df["conspiracy"].tolist()]
    test_texts = test_df["text"].tolist()
    test_labels = [label_to_id[l] for l in test_df["conspiracy"].tolist()]

    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        dataloader_num_workers=0,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ----- Training set -----
    print(f"\n{'='*60}")
    print("Training set evaluation")
    print(f"{'='*60}")
    print(f"  Samples: {len(train_dataset)}")
    train_results = trainer.evaluate(train_dataset)
    print(f"  Accuracy:      {train_results['eval_accuracy']:.4f}")
    print(f"  F1 (macro):    {train_results['eval_f1_macro']:.4f}")
    print(f"  F1 (weighted): {train_results['eval_f1_weighted']:.4f}")

    pred_train = trainer.predict(train_dataset)
    y_true = [id_to_label[int(x)] for x in train_dataset["labels"]]
    y_pred = [id_to_label[int(x)] for x in np.argmax(pred_train.predictions, axis=-1)]
    print("\n  Classification report (train):")
    print(classification_report(y_true, y_pred, labels=["No", "Yes"]))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    {confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])}")

    # ----- Validation set -----
    print(f"\n{'='*60}")
    print("Validation set evaluation")
    print(f"{'='*60}")
    print(f"  Samples: {len(val_dataset)}")
    val_results = trainer.evaluate(val_dataset)
    print(f"  Accuracy:      {val_results['eval_accuracy']:.4f}")
    print(f"  F1 (macro):    {val_results['eval_f1_macro']:.4f}")
    print(f"  F1 (weighted): {val_results['eval_f1_weighted']:.4f}")

    pred_val = trainer.predict(val_dataset)
    y_true = [id_to_label[int(x)] for x in val_dataset["labels"]]
    y_pred = [id_to_label[int(x)] for x in np.argmax(pred_val.predictions, axis=-1)]
    print("\n  Classification report (validation):")
    print(classification_report(y_true, y_pred, labels=["No", "Yes"]))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    {confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])}")

    # ----- Dev set (dev_rehydrated + dev_public ground truth) -----
    print(f"\n{'='*60}")
    print("Dev set evaluation (text: dev_rehydrated.jsonl, labels: dev_public.jsonl)")
    print(f"{'='*60}")
    print(f"  Samples: {len(test_dataset)}")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Accuracy:      {test_results['eval_accuracy']:.4f}")
    print(f"  F1 (macro):    {test_results['eval_f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_results['eval_f1_weighted']:.4f}")

    pred_test = trainer.predict(test_dataset)
    y_true = [id_to_label[int(x)] for x in test_dataset["labels"]]
    y_pred = [id_to_label[int(x)] for x in np.argmax(pred_test.predictions, axis=-1)]
    print("\n  Classification report (dev):")
    print(classification_report(y_true, y_pred, labels=["No", "Yes"]))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    {confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])}")

    # Save summary
    summary = {
        "checkpoint": str(checkpoint_dir),
        "train": {
            "n": len(train_dataset),
            "accuracy": float(train_results["eval_accuracy"]),
            "f1_macro": float(train_results["eval_f1_macro"]),
            "f1_weighted": float(train_results["eval_f1_weighted"]),
        },
        "validation": {
            "n": len(val_dataset),
            "accuracy": float(val_results["eval_accuracy"]),
            "f1_macro": float(val_results["eval_f1_macro"]),
            "f1_weighted": float(val_results["eval_f1_weighted"]),
        },
        "dev": {
            "n": len(test_dataset),
            "text_source": "dev_rehydrated.jsonl",
            "labels_source": "dev_public.jsonl",
            "accuracy": float(test_results["eval_accuracy"]),
            "f1_macro": float(test_results["eval_f1_macro"]),
            "f1_weighted": float(test_results["eval_f1_weighted"]),
        },
    }
    out_path = BASE / "eval_results_binary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
