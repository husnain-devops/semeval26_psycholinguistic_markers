#!/usr/bin/env python3
"""
Binary Conspiracy Detection with RoBERTa-Large + LoRA
Target: Beat baseline F1 weighted score of ~0.76 on dev set
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_and_filter_data(file_path):
    """Load JSONL and filter to binary classification (Yes/No only)"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'conspiracy' in item and item['conspiracy'] in ["Yes", "No"]:
                    data.append({
                        '_id': item.get('_id', ''),
                        'text': item.get('text', ''),
                        'conspiracy': item['conspiracy']
                    })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
    return data


def load_dev_data(file_path):
    """Load dev data for inference"""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                data.append({
                    "unique_sample_id": item.get("_id", f"sample_{i}"),
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i}")
    return data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def main():
    print("="*60)
    print("Binary Conspiracy Detection - RoBERTa-Large + LoRA")
    print("="*60)
    
    # Configuration
    BASE = Path('.')
    MODEL_NAME = "roberta-large"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0 #0.01
    WARMUP_RATIO = 0 #0.1
    DROPOUT_RATE = 0 #0.1
    
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["query", "value", "key", "dense"]
    
    OUTPUT_DIR = "./roberta-large-binary-conspiracy-lora"
    SPLITS_DIR = BASE / "data_splits"
    SPLITS_DIR.mkdir(exist_ok=True)
    
    label_to_id = {"No": 0, "Yes": 1}
    id_to_label = {0: "No", 1: "Yes"}
    num_labels = 2
    
    print(f"\n✓ Configuration loaded")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and filter data
    print(f"\n{'='*60}")
    print("Loading Data")
    print(f"{'='*60}")
    
    train_file = BASE / "train_rehydrated.jsonl"
    train_data = load_and_filter_data(train_file)
    df = pd.DataFrame(train_data)
    
    print(f"✓ Loaded {len(df)} samples (binary classification)")
    print(f"\nClass distribution:")
    print(df['conspiracy'].value_counts())
    
    # Create or load splits
    train_ids_file = SPLITS_DIR / "binary_train_ids.txt"
    val_ids_file = SPLITS_DIR / "binary_val_ids.txt"
    test_ids_file = SPLITS_DIR / "binary_test_ids.txt"
    
    if train_ids_file.exists() and val_ids_file.exists() and test_ids_file.exists():
        print(f"\n✓ Loading existing splits from {SPLITS_DIR}...")
        
        with open(train_ids_file, 'r') as f:
            train_ids = set(line.strip() for line in f)
        with open(val_ids_file, 'r') as f:
            val_ids = set(line.strip() for line in f)
        with open(test_ids_file, 'r') as f:
            test_ids = set(line.strip() for line in f)
        
        train_df = df[df['_id'].isin(train_ids)].copy()
        val_df = df[df['_id'].isin(val_ids)].copy()
        test_df = df[df['_id'].isin(test_ids)].copy()
    else:
        print(f"\n✓ Creating new splits...")
        
        temp_train, test_df = train_test_split(
            df, test_size=0.1, stratify=df['conspiracy'], random_state=42
        )
        
        train_df, val_df = train_test_split(
            temp_train, test_size=0.1, stratify=temp_train['conspiracy'], random_state=42
        )
        
        with open(train_ids_file, 'w') as f:
            f.write('\n'.join(train_df['_id'].values))
        with open(val_ids_file, 'w') as f:
            f.write('\n'.join(val_df['_id'].values))
        with open(test_ids_file, 'w') as f:
            f.write('\n'.join(test_df['_id'].values))
        
        print(f"  ✓ Saved split IDs to {SPLITS_DIR}")
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Prepare data
    print(f"\n{'='*60}")
    print("Preparing Data")
    print(f"{'='*60}")
    
    train_texts = train_df['text'].tolist()
    train_labels = [label_to_id[label] for label in train_df['conspiracy'].tolist()]
    
    val_texts = val_df['text'].tolist()
    val_labels = [label_to_id[label] for label in val_df['conspiracy'].tolist()]
    
    test_texts = test_df['text'].tolist()
    test_labels = [label_to_id[label] for label in test_df['conspiracy'].tolist()]
    
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print(f"✓ Datasets tokenized and ready")
    
    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing Model")
    print(f"{'='*60}")
    
    config = RobertaConfig.from_pretrained(MODEL_NAME)
    config.hidden_dropout_prob = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    config.num_labels = num_labels
    config.id2label = id_to_label
    config.label2id = label_to_id
    
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"✓ Model moved to GPU")
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ LoRA applied")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Training
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")
    
    use_fp16 = torch.cuda.is_available()
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            use_bf16 = True
            use_fp16 = False
        except:
            pass
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_dir='./logs',
        logging_steps=50,
        report_to="none",
        seed=42,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        save_total_limit=2,
        gradient_checkpointing=True
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Starting training...")
    train_result = trainer.train()
    
    print(f"\n✓ Training complete")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    
    # Validation evaluation
    print(f"\n{'='*60}")
    print("Validation Evaluation")
    print(f"{'='*60}")
    
    val_results = trainer.evaluate(val_dataset)
    print(f"Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"F1 (macro): {val_results['eval_f1_macro']:.4f}")
    print(f"F1 (weighted): {val_results['eval_f1_weighted']:.4f}")
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    
    test_results = trainer.evaluate(test_dataset)
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1 (macro): {test_results['eval_f1_macro']:.4f}")
    print(f"F1 (weighted): {test_results['eval_f1_weighted']:.4f}")
    
    predictions = trainer.predict(test_dataset)
    predicted_classes = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_dataset['labels']
    
    y_true_labels = [id_to_label[label] for label in true_labels]
    y_pred_labels = [id_to_label[label] for label in predicted_classes]
    
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    # Generate submission
    print(f"\n{'='*60}")
    print("Generating Submission")
    print(f"{'='*60}")
    
    DEV_FILE = BASE / "dev_rehydrated.jsonl"
    SUBMISSION_FILE = "submission.jsonl"
    SUBMISSION_ZIP = "submission.zip"
    
    dev_data = load_dev_data(DEV_FILE)
    dev_dataset = Dataset.from_list(dev_data)
    unique_ids = dev_dataset["unique_sample_id"]
    
    print(f"✓ Loaded {len(dev_dataset)} samples from dev set")
    
    dev_dataset_tokenized = dev_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True
    )
    dev_dataset_tokenized = dev_dataset_tokenized.remove_columns(["unique_sample_id", "text"])
    
    print("Generating predictions...")
    predictions_output = trainer.predict(dev_dataset_tokenized)
    logits = predictions_output.predictions
    predicted_class_ids = np.argmax(logits, axis=-1)
    predicted_labels = [id_to_label[int(id)] for id in predicted_class_ids]
    
    print(f"\n✓ Generated {len(predicted_labels)} predictions")
    print(f"\nPrediction distribution:")
    print(pd.Series(predicted_labels).value_counts())
    
    # Save submission
    print(f"\nSaving to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, 'w') as f:
        for i, label in enumerate(predicted_labels):
            jsonl_obj = {
                "_id": unique_ids[i],
                "conspiracy": label
            }
            f.write(json.dumps(jsonl_obj) + '\n')
    
    print(f"✓ Saved predictions")
    
    print(f"\nCreating {SUBMISSION_ZIP}...")
    with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUBMISSION_FILE, arcname='submission.jsonl')
    
    print(f"✓ Created submission zip")
    
    print(f"\n{'='*60}")
    print("SUBMISSION READY!")
    print(f"{'='*60}")
    print(f"✓ File: {SUBMISSION_ZIP}")
    print(f"✓ Predictions: {len(predicted_labels)}")
    print(f"\nNext steps:")
    print(f"1. Go to: https://www.codabench.org/competitions/10749/")
    print(f"2. Upload: {SUBMISSION_ZIP}")
    print(f"3. Check score (target: > 0.76 weighted F1)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
