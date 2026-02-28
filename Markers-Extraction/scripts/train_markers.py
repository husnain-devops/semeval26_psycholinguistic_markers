#!/usr/bin/env python3
"""
Train RoBERTa-LoRA model for psycholinguistic marker extraction
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback, RobertaConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Configuration
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL_NAME = 'roberta-large'
MAX_LENGTH = 256
VAL_SPLIT = 0.1
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']

# Training hyperparameters
BATCH_SIZE = 8
GRAD_ACC = 2
LR = 3e-4
EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP = 0.1

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ['query', 'value', 'key', 'dense']

def setup_labels():
    """Create BIO label scheme"""
    label_list = ['O']
    for mt in MARKER_TYPES:
        label_list.extend([f'B-{mt}', f'I-{mt}'])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_list, label_to_id, id_to_label

def load_data(file_path):
    """Load JSONL data"""
    data = []
    with open(file_path) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    """Tokenize and align labels with BIO tagging"""
    tok = tokenizer(
        examples['text'], truncation=True, max_length=MAX_LENGTH,
        return_offsets_mapping=True, is_split_into_words=False
    )
    labels, all_markers = [], examples.get('markers', [])
    
    for i, offsets in enumerate(tok['offset_mapping']):
        ex_labels = [label_to_id['O']] * len(offsets)
        markers = sorted(all_markers[i] if i < len(all_markers) else [], 
                        key=lambda x: x['startIndex'])
        
        for m in markers:
            b_lbl = label_to_id.get(f"B-{m['type']}")
            i_lbl = label_to_id.get(f"I-{m['type']}")
            if b_lbl is None:
                continue
            
            first = True
            for ti, (s, e) in enumerate(offsets):
                if s is None:
                    continue
                if s < m['endIndex'] and e > m['startIndex']:
                    if first:
                        ex_labels[ti] = b_lbl
                        first = False
                    elif ex_labels[ti] == label_to_id['O']:
                        ex_labels[ti] = i_lbl
        
        labels.append(ex_labels)
    
    tok['labels'] = labels
    return tok

def compute_metrics(eval_pred, id_to_label):
    """Compute F1 metrics"""
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=2)
    
    true_l, pred_l = [], []
    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != -100:
                true_l.append(id_to_label[li])
                pred_l.append(id_to_label[pi])
    
    f1_micro = f1_score(true_l, pred_l, average='micro')
    ent_l = [l for l in true_l if l != 'O']
    ent_p = [p for p, l in zip(pred_l, true_l) if l != 'O']
    f1_ent = f1_score(ent_l, ent_p, average='micro') if ent_l else 0.0
    
    return {'f1_overall': f1_micro, 'f1_entity': f1_ent}

def main():
    # Setup
    BASE = Path(__file__).parent.parent.parent
    TRAIN_FILE = BASE / 'train_rehydrated.jsonl'
    OUTPUT_DIR = Path(__file__).parent.parent / 'models' / 'roberta-base-markers-lora'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print('='*60)
    print('RoBERTa-LoRA Marker Extraction Training')
    print('='*60)
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print('='*60)
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Labels
    label_list, label_to_id, id_to_label = setup_labels()
    num_labels = len(label_list)
    print(f'Labels: {num_labels}')
    
    # Load data
    print('Loading data...')
    train_data = load_data(TRAIN_FILE)
    print(f'✓ Loaded {len(train_data)} examples')
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(train_data)), test_size=VAL_SPLIT, random_state=42
    )
    train_ds = Dataset.from_list([train_data[i] for i in train_idx])
    val_ds = Dataset.from_list([train_data[i] for i in val_idx])
    
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
    
    # Tokenize
    print('Tokenizing...')
    train_ds = train_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=['_id', 'text', 'markers', 'subreddit', 'conspiracy', 'annotator']
    )
    val_ds = val_ds.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=['_id', 'text', 'markers', 'subreddit', 'conspiracy', 'annotator']
    )
    print(f'✓ Train: {len(train_ds)}, Val: {len(val_ds)}')
    
    # Model
    print('Loading model...')
    config = RobertaConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels
    config.id2label = id_to_label
    config.label2id = label_to_id
    
    model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias='none'
    )
    model = get_peft_model(model, lora_config)
    
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'✓ LoRA: {train_p:,}/{total_p:,} params ({100*train_p/total_p:.2f}%)')
    
    # Training
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], dtype=torch.bfloat16)
            use_bf16 = True
        except:
            pass
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR), learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS, weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP, eval_strategy='epoch',
        save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='f1_entity', greater_is_better=True,
        logging_steps=50, report_to='none', seed=42,
        gradient_accumulation_steps=GRAD_ACC, bf16=use_bf16,
        gradient_checkpointing=True, save_total_limit=2
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id_to_label),
        callbacks=[early_stop]
    )
    
    print('\n' + '='*60)
    print('Starting Training')
    print('='*60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    
    print('\n' + '='*60)
    print('Training Complete!')
    print('='*60)
    print(f'Loss: {train_result.training_loss:.4f}')
    for k, v in eval_result.items():
        if k.startswith('eval_'):
            print(f'{k}: {v:.4f}')
    print('='*60)
    
    # Save
    final_dir = OUTPUT_DIR / 'final_model'
    final_dir.mkdir(exist_ok=True)
    
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    label_mapping = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'marker_types': MARKER_TYPES
    }
    with open(final_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f'\n✓ Model saved to: {final_dir}')
    print('✓ Training complete!')

if __name__ == '__main__':
    main()
