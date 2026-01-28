#!/usr/bin/env python3
"""
Train 5 separate RoBERTa-large-LoRA models for marker extraction
One model per marker type for better specialization and accuracy
"""
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import torch
import numpy as np
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Configuration
BASE = Path(__file__).parent.parent.parent
TRAIN_FILE = BASE / 'train_rehydrated.jsonl'
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
MODEL_NAME = 'roberta-large'
MAX_LENGTH = 128  # Shorter sequences to prevent over-prediction

# Training hyperparameters
BATCH_SIZE = 8
GRAD_ACC = 2
LR = 3e-4
EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP = 0.1

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ['query', 'value', 'key', 'dense']

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

def create_binary_labels(marker_type):
    """Create binary labels: O or MARKER"""
    label_list = ['O', marker_type]
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_to_id, id_to_label, 2

def tokenize_and_align_binary(examples, tokenizer, marker_type, label_to_id):
    """Tokenize and create binary labels - EXACT LOGIC from train_one_span.py"""
    tok = tokenizer(
        examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH,
        return_offsets_mapping=True, is_split_into_words=False
    )
    
    labels = []
    all_markers = examples.get('markers', [])
    
    for i, offsets in enumerate(tok['offset_mapping']):
        example_labels = [label_to_id['O']] * len(offsets)  # Initialize with 'O'
        example_markers = all_markers[i] if i < len(all_markers) else []
        
        for marker in example_markers:
            if marker.get('type') == marker_type:
                start_char = marker['startIndex']
                end_char = marker['endIndex']
                marker_label = label_to_id.get(marker_type)
                
                if marker_label is not None:
                    for token_idx, (start, end) in enumerate(offsets):
                        if start is not None and end is not None:
                            # Priority 1: Token START is within marker span
                            if start_char <= start < end_char:
                                if token_idx < len(example_labels):
                                    example_labels[token_idx] = marker_label
                            # Priority 2: Handle partial overlaps (only if not already marked)
                            elif start < end_char and end > start_char:
                                if token_idx < len(example_labels) and example_labels[token_idx] == label_to_id['O']:
                                    example_labels[token_idx] = marker_label
        
        labels.append(example_labels)
    
    tok['labels'] = labels
    return tok

def main():
    print('='*70)
    print('TRAINING 5 SEPARATE ROBERTA-LARGE-LORA MODELS')
    print('='*70)
    print(f'Base model: {MODEL_NAME}')
    print(f'Marker types: {MARKER_TYPES}')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print('='*70)
    
    # Load data once
    print('\nLoading data...')
    train_data = load_data(TRAIN_FILE)
    print(f'✓ Loaded {len(train_data)} examples')
    
    # Split once
    VAL_SPLIT = 0.1
    train_idx, val_idx = train_test_split(range(len(train_data)), test_size=VAL_SPLIT, random_state=42)
    print(f'✓ Train: {len(train_idx)}, Val: {len(val_idx)}')
    
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
    
    # Train each model
    for marker_idx, marker_type in enumerate(MARKER_TYPES):
        print('\n' + '='*70)
        print(f'TRAINING MODEL {marker_idx+1}/5: {marker_type}')
        print('='*70)
        
        # Create binary labels
        label_to_id, id_to_label, num_labels = create_binary_labels(marker_type)
        print(f'Labels: {label_to_id}')
        
        # Prepare datasets
        print('Tokenizing...')
        train_ds = Dataset.from_list([train_data[i] for i in train_idx])
        val_ds = Dataset.from_list([train_data[i] for i in val_idx])
        
        train_ds_tok = train_ds.map(
            lambda x: tokenize_and_align_binary(x, tokenizer, marker_type, label_to_id),
            batched=True,
            remove_columns=['_id', 'text', 'markers', 'subreddit', 'conspiracy', 'annotator']
        )
        val_ds_tok = val_ds.map(
            lambda x: tokenize_and_align_binary(x, tokenizer, marker_type, label_to_id),
            batched=True,
            remove_columns=['_id', 'text', 'markers', 'subreddit', 'conspiracy', 'annotator']
        )
        print(f'✓ Tokenized: {len(train_ds_tok)} train, {len(val_ds_tok)} val')
        
        # Load model
        print('Loading model...')
        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias='none'
        )
        model = get_peft_model(model, lora_config)
        
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'✓ LoRA applied: {train_p:,}/{total_p:,} params ({100*train_p/total_p:.2f}%)')
        
        # Output directory
        output_dir = BASE / 'Markers-Extraction' / 'models' / f'roberta-large-lora-{marker_type}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        use_bf16 = False
        if torch.cuda.is_available():
            try:
                _ = torch.tensor([1.0], dtype=torch.bfloat16)
                use_bf16 = True
            except:
                pass
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            logging_steps=50,
            report_to='none',
            seed=42,
            gradient_accumulation_steps=GRAD_ACC,
            bf16=use_bf16,
            gradient_checkpointing=True,
            save_total_limit=2
        )
        
        # Metrics
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=2)
            
            true_l, pred_l = [], []
            for p, l in zip(preds, labels):
                for pi, li in zip(p, l):
                    if li != -100:
                        true_l.append(id_to_label[li])
                        pred_l.append(id_to_label[pi])
            
            marker_true = [l for l in true_l if l != 'O']
            marker_pred = [p for p, l in zip(pred_l, true_l) if l != 'O']
            
            if marker_true:
                f1 = f1_score(marker_true, marker_pred, average='binary', pos_label=marker_type)
            else:
                f1 = 0.0
            
            return {'f1': f1}
        
        # Trainer
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
        early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_tok,
            eval_dataset=val_ds_tok,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stop]
        )
        
        # Train
        print(f'\nStarting training for {marker_type}...')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        
        print(f'\n✅ {marker_type} - Training complete!')
        print(f'   Loss: {train_result.training_loss:.4f}')
        print(f'   Val F1: {eval_result.get("eval_f1", 0):.4f}')
        
        # Save model
        final_dir = output_dir / 'final_model'
        final_dir.mkdir(exist_ok=True)
        trainer.model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        
        with open(final_dir / 'label_mapping.json', 'w') as f:
            json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)
        
        print(f'✓ Model saved to: {final_dir}')
    
    print('\n' + '='*70)
    print('🎉 ALL 5 MODELS TRAINED SUCCESSFULLY!')
    print('='*70)
    print('\nNext steps:')
    print('  1. Run inference: python scripts/infer_5_models.py')
    print('  2. Or use the notebook for interactive inference')

if __name__ == '__main__':
    main()
