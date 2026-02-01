import json
import warnings
import torch
import numpy as np
from pathlib import Path
from torch import nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE = Path(__file__).parent.parent
TRAIN_FILE = BASE / 'train_rehydrated.jsonl'
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
MODEL_NAME = 'roberta-large'
MAX_LENGTH = 512

# LoRA & Training Hyperparams
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 10
LORA_R = 16

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # FIX: Use labels.device instead of model.device to avoid DataParallel attribute errors
        device = labels.device
        weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float, device=device)
        
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def create_labels(marker_type):
    label_list = ['O', f'B-{marker_type}', f'I-{marker_type}']
    return {l: i for i, l in enumerate(label_list)}, {i: l for i, l in enumerate(label_list)}

def tokenize_and_align(examples, tokenizer, marker_type, label_to_id):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', 
                          max_length=MAX_LENGTH, return_offsets_mapping=True)
    labels = []
    for i, offsets in enumerate(tokenized['offset_mapping']):
        doc_labels = [label_to_id['O']] * len(offsets)
        for m in examples['markers'][i]:
            if m['type'] == marker_type:
                start, end = m['startIndex'], m['endIndex']
                first = True
                for idx, (t_start, t_end) in enumerate(offsets):
                    if t_start == t_end == 0 or t_start is None: continue
                    # Overlap logic for sub-tokens
                    if t_start < end and t_end > start:
                        doc_labels[idx] = label_to_id[f'B-{marker_type}'] if first else label_to_id[f'I-{marker_type}']
                        first = False
        labels.append(doc_labels)
    tokenized['labels'] = labels
    return tokenized

def train():
    print("Loading data...")
    with open(TRAIN_FILE) as f:
        data = [json.loads(line) for line in f]
    
    train_idx, val_idx = train_test_split(range(len(data)), test_size=0.1, random_state=42)
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)

    for m_type in MARKER_TYPES:
        print(f"\n>>> STARTING TRAINING: {m_type}")
        l2id, id2l = create_labels(m_type)
        
        train_ds = Dataset.from_list([data[i] for i in train_idx]).map(
            lambda x: tokenize_and_align(x, tokenizer, m_type, l2id), batched=True)
        val_ds = Dataset.from_list([data[i] for i in val_idx]).map(
            lambda x: tokenize_and_align(x, tokenizer, m_type, l2id), batched=True)

        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3)
        
        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS, 
            inference_mode=False, 
            r=LORA_R, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=['query', 'value', 'key', 'dense']
        )
        model = get_peft_model(model, peft_config)
        
        output_dir = BASE / 'models' / f'roberta-{m_type}'
        
        # UPDATED: eval_strategy instead of evaluation_strategy
        args = TrainingArguments(
                    output_dir=str(output_dir), 
                    eval_strategy='epoch', 
                    save_strategy='epoch',
                    learning_rate=LR, 
                    per_device_train_batch_size=16, # Increased since you have high VRAM
                    gradient_accumulation_steps=1,
                    num_train_epochs=EPOCHS,
                    load_best_model_at_end=True, 
                    metric_for_best_model='f1', 
                    weight_decay=0.01,
                    logging_steps=10,
                    report_to="none",
                    fp16=True, # Enable mixed precision for much faster training on ROCm/CUDA
                    ddp_find_unused_parameters=False
                )

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=2).flatten()
            labs = p.label_ids.flatten()
            mask = labs != -100
            return {'f1': f1_score(labs[mask], preds[mask], average='macro')}

        trainer = WeightedTrainer(
            model=model, 
            args=args, 
            train_dataset=train_ds, 
            eval_dataset=val_ds,
            data_collator=DataCollatorForTokenClassification(tokenizer), 
            compute_metrics=compute_metrics, 
            callbacks=[EarlyStoppingCallback(3)]
        )
        
        trainer.train()
        
        # Save results
        final_path = output_dir / 'final'
        trainer.model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        with open(final_path / 'config_labels.json', 'w') as f:
            json.dump({'l2id': l2id, 'id2l': id2l}, f)
        print(f"✓ {m_type} training complete.")

if __name__ == '__main__':
    train()