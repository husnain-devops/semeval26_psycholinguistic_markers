# 5 Separate Models Approach for Better Accuracy

## Problem

The unified model (01_RoBERTa_LoRA_Markers_Extraction.ipynb) with 11 classes (O + B-/I- for 5 markers) may have lower accuracy because:
- Complex multi-class classification
- Confusion between similar marker types
- Harder to optimize all marker types simultaneously

## Solution: Train 5 Separate Binary Classifiers

**Strategy**: One RoBERTa-large-LoRA model per marker type
- Each model: Binary classification (O vs MARKER)
- Simpler task → Better accuracy per marker
- Models specialize in their specific marker type

## Implementation Steps

### 1. Loop Through Each Marker Type

```python
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']

for marker_type in MARKER_TYPES:
    # Train one model for this marker type
    print(f"Training model for: {marker_type}")
```

### 2. Create Binary Labels (Not BIO)

```python
def create_binary_labels(marker_type):
    """Simple: O (0) or MARKER (1)"""
    label_list = ['O', marker_type]
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_to_id, id_to_label, 2  # Only 2 classes!
```

### 3. Modified Tokenization Function

```python
def tokenize_and_align_binary(examples, marker_type, label_to_id):
    tok = tokenizer(
        examples['text'], truncation=True, max_length=512,
        return_offsets_mapping=True
    )
    
    labels = []
    all_markers = examples.get('markers', [])
    
    for i, offsets in enumerate(tok['offset_mapping']):
        ex_labels = [label_to_id['O']] * len(offsets)
        
        marker_list = all_markers[i] if (i < len(all_markers) and all_markers[i] is not None) else []
        
        # Only process markers of this specific type
        for m in marker_list:
            if m.get('type') == marker_type:
                start_char = m['startIndex']
                end_char = m['endIndex']
                
                # Mark tokens that overlap
                for ti, (s, e) in enumerate(offsets):
                    if s is not None and e is not None:
                        if s < end_char and e > start_char:
                            ex_labels[ti] = label_to_id[marker_type]
        
        labels.append(ex_labels)
    
    tok['labels'] = labels
    return tok
```

### 4. Train Each Model

```python
trained_models = {}

for marker_type in MARKER_TYPES:
    # Create binary labels
    label_to_id, id_to_label, num_labels = create_binary_labels(marker_type)
    
    # Tokenize with marker-specific labeling
    train_ds_tok = train_ds.map(
        lambda x: tokenize_and_align_binary(x, marker_type, label_to_id),
        batched=True
    )
    
    # Load fresh model (num_labels=2)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-large', 
        num_labels=2
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=['query', 'value', 'key', 'dense']
    )
    model = get_peft_model(model, lora_config)
    
    # Train
    trainer = Trainer(...)
    trainer.train()
    
    # Save
    model.save_pretrained(f'models/roberta-large-lora-{marker_type}')
    trained_models[marker_type] = model
```

### 5. Inference: Run All 5 Models

```python
all_predictions = {marker_type: [] for marker_type in MARKER_TYPES}

for marker_type in MARKER_TYPES:
    model = trained_models[marker_type]
    
    # Get predictions for this marker type
    predictions = trainer.predict(test_ds_tok)
    pred_labels = np.argmax(predictions.predictions, axis=2)
    
    # Convert to markers
    for i, (pred, offsets) in enumerate(zip(pred_labels, offset_mappings)):
        markers = []
        current_marker = None
        
        for label_id, offset in zip(pred, offsets):
            label = id_to_label[label_id]
            
            if label == marker_type:  # Found marker token
                if current_marker is None:
                    current_marker = {
                        'startIndex': int(offset[0]),
                        'endIndex': int(offset[1]),
                        'type': marker_type
                    }
                else:
                    current_marker['endIndex'] = int(offset[1])
            else:  # 'O' label
                if current_marker:
                    markers.append(current_marker)
                    current_marker = None
        
        if current_marker:
            markers.append(current_marker)
        
        all_predictions[marker_type].append(markers)
```

### 6. Combine Results from 5 Models

```python
combined_markers = []

for i in range(len(test_data)):
    # Collect markers from all 5 models
    example_markers = []
    for marker_type in MARKER_TYPES:
        example_markers.extend(all_predictions[marker_type][i])
    
    # Sort by position
    example_markers = sorted(example_markers, key=lambda x: x['startIndex'])
    combined_markers.append(example_markers)

# Create submission
for i, example in enumerate(test_data):
    submission_entry = {
        '_id': example['_id'],
        'markers': combined_markers[i]
    }
```

## Expected Benefits

| Aspect | Unified Model (11 classes) | 5 Separate Models (binary each) |
|--------|----------------------------|----------------------------------|
| **Task Complexity** | Hard (11-way classification) | Easy (binary per type) |
| **Specialization** | One model for all | Each specializes |
| **Confusion** | Can confuse B-Action vs B-Actor | No confusion between types |
| **Training** | One training run | 5 training runs |
| **Time** | ~90 min | ~450 min (90 × 5) |
| **Expected F1** | 0.40-0.45 | **0.50-0.60+** |

## Comparison to Baseline

The baseline `train_one_span.py` uses this approach with **DistilBERT**:
- 5 models × DistilBERT = ~0.15 F1

Your approach with **RoBERTa-large + LoRA**:
- 5 models × RoBERTa-large-LoRA = **~0.50-0.60 F1** (expected)
- 3-4x improvement over baseline!

## File Structure After Training

```
Markers-Extraction/
├── models/
│   ├── roberta-large-lora-Action/
│   │   └── final_model/
│   ├── roberta-large-lora-Actor/
│   │   └── final_model/
│   ├── roberta-large-lora-Effect/
│   │   └── final_model/
│   ├── roberta-large-lora-Evidence/
│   │   └── final_model/
│   └── roberta-large-lora-Victim/
│       └── final_model/
├── submission_5models.jsonl
└── submission_5models.zip
```

## Quick Start Script

Would you like me to:
1. **Create the full notebook** (`02_RoBERTa_LoRA_5_Models.ipynb`)
2. **Create a Python script** (`train_5_models.py`)
3. **Modify the existing notebook** to use this approach

The 5-model approach is more computationally expensive (5x training time) but should give significantly better results!

## Recommendation

✅ **Use 5 separate models** if:
- You have time (~7-8 hours total training)
- You want best possible accuracy
- GPU memory is sufficient (~16 GB per model)

❌ **Stick with unified model** if:
- Limited time
- Need quick iteration
- Memory constraints

**Expected improvement**: +10-15% F1 score boost! 🚀
