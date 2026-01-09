# Psycholinguistic Marker Extraction

This folder contains the implementation for **SemEval 2026 Task 10 - Marker Extraction Subtask**.

## Task Description

Extract psycholinguistic markers from conspiracy-related text. The task involves identifying text spans that correspond to:

- **Action**: Actions or events (e.g., "coups", "invasion")
- **Actor**: People, organizations, or entities involved (e.g., "US", "government")
- **Effect**: Consequences or results (e.g., "collapse", "death")
- **Evidence**: Supporting information or proof (e.g., "article", "data")
- **Victim**: Affected parties or targets (e.g., "Bolivia", "citizens")

## Approach

- **Model**: RoBERTa-base with LoRA (Low-Rank Adaptation)
- **Task Type**: Token Classification / Named Entity Recognition (NER)
- **Tagging Scheme**: BIO (Beginning-Inside-Outside)
- **Fine-tuning**: Parameter-efficient with LoRA adapters

## Folder Structure

```
Markers-Extraction/
├── notebooks/
│   └── 01_RoBERTa_LoRA_Markers_Extraction.ipynb  # Training notebook
├── data_processed/                                 # Processed datasets
├── figures/                                        # Visualizations
├── models/                                         # Trained models
│   └── roberta-base-markers-lora/                 # LoRA checkpoints
└── scripts/
    ├── train_markers.py                           # Training script
    └── infer_markers.py                           # Inference script
```

## Quick Start

### Training

```bash
# Run Jupyter notebook
cd notebooks
jupyter notebook 01_RoBERTa_LoRA_Markers_Extraction.ipynb

# Or use Python script
python scripts/train_markers.py
```

### Inference

```bash
python scripts/infer_markers.py --input dev_rehydrated.jsonl --output predictions.jsonl
```

## Model Details

### Architecture
- **Base Model**: RoBERTa-base (125M parameters)
- **LoRA Config**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: query, value, key, dense
  - Trainable parameters: ~2-3% of total

### Training Configuration
- **Batch Size**: 8 (effective: 16 with gradient accumulation)
- **Learning Rate**: 3e-4
- **Epochs**: 10 (with early stopping)
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Max Sequence Length**: 256 tokens

### Label Scheme (BIO Tagging)
- `O`: Outside any marker
- `B-Action, I-Action`: Beginning/Inside Action markers
- `B-Actor, I-Actor`: Beginning/Inside Actor markers
- `B-Effect, I-Effect`: Beginning/Inside Effect markers
- `B-Evidence, I-Evidence`: Beginning/Inside Evidence markers
- `B-Victim, I-Victim`: Beginning/Inside Victim markers

Total: 11 labels (1 + 5×2)

## Performance Metrics

The model is evaluated using:
- **Overall F1**: Token-level F1 score across all labels
- **Entity F1**: F1 score excluding `O` labels (entity-level performance)
- **Per-Marker F1**: Individual F1 scores for each marker type

## Data Format

### Input Format (train_rehydrated.jsonl)
```json
{
  "_id": "t1_example",
  "text": "US backed coups in Bolivia...",
  "markers": [
    {
      "startIndex": 0,
      "endIndex": 2,
      "type": "Actor",
      "text": "US"
    },
    {
      "startIndex": 10,
      "endIndex": 15,
      "type": "Action",
      "text": "coups"
    }
  ]
}
```

### Output Format (predictions.jsonl)
```json
{
  "_id": "t1_example",
  "markers": [
    {
      "startIndex": 0,
      "endIndex": 2,
      "type": "Actor"
    }
  ]
}
```

## Advantages of RoBERTa-LoRA

1. **Parameter Efficiency**: Train only ~2-3% of parameters
2. **Fast Training**: Reduced memory and compute requirements
3. **Better Generalization**: LoRA prevents overfitting
4. **Contextual Understanding**: Pre-trained RoBERTa captures semantic relationships
5. **Multi-Label**: Single model handles all marker types simultaneously

## Next Steps

- [ ] Experiment with roberta-large for better performance
- [ ] Try different LoRA configurations (rank, alpha)
- [ ] Implement ensemble with multiple models
- [ ] Add post-processing for better span extraction
- [ ] Experiment with focal loss for class imbalance

## References

- [SemEval 2026 Task 10](https://sites.google.com/view/semeval2026-task10)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
