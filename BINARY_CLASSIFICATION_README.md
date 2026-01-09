# Binary Conspiracy Detection - Complete Implementation

This implementation provides a reproducible pipeline for binary conspiracy detection using RoBERTa-Large with LoRA.

## Quick Start

### Option 1: Run as Python Script (Recommended)

```bash
cd /home/husnain/semeval26_psycholinguistic_markers
python train_and_infer_binary.py
```

This will:
1. Load and filter data (Yes/No only, removes "Can't tell")
2. Create reproducible train/val/test splits (saved to `data_splits/`)
3. Train RoBERTa-large with LoRA
4. Evaluate on validation and test sets
5. Generate `submission.jsonl` and `submission.zip`

### Option 2: Use Jupyter Notebook

Convert the script to notebook:
```bash
jupytext --to notebook train_and_infer_binary.py
```

Or manually copy cells from the script into a new notebook.

## Output Files

After running, you'll have:

### Data Splits (Reusable)
```
data_splits/
├── binary_train_ids.txt    # Training sample IDs
├── binary_val_ids.txt       # Validation sample IDs
└── binary_test_ids.txt      # Test sample IDs
```

These splits are **saved and reused** in future runs for reproducibility.

### Model Checkpoint
```
roberta-large-binary-conspiracy-lora/
├── checkpoint-*/            # Best model checkpoint
└── ...
```

### Submission Files
```
submission.jsonl            # Predictions in CodaLab format
submission.zip              # Ready to upload to CodaLab
```

## Configuration

Key parameters in `train_and_infer_binary.py`:

```python
MODEL_NAME = "roberta-large"        # Can change to any model
BATCH_SIZE = 16                     # Adjust for your GPU
LEARNING_RATE = 2e-5               
NUM_EPOCHS = 10                     
LORA_R = 16                         # LoRA rank
LORA_ALPHA = 32                     # LoRA scaling
```

## Data Format

### Input: `train_rehydrated.jsonl`
```json
{"_id": "t1_xxx", "text": "...", "conspiracy": "Yes"}
{"_id": "t1_yyy", "text": "...", "conspiracy": "No"}
{"_id": "t1_zzz", "text": "...", "conspiracy": "Can't tell"}  # Filtered out
```

### Output: `submission.jsonl`
```json
{"_id": "t1_emz6exn", "conspiracy": "Yes"}
{"_id": "t1_f07ejkp", "conspiracy": "No"}
```

## Submission to CodaLab

1. **Upload**: Go to https://www.codabench.org/competitions/10749/
2. **Submit**: Upload `submission.zip`
3. **Evaluate**: Wait ~5 minutes for results
4. **Check Score**: Target > 0.76 weighted F1

## Reproducibility

### First Run
- Creates train/val/test splits
- Saves split IDs to `data_splits/`
- Trains model

### Subsequent Runs
- **Reuses existing splits** from `data_splits/`
- Ensures same train/val/test data
- Only retrains model with new hyperparameters

### Force New Splits
```bash
rm -rf data_splits/
python train_and_infer_binary.py
```

## Performance

### Expected Results

| Split | Metric | Expected |
|-------|--------|----------|
| Validation | F1 (weighted) | ~0.78-0.82 |
| Test | F1 (weighted) | ~0.78-0.82 |
| Dev (CodaLab) | F1 (weighted) | **> 0.76** (target) |

RoBERTa-large should significantly outperform the DistilBERT baseline (~0.76).

## Troubleshooting

### Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 8  # or 4
GRADIENT_ACCUMULATION_STEPS = 4  # Keep effective batch size
```

### Slow Training
- Use smaller model: `MODEL_NAME = "roberta-base"`
- Reduce epochs: `NUM_EPOCHS = 5`
- Use fp16: Automatic if GPU supports it

### Different Results Each Run
The script uses fixed seeds (42) and saved splits, so results should be reproducible. If not:
- Check if `data_splits/` exists
- Verify GPU determinism settings

## Customization

### Use Different Model
```python
MODEL_NAME = "microsoft/deberta-v3-large"  # Or any HuggingFace model
```

### Adjust LoRA Settings
```python
LORA_R = 32           # Higher = more parameters
LORA_ALPHA = 64       # Higher = stronger adaptation
LORA_DROPOUT = 0.05   # Lower = less regularization
```

### Change Split Ratios
```python
# In train_test_split calls
test_size=0.15  # 15% for test instead of 10%
```

## Files Overview

```
.
├── train_and_infer_binary.py           # Main script
├── train_rehydrated.jsonl              # Input data
├── dev_rehydrated.jsonl                # Dev set (for submission)
├── data_splits/                        # Saved splits (reproducibility)
├── roberta-large-binary-conspiracy-lora/  # Model checkpoints
├── submission.jsonl                    # Predictions
└── submission.zip                      # CodaLab submission
```

## Next Steps

1. **Run the script**: `python train_and_infer_binary.py`
2. **Check validation F1**: Should be > 0.78
3. **Submit to CodaLab**: Upload `submission.zip`
4. **Iterate**: Adjust hyperparameters if needed
5. **Share results**: Report your F1 score!

## Citation

If you use this implementation, please cite:
- SemEval 2026 Task 10: PsyCoMark
- RoBERTa: Liu et al., 2019
- LoRA: Hu et al., 2021

## Support

For issues:
- Check CodaLab: https://www.codabench.org/competitions/10749/
- Review README: This file
- Check logs: `./logs/` directory

Good luck! 🚀
