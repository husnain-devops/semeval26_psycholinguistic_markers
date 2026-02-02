# Model Sharing Guide - Cross-Machine Compatibility

## The Problem

Your models were saved with **PEFT 0.18.0** (newer version) but have compatibility issues when loading with older PEFT versions. The saved adapter configs contain fields that older PEFT versions don't recognize.

## The Solution ✅

**Merge LoRA weights into the base model** → Create standard PyTorch models

### What Was Done:

Ran `python3 save_models_for_sharing.py` which:

1. ✅ **Loaded** each model with LoRA adapters
2. ✅ **Merged** LoRA weights into base RoBERTa-large weights
3. ✅ **Saved** as standard PyTorch models (no PEFT dependency)

### Result:

```
models_for_sharing/
├── extraction-Action-roberta-large-merged/       ✅
├── extraction-Actor-roberta-large-merged/        ✅
├── extraction-Effect-roberta-large-merged/       ✅
├── extraction-Evidence-roberta-large-merged/     ✅
└── extraction-Victim-roberta-large-merged/       ✅
```

Each directory contains:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `label_mapping.json` - Label definitions

## Using These Models

### On ANY Machine (No PEFT Required):

```python
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

# Load model
model = RobertaForTokenClassification.from_pretrained(
    'models_for_sharing/extraction-Action-roberta-large-merged'
)

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(
    'models_for_sharing/extraction-Action-roberta-large-merged',
    add_prefix_space=True
)

# Use for inference
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
```

**Requirements**: Just `pip install transformers torch` - no PEFT needed!

## Uploading to Hugging Face

### Option 1: Use the New Upload Script (Recommended)

```bash
python3 upload_merged_models_hf.py
```

This script:
- Loads merged models (no PEFT compatibility issues)
- Uploads using standard transformers API
- Works on any HF account

### Option 2: Manual Upload

```python
from transformers import RobertaForTokenClassification

model = RobertaForTokenClassification.from_pretrained(
    'models_for_sharing/extraction-Action-roberta-large-merged'
)

# Upload
model.push_to_hub('your-username/your-repo-name')
```

## Comparison: LoRA vs Merged Models

| Aspect | LoRA Adapters | Merged Models |
|--------|---------------|---------------|
| **File Size** | Small (~10MB adapters) | Large (~1.4GB each) |
| **Dependencies** | transformers + peft | transformers only |
| **Compatibility** | Version-sensitive | Universal ✅ |
| **Loading** | 2-step (base + adapter) | 1-step |
| **Portability** | Limited | Excellent ✅ |
| **Use Case** | Development, iteration | Deployment, sharing ✅ |

## Why Merged Models Are Better For Sharing

✅ **No version conflicts** - Works with any transformers version  
✅ **Simpler** - One-step loading  
✅ **Portable** - Transfer to any machine  
✅ **Standard** - No special libraries needed  
✅ **Reliable** - No adapter compatibility issues  

## Your Situation

### What Works ✅:
- **5 extraction models** successfully merged and saved
- Ready to upload to Hugging Face
- Can be used on any machine with just transformers

### What Needs Attention:
- **Detection model** had a `ModulesToSaveWrapper` error (likely due to how it was saved)
- Can retrain or use the best checkpoint from trainer directly

## Next Steps

### 1. Upload Extraction Models (READY NOW):

```bash
python3 upload_merged_models_hf.py
```

### 2. For Detection Model:

If you need to share it, you have two options:

**Option A**: Use the model from the trainer directly (if available in memory):
```python
# In your training notebook
trainer.model.merge_and_unload().save_pretrained('models_for_sharing/detection-merged')
```

**Option B**: Retrain with a clean LoRA config or use a checkpoint that loads successfully

### 3. Use on Another Machine:

```python
# Installation (one time)
pip install transformers torch

# Usage
from transformers import RobertaForTokenClassification
model = RobertaForTokenClassification.from_pretrained(
    'your-username/semeval26-extraction-Action-2024-02-01'
)
```

## Summary

✅ **Problem Solved**: PEFT version conflicts eliminated  
✅ **5 Models Ready**: All extraction models merged and saved  
✅ **Portable**: Work on any machine, no PEFT needed  
✅ **Upload Ready**: Use `upload_merged_models_hf.py`  

Your models are now in the **best format for sharing and deployment**! 🚀

## Files Overview

| File | Purpose |
|------|---------|
| `fix_adapter_configs.py` | Clean adapter configs (tried, has version issues) |
| `save_models_for_sharing.py` | **Merge LoRA → Standard models** ✅ |
| `upload_merged_models_hf.py` | **Upload merged models** ✅ |
| `models_for_sharing/` | **Portable models directory** ✅ |

**Recommendation**: Use the merged models from `models_for_sharing/` going forward!
