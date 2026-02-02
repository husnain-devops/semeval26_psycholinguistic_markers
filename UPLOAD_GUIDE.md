# Model Upload Guide - SemEval 2026 Psycholinguistic Markers

## Current Status

### ✅ What's Ready:

**5 Extraction Models (Merged, No PEFT)**
- `models_for_sharing/extraction-Action-roberta-large-merged/`
- `models_for_sharing/extraction-Actor-roberta-large-merged/`
- `models_for_sharing/extraction-Effect-roberta-large-merged/`
- `models_for_sharing/extraction-Evidence-roberta-large-merged/`
- `models_for_sharing/extraction-Victim-roberta-large-merged/`

**Status**: ✅ Ready to upload to Hugging Face

### ⚠️ What Needs Attention:

**Detection Model**
- Location: `roberta-large-binary-conspiracy-lora/checkpoint-168`
- Issue: Corrupted PEFT state (`ModulesToSaveWrapper` error)
- Status: ❌ Cannot merge LoRA weights

## Upload Instructions

### Step 1: Upload Extraction Models (Ready Now!)

```bash
# Login to Hugging Face (one time)
huggingface-cli login

# Upload all 5 extraction models
python3 upload_extraction_merged.py
```

This will upload to:
- `MMHusnain/semeval26-extraction-Action-YYYY-MM-DD-HHMM`
- `MMHusnain/semeval26-extraction-Actor-YYYY-MM-DD-HHMM`
- `MMHusnain/semeval26-extraction-Effect-YYYY-MM-DD-HHMM`
- `MMHusnain/semeval26-extraction-Evidence-YYYY-MM-DD-HHMM`
- `MMHusnain/semeval26-extraction-Victim-YYYY-MM-DD-HHMM`

### Step 2: Detection Model Options

The detection model has a corrupted state. Here are your options:

#### Option A: Use Local Model (Recommended for Challenge)

The model works fine locally for inference! You can:
1. Keep using it locally for predictions
2. Create your submission files locally
3. Submit those to the challenge

The challenge only requires submission files, not the model itself!

#### Option B: Retrain Detection Model

If you need to share it, retrain with clean LoRA config:

```python
# In your notebook, after training completes:
trainer.model.merge_and_unload().save_pretrained('models_for_sharing/detection-merged')
```

Then upload directly from the notebook.

#### Option C: Save from Notebook Memory

If you have the model in notebook memory after training:

```python
# In the training notebook, after training:
merged = trainer.model.merge_and_unload()
merged.save_pretrained('models_for_sharing/detection-merged')
tokenizer.save_pretrained('models_for_sharing/detection-merged')

# Then upload
from huggingface_hub import login
login()
merged.push_to_hub('MMHusnain/semeval26-detection-YYYY-MM-DD')
```

## Using Models on Another Machine

### Extraction Models (After Upload):

```python
# Install dependencies
pip install transformers torch

# Load model
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

model = RobertaForTokenClassification.from_pretrained(
    'MMHusnain/semeval26-extraction-Action-2026-02-01-1234'
)
tokenizer = RobertaTokenizerFast.from_pretrained(
    'MMHusnain/semeval26-extraction-Action-2026-02-01-1234',
    add_prefix_space=True
)

# Run inference
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

### Detection Model (Local Use):

The detection model works fine locally! Use it for your challenge submissions:

```python
# Your existing inference code works fine
from transformers import RobertaForSequenceClassification
from peft import PeftModel

base_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=3
)
model = PeftModel.from_pretrained(
    base_model,
    "roberta-large-binary-conspiracy-lora/checkpoint-168"
)

# Use for predictions - this works!
# The issue is only with uploading to HF
```

## For the Challenge

### What You Need to Submit:

1. **Markers Extraction Task**:
   - File: `Markers-Extraction/submission.zip`
   - Contains: `submission.jsonl` with predicted markers
   - ✅ You already have this!

2. **Detection Task** (if participating):
   - File: Predictions for conspiracy detection
   - ✅ You can generate this locally with your working model

### You DO NOT need to upload models to complete the challenge!

The challenge only asks for prediction files (`.jsonl` or `.zip`), not the models themselves.

## Summary

| Item | Status | Action |
|------|--------|--------|
| Extraction Models | ✅ Ready | Upload with `upload_extraction_merged.py` |
| Detection Model | ⚠️ Corrupted upload state | Use locally or retrain |
| Challenge Submission | ✅ Ready | Use `Markers-Extraction/submission.zip` |

## Next Steps

1. ✅ **Upload extraction models** (if you want to share them):
   ```bash
   python3 upload_extraction_merged.py
   ```

2. ✅ **Use detection model locally** for challenge:
   - Model works fine for inference
   - Generate submission files
   - Submit to challenge

3. ⏭️ **Optional**: Retrain detection model if you need to share it

## Questions?

- **Q**: Can I still submit to the challenge?
  - **A**: Yes! Use your models locally to generate submission files

- **Q**: Do I need to fix the detection model upload?
  - **A**: No, not for the challenge. Only if you want to share it publicly

- **Q**: Are the extraction models good to go?
  - **A**: Yes! They're in perfect shape for upload and sharing

## Files Created

- `save_models_for_sharing.py` - Merge LoRA weights (done)
- `save_detection_model_merged.py` - Attempt detection model merge (failed)
- `upload_extraction_merged.py` - Upload extraction models ✅
- `MODEL_SHARING_GUIDE.md` - Detailed technical guide
- `UPLOAD_GUIDE.md` - This file

---

**TL;DR**: 
- ✅ Extraction models are ready to upload
- ⚠️ Detection model works locally, upload is broken (not critical for challenge)
- ✅ You have everything needed for challenge submission
