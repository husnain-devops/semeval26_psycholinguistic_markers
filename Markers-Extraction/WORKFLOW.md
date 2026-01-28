# Complete Workflow: Training → Inference → Submission

## Quick Start (All-in-One Notebook)

```bash
cd Markers-Extraction/notebooks
jupyter notebook 01_RoBERTa_LoRA_Markers_Extraction.ipynb
```

Then in the notebook:
1. **Run Cells 0-10**: Train the model (~40 minutes on GPU)
2. **Run Cells 11-17**: Generate predictions + create submission
3. **Done!** Your `submission.jsonl` is ready

## What Each Section Does

### 📚 Training Section (Cells 0-10)

```
Cell 0: Environment Setup
├─ Check GPU availability
└─ Display PyTorch version

Cell 1: Import Libraries
├─ transformers (RoBERTa, Trainer)
├─ peft (LoRA)
└─ sklearn (metrics, splitting)

Cell 2: Load Training Data
├─ Read train_rehydrated.jsonl
└─ 3,213 examples with markers

Cell 3: Create BIO Labels
├─ Action: B-Action, I-Action
├─ Actor: B-Actor, I-Actor
├─ Effect: B-Effect, I-Effect
├─ Evidence: B-Evidence, I-Evidence
├─ Victim: B-Victim, I-Victim
└─ O (Outside): No marker
    Total: 11 labels

Cell 4: Tokenization
├─ RoBERTa tokenizer with prefix space
├─ Max length: 256 tokens
├─ Align BIO labels to tokens
└─ Use offset mapping

Cell 5: Train/Val Split
├─ 90% training (2,891 examples)
└─ 10% validation (322 examples)

Cell 6: Model Setup
├─ Base: roberta-base (125M params)
├─ LoRA: rank=16, alpha=32, dropout=0.1
└─ Trainable: ~3M params (~2.4%)

Cell 7: Metrics
├─ Precision, Recall, F1
└─ Per-marker-type breakdown

Cell 8: Training
├─ Batch size: 8 (effective 16)
├─ Learning rate: 3e-4
├─ Epochs: 10 (early stop: patience=3)
├─ Optimizer: AdamW
└─ Warmup: 10%

Cell 9: Validation Results
├─ Loss, Accuracy
└─ F1 per marker type

Cell 10: Save Model
└─ Saved to: ../models/roberta-base-markers-lora/
```

### 🔮 Inference Section (Cells 11-17)

```
Cell 11: Header
└─ "Inference on Dev/Test Data"

Cell 12: Load Test Data
├─ Read dev_rehydrated.jsonl (or test file)
└─ Keep original data for submission

Cell 13: Tokenize Test Data
├─ Same tokenization as training
└─ Offset mapping for span extraction

Cell 14: Generate Predictions
├─ Use trained model
├─ Get logits from RoBERTa
└─ Argmax to get predicted labels

Cell 15: Extract Character Spans
├─ Convert token predictions to spans
├─ BIO-aware span assembly:
│  ├─ B-X: Start new marker of type X
│  ├─ I-X: Continue marker of type X
│  └─ O: No marker
└─ Output: [{"startIndex": 10, "endIndex": 15, "type": "Action"}, ...]

Cell 16: Create Submission File
├─ Format: {"_id": "...", "markers": [...]}
├─ One line per example (JSONL)
└─ Saved to: ../submission.jsonl

Cell 17: Sample Predictions
├─ Display first 3 examples
├─ Show text snippets
└─ Show extracted markers
```

## Output: submission.jsonl

**Format**:
```json
{"_id": "t1_abcd123", "markers": [{"startIndex": 0, "endIndex": 2, "type": "Actor"}, {"startIndex": 10, "endIndex": 15, "type": "Action"}]}
{"_id": "t1_efgh456", "markers": []}
{"_id": "t1_ijkl789", "markers": [{"startIndex": 5, "endIndex": 12, "type": "Evidence"}]}
```

**Key Points**:
- ✅ One JSON object per line (JSONL format)
- ✅ `_id`: Must match input document ID
- ✅ `markers`: List of marker objects
- ✅ `startIndex`, `endIndex`: **Character positions** (not tokens)
- ✅ `type`: One of [Action, Actor, Effect, Evidence, Victim]

## Submission to CodaBench

### 1. Create ZIP
```bash
cd /home/husnain/semeval26_psycholinguistic_markers
zip submission.zip submission.jsonl
```

### 2. Upload to CodaBench
1. Go to: https://www.codabench.org/competitions/10751/
2. Click: "My Submissions" tab
3. Upload: `submission.zip`
4. Wait: A few minutes for evaluation
5. Add to leaderboard and make public!

### 3. Expected Score
- **Baseline** (5 separate DistilBERT models): ~0.15 F1
- **This notebook** (RoBERTa-LoRA): **~0.40-0.45 F1** 🎯

## Alternative: Standalone Scripts

If you prefer command-line scripts:

### Training
```bash
cd Markers-Extraction
python scripts/train_markers.py
```

### Inference
```bash
python scripts/infer_markers.py \
  --input ../dev_rehydrated.jsonl \
  --output submission.jsonl \
  --model models/roberta-base-markers-lora/final_model
```

### Submission
```bash
cd ..
zip submission.zip submission.jsonl
# Upload to CodaBench
```

## Troubleshooting

### Issue: "Can't find dev_rehydrated.jsonl"
**Solution**: Update `TEST_FILE` path in Cell 12:
```python
TEST_FILE = Path('/absolute/path/to/dev_rehydrated.jsonl')
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in Cell 6:
```python
per_device_train_batch_size=4,  # Was 8
gradient_accumulation_steps=4,  # Was 2
```

### Issue: "Model not found for inference"
**Solution**: Make sure Cell 10 (Save Model) ran successfully. Check:
```bash
ls Markers-Extraction/models/roberta-base-markers-lora/final_model/
# Should see: config.json, adapter_model.bin, adapter_config.json, label_mapping.json
```

## File Structure After Training

```
Markers-Extraction/
├── notebooks/
│   ├── 01_RoBERTa_LoRA_Markers_Extraction.ipynb  ← Main notebook
│   └── submission.jsonl                            ← Created by Cell 16
├── models/
│   └── roberta-base-markers-lora/
│       ├── checkpoint-XXX/                         ← Intermediate checkpoints
│       └── final_model/                            ← Final model (best)
│           ├── adapter_model.bin
│           ├── adapter_config.json
│           ├── config.json
│           ├── tokenizer files...
│           └── label_mapping.json
├── scripts/
│   ├── train_markers.py                            ← Alternative training
│   └── infer_markers.py                            ← Alternative inference
└── README.md                                       ← Documentation
```

## Summary

| Step | Tool | Time | Output |
|------|------|------|--------|
| **1. Train** | Cells 0-10 | ~40 min | Trained model |
| **2. Inference** | Cells 11-17 | ~2 min | submission.jsonl |
| **3. Submit** | Command line | 1 min | CodaBench score |

**Total**: ~45 minutes from zero to leaderboard! 🚀

## Next Steps After Submission

1. **Check Score**: Compare against baseline (~0.15 F1)
2. **Analyze Errors**: Look at misclassified markers
3. **Improve**:
   - Adjust LoRA hyperparameters
   - Increase training epochs
   - Try different base models (roberta-large)
   - Ensemble multiple runs
4. **Re-submit**: Iterate and improve!

---

**🎯 Goal**: Beat the 0.15 F1 baseline by 3x with parameter-efficient fine-tuning!
