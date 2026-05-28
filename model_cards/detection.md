---
base_model: roberta-large
library_name: peft
license: mit
language: en
tags:
- semeval2026
- task10
- conspiracy-detection
- psycholinguistic-markers
- lora
- roberta-large
datasets:
- semeval26-task10-psycholinguistic-markers
pipeline_tag: text-classification
---

# SemEval 2026 Task 10 — Conspiracy Detection

RoBERTa-large with LoRA for **binary conspiracy detection** on Reddit submission statements (SemEval 2026 Task 10, detection subtask).

## Model description

- **Task:** Classify each document as conspiracy-related (`Yes`) or not (`No`). Samples labeled `Can't tell` in training are excluded from the binary setup.
- **Architecture:** `roberta-large` + PEFT LoRA adapter (sequence classification, 2 labels).
- **Labels:** `No` (0), `Yes` (1).
- **Max length:** 512 tokens.

## Intended use

Research and evaluation on the SemEval 2026 psycholinguistic markers shared task. Not intended for high-stakes or real-time moderation without further validation.

## How to load

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

repo_id = "{repo_id}"
base_model = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
model = PeftModel.from_pretrained(model, repo_id)
```

For merged weights (if you merge LoRA locally), load with `AutoModelForSequenceClassification.from_pretrained(repo_id)` only.

## Training

Trained with parameter-efficient fine-tuning (LoRA) on rehydrated task data. See the project repository for splits, hyperparameters, and `train_and_infer_binary.py` / `infer_binary.py`.

## Evaluation

Binary metrics on `Yes` / `No` dev labels (e.g. `dev_public.jsonl`). Use the companion extraction models for span-level psycholinguistic markers.

## Limitations

- Trained on English Reddit submission statements; may not generalize to other domains or languages.
- Task labels have moderate annotator agreement; treat outputs as research signals, not ground truth.
- Oversampled conspiracy-related content in the corpus — prevalence on open web may differ.

## Citation

SemEval-2026 Task 10: Psycholinguistic Markers of Conspiracy Theories in Social Media Conversations.

## Model metadata

| Field | Value |
|-------|--------|
| Hub repo | `{repo_id}` |
| Upload batch | `{upload_datetime}` |
| Base model | `{base_model}` |
