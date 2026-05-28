---
base_model: roberta-large
library_name: peft
license: mit
language: en
tags:
- semeval2026
- task10
- token-classification
- ner
- psycholinguistic-markers
- lora
- roberta-large
- marker-action
datasets:
- semeval26-task10-psycholinguistic-markers
pipeline_tag: token-classification
---

# SemEval 2026 Task 10 — Marker Extraction (Action)

RoBERTa-large with LoRA for **Action** span extraction (SemEval 2026 Task 10, marker extraction subtask).

## Marker: Action

Spans denoting **actions or events** (e.g. coups, invasions, operations) in conspiracy-related discourse.

## Model description

- **Task:** Token-level binary classification (BIO-style labels collapsed to token labels in this checkpoint).
- **Architecture:** `roberta-large` + PEFT LoRA (`RobertaForTokenClassification`).
- **This repo:** Single-marker model for **Action** only. Sibling repos cover Actor, Effect, Evidence, and Victim.

## Intended use

Research and evaluation on SemEval 2026 marker extraction. Combine predictions from all five marker models for full submission format.

## How to load

```python
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from peft import PeftModel

repo_id = "{repo_id}"
base_model = "roberta-large"

tokenizer = RobertaTokenizerFast.from_pretrained(repo_id, add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained(base_model, num_labels=2)
model = PeftModel.from_pretrained(model, repo_id)
```

`label_mapping.json` in the repo describes label ids for this marker type.

## Training

Trained under `Markers-Extraction/` with LoRA on rehydrated annotations. See `train_5_models.py` and related scripts.

## Limitations

- English Reddit submission statements only.
- One marker type per model; aggregate spans externally for full task output.
- Span boundaries depend on tokenizer alignment and preprocessing matching the shared task starter pack.

## Citation

SemEval-2026 Task 10: Psycholinguistic Markers of Conspiracy Theories in Social Media Conversations.

## Model metadata

| Field | Value |
|-------|--------|
| Hub repo | `{repo_id}` |
| Marker type | Action |
| Upload batch | `{upload_datetime}` |
| Base model | `{base_model}` |
