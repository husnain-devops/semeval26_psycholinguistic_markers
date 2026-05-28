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
- marker-actor
datasets:
- semeval26-task10-psycholinguistic-markers
pipeline_tag: token-classification
---

# SemEval 2026 Task 10 — Marker Extraction (Actor)

RoBERTa-large with LoRA for **Actor** span extraction (SemEval 2026 Task 10, marker extraction subtask).

## Marker: Actor

Spans denoting **people, organizations, or entities** involved in the narrative (e.g. government, agencies, named groups).

## Model description

- **Task:** Token-level binary classification for Actor spans.
- **Architecture:** `roberta-large` + PEFT LoRA (`RobertaForTokenClassification`).
- **This repo:** Single-marker model for **Actor** only.

## Intended use

Research and evaluation on SemEval 2026 marker extraction. Use with the Action, Effect, Evidence, and Victim models for complete marker coverage.

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

## Training

Trained under `Markers-Extraction/` with LoRA. See project training scripts for hyperparameters.

## Limitations

- English Reddit submission statements only.
- One marker type per model.
- Requires task-consistent text rehydration and preprocessing.

## Citation

SemEval-2026 Task 10: Psycholinguistic Markers of Conspiracy Theories in Social Media Conversations.

## Model metadata

| Field | Value |
|-------|--------|
| Hub repo | `{repo_id}` |
| Marker type | Actor |
| Upload batch | `{upload_datetime}` |
| Base model | `{base_model}` |
