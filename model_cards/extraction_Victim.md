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
- marker-victim
datasets:
- semeval26-task10-psycholinguistic-markers
pipeline_tag: token-classification
---

# SemEval 2026 Task 10 — Marker Extraction (Victim)

RoBERTa-large with LoRA for **Victim** span extraction (SemEval 2026 Task 10, marker extraction subtask).

## Marker: Victim

Spans denoting **affected parties or targets** (e.g. populations, countries, groups harmed in the narrative).

## Model description

- **Task:** Token-level binary classification for Victim spans.
- **Architecture:** `roberta-large` + PEFT LoRA (`RobertaForTokenClassification`).
- **This repo:** Single-marker model for **Victim** only.

## Intended use

Research and evaluation on SemEval 2026 marker extraction.

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

Trained under `Markers-Extraction/` with LoRA.

## Limitations

- English Reddit submission statements only.
- One marker type per model.

## Citation

SemEval-2026 Task 10: Psycholinguistic Markers of Conspiracy Theories in Social Media Conversations.

## Model metadata

| Field | Value |
|-------|--------|
| Hub repo | `{repo_id}` |
| Marker type | Victim |
| Upload batch | `{upload_datetime}` |
| Base model | `{base_model}` |
