# SCAFFOLD layer alignment

Rule-layer diagnostic for the thesis Micro / Meso / Macro extractors. The pipeline emits PsyCoMark `{startIndex, endIndex, type}` spans and scores them against gold with the official token-IoU overlap-F1 protocol (`eval_token.py`). There is no LoRA and no detection head.

## What is being tested

Gold Actor is a conspiratorial agent, not every syntactic subject. These extractors will over-generate. A layer is useful as a **prior** if it covers a non-trivial fraction of gold spans of its assigned types, even if official O-F1 is below RB-EXT-5M (~0.22 on dev).

| Layer | NLP operation | Marker spans |
|-------|----------------|--------------|
| Micro | spaCy POS (features only); textacy / UD SVO; PropBank-style SRL or UD proxy | Actor, Action, Victim |
| Meso | fastcoref (or lexical repetition); PDTB-style causal connectives | Effect; inherited Actor/Victim |
| Macro | hedge / certainty lexicon (features only); cue-based argument miner | Evidence |

## Data

Uses canonical PsyCoMark files in this repo, not the SemEval-2025 COVID export.

- Train gold: `EDA-Rehydrated/data/train_rehydrated.jsonl`
- Dev text: `EDA-Rehydrated/data/dev_rehydrated.jsonl`
- Dev gold: `dev_public.jsonl` (joined on `_id`)
- Test is unlabeled and is not scored

## Setup

Reuse the EDA virtualenv if you have it (`spacy` + `en_core_web_sm`). Otherwise:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../EDA-Rehydrated/requirements.txt
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`textacy` and `fastcoref` are optional at runtime: SVO falls back to Universal Dependencies, coref falls back to exact-string repetition. HuggingFace SRL is opt-in (`--srl hf`) and otherwise uses the UD Arg0/predicate/Arg1 proxy.

## Run

From this directory:

```bash
python run_extract.py --split dev --layers all
```

Options:

- `--split dev|train` (primary report is `dev`)
- `--layers all|micro|meso|macro`
- `--srl ud|hf|auto`
- `--limit N` smoke-test cap
- `--no-score` write preds only

## Outputs (`outputs/`, gitignored)

- `preds_dev.jsonl` — all predicted spans, with `layer` / `source` tags
- `preds_dev_{micro,micro_meso,all}.jsonl` — ablation views
- `gold_dev.jsonl` — joined text + gold markers
- `scores_dev_{tag}.json` — official CodaBench-style scores
- `ablation_dev.json` — official O-F1 plus coverage
- `features_dev.jsonl` — POS and modality sidecars (no spans)
- `alignment_report.md` — tables and qualitative TP / FP / FN examples

## Scoring

Official: token IoU ≥ 0.5, one-to-one, type-constrained (`../eval_token.py`).

Coverage: share of gold spans that overlap any same-type prediction at IoU ≥ 0.5. This is the prior-usefulness number.

Ablations: micro only; micro+meso; all three. Meso should move Effect; macro should move Evidence.
