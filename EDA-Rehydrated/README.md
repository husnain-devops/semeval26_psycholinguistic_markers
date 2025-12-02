# EDA-Rehydrated

This folder contains the EDA pipeline for the rehydrated train dataset.

How to create a virtual environment and install dependencies (macOS / zsh):

1. Create and activate venv

```
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install requirements

```
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Some packages (e.g., `hdbscan`, `spacy`, `torch`) can be heavy. If installation fails, install a subset first (pandas, numpy, scikit-learn, spacy, sentence-transformers) and re-run the pipeline; the scripts will log any missing features.
- To download spaCy model run:

```
python -m spacy download en_core_web_sm
```

Run the full pipeline:

```
./run_all.sh
```

Outputs are saved under the folder structure in this directory.
