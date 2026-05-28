#!/usr/bin/env python3
"""
Upload Task 2 (Conspiracy Detection) model to Hugging Face Hub.
Use with models trained by train_and_infer_binary.py (RoBERTa-large + LoRA).

Repo ID format: {HF_USER}/semeval26-psysemev-detection-{YYYY-MM-DD-HHMM}
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from datetime import datetime

from upload_hf_model_card import (
    card_path_for,
    default_cards_dir,
    render_model_card,
    upload_model_card,
)


def require():
    try:
        import huggingface_hub  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print("Missing dependencies. Install with:", file=sys.stderr)
        print("  pip install huggingface_hub peft transformers", file=sys.stderr)
        raise SystemExit(1) from e


def find_latest_checkpoint(base_path: str | Path) -> Path:
    base = Path(base_path)
    pattern = str(base / "checkpoint-*")
    dirs = glob.glob(pattern)

    def _key(p: str) -> int:
        name = Path(p).name
        suffix = name.split("-")[-1]
        return int(suffix) if suffix.isdigit() else 0

    dirs = sorted(dirs, key=_key)
    if not dirs:
        return base
    return Path(dirs[-1])


def main():
    require()
    from huggingface_hub import HfApi, login
    from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast
    from peft import PeftModel

    parser = argparse.ArgumentParser(description="Upload Task 2 detection model to Hugging Face")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./roberta-large-binary-conspiracy-lora",
        help="Path to model directory (train_and_infer_binary output)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.environ.get("HF_USERNAME"),
        help="Hugging Face username (or set HF_USERNAME). Uses whoami if omitted.",
    )
    parser.add_argument(
        "--datetime-fmt",
        type=str,
        default="%Y-%m-%d-%H%M",
        help="DateTime suffix format for repo ID",
    )
    parser.add_argument(
        "--datetime-override",
        type=str,
        default=os.environ.get("UPLOAD_DATETIME"),
        help="Use this string as datetime suffix instead of now() (e.g. from upload_all_hf)",
    )
    parser.add_argument("--token", type=str, default=None, help="HF token (or use huggingface-cli login)")
    parser.add_argument(
        "--model-cards-dir",
        type=str,
        default=str(default_cards_dir()),
        help="Directory with detection.md model card template (set empty to skip)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print repo ID and paths only, do not upload")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    checkpoint = find_latest_checkpoint(model_dir)
    dt = args.datetime_override or datetime.now().strftime(args.datetime_fmt)

    if args.token:
        login(token=args.token)
    api = HfApi()
    user = args.user
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print("Could not determine HF user. Set --user or HF_USERNAME, or run 'huggingface-cli login'.")
            raise SystemExit(1) from e

    repo_id = f"{user}/semeval26-psysemev-detection-{dt}"
    print(f"Model dir: {model_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Repo ID: {repo_id}")

    if args.dry_run:
        print("Dry run. Exiting.")
        return

    base_model = "roberta-large"
    print(f"Loading base {base_model} and adapter from {checkpoint}...")
    config = RobertaConfig.from_pretrained(base_model)
    config.num_labels = 2
    config.id2label = {0: "No", 1: "Yes"}
    config.label2id = {"No": 0, "Yes": 1}
    model = RobertaForSequenceClassification.from_pretrained(base_model, config=config)
    model = PeftModel.from_pretrained(model, str(checkpoint))
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model)

    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)

    if args.model_cards_dir:
        cards_dir = Path(args.model_cards_dir)
        template = card_path_for(cards_dir, detection=True)
        if template.exists():
            readme = render_model_card(
                template,
                repo_id=repo_id,
                base_model=base_model,
                upload_datetime=dt,
            )
            print(f"Uploading model card from {template}...")
            upload_model_card(api, repo_id, readme)
        else:
            print(f"Warning: model card not found at {template}, skipping README upload.")

    print(f"Done. https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
