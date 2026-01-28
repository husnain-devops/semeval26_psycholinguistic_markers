#!/usr/bin/env python3
"""
Upload Task 1 (Marker Extraction) five models to Hugging Face Hub.
Use with models trained by Markers-Extraction/scripts/train_5_models.py.

Repo ID format: {HF_USER}/semeval26-psysemev-extraction-{Action|Actor|Effect|Evidence|Victim}-{YYYY-MM-DD-HHMM}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datetime import datetime

MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]


def require():
    try:
        import huggingface_hub  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print("Missing dependencies. Install with:", file=sys.stderr)
        print("  pip install huggingface_hub peft transformers", file=sys.stderr)
        raise SystemExit(1) from e


def main():
    require()
    from huggingface_hub import HfApi, login
    from transformers import RobertaForTokenClassification, RobertaTokenizerFast
    from peft import PeftModel

    parser = argparse.ArgumentParser(description="Upload Task 1 (5 extraction models) to Hugging Face")
    parser.add_argument(
        "--models-base",
        type=str,
        default=os.path.join("Markers-Extraction", "models"),
        help="Base dir containing roberta-large-lora-{Action,Actor,...}",
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
        help="DateTime suffix format for repo IDs",
    )
    parser.add_argument(
        "--datetime-override",
        type=str,
        default=os.environ.get("UPLOAD_DATETIME"),
        help="Use this string as datetime suffix instead of now() (e.g. from upload_all_hf)",
    )
    parser.add_argument("--token", type=str, default=None, help="HF token (or use huggingface-cli login)")
    parser.add_argument("--dry-run", action="store_true", help="Print repo IDs and paths only, do not upload")
    parser.add_argument(
        "--markers",
        type=str,
        nargs="+",
        default=MARKER_TYPES,
        help="Marker types to upload",
    )
    args = parser.parse_args()

    base = Path(args.models_base)
    if not base.exists():
        print(f"Error: Models base not found: {base}")
        sys.exit(1)

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

    for marker_type in args.markers:
        model_dir = base / f"roberta-large-lora-{marker_type}" / "final_model"
        if not model_dir.exists():
            print(f"Skip {marker_type}: not found at {model_dir}")
            continue

        repo_id = f"{user}/semeval26-psysemev-extraction-{marker_type}-{dt}"
        print(f"[{marker_type}] {model_dir} -> {repo_id}")

        if args.dry_run:
            continue

        print(f"  Loading base roberta-large + adapter from {model_dir}...")
        model = RobertaForTokenClassification.from_pretrained("roberta-large", num_labels=2)
        model = PeftModel.from_pretrained(model, str(model_dir))
        tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir), add_prefix_space=True)

        print(f"  Pushing to {repo_id}...")
        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id, private=False)

        label_path = model_dir / "label_mapping.json"
        if label_path.exists():
            api.upload_file(
                path_or_fileobj=str(label_path),
                path_in_repo="label_mapping.json",
                repo_id=repo_id,
                repo_type="model",
            )
        print(f"  Done. https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("Dry run. No uploads performed.")
    else:
        print("\nAll 5 extraction models uploaded.")


if __name__ == "__main__":
    main()
