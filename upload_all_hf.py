#!/usr/bin/env python3
"""
Run both Task 2 (detection) and Task 1 (5 extraction models) upload scripts.
Uses a shared datetime suffix so all repos are grouped by upload time.

Usage:
  python upload_all_hf.py [--detection-only] [--extraction-only] [shared options]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from datetime import datetime


def main():
    root = Path(__file__).resolve().parent
    detection_script = root / "upload_detection_hf.py"
    extraction_script = root / "upload_extraction_5models_hf.py"

    parser = argparse.ArgumentParser(description="Upload detection + extraction models to Hugging Face")
    parser.add_argument("--detection-only", action="store_true", help="Upload only Task 2 detection model")
    parser.add_argument("--extraction-only", action="store_true", help="Upload only Task 1 (5 extraction models)")
    parser.add_argument("--model-dir", type=str, default="./roberta-large-binary-conspiracy-lora", help="Detection model dir")
    parser.add_argument("--models-base", type=str, default=os.path.join("Markers-Extraction", "models"), help="Extraction models base")
    parser.add_argument("--user", type=str, default=os.environ.get("HF_USERNAME"), help="HF username")
    parser.add_argument("--token", type=str, default=None, help="HF token")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for both scripts")
    parser.add_argument("--datetime-fmt", type=str, default="%Y-%m-%d-%H%M", help="DateTime format for repo IDs")
    args = parser.parse_args()

    do_detection = not args.extraction_only
    do_extraction = not args.detection_only
    if not do_detection and not do_extraction:
        print("Use --detection-only XOR --extraction-only, or neither to upload both.")
        sys.exit(1)

    dt = datetime.now().strftime(args.datetime_fmt)
    env = os.environ.copy()
    env["UPLOAD_DATETIME"] = dt

    def run(script: Path, extra: list[str]) -> int:
        cmd = [sys.executable, str(script), "--datetime-fmt", args.datetime_fmt, "--datetime-override", dt]
        if args.user:
            cmd += ["--user", args.user]
        if args.token:
            cmd += ["--token", args.token]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd += extra
        return subprocess.run(cmd, cwd=str(root), env=env).returncode

    failed = 0
    if do_detection:
        print("=" * 60)
        print("Uploading Task 2 (Detection) model")
        print("=" * 60)
        failed += run(detection_script, ["--model-dir", args.model_dir]) != 0
    if do_extraction:
        print("\n" + "=" * 60)
        print("Uploading Task 1 (5 Extraction) models")
        print("=" * 60)
        failed += run(extraction_script, ["--models-base", args.models_base]) != 0

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
