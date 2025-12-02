#!/usr/bin/env python3
"""
extract_keys.py

Read a JSONL file and print all unique keys found across the records.
Nested keys are shown with dot notation (e.g. markers.startIndex).

Usage examples:
  python3 extract_keys.py train_redacted.jsonl
  python3 extract_keys.py train_redacted.jsonl -o keys.txt --show-counts
"""

import argparse
import json
import sys
from typing import Set, Dict, Any


def collect_keys(obj: Any, parent: str = "", keys: Set[str] = None) -> Set[str]:
    """Recursively collect keys from dictionaries and lists.

    - dict keys become 'parent.key' (or just 'key' if parent is empty)
    - for lists, we recurse into elements using the same parent
    """
    if keys is None:
        keys = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{parent}.{k}" if parent else k
            keys.add(full)
            collect_keys(v, full, keys)
    elif isinstance(obj, list):
        for item in obj:
            collect_keys(item, parent, keys)
    # primitives (str/int/float/bool/None) have no keys
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract keys from a JSONL file (flatten nested keys).")
    parser.add_argument("file", nargs="?", default="train_redacted.jsonl", help="path to the JSONL file (default: train_redacted.jsonl)")
    parser.add_argument("-o", "--output", help="write keys to this file instead of stdout")
    parser.add_argument("--show-counts", action="store_true", help="also show how many records contain each key")
    args = parser.parse_args()

    all_keys: Set[str] = set()
    counts: Dict[str, int] = {}

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping malformed JSON at line {lineno}: {e}", file=sys.stderr)
                    continue

                local_keys = collect_keys(obj)
                for k in local_keys:
                    counts[k] = counts.get(k, 0) + 1
                all_keys.update(local_keys)
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    keys_sorted = sorted(all_keys)
    output_text = "\n".join(keys_sorted)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            out_f.write(output_text)
        print(f"Wrote {len(keys_sorted)} unique keys to {args.output}")
    else:
        print(output_text)

    if args.show_counts:
        print("\nKey occurrence counts (number of records containing the key):")
        for k in sorted(counts.keys(), key=lambda x: (-counts[x], x)):
            print(f"{counts[k]:6d}  {k}")


if __name__ == "__main__":
    main()
