"""Join rehydrated text with gold markers for eval_token.py."""

from __future__ import annotations

import json
from pathlib import Path

from layers.paths import OUTPUTS, split_paths


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_gold_records(split: str) -> list[dict]:
    paths = split_paths(split)
    texts = load_jsonl(paths["text"])
    if split == "train":
        gold_by_id = {r["_id"]: (r.get("markers") or []) for r in texts}
    else:
        gold_rows = load_jsonl(paths["gold"])
        gold_by_id = {r["_id"]: (r.get("markers") or []) for r in gold_rows}
    records = []
    for row in texts:
        doc_id = row.get("_id")
        text = row.get("text") or ""
        if not doc_id or not text:
            continue
        records.append(
            {
                "_id": doc_id,
                "text": text,
                "markers": gold_by_id.get(doc_id) or [],
            }
        )
    return records


def write_gold(split: str, dest: Path | None = None) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dest = dest or (OUTPUTS / f"gold_{split}.jsonl")
    records = build_gold_records(split)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write joined gold JSONL for a split.")
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    args = parser.parse_args()
    path = write_gold(args.split)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
