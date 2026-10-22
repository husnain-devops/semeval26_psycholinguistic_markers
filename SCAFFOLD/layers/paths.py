"""Resolve PsyCoMark JSONL locations inside the SemEval repo."""

from __future__ import annotations

from pathlib import Path

SCAFFOLD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCAFFOLD_ROOT.parent
EDA_DATA = REPO_ROOT / "EDA-Rehydrated" / "data"
OUTPUTS = SCAFFOLD_ROOT / "outputs"


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def train_path() -> Path:
    return _first_existing(
        EDA_DATA / "train_rehydrated.jsonl",
        REPO_ROOT / "train_rehydrated.jsonl",
    )


def dev_text_path() -> Path:
    return _first_existing(
        EDA_DATA / "dev_rehydrated.jsonl",
        REPO_ROOT / "dev_rehydrated.jsonl",
    )


def dev_gold_path() -> Path:
    return _first_existing(
        EDA_DATA / "dev_public.jsonl",
        REPO_ROOT / "dev_public.jsonl",
    )


def eval_token_path() -> Path:
    return REPO_ROOT / "eval_token.py"


def split_paths(split: str) -> dict[str, Path]:
    if split == "train":
        return {"text": train_path(), "gold": train_path()}
    if split == "dev":
        return {"text": dev_text_path(), "gold": dev_gold_path()}
    raise ValueError(f"unsupported split: {split} (use train or dev)")
