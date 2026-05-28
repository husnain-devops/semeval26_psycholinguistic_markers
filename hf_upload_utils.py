"""Shared helpers for Hugging Face model uploads."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_CARDS_DIR = Path(__file__).resolve().parent / "model_cards"


def model_card_path(cards_dir: Path, *, detection: bool = False, marker_type: str | None = None) -> Path:
    if detection:
        return cards_dir / "detection.md"
    if not marker_type:
        raise ValueError("marker_type required for extraction model cards")
    return cards_dir / f"extraction_{marker_type}.md"


def render_model_card(template_path: Path, **kwargs: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    return text.format_map(_SafeFormatDict(kwargs))


def upload_model_card(api, repo_id: str, readme: str) -> None:
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
