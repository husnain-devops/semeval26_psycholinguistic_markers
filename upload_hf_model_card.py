"""Load and upload Hugging Face model cards (README.md) for SemEval 2026 models."""

from __future__ import annotations

from pathlib import Path


def render_model_card(template_path: Path, **kwargs: str) -> str:
    """Fill {placeholders} in a markdown template."""
    text = template_path.read_text(encoding="utf-8")
    try:
        return text.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Model card template {template_path} is missing placeholder: {{{missing}}}") from e


def upload_model_card(api, repo_id: str, readme: str) -> None:
    """Upload rendered README as the Hub model card."""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )


def default_cards_dir() -> Path:
    return Path(__file__).resolve().parent / "model_cards"


def card_path_for(cards_dir: Path, *, detection: bool = False, marker_type: str | None = None) -> Path:
    if detection:
        return cards_dir / "detection.md"
    if marker_type is None:
        raise ValueError("marker_type is required for extraction model cards")
    return cards_dir / f"extraction_{marker_type}.md"
