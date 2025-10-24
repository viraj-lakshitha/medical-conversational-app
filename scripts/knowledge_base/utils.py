"""
Utility helpers shared across the knowledge base tooling.

The functions in this module are intentionally lightweight so they can be
imported both as part of the ``knowledge_base`` package and from standalone
scripts executed directly via ``python path/to/file.py``.
"""

from __future__ import annotations

import re
from typing import Iterable, List


def make_symptom_id(symptom: str) -> str:
    """
    Create a deterministic identifier for a symptom.

    The identifier mirrors the format used when symptoms are persisted to
    Neo4j (``symptom::<slug>``). Keeping this logic in one place prevents the
    accidental creation of mismatched IDs across different scripts.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", symptom.lower()).strip("_")
    if not slug:
        slug = f"symptom_{abs(hash(symptom))}"
    return f"symptom::{slug}"


def normalise_symptom_text(symptom: str) -> str:
    """
    Apply minimal normalisation to raw symptom text.

    The builder performs additional medical-domain replacements; this helper
    focuses on consistent casing and whitespace so other modules (e.g.
    query utilities) can align user input with stored values.
    """
    return re.sub(r"\s+", " ", symptom.strip().lower())


def ensure_list(value: Iterable[str] | str) -> List[str]:
    """Force input into a list of strings."""
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


__all__ = ["make_symptom_id", "normalise_symptom_text", "ensure_list"]
