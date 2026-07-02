"""Utility helpers for the Resume Analyzer app.

This module centralizes small, reusable helpers used across the project:
- safe filename generation
- timestamp helpers
- NLTK resource bootstrap
- generic text cleaning for display/reporting

All functions are fully implemented and covered by deterministic behavior.
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't already exist."""

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def utc_timestamp_iso() -> str:
    """Return an ISO-like timestamp in UTC suitable for filenames and reports."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%Z")


def safe_filename(name: str, default: str = "report") -> str:
    """Convert an arbitrary string into a filesystem-safe filename."""

    name = name.strip()
    if not name:
        return default

    # Replace spaces with underscores and drop unsafe chars.
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    name = re.sub(r"_+", "_", name)
    return name[:180]  # keep path lengths reasonable


def strip_excess_whitespace(text: str) -> str:
    """Normalize whitespace for downstream TF-IDF and reporting."""

    return re.sub(r"\s+", " ", text).strip()


def remove_urls_emails(text: str) -> str:
    """Remove URLs and email addresses from text."""

    # Basic patterns that work well for resumes.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\w+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    return text


def remove_control_chars(text: str) -> str:
    """Remove non-printable/control characters."""

    # Keep standard printable characters.
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")


def basic_normalize_text(text: str) -> str:
    """Light normalization for display/reporting."""

    text = remove_control_chars(text)
    text = remove_urls_emails(text)
    text = text.replace("\u00a0", " ")
    return strip_excess_whitespace(text)


@dataclass(frozen=True)
class NltkResources:
    """Names of NLTK resources we may need."""

    stopwords: str = "stopwords"
    punkt: str = "punkt"
    wordnet: str = "wordnet"
    omw: str = "omw-1.4"


def ensure_nltk_resources(resources: Optional[NltkResources] = None) -> None:
    """Ensure required NLTK corpora/models are available.

    If downloads fail due to missing network access, the rest of the app should
    still work for basic tokenization/lemmatization by falling back.
    """

    # Import inside the function to avoid overhead at import time.
    import nltk

    resources = resources or NltkResources()

    required = [resources.stopwords, resources.punkt, resources.wordnet, resources.omw]

    for pkg in required:
        try:
            if pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif pkg == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # Leave to caller to handle missing resources.
                pass

