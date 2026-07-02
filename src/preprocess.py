"""Text cleaning and NLP preprocessing.

Implements the required pipeline:
- lowercase
- remove punctuation
- remove numbers
- tokenization
- stopword removal
- lemmatization

Designed to be deterministic and safe for production usage.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .utils import ensure_nltk_resources


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = True
    remove_urls_emails: bool = True
    keep_internal_hyphens: bool = True
    use_lemmatization: bool = True


_PUNCT_TRANS = str.maketrans({c: " " for c in string.punctuation})


def _maybe_download_nltk() -> None:
    # Attempts download; if fails, we still can do tokenization + stopword fallback.
    ensure_nltk_resources()


def clean_text(text: str, *, remove_urls_emails: bool = True) -> str:
    """Basic cleaning for downstream processing."""

    from .utils import basic_normalize_text

    if remove_urls_emails:
        # basic_normalize_text already removes URLs/emails.
        return basic_normalize_text(text)

    # Still remove control chars and normalize whitespace.
    text = text.replace("\u00a0", " ")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_model(text: str, config: PreprocessConfig) -> str:
    """Normalization step that removes punctuation/numbers etc."""

    if config.lowercase:
        text = text.lower()

    if config.remove_urls_emails:
        from .utils import remove_urls_emails

        text = remove_urls_emails(text)

    # Remove numbers.
    if config.remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # Punctuation: replace with spaces to preserve word boundaries.
    if config.remove_punctuation:
        if config.keep_internal_hyphens:
            # Temporarily protect hyphens inside words.
            text = re.sub(r"(\w)-(\w)", r"\1__HYP__\2", text)
            text = text.translate(_PUNCT_TRANS)
            text = text.replace("__HYP__", "-")
        else:
            text = text.translate(_PUNCT_TRANS)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize using NLTK if available; fallback to regex."""

    _maybe_download_nltk()

    try:
        import nltk

        tokens = nltk.word_tokenize(text)
        return [t for t in tokens if t.strip()]
    except Exception:
        # Fallback: word-ish tokens.
        return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", text)


def remove_stopwords(tokens: Sequence[str]) -> List[str]:
    """Remove stopwords using NLTK when available."""

    _maybe_download_nltk()

    try:
        import nltk

        stop = set(nltk.corpus.stopwords.words("english"))
        return [t for t in tokens if t.lower() not in stop]
    except Exception:
        # Small built-in fallback.
        stop = {
            "the",
            "and",
            "for",
            "with",
            "to",
            "of",
            "in",
            "a",
            "an",
            "on",
            "by",
            "at",
            "from",
            "is",
            "it",
            "as",
            "are",
            "be",
            "this",
            "that",
            "or",
            "will",
            "can",
            "may",
            "have",
            "has",
            "had",
            "not",
            "but",
            "we",
            "you",
            "they",
            "i",
        }
        return [t for t in tokens if t.lower() not in stop]


def lemmatize(tokens: Sequence[str]) -> List[str]:
    """Lemmatize tokens using WordNet."""

    if not tokens:
        return []

    _maybe_download_nltk()

    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        return list(tokens)


def preprocess(text: str, config: PreprocessConfig | None = None) -> str:
    """Full preprocessing pipeline producing a single string.

    Output is suitable for feeding into TF-IDF vectorizer.
    """

    config = config or PreprocessConfig()

    text = clean_text(text, remove_urls_emails=config.remove_urls_emails)
    text = normalize_for_model(text, config)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if config.use_lemmatization:
        tokens = lemmatize(tokens)

    return " ".join(tokens)

