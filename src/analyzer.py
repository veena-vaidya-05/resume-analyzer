"""TF-IDF based resume/job matching analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class QualificationThresholds:
    qualified: float = 80.0
    partially_qualified: float = 65.0


@dataclass(frozen=True)
class SimilarityResult:
    similarity_score: float  # cosine similarity [0..1] (typically)
    match_percentage: float  # 0..100
    status: str


def compute_tfidf_cosine_similarity(
    resume_text: str,
    job_description_text: str,
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50000,
) -> SimilarityResult:
    """Compute cosine similarity between resume and job description using TF-IDF."""

    # Guard empty inputs.
    if not resume_text or not resume_text.strip():
        return SimilarityResult(similarity_score=0.0, match_percentage=0.0, status="Not Qualified")
    if not job_description_text or not job_description_text.strip():
        return SimilarityResult(similarity_score=0.0, match_percentage=0.0, status="Not Qualified")

    vectorizer = TfidfVectorizer(
        stop_words=None,  # preprocessing handles stopwords
        lowercase=False,
        ngram_range=ngram_range,
        max_features=max_features,
    )

    tfidf = vectorizer.fit_transform([resume_text, job_description_text])
    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    # Cosine sim theoretically can be >1 due to numerical issues? Clamp.
    sim = float(np.clip(sim, 0.0, 1.0))
    match_percentage = sim * 100.0

    thresholds = QualificationThresholds()
    if match_percentage >= thresholds.qualified:
        status = "Qualified"
    elif match_percentage >= thresholds.partially_qualified:
        status = "Partially Qualified"
    else:
        status = "Not Qualified"

    return SimilarityResult(
        similarity_score=sim,
        match_percentage=match_percentage,
        status=status,
    )


def status_from_percentage(match_percentage: float, thresholds: QualificationThresholds) -> str:
    """Map match percentage to qualification status."""

    if match_percentage >= thresholds.qualified:
        return "Qualified"
    if match_percentage >= thresholds.partially_qualified:
        return "Partially Qualified"
    return "Not Qualified"


def compute_tfidf_cosine_similarity_with_thresholds(
    resume_text: str,
    job_description_text: str,
    *,
    thresholds: QualificationThresholds,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50000,
) -> SimilarityResult:
    """Same as compute_tfidf_cosine_similarity but configurable thresholds."""

    if not resume_text or not resume_text.strip():
        return SimilarityResult(similarity_score=0.0, match_percentage=0.0, status=status_from_percentage(0.0, thresholds))
    if not job_description_text or not job_description_text.strip():
        return SimilarityResult(similarity_score=0.0, match_percentage=0.0, status=status_from_percentage(0.0, thresholds))

    vectorizer = TfidfVectorizer(
        stop_words=None,
        lowercase=False,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    tfidf = vectorizer.fit_transform([resume_text, job_description_text])
    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    sim = float(np.clip(sim, 0.0, 1.0))

    match_percentage = sim * 100.0
    status = status_from_percentage(match_percentage, thresholds)

    return SimilarityResult(similarity_score=sim, match_percentage=match_percentage, status=status)

