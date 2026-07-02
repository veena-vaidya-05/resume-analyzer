"""Summary generation and improvement recommendations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RecommendationResult:
    resume_summary: str
    recommendations: List[str]
    strengths: List[str]


_ACTION_VERBS = [
    "Developed",
    "Designed",
    "Built",
    "Implemented",
    "Led",
    "Optimized",
    "Improved",
    "Delivered",
    "Architected",
    "Automated",
    "Integrated",
    "Validated",
]


def _extract_top_keywords(text: str, k: int = 12) -> List[str]:
    # Very lightweight heuristic for strengths; TF-IDF similarity handles scoring.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]*", (text or "").lower())
    freq: Dict[str, int] = {}
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "you",
        "your",
        "have",
        "has",
        "had",
        "will",
        "can",
        "may",
        "use",
        "used",
        "using",
        "including",
        "into",
        "within",
        "about",
        "such",
        "our",
        "i",
        "we",
        "they",
        "them",
        "their",
        "at",
        "as",
        "by",
        "on",
        "be",
        "or",
        "an",
        "a",
        "to",
        "of",
        "in",
        "is",
        "it",
        "not",
        "but",
        "may",
        "also",
        "more",
        "most",
    }

    for t in tokens:
        if len(t) < 3:
            continue
        if t in stop:
            continue
        freq[t] = freq.get(t, 0) + 1

    # Sort by freq then alpha for stability.
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ordered[:k]]


def generate_resume_summary(resume_text: str, *, matching_skills: Sequence[str], categories: Optional[Dict[str, List[str]]] = None) -> str:
    """Create a structured summary using heuristics."""

    top = _extract_top_keywords(resume_text, k=10)

    # Attempt to infer experience years from common patterns.
    years = None
    m = re.search(r"(\d+\.?\d*)\s*(years|yrs|year)", resume_text, flags=re.IGNORECASE)
    if m:
        try:
            years = float(m.group(1))
        except Exception:
            years = None

    role_hint = None
    for pattern in [r"software engineer", r"data scientist", r"machine learning", r"ai engineer", r"backend developer", r"full stack"]:
        if re.search(pattern, resume_text, flags=re.IGNORECASE):
            role_hint = pattern
            break

    techs = sorted(set([s for s in matching_skills]))
    techs_short = ", ".join(techs[:8]) if techs else ", ".join(top[:6])

    # Categories strengths.
    cat_bits: List[str] = []
    if categories:
        for cat, items in categories.items():
            if items:
                cat_bits.append(f"{cat}: {', '.join(items[:4])}")
    cat_str = "\n".join(cat_bits[:4])

    header = "AI/ML / Engineering Candidate"
    if role_hint:
        header = role_hint.title()

    years_str = f"~{years:g}+ years" if years is not None else ""


    summary_lines = [
        header,
        (f"Experience: {years_str}" if years is not None else "Experience: Project & coursework driven"),
        f"Core Technologies: {techs_short}",
        "Highlights:",
        f"- {top[0] if top else 'TF-IDF/Cosine Similarity'} focused execution",
        "- Strong emphasis on measurable outcomes and applied problem solving",
    ]

    if cat_str:
        summary_lines.append("\nSkill Clusters:")
        summary_lines.append(cat_str)

    # Ensure no special placeholder.
    return "\n".join(summary_lines).replace("\u0000", "")


def generate_recommendations(missing_skills: Sequence[str]) -> List[str]:
    """Create concrete suggestions for missing skills."""

    recommendations: List[str] = []

    # Prioritize common themes.
    missing_lower = [s.lower() for s in missing_skills]

    # Generic skill-to-action mapping.
    def add_if(keyword: str, rec: str) -> None:
        if any(keyword in s for s in missing_lower):
            if rec not in recommendations:
                recommendations.append(rec)

    add_if("docker", "Add Docker/containers experience: create a Dockerfile + CI pipeline for one of your projects.")
    add_if("kubernetes", "Add Kubernetes experience: deploy a small service with basic manifests (Deployment/Service).")
    add_if("aws", "Learn AWS essentials: design a small architecture using S3 + EC2/Lambda and document trade-offs.")
    add_if("azure", "Learn Azure basics: deploy a small service and reference it in your resume with metrics.")
    add_if("gcp", "Learn GCP fundamentals: build a simple pipeline on Cloud Run/BigQuery and mention performance improvements.")
    add_if("fastapi", "If your domain is Python APIs, mention FastAPI: implement one endpoint-based project with validation.")
    add_if("django", "Add framework keywords: implement a small Django app with auth + a REST endpoint.")
    add_if("sql", "Include SQL examples: add 2–3 query snippets (joins, aggregations, indexing concepts).")
    add_if("mongodb", "Add NoSQL experience: include one project that demonstrates data modeling and query patterns.")
    add_if("react", "If frontend is missing, add a React project and highlight component state + API integration.")
    add_if("node", "Add backend fundamentals: create a Node/Express service and mention logging + testing.")
    add_if("security", "Strengthen security: document secure coding practices and threat modeling for one feature.")
    add_if("owasp", "Mention OWASP practices: add input validation, authz/authentication, and API security notes.")

    # Skill-specific recommendations: cap to avoid noise.
    for s in missing_skills[:18]:
        # Make it readable.
        clean = s
        rec = f"Add/expand on: {clean}. Demonstrate it with a concrete project bullet (what you built + the result)."
        if rec not in recommendations:
            recommendations.append(rec)

    # Always include resume-writing improvements.
    base = [
        "Mirror job-description keywords naturally (same terms, not just synonyms) while keeping bullets ATS-friendly.",
        "Quantify impact: include metrics (latency, accuracy, throughput, cost) for each major project.",
        "Use stronger action verbs and consistent formatting across sections (Experience, Projects, Skills).",
    ]
    for b in base:
        if b not in recommendations:
            recommendations.append(b)

    # Ensure non-empty.
    return recommendations[:25] if recommendations else [
        "Improve resume keywords by matching the job description language and adding measurable outcomes."
    ]


def generate_recommendation_result(
    *,
    resume_text: str,
    matching_skills: Sequence[str],
    missing_skills: Sequence[str],
    categories: Optional[Dict[str, List[str]]] = None,
) -> RecommendationResult:
    """Produce resume summary + recommendation list."""

    strengths = list(matching_skills)[:10]
    summary = generate_resume_summary(resume_text, matching_skills=matching_skills, categories=categories)
    recs = generate_recommendations(missing_skills)
    return RecommendationResult(resume_summary=summary, recommendations=recs, strengths=strengths)

