"""Streamlit UI for the TF-IDF + Cosine Similarity Resume Analyzer.

This file wires together:
- Resume parsing (PDF/DOCX/TXT/RTF/ODT best-effort)
- Preprocessing (NLTK tokenization/stopwords/lemmatization)
- TF-IDF + cosine similarity scoring
- Skill extraction (300+ skills database)
- Summary + recommendations
- Downloadable PDF report

No LLMs are used.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.analyzer import QualificationThresholds, compute_tfidf_cosine_similarity_with_thresholds
from src.preprocess import PreprocessConfig, preprocess
from src.recommender import generate_recommendation_result
from src.report_generator import ReportData, generate_pdf_report
from src.resume_parser import extract_text
from src.skills import build_skill_database, skill_match_report
from src.utils import ensure_dirs, safe_filename


# -----------------------------
# Streamlit page configuration
# -----------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        r"""
<style>
:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.14);
  --text:#E5E7EB;
  --muted:#9CA3AF;
  --primary:#7C3AED; /* violet */
  --primary2:#22C55E; /* green */
  --warn:#F59E0B;
  --danger:#EF4444;
}

html, body, [class*="css"]{background: linear-gradient(135deg, var(--bg0), var(--bg1)) !important; color: var(--text);}

.hero{
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(34,197,94,0.10));
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
  position: relative;
  overflow: hidden;
}
.hero:before{
  content:"";
  position:absolute;
  top:-140px;
  left:-140px;
  width:320px;
  height:320px;
  background: radial-gradient(circle at center, rgba(124,58,237,0.35), transparent 60%);
  animation: floatGlow 7s ease-in-out infinite;
}
.hero:after{
  content:"";
  position:absolute;
  bottom:-160px;
  right:-160px;
  width:380px;
  height:380px;
  background: radial-gradient(circle at center, rgba(34,197,94,0.28), transparent 62%);
  animation: floatGlow 9s ease-in-out infinite;
}
@keyframes floatGlow{ 0%,100%{ transform: translate(0px,0px)} 50%{ transform: translate(45px,30px)} }

.card{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card:hover{ background: var(--card2); transform: translateY(-1px); transition: all 0.15s ease; }

.kpi{
  display:flex;
  align-items:center;
  justify-content:space-between;
}
.kpi .label{color: var(--muted); font-size: 12px;}
.kpi .value{font-size: 26px; font-weight: 800;}

.progress{
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  overflow:hidden;
}
.progress > div{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--primary), rgba(34,197,94,0.9));
}

.badge{
  display:inline-flex;
  align-items:center;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}

.badge.ok{ border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); }
.badge.mid{ border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.12); }
.badge.bad{ border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.12); }

.metric-row{
  display:flex;
  gap: 12px;
}

@media (max-width: 900px){
  .metric-row{flex-direction:column;}
}

</style>
""",
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> str:
    s = (status or "").strip().lower()
    if "qualified" in s and "partially" not in s:
        cls = "ok"
        icon = "✅"
    elif "partially" in s:
        cls = "mid"
        icon = "⚠️"
    else:
        cls = "bad"
        icon = "❌"
    return f"<span class='badge {cls}'>{icon} {status}</span>"


def _circular_indicator_html(percentage: float, status: str) -> str:
    # percentage: 0..100
    pct = float(np.clip(percentage, 0.0, 100.0))
    stroke = 10
    r = 40
    c = 2 * np.pi * r
    dash = (pct / 100.0) * c

    s = (status or "").strip().lower()
    if "qualified" in s and "partially" not in s:
        color = "#22C55E"
    elif "partially" in s:
        color = "#F59E0B"
    else:
        color = "#EF4444"

    return f"""
<div class='card' style='display:flex; align-items:center; justify-content:space-between; gap:14px;'>
  <div>
    <div style='color: #9CA3AF; font-size: 12px;'>Match Indicator</div>
    <div style='font-size: 34px; font-weight: 900;'>{pct:.1f}%</div>
    <div style='margin-top:4px;'>{_status_badge(status)}</div>
  </div>
  <svg width='110' height='110' viewBox='0 0 110 110'>
    <circle cx='55' cy='55' r='{r}' stroke='rgba(255,255,255,0.14)' stroke-width='{stroke}' fill='none'/>
    <circle cx='55' cy='55' r='{r}' stroke='{color}' stroke-width='{stroke}' fill='none'
      stroke-linecap='round' stroke-dasharray='{c}' stroke-dashoffset='{c - dash}' transform='rotate(-90 55 55)'/>
  </svg>
</div>
"""


def _extract_uploaded_text(uploaded_file) -> str:
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    return extract_text(file_bytes, filename)


# -----------------------------
# Sidebar / Configuration
# -----------------------------

_inited = False
if "skill_db" not in st.session_state:
    st.session_state.skill_db = build_skill_database()

_ensure_once = ensure_dirs("uploads", "reports")

_inited = True

_invoke_css = _inject_css()

st.sidebar.header("Qualification Thresholds")
qualified_thr = st.sidebar.slider("Qualified (match % >=)", 0, 100, 80, step=1)
partial_thr = st.sidebar.slider("Partially Qualified (match % >=)", 0, 100, 65, step=1)
if partial_thr > qualified_thr:
    st.sidebar.error("Partially Qualified threshold must be <= Qualified threshold.")

st.sidebar.divider()

st.sidebar.header("Preprocessing")
use_lemmatization = st.sidebar.checkbox("Lemmatization", value=True)
remove_urls_emails = st.sidebar.checkbox("Remove URLs & emails", value=True)

st.sidebar.header("TF-IDF")
ngram_1 = st.sidebar.selectbox("TF-IDF n-gram start", options=[1], index=0)
ngram_2 = st.sidebar.selectbox("TF-IDF n-gram end", options=[1, 2, 3], index=1)
max_features = st.sidebar.number_input("Max TF-IDF features", min_value=1000, max_value=250000, value=50000, step=5000)


# -----------------------------
# Main UI
# -----------------------------

st.markdown("""
<div class='hero'>
  <div style='position:relative; z-index:1;'>
    <h1 style='margin:0; font-size: 32px;'>🧠 AI Resume Analyzer</h1>
    <p style='margin:6px 0 0 0; color: #9CA3AF;'>TF-IDF + Cosine Similarity based matching — no LLMs.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("1) Upload Resume")
    uploaded = st.file_uploader(
        "Upload your resume (PDF/DOCX/TXT/RTF; ODT supported best-effort)",
        type=["pdf", "docx", "txt", "rtf", "odt"],
    )

with right:
    st.subheader("2) Job Description")
    jd = st.text_area(
        "Paste the job description",
        height=250,
        placeholder="Paste the role requirements and responsibilities here...",
    )

st.write("")

c1, c2, c3 = st.columns([1, 1, 2])

thresholds = QualificationThresholds(qualified=float(qualified_thr), partially_qualified=float(partial_thr))

run_disabled = not uploaded or not jd or partial_thr > qualified_thr

analyze_btn = c2.button("Analyze Match", type="primary", disabled=run_disabled)

if analyze_btn:
    with st.spinner("Extracting resume text..."):
        resume_raw = _extract_uploaded_text(uploaded)

    if not resume_raw or len(resume_raw.strip()) < 20:
        st.error("Could not extract enough text from the uploaded resume. Try a different file or ensure it contains selectable text.")
        st.stop()

    with st.spinner("Preprocessing text (NLTK) and computing TF-IDF similarity..."):
        config = PreprocessConfig(
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True,
            remove_urls_emails=remove_urls_emails,
            keep_internal_hyphens=True,
            use_lemmatization=use_lemmatization,
        )

        resume_clean = preprocess(resume_raw, config=config)
        jd_clean = preprocess(jd, config=config)

        ngram_range = (int(ngram_1), int(ngram_2))

        similarity_res = compute_tfidf_cosine_similarity_with_thresholds(
            resume_clean,
            jd_clean,
            thresholds=thresholds,
            ngram_range=ngram_range,
            max_features=int(max_features),
        )

        # Skill matching uses raw (cleaned lightly) to keep multiword phrases.
        # We still leverage preprocessing-like normalization implicitly via preprocess.
        skill_report = skill_match_report(
            resume_clean,
            jd_clean,
            skill_db=st.session_state.skill_db,
        )

        recommender = generate_recommendation_result(
            resume_text=resume_raw,
            matching_skills=skill_report.matching_skills,
            missing_skills=skill_report.missing_skills,
            categories=skill_report.by_category,
        )

    # Persist report data
    generated_at_iso = datetime.now(timezone.utc).isoformat()
    stem = safe_filename(f"resume_analysis_{uploaded.name}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{stem}_{generated_at_iso.replace(':','-')}.pdf"

    report_data = ReportData(
        match_percentage=similarity_res.match_percentage,
        status=similarity_res.status,
        matching_skills=skill_report.matching_skills,
        missing_skills=skill_report.missing_skills,
        resume_summary=recommender.resume_summary,
        recommendations=recommender.recommendations,
        generated_at_iso=generated_at_iso,
        filename_stem=stem,
    )

    with st.spinner("Generating PDF report..."):
        generate_pdf_report(report_data, str(out_path))

    st.write("")

    st.markdown(_circular_indicator_html(similarity_res.match_percentage, similarity_res.status), unsafe_allow_html=True)

    # Two-column dashboard
    colA, colB = st.columns([1.05, 1.0], gap="large")

    with colA:
        st.subheader("3) Matching Skills")
        if skill_report.matching_skills:
            st.markdown(
                " ".join([f"<span class='badge ok' style='margin:4px;'>{s}</span>" for s in skill_report.matching_skills[:40]]),
                unsafe_allow_html=True,
            )
        else:
            st.info("No matching skills found from the predefined skill database.")

        st.subheader("4) Missing Skills")
        if skill_report.missing_skills:
            st.markdown(
                " ".join([f"<span class='badge bad' style='margin:4px;'>{s}</span>" for s in skill_report.missing_skills[:40]]),
                unsafe_allow_html=True,
            )
        else:
            st.success("No missing skills identified. Great alignment with the job description keywords!")

    with colB:
        st.subheader("5) Resume Summary")
        st.write(recommender.resume_summary)

        st.subheader("6) Improvement Suggestions")
        for i, rec in enumerate(recommender.recommendations[:12], start=1):
            st.write(f"{i}. {rec}")

    st.divider()

    st.subheader("7) Download Analysis Report")
    try:
        with open(out_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=out_path.name,
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Could not generate/download the PDF report: {e}")


# Footer
st.markdown(
    """
<hr/>
<div style='color:#9CA3AF; font-size: 12px; text-align:center; padding-bottom: 10px;'>
  Built for capstone/placements using TF-IDF + Cosine Similarity • Secure, ATS-friendly output
</div>
""",
    unsafe_allow_html=True,
)

