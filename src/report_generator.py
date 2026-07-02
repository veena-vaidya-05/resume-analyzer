"""PDF report generation for the Resume Analyzer.

Uses reportlab to create a downloadable PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_LEFT

from .utils import safe_filename


@dataclass(frozen=True)
class ReportData:
    match_percentage: float
    status: str
    matching_skills: Sequence[str]
    missing_skills: Sequence[str]
    resume_summary: str
    recommendations: Sequence[str]
    generated_at_iso: str
    filename_stem: str


def _bullets(items: Sequence[str]) -> str:
    return "\n".join([f"• {i}" for i in items]) if items else "(none)"


def generate_pdf_report(data: ReportData, output_path: str) -> None:
    """Generate a PDF report."""

    doc = SimpleDocTemplate(output_path, pagesize=A4)

    style_title = ParagraphStyle(
        "title",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    style_h = ParagraphStyle(
        "h",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
        spaceBefore=12,
    )
    style_body = ParagraphStyle(
        "body",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )

    story: List = []

    story.append(Paragraph("AI Resume Analyzer Report", style_title))
    story.append(Paragraph(f"Generated At: {data.generated_at_iso}", style_body))

    # Metric table
    table = Table(
        [
            ["Match Percentage", f"{data.match_percentage:.2f}%"],
            ["Qualification Status", data.status],
        ],
        colWidths=[2.0 * inch, 2.8 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C7D2FE")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(Spacer(1, 12))
    story.append(table)

    story.append(Paragraph("Matching Skills", style_h))
    story.append(Paragraph(_bullets(data.matching_skills).replace("\n", "<br/>"), style_body))

    story.append(Paragraph("Missing Skills", style_h))
    story.append(Paragraph(_bullets(data.missing_skills).replace("\n", "<br/>"), style_body))

    story.append(Paragraph("Resume Summary", style_h))
    story.append(Paragraph(data.resume_summary.replace("\n", "<br/>"), style_body))

    story.append(Paragraph("Recommendations", style_h))
    story.append(Paragraph(_bullets(data.recommendations).replace("\n", "<br/>"), style_body))

    doc.build(story)

