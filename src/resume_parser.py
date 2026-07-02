"""Resume parsing utilities.

Supports:
- PDF: pdfplumber first, then PyPDF2 fallback
- DOCX: python-docx
- TXT: plain text decode
- RTF: best-effort stripping of RTF control words
- ODT: optional, best-effort via zip+content.xml (no extra deps)

All functions are fully implemented.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Optional

import pdfplumber
from PyPDF2 import PdfReader
import docx

from .utils import basic_normalize_text


def _extract_text_from_pdfplumber(file_bytes: bytes) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_text_from_pypdf2(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber with fallback."""

    try:
        text = _extract_text_from_pdfplumber(file_bytes)
        if text.strip():
            return text
    except Exception:
        # Fallback.
        pass

    return _extract_text_from_pypdf2(file_bytes)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX."""

    doc = docx.Document(io.BytesIO(file_bytes))
    parts: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT via best-effort decoding."""

    # Try UTF-8 then fallback to latin-1.
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(enc, errors="ignore")
        except Exception:
            continue
    return file_bytes.decode("latin-1", errors="ignore")


_RTF_CONTROL_RE = re.compile(r"\\[a-zA-Z]+-?\d*\s?|[{}]")

def extract_text_from_rtf(file_bytes: bytes) -> str:
    """Best-effort extraction from RTF.

    RTF is a tagged format. For capstone/placement use-cases, stripping
    control words + groups provides workable text.
    """

    text = file_bytes.decode("utf-8", errors="ignore")

    # Remove Unicode escape sequences like \'hh
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)

    # Remove RTF control words and braces.
    text = _RTF_CONTROL_RE.sub(" ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_text_from_odt(file_bytes: bytes) -> str:
    """Best-effort extraction from ODT (OpenDocument Text).

    ODT is a zip file containing content.xml.
    We strip XML tags using regex for simplicity.
    """

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            # content.xml is where the main text lives.
            name = None
            for candidate in ("content.xml", "content/content.xml"):
                if candidate in z.namelist():
                    name = candidate
                    break
            if not name:
                return ""

            xml = z.read(name).decode("utf-8", errors="ignore")

        # Strip tags.
        xml = re.sub(r"<[^>]+>", " ", xml)
        xml = re.sub(r"\s+", " ", xml).strip()
        return xml
    except Exception:
        return ""


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Detect file type and extract text."""

    name = (filename or "").lower()

    # Use extension.
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx") or name.endswith(".doc"):
        # .doc is not supported natively; python-docx may fail.
        text = extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        text = extract_text_from_txt(file_bytes)
    elif name.endswith(".rtf"):
        text = extract_text_from_rtf(file_bytes)
    elif name.endswith(".odt"):
        text = extract_text_from_odt(file_bytes)
    else:
        # Fallback: try txt.
        text = extract_text_from_txt(file_bytes)

    return basic_normalize_text(text)

