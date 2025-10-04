"""
Utility helpers shared between document processors.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# A global lock guards concurrent read/write operations performed via to_thread
_FILE_IO_LOCK = asyncio.Lock()


async def read_text_async(path: str | Path, encoding: str = "utf-8", errors: str = "ignore") -> str:
    """Read text file contents asynchronously using a shared lock."""
    file_path = Path(path)
    async with _FILE_IO_LOCK:
        return await asyncio.to_thread(file_path.read_text, encoding=encoding, errors=errors)


def is_image_file(file_path: str | Path) -> bool:
    """Return True when the path looks like a common image file."""
    extension = Path(file_path).suffix.lower()
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
    }
    return extension in image_extensions


class FileFormatHandler:
    """Helpers for extracting raw text content from user supplied documents."""

    TEXT_EXTENSIONS = {".txt", ".md", ".rtf", ".log"}
    DOCX_EXTENSIONS = {".docx"}
    PDF_EXTENSIONS = {".pdf"}

    @staticmethod
    async def extract_text_from_file(file_path: str | Path) -> tuple[bool, str]:
        """Extract text from supported document formats.

        Returns
        -------
        tuple[bool, str]
            ``(True, text)`` when extraction succeeds and ``(False, reason)`` otherwise.
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"

        suffix = path.suffix.lower()
        try:
            if suffix in FileFormatHandler.TEXT_EXTENSIONS:
                text = await read_text_async(path)
                return True, text
            if suffix in FileFormatHandler.DOCX_EXTENSIONS:
                text = await FileFormatHandler._extract_docx(path)
                return True, text
            if suffix in FileFormatHandler.PDF_EXTENSIONS:
                text = await FileFormatHandler._extract_pdf(path)
                return True, text

            # Fallback – treat unknown formats as UTF-8 text to avoid hard failures.
            text = await read_text_async(path, errors="ignore")
            return True, text
        except Exception as exc:  # noqa: BLE001 – propagate readable message
            logger.exception("Failed to extract text from %s", path)
            return False, str(exc)

    @staticmethod
    async def _extract_docx(path: Path) -> str:
        def load_docx(doc_path: Path) -> str:
            from docx import Document  # Imported lazily to keep startup cheap

            document = Document(doc_path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            return "\n".join(paragraphs)

        return await asyncio.to_thread(load_docx, path)

    @staticmethod
    async def _extract_pdf(path: Path) -> str:
        async def _pdf_with_pdfplumber() -> str:
            def load_pdfplumber(pdf_path: Path) -> str:
                import pdfplumber

                with pdfplumber.open(pdf_path) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)

            return await asyncio.to_thread(load_pdfplumber, path)

        async def _pdf_with_pypdf() -> str:
            def load_pypdf(pdf_path: Path) -> str:
                from PyPDF2 import PdfReader

                reader = PdfReader(str(pdf_path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)

            return await asyncio.to_thread(load_pypdf, path)

        # Try the richer pdfplumber backend first and fall back to PyPDF2 when absent
        try:
            text = await _pdf_with_pdfplumber()
            if text.strip():
                return text
        except Exception:  # noqa: BLE001 – fall back to PyPDF2
            logger.debug("pdfplumber failed for %s, falling back to PyPDF2", path, exc_info=True)

        return await _pdf_with_pypdf()

    @staticmethod
    def guess_mime_type(file_name: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_name)
        return mime_type or "application/octet-stream"


class TextProcessor:
    """Text normalisation and analysis helpers shared across processors."""

    _WHITESPACE_RE = re.compile(r"[\t\r\f]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _WORD_RE = re.compile(r"\b[\w\-']+\b", re.UNICODE)
    _SENTENCE_END_RE = re.compile(r"[.!?]+")

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise whitespace and strip zero-width characters."""
        if not text:
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u0000", "")
        normalized = TextProcessor._WHITESPACE_RE.sub(" ", normalized)
        normalized = TextProcessor._MULTI_NEWLINE_RE.sub("\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def split_into_chunks(text: str, *, max_chunk_size: int, overlap: int = 0) -> list[str]:
        """Split long text into overlapping chunks preserving paragraphs when possible."""
        text = text.strip()
        if not text:
            return []
        if max_chunk_size <= 0 or len(text) <= max_chunk_size:
            return [text]

        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush_chunk() -> None:
            nonlocal current, current_len
            if not current:
                return
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            if overlap > 0 and len(chunk_text) > overlap:
                overlap_text = chunk_text[-overlap:]
                current = [overlap_text]
                current_len = len(overlap_text)
            else:
                current = []
                current_len = 0

        for paragraph in paragraphs:
            para_len = len(paragraph)
            if para_len > max_chunk_size:
                start = 0
                while start < para_len:
                    end = min(para_len, start + max_chunk_size)
                    slice_text = paragraph[start:end]
                    current.append(slice_text)
                    current_len += len(slice_text)
                    if current_len >= max_chunk_size:
                        flush_chunk()
                    start = end
                continue

            additional = para_len if not current else para_len + 2
            if current_len + additional > max_chunk_size:
                flush_chunk()

            current.append(paragraph)
            current_len += additional

        flush_chunk()
        return [chunk for chunk in chunks if chunk.strip()]

    @staticmethod
    def extract_metadata(text: str) -> dict[str, Any]:
        """Compute simple statistics about the provided text."""
        cleaned = TextProcessor.clean_text(text)
        words = TextProcessor._WORD_RE.findall(cleaned)
        sentences = TextProcessor._SENTENCE_END_RE.split(cleaned)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        word_count = len(words)
        unique_words = len({word.lower() for word in words})
        sentence_count = len(sentences) if sentences else 0
        avg_sentence_length = (word_count / sentence_count) if sentence_count else 0.0
        lengths = [len(sentence.split()) for sentence in sentences] or [0]
        median_sentence_length = statistics.median(lengths)
        estimated_read_minutes = round(word_count / 180.0, 2) if word_count else 0.0

        return {
            "characters": len(cleaned),
            "words": word_count,
            "unique_words": unique_words,
            "sentences": sentence_count,
            "average_sentence_length": round(avg_sentence_length, 2),
            "median_sentence_length": median_sentence_length,
            "estimated_read_time_minutes": estimated_read_minutes,
        }

    @staticmethod
    def top_keywords(text: str, limit: int = 10, *, min_length: int = 4) -> list[str]:
        """Return the most frequent keywords from the text."""
        words = [word.lower() for word in TextProcessor._WORD_RE.findall(text)]
        counts: dict[str, int] = {}
        for word in words:
            if len(word) < min_length:
                continue
            counts[word] = counts.get(word, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in sorted_words[:limit]]


async def write_text_async(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Persist text asynchronously and return the resulting path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    async with _FILE_IO_LOCK:
        await asyncio.to_thread(file_path.write_text, content, encoding, "ignore")
    return file_path
