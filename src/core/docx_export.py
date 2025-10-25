from __future__ import annotations

import re
import tempfile
import uuid
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence


def _require_docx():
    try:
        from docx import Document  # type: ignore
        from docx.shared import Pt, Cm  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx не установлен, DOCX-экспорт недоступен") from exc
    return Document, Pt, Cm


def _html_to_plain(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", "", clean)
    return unescape(clean).strip()


def _temp_path(stem: str, suffix: str = ".docx") -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "export"
    return Path(tempfile.gettempdir()) / f"{safe}_{uuid.uuid4().hex}{suffix}"


def _setup_page_and_styles(doc) -> None:
    from docx.shared import Pt, Cm  # type: ignore

    for section in doc.sections:
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)

    try:
        normal = doc.styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(12)
    except Exception:
        pass

    for name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 13)):
        try:
            style = doc.styles[name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(size)
            style.font.bold = True
        except Exception:
            continue


def _add_hyperlink(paragraph, text: str, url: str) -> None:
    if not url:
        paragraph.add_run(text)
        return
    try:
        from docx.opc.constants import RELATIONSHIP_TYPE as RT  # type: ignore
        from docx.oxml import OxmlElement  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
    except Exception:  # pragma: no cover
        paragraph.add_run(f"{text} ({url})")
        return

    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    r_style = OxmlElement("w:rStyle")
    r_style.set(qn("w:val"), "Hyperlink")
    r_pr.append(r_style)
    run.append(r_pr)

    text_elem = OxmlElement("w:t")
    text_elem.text = text or url
    run.append(text_elem)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


class _HTML2Docx(HTMLParser):
    def __init__(self, doc):
        super().__init__(convert_charrefs=True)
        self.doc = doc
        self.paragraph = None
        self.bold = False
        self.italic = False
        self.underline = False
        self.code = False
        self.pre = False
        self.list_stack: list[str] = []
        self.href: str | None = None

    def _ensure_paragraph(self, style: str | None = None):
        if self.paragraph is None:
            self.paragraph = self.doc.add_paragraph(style=style) if style else self.doc.add_paragraph()
        return self.paragraph

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"p", "div"}:
            self.paragraph = self.doc.add_paragraph()
        elif tag in {"h1", "h2", "h3"}:
            style = {"h1": "Heading 1", "h2": "Heading 2", "h3": "Heading 3"}[tag]
            self.paragraph = self.doc.add_paragraph(style=style)
        elif tag == "br":
            self._ensure_paragraph().add_run("\n")
        elif tag in {"b", "strong"}:
            self.bold = True
        elif tag in {"i", "em"}:
            self.italic = True
        elif tag == "u":
            self.underline = True
        elif tag == "code":
            self.code = True
        elif tag == "pre":
            self.pre = True
            self.paragraph = self.doc.add_paragraph()
        elif tag == "a":
            self.href = None
            for attr_name, attr_value in attrs:
                if attr_name.lower() == "href":
                    self.href = attr_value
                    break
        elif tag == "ul":
            self.list_stack.append("ul")
        elif tag == "ol":
            self.list_stack.append("ol")
        elif tag == "li":
            style = "List Bullet" if (self.list_stack and self.list_stack[-1] == "ul") else "List Number"
            self.paragraph = self.doc.add_paragraph(style=style)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"p", "div", "li", "pre", "h1", "h2", "h3"}:
            self.paragraph = None
        elif tag in {"ul", "ol"}:
            if self.list_stack and self.list_stack[-1] == tag:
                self.list_stack.pop()
        elif tag in {"b", "strong"}:
            self.bold = False
        elif tag in {"i", "em"}:
            self.italic = False
        elif tag == "u":
            self.underline = False
        elif tag == "code":
            self.code = False
        elif tag == "pre":
            self.pre = False
        elif tag == "a":
            self.href = None

    def handle_data(self, data):
        if not data:
            return
        paragraph = self._ensure_paragraph()
        if self.href:
            _add_hyperlink(paragraph, data.strip(), self.href)
            return
        run = paragraph.add_run(data)
        run.bold = self.bold
        run.italic = self.italic
        run.underline = self.underline
        if self.code or self.pre:
            try:
                run.font.name = "Courier New"
            except Exception:
                pass


def _render_html(doc, html: str):
    if not html or not html.strip():
        return
    parser = _HTML2Docx(doc)
    parser.feed(html)
    parser.close()


_TOPIC_RE = re.compile(r"/document/(\d+)")


def _topic_from_url(url: str) -> int | None:
    if not url:
        return None
    match = _TOPIC_RE.search(url)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def build_practice_docx(
    *,
    summary_html: str,
    fragments: Sequence[Any],
    structured: Mapping[str, Any] | None = None,
    full_texts: Mapping[int, str] | None = None,
    file_stub: str | None = None,
) -> Path:
    Document, Pt, Cm = _require_docx()
    full_texts = dict(full_texts or {})

    doc = Document()
    _setup_page_and_styles(doc)
    doc.add_paragraph("Подборка судебной практики", style="Heading 1")
    doc.add_paragraph(datetime.now().strftime("%d.%m.%Y %H:%M"))
    doc.add_paragraph()

    doc.add_paragraph("Краткий вывод", style="Heading 2")
    summary = None
    if structured and isinstance(structured.get("summary"), str):
        summary = structured["summary"]
    if isinstance(summary, str) and summary.strip():
        _render_html(doc, summary)
    elif summary_html and summary_html.strip():
        _render_html(doc, summary_html)
    else:
        doc.add_paragraph("нет данных")
    doc.add_paragraph()

    cases: list[Mapping[str, Any]] = []
    if structured and isinstance(structured.get("cases"), list):
        cases = [c for c in structured["cases"] if isinstance(c, Mapping)]
    elif fragments:
        for fragment in fragments:
            match = getattr(fragment, "match", None)
            metadata = getattr(match, "metadata", {}) if match else {}
            if not isinstance(metadata, Mapping):
                continue
            cases.append(
                {
                    "title": metadata.get("title") or metadata.get("name") or getattr(fragment, "header", ""),
                    "case_number": metadata.get("case_number") or metadata.get("case"),
                    "url": metadata.get("url") or metadata.get("link"),
                    "facts": metadata.get("summary") or getattr(fragment, "excerpt", ""),
                    "holding": metadata.get("decision_summary") or "",
                    "norms": metadata.get("norms_summary") or "",
                    "topic": metadata.get("topic"),
                    "entry": metadata.get("entry"),
                }
            )

    if not cases:
        doc.add_paragraph().add_run("Нет данных по судебной практике").italic = True
    else:
        for idx, case in enumerate(cases, start=1):
            title = str(case.get("title") or "").strip()
            case_number = str(case.get("case_number") or "").strip()
            url = str(case.get("link") or case.get("url") or "").strip()
            facts = str(case.get("facts") or "").strip()
            holding = str(case.get("holding") or "").strip()
            norms_raw = case.get("norms")
            if isinstance(norms_raw, (list, tuple)):
                norms_text = ", ".join(str(item).strip() for item in norms_raw if str(item).strip())
            else:
                norms_text = str(norms_raw or "").strip()
            topic = case.get("topic")
            try:
                if topic is None:
                    topic_candidate = _topic_from_url(url)
                    topic = int(topic_candidate) if topic_candidate is not None else None
                else:
                    topic = int(topic)
            except Exception:
                topic = None

            doc.add_paragraph()
            doc.add_paragraph(f"{idx}) {title}", style="Heading 3")
            if case_number:
                doc.add_paragraph(f"Номер дела: {case_number}")
            if url:
                paragraph = doc.add_paragraph("Ссылка: ")
                _add_hyperlink(paragraph, "перейти", url)
            if facts:
                doc.add_paragraph("Выдержка", style="Heading 3")
                _render_html(doc, facts)
            if holding:
                doc.add_paragraph("Вывод суда", style="Heading 3")
                _render_html(doc, holding)
            if norms_text:
                doc.add_paragraph("Нормы", style="Heading 3")
                for line in [s.strip() for s in norms_text.splitlines() if s.strip()]:
                    doc.add_paragraph(line, style="List Bullet")

            doc.add_paragraph("Полный текст", style="Heading 3")
            body = str(case.get("fulltext_html") or "").strip()
            if not body and isinstance(topic, int):
                body = full_texts.get(topic)
            if body:
                _render_html(doc, body)
            else:
                doc.add_paragraph("Полный текст недоступен через API, используйте ссылку выше.").italic = True

    output = _temp_path(file_stub or "practice_fulltext")
    doc.save(str(output))
    return output
