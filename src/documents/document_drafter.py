"""Utilities for interactive legal document drafting."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class DocumentDraftingError(RuntimeError):
    """Raised when the LLM response cannot be processed."""


@dataclass(slots=True)
class DraftPlan:
    title: str
    questions: list[dict[str, str]]
    notes: list[str]


@dataclass(slots=True)
class DraftResult:
    status: str
    title: str
    markdown: str
    validated: list[str]
    issues: list[str]
    follow_up_questions: list[str]


DOCUMENT_PLANNER_SYSTEM_PROMPT = """
Ты — юридический ассистент, который помогает юристу подготовить точный правовой документ под конкретное дело.
Работай строго по этапам:

1. Если юрист уже назвал тип документа — найди подходящий образец и сформулируй точное название документа. Если тип не очевиден, задай наводящие вопросы, чтобы определить, какой именно документ нужен.
2. Подготовь перечень вопросов, которые необходимо задать юристу для заполнения документа. При необходимости группируй их по этапам и обязательно поясняй цель каждого вопроса.
3. Сформируй заметки контекста (context_notes), которые пригодятся при генерации документа и помогут ИИ придерживаться юридической точности.

Всегда возвращай валидный JSON.
""".strip()

DOCUMENT_PLANNER_USER_TEMPLATE = """
Запрос юриста:
{request}

Сформируй структуру ответа:
- Поле document_title — точное наименование документа.
- Поле need_more_info = true, если требуются ответы юриста; false, если данных достаточно для генерации.
- Массив questions: элементы {{"id": "...", "text": "...", "purpose": "..."}} с пояснением цели вопроса.
- Массив context_notes: короткие заметки, которые помогут при подготовке документа.

Ответ верни строго в JSON.
""".strip()

DOCUMENT_GENERATOR_SYSTEM_PROMPT = """
Ты — юридический ассистент-документалист. На основе запроса и ответов юриста подготовь юридически точный документ.
Следуй инструкции:

1. Проверь, достаточно ли данных. Если чего-то не хватает, верни status = "need_more_info" и сформируй follow_up_questions с уточнениями.
2. Если данных достаточно, составь документ в Markdown, опираясь на найденный образец и профессиональные стандарты. Соблюдай структуру, стиль и формулировки, характерные для соответствующего вида документа.
3. Перед выдачей результата выполни самопроверку: перечисли учтённые ключевые данные и выявленные риски/пробелы.

Формат ответа — валидный JSON вида:
{
  "status": "ok" | "need_more_info" | "abort",
  "document_title": "...",
  "document_markdown": "...",
  "self_check": {
    "validated": ["..."],
    "issues": ["..."]
  },
  "follow_up_questions": ["..."]
}
Если status = "ok", поле document_markdown должно содержать полный текст, готовый к конвертации в Word.
""".strip()

DOCUMENT_GENERATOR_USER_TEMPLATE = """
Запрос юриста и вводные данные:
{request}

Предполагаемое название документа: {title}

Ответы юриста:
{answers}

Собери итог по схеме JSON:
{{
  "status": "ok" | "need_more_info" | "abort",
  "document_title": "...",
  "document_markdown": "...",
  "self_check": {{
    "validated": ["..."],
    "issues": ["..."]
  }},
  "follow_up_questions": ["..."]
}}
""".strip()

_JSON_START_RE = re.compile(r"[{\[]")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_JSON_DECODER = json.JSONDecoder()


def _strip_code_fences(payload: str) -> str:
    stripped = payload.strip()
    lowered = stripped.lower()
    for prefix in ("```json", "```", "~~~json", "~~~"):
        if lowered.startswith(prefix):
            stripped = stripped[len(prefix) :]
            stripped = stripped.lstrip("\r\n")
            break

    for suffix in ("```", "~~~"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            stripped = stripped.rstrip()
            break

    return stripped.strip()


def _extract_json(text: Any) -> Any:
    """Extract the first JSON structure found in text."""

    if isinstance(text, (dict, list)):
        return text
    if text is None:
        raise DocumentDraftingError("Не удалось найти JSON в ответе модели")
    if not isinstance(text, str):
        raise DocumentDraftingError("Не удалось найти JSON в ответе модели")

    cleaned_text = _strip_code_fences(text)

    def _try_parse(candidate: str) -> Any:
        candidate = candidate.strip()
        if not candidate:
            raise json.JSONDecodeError("Empty JSON candidate", candidate, 0)
        normalized = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        obj, _ = _JSON_DECODER.raw_decode(normalized)
        return obj

    try:
        return _try_parse(cleaned_text)
    except (json.JSONDecodeError, ValueError) as err:
        logger.debug("Primary JSON parse failed, scanning for nested JSON: %s", err)
        for match in _JSON_START_RE.finditer(cleaned_text):
            start_idx = match.start()
            try:
                return _try_parse(cleaned_text[start_idx:])
            except (json.JSONDecodeError, ValueError):
                continue

    raise DocumentDraftingError("Не удалось найти JSON в ответе модели")


def _format_answers(answers: Iterable[dict[str, str]]) -> str:
    lines = []
    for idx, item in enumerate(answers, 1):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        lines.append(f"{idx}. {question}\n   Ответ: {answer}")
    return "\n".join(lines) if lines else "(Ответы пока не получены)"



async def plan_document(openai_service, request_text: str) -> DraftPlan:
    if not request_text.strip():
        raise DocumentDraftingError("Пустой запрос")

    user_prompt = DOCUMENT_PLANNER_USER_TEMPLATE.format(request=request_text.strip())
    response = await openai_service.ask_legal(
        system_prompt=DOCUMENT_PLANNER_SYSTEM_PROMPT,
        user_text=user_prompt,
    )
    if not response.get("ok"):
        raise DocumentDraftingError(response.get("error") or "Не удалось получить ответ от модели")

    raw_text = response.get("text", "")
    if not raw_text:
        raise DocumentDraftingError("Пустой ответ модели")

    data = _extract_json(raw_text)
    title = str(data.get("document_title") or "Документ").strip()
    questions = [
        {
            "id": str(q.get("id") or f"q{i+1}"),
            "text": str(q.get("text") or "").strip(),
            "purpose": str(q.get("purpose") or "").strip(),
        }
        for i, q in enumerate(data.get("questions") or [])
        if str(q.get("text") or "").strip()
    ]
    notes = [str(note).strip() for note in data.get("context_notes") or [] if str(note).strip()]

    need_more_raw = data.get("need_more_info")
    if isinstance(need_more_raw, bool):
        need_more = need_more_raw
    elif isinstance(need_more_raw, str):
        lowered = need_more_raw.strip().lower()
        need_more = lowered in {"true", "yes", "1", "y"}
    elif need_more_raw is None:
        need_more = False
    else:
        need_more = bool(need_more_raw)
    if not need_more:
        questions = []

    return DraftPlan(title=title or "Документ", questions=questions, notes=notes)


async def generate_document(
    openai_service,
    request_text: str,
    title: str,
    answers: list[dict[str, str]],
) -> DraftResult:
    answers_formatted = _format_answers(answers)
    user_prompt = DOCUMENT_GENERATOR_USER_TEMPLATE.format(
        request=request_text.strip(),
        title=title,
        answers=answers_formatted or "(Ответы пока не получены)",
    )
    response = await openai_service.ask_legal(
        system_prompt=DOCUMENT_GENERATOR_SYSTEM_PROMPT,
        user_text=user_prompt,
    )
    if not response.get("ok"):
        raise DocumentDraftingError(response.get("error") or "Не удалось получить ответ от модели")

    raw_text = response.get("text", "")
    if not raw_text:
        raise DocumentDraftingError("Пустой ответ модели")

    data = _extract_json(raw_text)
    status = str(data.get("status") or "ok").lower()
    doc_title = str(data.get("document_title") or title).strip()
    markdown = str(data.get("document_markdown") or "")
    self_check = data.get("self_check") or {}
    validated = [str(item).strip() for item in self_check.get("validated") or [] if str(item).strip()]
    issues = [str(item).strip() for item in self_check.get("issues") or [] if str(item).strip()]
    follow_up = [str(item).strip() for item in data.get("follow_up_questions") or [] if str(item).strip()]

    return DraftResult(
        status=status,
        title=doc_title or title,
        markdown=markdown,
        validated=validated,
        issues=issues,
        follow_up_questions=follow_up,
    )


def build_docx_from_markdown(markdown: str, output_path: str) -> None:
    try:
        from docx import Document  # type: ignore
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
        from docx.shared import Cm, Pt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise DocumentDraftingError("Не удалось подготовить DOCX без пакета python-docx") from exc

    if not markdown.strip():
        raise DocumentDraftingError("Пустой текст документа")

    document = Document()
    section = document.sections[0]
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, attr, Cm(2.5))

    def _ensure_font(style_name: str, *, size: int, bold: bool = False, italic: bool = False, align: int | None = None) -> None:
        try:
            style = document.styles[style_name]
        except KeyError:
            return
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(size)
        font.bold = bold
        font.italic = italic
        if style.element.rPr is not None:
            r_fonts = style.element.rPr.rFonts
            if r_fonts is not None:
                r_fonts.set(qn("w:eastAsia"), "Times New Roman")
        paragraph_format = getattr(style, "paragraph_format", None)
        if paragraph_format:
            paragraph_format.space_after = Pt(6)
            if align is not None:
                paragraph_format.alignment = align

    normal_style = document.styles["Normal"]
    normal_font = normal_style.font
    normal_font.name = "Times New Roman"
    normal_font.size = Pt(12)
    if normal_style.element.rPr is not None:
        r_fonts = normal_style.element.rPr.rFonts
        if r_fonts is not None:
            r_fonts.set(qn("w:eastAsia"), "Times New Roman")
    normal_paragraph = normal_style.paragraph_format
    normal_paragraph.space_after = Pt(6)
    normal_paragraph.first_line_indent = Cm(1.25)
    normal_paragraph.line_spacing = 1.5

    _ensure_font("Title", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    _ensure_font("Heading 1", size=14, bold=True)
    _ensure_font("Heading 2", size=13, bold=True)
    _ensure_font("Heading 3", size=12, bold=True)

    def _add_runs(paragraph, text: str) -> None:
        pattern = re.compile(r"(\*\*.+?\*\*|__.+?__|\*.+?\*|_.+?_|`.+?`)", re.DOTALL)
        for chunk in pattern.split(text):
            if not chunk:
                continue
            run = paragraph.add_run()
            if chunk.startswith(("**", "__")) and chunk.endswith(chunk[:2]):
                run.text = chunk[2:-2]
                run.bold = True
            elif chunk.startswith(("*", "_")) and chunk.endswith(chunk[0]):
                run.text = chunk[1:-1]
                run.italic = True
            elif chunk.startswith("`") and chunk.endswith("`"):
                run.text = chunk[1:-1]
                run.font.name = "Consolas"
                run.font.size = Pt(11)
            else:
                run.text = chunk

    list_mode: str | None = None

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if not line:
            document.add_paragraph("")
            list_mode = None
            continue

        if line.startswith("# "):
            content = line[2:].strip()
            title_paragraph = document.add_paragraph(style="Title")
            title_paragraph.paragraph_format.space_after = Pt(12)
            title_paragraph.paragraph_format.keep_with_next = True
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(title_paragraph, content.upper())
            list_mode = None
            continue

        if line.startswith("## "):
            content = line[3:].strip()
            paragraph = document.add_paragraph(style="Heading 1")
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(12)
            _add_runs(paragraph, content)
            list_mode = None
            continue

        if line.startswith("### "):
            content = line[4:].strip()
            paragraph = document.add_paragraph(style="Heading 2")
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(10)
            _add_runs(paragraph, content)
            list_mode = None
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)", line)
        number_match = re.match(r"^\d+[.)]\s+(.*)", line)
        if bullet_match:
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Cm(0.75)
            _add_runs(paragraph, bullet_match.group(1).strip())
            list_mode = "bullet"
            continue
        if number_match:
            paragraph = document.add_paragraph(style="List Number")
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Cm(0.75)
            _add_runs(paragraph, number_match.group(1).strip())
            list_mode = "number"
            continue

        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if list_mode:
            paragraph.paragraph_format.first_line_indent = Cm(1.25)
        _add_runs(paragraph, line.strip())
        list_mode = None

    document.save(output_path)


def format_plan_summary(plan: DraftPlan) -> str:
    from html import escape as html_escape

    lines: list[str] = []

    title = (plan.title or "Документ").strip()
    title_escaped = html_escape(title)

    # Красивый заголовок документа
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  ✨ <b>Документ готов</b>          ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append(f"📄 <b>{title_escaped}</b>")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")



    # Информация о вопросах
    lines.append(f"❓ <b>Уточняющих вопросов:</b> <i>{len(plan.questions)}</i>")

    if plan.questions:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("💡 <b>Как отвечать</b>")
        lines.append("")
        lines.append("✅ Напишите все ответы")
        lines.append("   <b>одним сообщением</b>")
        lines.append("")
        lines.append("📝 <b>Варианты оформления:</b>")
        lines.append("")
        lines.append("   <code>1) Первый ответ</code>")
        lines.append("   <code>2) Второй ответ</code>")
        lines.append("   <code>3) Третий ответ</code>")
        lines.append("")
        lines.append("   <i>или разделяйте пустой строкой</i>")
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("👇 <i>Вопросы будут отправлены</i>")
        lines.append("   <i>следующим сообщением</i>")
    else:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("✅ <b>Дополнительных уточнений</b>")
        lines.append("   <b>не требуется</b>")
        lines.append("")
        lines.append("🚀 <i>Можно формировать документ</i>")

    return "\n".join(lines)
