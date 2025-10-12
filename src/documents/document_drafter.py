"""Utilities for interactive legal document drafting."""

from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
3. Структура `document_markdown`:
   • В начале размести блок реквизитов строками формата `Поле: Значение` (обязательно включи: `Суд`, `Истец`, `Ответчик`, `Цена иска` или аналогичную стоимость, `Госпошлина`/сбор, `Дата`, `Подпись` с расшифровкой; при наличии — `Банковские реквизиты`, `Способ подачи`). Не используй псевдографику и обратные слэши.
   • После реквизитов сделай пустую строку и добавь заголовок документа через `# {название}`.
   • Основные разделы оформляй подзаголовками `## …`, внутри разделов используй пронумерованные списки `1.` для ключевых пунктов и маркированные списки `-` для подпунктов.
   • Итоговые блоки (просительная часть, приложения, подпись) также подай в виде структурированных списков, без `\` в конце строк.
   • Для таблиц расчетов или смет используй Markdown-таблицы (`| колонка | колонка |`), чтобы данные можно было корректно перенести в Word.
4. Перед выдачей результата выполни самопроверку: перечисли учтённые ключевые данные и выявленные риски/пробелы.

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
_INLINE_MARKUP_RE = re.compile(r"(\*\*.+?\*\*|__.+?__|\*.+?\*|_.+?_|`.+?`)", re.DOTALL)
_BULLET_ITEM_RE = re.compile(r"^([-*•])\s+(.*)")
_NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s+(.*)")
_METADATA_LINE_RE = re.compile(r"^([^:]{1,80}):\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r":?-{3,}:?")


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


def _escape_unescaped_newlines(payload: str) -> str:
    """Escape literal newlines that appear inside JSON strings."""

    if "\n" not in payload and "\r" not in payload:
        return payload

    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    length = len(payload)

    while index < length:
        char = payload[index]

        if in_string:
            if escaped:
                result.append(char)
                escaped = False
            else:
                if char == "\\":
                    result.append(char)
                    escaped = True
                elif char == '"':
                    result.append(char)
                    in_string = False
                elif char == "\r":
                    result.append("\\r")
                    if index + 1 < length and payload[index + 1] == "\n":
                        result.append("\\n")
                        index += 1
                elif char == "\n":
                    result.append("\\n")
                else:
                    result.append(char)
        else:
            result.append(char)
            if char == '"':
                in_string = True
                escaped = False

        index += 1

    repaired = "".join(result)
    return repaired


def _extract_json(text: Any) -> Any:
    """Extract the first JSON structure found in text."""

    if isinstance(text, (dict, list)):
        return text
    if text is None:
        raise DocumentDraftingError("Не удалось найти JSON в ответе модели")
    if not isinstance(text, str):
        raise DocumentDraftingError("Не удалось найти JSON в ответе модели")

    cleaned_text = _strip_code_fences(text)

    def _decode_from_index(source: str, start_index: int = 0) -> Any:
        idx = start_index
        length = len(source)
        while idx < length and source[idx].isspace():
            idx += 1
        if idx >= length:
            raise json.JSONDecodeError("Empty JSON candidate", source, idx)
        try:
            obj, _ = _JSON_DECODER.raw_decode(source, idx)
            return obj
        except (json.JSONDecodeError, ValueError) as primary_err:
            candidate = source[idx:]
            stripped_candidate = candidate.strip()
            if not stripped_candidate:
                raise json.JSONDecodeError("Empty JSON candidate", source, idx) from primary_err

            normalized = _TRAILING_COMMA_RE.sub(r"\1", stripped_candidate)
            attempts = [normalized]

            repaired = _escape_unescaped_newlines(normalized)
            if repaired != normalized:
                attempts.append(repaired)

            for attempt in attempts:
                with suppress(json.JSONDecodeError, ValueError):
                    obj, _ = _JSON_DECODER.raw_decode(attempt)
                    return obj

            raise primary_err

    try:
        return _decode_from_index(cleaned_text)
    except (json.JSONDecodeError, ValueError) as err:
        logger.debug("Primary JSON parse failed, scanning for nested JSON: %s", err)
        for match in _JSON_START_RE.finditer(cleaned_text):
            start_idx = match.start()
            try:
                return _decode_from_index(cleaned_text, start_idx)
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

    structured = response.get("structured")
    if isinstance(structured, Mapping):
        data = dict(structured)
    else:
        raw_text = response.get("text", "")
        if not raw_text:
            raise DocumentDraftingError("Пустой ответ модели")
        data = _extract_json(raw_text)
    title = str(data.get("document_title") or "Документ").strip()
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
    questions: list[dict[str, str]] = []
    if need_more:
        raw_questions = data.get("questions") or []
        for idx, raw in enumerate(raw_questions):
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            purpose = str(raw.get("purpose") or "").strip()
            question_id = str(raw.get("id") or f"q{idx + 1}")
            questions.append(
                {
                    "id": question_id,
                    "text": text,
                    "purpose": purpose,
                }
            )

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

    structured = response.get("structured")
    if isinstance(structured, Mapping):
        data = dict(structured)
    else:
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
    normal_paragraph.first_line_indent = Cm(0)
    normal_paragraph.line_spacing = 1.5

    _ensure_font("Title", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    _ensure_font("Heading 1", size=14, bold=True)
    _ensure_font("Heading 2", size=13, bold=True)
    _ensure_font("Heading 3", size=12, bold=True)

    def _add_runs(paragraph, text: str) -> None:
        for chunk in _INLINE_MARKUP_RE.split(text):
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
    metadata_buffer: list[tuple[str, str]] = []

    def _is_table_row(text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

    def _parse_table_row(text: str) -> list[str]:
        cells = [cell.strip() for cell in text.strip()[1:-1].split("|")]
        return cells

    def flush_metadata(force_plain: bool = False) -> None:
        nonlocal metadata_buffer
        if not metadata_buffer:
            return

        if not force_plain and len(metadata_buffer) >= 2:
            table = document.add_table(rows=0, cols=2)
            with suppress(Exception):
                table.style = "Table Grid"
            for field, value in metadata_buffer:
                row = table.add_row()
                field_cell, value_cell = row.cells

                field_paragraph = field_cell.paragraphs[0]
                field_paragraph.paragraph_format.space_before = Pt(0)
                field_paragraph.paragraph_format.space_after = Pt(0)
                field_paragraph.paragraph_format.first_line_indent = Cm(0)
                field_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                field_run = field_paragraph.add_run(field)
                field_run.bold = True

                value_paragraph = value_cell.paragraphs[0]
                value_paragraph.paragraph_format.space_before = Pt(0)
                value_paragraph.paragraph_format.space_after = Pt(0)
                value_paragraph.paragraph_format.first_line_indent = Cm(0)
                value_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                _add_runs(value_paragraph, value)

            document.add_paragraph("")
        else:
            for field, value in metadata_buffer:
                paragraph = document.add_paragraph(style="Normal")
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.left_indent = Cm(0)
                paragraph.paragraph_format.space_after = Pt(6)
                _add_runs(paragraph, f"{field}: {value}")

        metadata_buffer = []

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        raw_line = lines[idx]
        idx += 1
        line = raw_line.rstrip()

        if line.endswith("\\"):
            line = line[:-1].rstrip()

        if not line:
            flush_metadata()
            document.add_paragraph("")
            list_mode = None
            continue

        plain_line = line.strip()
        plain_lower = plain_line.lower()
        if plain_line and plain_lower.startswith("исковое заявление"):
            title_paragraph = document.add_paragraph(style="Title")
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_paragraph.paragraph_format.space_before = Pt(12)
            title_paragraph.paragraph_format.space_after = Pt(12)
            title_paragraph.paragraph_format.keep_with_next = True
            _add_runs(title_paragraph, plain_line)
            list_mode = None
            continue

        if list_mode is None and plain_line:
            if (
                not plain_line.startswith(("-", "*", "•", "#"))
                and not _NUMBERED_ITEM_RE.match(plain_line)
            ):
                colon_match = _METADATA_LINE_RE.match(plain_line)
                if colon_match:
                    field = colon_match.group(1).strip()
                    value = colon_match.group(2).strip()
                    if value:
                        metadata_buffer.append((field, value))
                        continue

        if _is_table_row(line):
            flush_metadata()
            table_lines = [line]
            while idx < total_lines:
                peek_line = lines[idx].rstrip()
                if _is_table_row(peek_line):
                    table_lines.append(peek_line)
                    idx += 1
                else:
                    break

            parsed_rows = [_parse_table_row(row) for row in table_lines]
            if len(parsed_rows) >= 2 and all(
                _TABLE_SEPARATOR_RE.fullmatch(cell.replace(" ", ""))
                for cell in parsed_rows[1]
            ):
                data_rows = [parsed_rows[0]] + parsed_rows[2:]
            else:
                data_rows = parsed_rows

            if data_rows:
                num_cols = max(len(row) for row in data_rows)
                table = document.add_table(rows=0, cols=num_cols)
                with suppress(Exception):
                    table.style = "Table Grid"
                for row_idx, cells in enumerate(data_rows):
                    row = table.add_row()
                    for col in range(num_cols):
                        cell_text = cells[col].strip() if col < len(cells) else ""
                        cell = row.cells[col]
                        paragraph = cell.paragraphs[0]
                        paragraph.paragraph_format.space_before = Pt(0)
                        paragraph.paragraph_format.space_after = Pt(0)
                        paragraph.paragraph_format.first_line_indent = Cm(0)
                        paragraph.alignment = (
                            WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
                        )
                        _add_runs(paragraph, cell_text)
                        if row_idx == 0:
                            for run in paragraph.runs:
                                run.bold = True
                document.add_paragraph("")
            list_mode = None
            continue

        flush_metadata()

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

        bullet_match = _BULLET_ITEM_RE.match(line)
        number_match = _NUMBERED_ITEM_RE.match(line)
        if bullet_match:
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Cm(0.75)
            _add_runs(paragraph, bullet_match.group(2).strip())
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
        paragraph.paragraph_format.space_after = Pt(6)
        if list_mode:
            paragraph.paragraph_format.left_indent = Cm(1.25)
            paragraph.paragraph_format.first_line_indent = Cm(0)
        else:
            if ":" in plain_line and not plain_line.endswith(":"):
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.left_indent = Cm(0)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                paragraph.paragraph_format.first_line_indent = Cm(1.25)
        _add_runs(paragraph, plain_line)
        list_mode = None

    flush_metadata()

    document.save(output_path)


def _pluralize_questions(count: int) -> str:
    """Склонение слова 'вопрос' в зависимости от числа."""
    if count % 10 == 1 and count % 100 != 11:
        return "вопрос"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "вопроса"
    else:
        return "вопросов"


def format_plan_summary(plan: DraftPlan) -> str:
    from html import escape as html_escape

    lines: list[str] = []

    title = (plan.title or "Документ").strip()
    title_escaped = html_escape(title)

    # Красивый заголовок документа
    lines.append("📋 <b>Подготовка документа</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"<b>Документ:</b> {title_escaped}")
    lines.append("")

    if plan.questions:
        lines.append(f"<b>Требуется информация:</b> {len(plan.questions)} {_pluralize_questions(len(plan.questions))}")
        lines.append("")
        lines.append("")
        lines.append("💡 <b>Инструкция по ответу</b>")
        lines.append("")
        lines.append("<b>Как отвечать:</b>")
        lines.append("")
        lines.append("  ✓ Напишите все ответы в одном")
        lines.append("    сообщении, не разделяя их")
        lines.append("")
        lines.append("<b>Формат ответов:</b>")
        lines.append("")
        lines.append("  <b>Вариант 1</b> — нумерация:")
        lines.append("  <code>1) Ваш первый ответ</code>")
        lines.append("  <code>2) Ваш второй ответ</code>")
        lines.append("  <code>3) Ваш третий ответ</code>")
        lines.append("")
        lines.append("  <b>Вариант 2</b> — пустая строка:")
        lines.append("  <code>Первый ответ</code>")
        lines.append("  <code></code>")
        lines.append("  <code>Второй ответ</code>")
        lines.append("")
        lines.append("")
        lines.append("👇 <i>Вопросы отправлены отдельным</i>")
        lines.append("   <i>сообщением ниже</i>")
    else:
        lines.append("<b>Статус:</b> Готов к созданию ✅")
        lines.append("")
        lines.append("")
        lines.append("✅ <b>Всё готово!</b>")
        lines.append("")
        lines.append("Дополнительная информация")
        lines.append("не требуется.")
        lines.append("")
        lines.append("🚀 Можно приступать к")
        lines.append("   формированию документа")

    return "\n".join(lines)
