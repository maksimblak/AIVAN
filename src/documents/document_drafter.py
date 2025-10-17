"""Utilities for interactive legal document drafting."""

from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


# ============================== Errors & Models ===============================

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


# ============================ System & User Prompts ===========================

DOCUMENT_ASSISTANT_SYSTEM_PROMPT = """
Ты — ИИ-Иван, юридический ИИ-ассистент.
Твоя цель — помогать юристам, снимая с них рутинные задачи
Твоя задача  аккуратно, формально корректно и прагматично составлять процессуальные документы по запросу юриста.
Делай эту задачу как опытный юрист эксперт по документам знающий все шаблоны и нюансы , эксперт по процессуальному праву и составлению процессуальных документов (исковые заявления, возражения, ходатайства, апелляции, определения, заявления об обеспечительных мерах и т.д.). 


ОСНОВНЫЕ ПРИНЦИПЫ РАБОТЫ: 
Всегда соблюдай формальные требования к форме и содержанию искового заявления/ходатайства/апелляции и т.д. 
Не додумывай факты. Работай только с теми данными, которые дал пользователь. Если чего-то не хватает — чётко попроси недостающую информацию. 
Перед финальной генерацией документа выполни независимую проверку (self-checklist) и укажи все исправления, которые сделал. 
Версионность и сохранение: если пользователь явно просит вести версии, предложи обозначение версии и рекомендации по доработкам во `self_check["issues"]` или во `follow_up_questions`.
Тон и стиль: Строго юридический, сухой, без эмоционального языка. Используй формулировки "рекомендуется", "следует", "целесообразно". • Не давай внештатных рекомендаций по вопросам, выходящим за рамки документа 
Когда тебя просят сделать какой-то документ, действуй по следующим шагам


1 шаг Найди образец требуемого документа, если его сразу назвали, или задай наводящие вопросы, чтобы понять какой именно документ нужен
2 шаг Задай все необходимые вопросы, чтобы его подготовить. Возможно, это будут вопросы в два этапа, чтобы все точно выяснить.
3 шаг После того как получил все ответы составь документ. Соблюдай все правила составления документов

После получения полных ответов составь документ в типовой процессуальной форме: шапка → вводная часть (реквизиты) → изложение фактов → доводы и правовое обоснование → расчёты (если применимо) → требования → перечень приложений → подпись/дата. 
Применяй чёткую, юридическую формулировку, краткость и логическую последовательность. Ссылайся на релевантные нормы (статьи ГПК/АПК/ГК РФ и т.п.) там, где это уместно. 
Подготовь также краткое «обоснование стратегии» (2–4 пункта): почему такой иск/формула требований обоснована, какие слабые места у оппонента, какие дополнительные доказательства нужно собрать. Включи этот блок в конце `document_markdown` под заголовком `## Обоснование стратегии` в виде маркированного списка.


4 шаг Перепроверь себя

Перепроверь себя на соблюдение всех процессуальных норм и правил
Проверь наличие всех обязательных реквизитов (шапка, подпись, дата, приложения).
Сверь указанные суммы с расчётами и приложениями. 
Проверь подсудность и правильность адресования суду (место подачи). 
Проверь сроки исковой давности и процессуальные сроки. 
Выпиши в отдельный блок все обнаруженные пробелы/некорректности и предложи конкретную правку.

5 шаг Сделай документ

Общие правила:
- Всегда возвращай валидный JSON и не добавляй текст вне JSON. Не используй кодовые блоки (```), псевдографику, backslashes для визуального форматирования.
- Соблюдай юридическую точность, используй профессиональные стандарты и не выдумывай данные.
- Если информации недостаточно, запроси уточнения через follow_up_questions или вопросы.

Режим «Планирование» (запрос содержит инструкции вроде «Сформируй структуру ответа», «Поле document_title…»):
1. Если тип документа понятен — найди подходящий образец и сформулируй точное название документа. Если тип неочевиден, предложи уточняющие вопросы.
2. Подготовь перечень вопросов, необходимых юристу для заполнения документа; при необходимости группируй и поясняй цель каждого вопроса.
3. Сформируй заметки контекста (context_notes), которые пригодятся при генерации документа и помогут соблюдать юридическую точность.

Режим «Генерация» (запрос содержит требования вернуть status/document_markdown/self_check и подготовить итоговый текст документа):
1. Проверь полноту данных. Если сведений недостаточно, верни status = "need_more_info" и сформируй follow_up_questions с уточнениями.
2. Если выполнить запрос невозможно (противоречит политике, отсутствуют ключевые данные и их нельзя запросить, либо требуется вмешательство юриста), верни status = "abort" и опиши причину в `self_check["issues"]` или во `follow_up_questions`.
3. При достаточности данных составь документ в Markdown, опираясь на образцы и профессиональные стандарты, соблюдая структуру, стиль и формулировки, характерные для соответствующего вида документа.
4. Структура `document_markdown`:
   • В начале размести блок реквизитов строками вида `Поле: Значение` (включи: `Суд`, `Истец`, `Ответчик`, `Цена иска` или иную стоимость, `Госпошлина`/сбор, `Дата`, `Подпись` с расшифровкой; при наличии — `Банковские реквизиты`, `Способ подачи`). Не используй псевдографику и обратные слэши.
   • После реквизитов добавь пустую строку и заголовок `# {название}`.
   • Основные разделы оформляй подзаголовками `## …`; используй нумерованные списки `1.`/`1)` для ключевых пунктов и маркированные списки `-`, `*`, `•`, `—` для подпунктов.
   • Итоговые блоки (просительная часть, приложения, подпись) подавай как структурированные списки без обратных слэшей в конце строк.
   • Для расчётов и смет используй Markdown-таблицы (`| колонка | колонка |`), чтобы их можно было перенести в Word.
5. Перед выдачей результата выполни самопроверку: перечисли учтённые ключевые данные и выявленные риски/пробелы в self_check.
6. Формат ответа:
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

USER_TEMPLATE = """
{request_heading}
{request}

{details}
""".strip()


# ================================ Regexes =====================================

_JSON_START_RE = re.compile(r"[{\[]")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_JSON_DECODER = json.JSONDecoder()
_INLINE_MARKUP_RE = re.compile(r"(\*\*.+?\*\*|__.+?__|\*.+?\*|_.+?_|`.+?`)", re.DOTALL)
_BULLET_ITEM_RE = re.compile(r"^([-*•—–])\s+(.*)")
_NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s+(.*)")
_METADATA_LINE_RE = re.compile(r"^([^:]{1,80}):\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r":?-{3,}:?")


# =============================== JSON helpers =================================

def _strip_code_fences(payload: str) -> str:
    stripped = payload.strip()
    lowered = stripped.lower()
    for prefix in ("```json", "```", "~~~json", "~~~"):
        if lowered.startswith(prefix):
            stripped = stripped[len(prefix):]
            stripped = stripped.lstrip("\r\n")
            break

    for suffix in ("```", "~~~"):
        if stripped.endswith(suffix):
            stripped = stripped[:-len(suffix)]
            stripped = stripped.rstrip()
            break

    return stripped.strip()


def _sanitize_json_string(payload: str) -> str:
    """
    Escape literal newlines and stray double quotes that appear inside JSON strings.

    This function walks the payload character-by-character, keeping track of whether
    we are inside a JSON string literal. When inside a string it converts raw CR/LF
    characters to their escaped versions and protects accidental quote characters
    (e.g. "Мой арбитр") that the model may emit without escaping.
    """

    if ("\n" not in payload and "\r" not in payload and '"' not in payload):
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
                    # Determine whether this quote should terminate the string.
                    lookahead = index + 1
                    while lookahead < length and payload[lookahead] in {" ", "\t"}:
                        lookahead += 1
                    next_char = payload[lookahead] if lookahead < length else ""

                    if next_char in {"\r", "\n"}:
                        while lookahead < length and payload[lookahead] in {" ", "\t", "\r", "\n"}:
                            lookahead += 1
                        next_char = payload[lookahead] if lookahead < length else ""

                    if next_char == ",":
                        lookahead += 1
                        while lookahead < length and payload[lookahead] in {" ", "\t", "\r", "\n"}:
                            lookahead += 1
                        after_comma = payload[lookahead] if lookahead < length else ""
                        if after_comma in {'"', "}", "]"}:
                            result.append(char)
                            in_string = False
                        else:
                            result.append('\\"')
                    elif next_char in {"", "}", "]", ":"}:
                        result.append(char)
                        in_string = False
                    else:
                        result.append('\\"')
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

    sanitized = "".join(result)
    return sanitized


def _deduplicate_consecutive_properties(payload: str) -> str:
    """Remove consecutive duplicate object properties that often appear due to streaming glitches."""

    lines = payload.splitlines()
    if not lines:
        return payload

    result: list[str] = []
    prev_key: str | None = None
    prev_value: str | None = None

    for line in lines:
        stripped = line.strip()
        normalized_line = stripped.lstrip(",")

        if normalized_line.startswith('"') and ":" in normalized_line:
            key_part, value_part = normalized_line.split(":", 1)
            key = key_part.strip().strip('"').strip()
            value = value_part.strip()
            value_no_comma = value[:-1].rstrip() if value.endswith(",") else value

            if prev_key == key:
                if prev_value != value_no_comma:
                    logger.debug(
                        "Dropping duplicate key with differing value in JSON repair: %s",
                        key,
                    )
                # Skip exact duplicate of the previous property entry
                continue

            prev_key = key
            prev_value = value_no_comma
        else:
            prev_key = None
            prev_value = None

        result.append(line)

    return "\n".join(result)


def _extract_structured_payload(raw: Any) -> Mapping[str, Any] | None:
    """Attempt to locate the actual JSON payload within structured response metadata."""

    def _unwrap(candidate: Any) -> Mapping[str, Any] | None:
        if isinstance(candidate, Mapping):
            lower_keys = {str(key).lower() for key in candidate.keys()}
            if (
                "document_title" in lower_keys
                or "document_markdown" in lower_keys
                or "status" in lower_keys
                or "questions" in lower_keys
                or "context_notes" in lower_keys
            ):
                return dict(candidate)
            nested_keys = (
                "parsed",
                "data",
                "json_schema",
                "content",
                "output",
                "outputs",
                "message",
                "messages",
                "result",
                "response",
                "payload",
                "body",
                "value",
            )
            for key in nested_keys:
                if key in candidate:
                    nested = candidate.get(key)
                    unwrapped = _unwrap(nested)
                    if unwrapped is not None:
                        return unwrapped
            for nested_value in candidate.values():
                if isinstance(nested_value, (Mapping, list, tuple)):
                    unwrapped = _unwrap(nested_value)
                    if unwrapped is not None:
                        return unwrapped
            return None

        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                unwrapped = _unwrap(item)
                if unwrapped is not None:
                    return unwrapped
        return None

    return _unwrap(raw)


def _extract_json(text: Any) -> Any:
    """Extract the first JSON structure found in text."""
    if isinstance(text, Mapping):
        return text
    if isinstance(text, list):
        for item in text:
            if isinstance(item, Mapping):
                return item
        if text:
            return text[0]
        raise DocumentDraftingError("Не удалось найти JSON в ответе модели")
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

            candidates: list[str] = [normalized]

            deduped = _deduplicate_consecutive_properties(normalized)
            if deduped != normalized:
                candidates.append(deduped)

            sanitized_variants: list[str] = []
            for candidate in candidates:
                sanitized = _sanitize_json_string(candidate)
                if sanitized != candidate:
                    sanitized_variants.append(sanitized)
            candidates.extend(sanitized_variants)

            for attempt in candidates:
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


def _looks_like_document_payload(data: Mapping[str, Any]) -> bool:
    status_value = str(data.get("status") or "").lower()
    if status_value in {"ok", "need_more_info", "abort"}:
        return True
    candidate_keys = {
        "document_title",
        "document_markdown",
        "questions",
        "follow_up_questions",
        "context_notes",
        "need_more_info",
        "self_check",
    }
    return any(key in data for key in candidate_keys)


def _coerce_payload(structured: Any, raw_text: str) -> Mapping[str, Any]:
    if isinstance(structured, Mapping):
        mapping = dict(structured)
        if _looks_like_document_payload(mapping):
            return mapping
        for key in ("parsed", "data", "json", "json_schema"):
            nested = mapping.get(key)
            if nested:
                try:
                    return _coerce_payload(nested, raw_text)
                except DocumentDraftingError:
                    continue
    if isinstance(structured, list):
        for item in structured:
            try:
                return _coerce_payload(item, raw_text)
            except DocumentDraftingError:
                continue
    if raw_text:
        extracted = _extract_json(raw_text)
        if isinstance(extracted, Mapping) and _looks_like_document_payload(extracted):
            return extracted
    if structured is not None:
        extracted = _extract_json(structured)
        if isinstance(extracted, Mapping) and _looks_like_document_payload(extracted):
            return extracted
    raise DocumentDraftingError("Не удалось получить структурированный ответ модели")


def _format_answers(answers: Iterable[dict[str, str]]) -> str:
    lines = []
    for idx, item in enumerate(answers, 1):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        lines.append(f"{idx}. {question}\n   Ответ: {answer}")
    return "\n".join(lines) if lines else "(Ответы пока не получены)"


# ============================ High-level API calls ============================

async def plan_document(openai_service, request_text: str) -> DraftPlan:
    if not request_text.strip():
        raise DocumentDraftingError("Пустой запрос")

    planner_details = """
Сформируй структуру ответа:
- Поле document_title — точное наименование документа.
- Поле need_more_info = true, если требуются ответы юриста; false, если данных достаточно для генерации.
- Массив questions: элементы {"id": "...", "text": "...", "purpose": "..."} с пояснением цели вопроса.
- Массив context_notes: короткие заметки, которые помогут при подготовке документа.

Ответ верни строго в JSON.
""".strip()

    user_prompt = USER_TEMPLATE.format(
        request_heading="Запрос юриста:",
        request=request_text.strip(),
        details=planner_details,
    )
    response = await openai_service.ask_legal(
        system_prompt=DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
        user_text=user_prompt,
    )
    if not response.get("ok"):
        raise DocumentDraftingError(response.get("error") or "Не удалось получить ответ от модели")

    raw_text = response.get("text", "")
    structured = _extract_structured_payload(response.get("structured"))
    if structured:
        logger.debug("Using structured planner payload (keys: %s)", list(structured.keys()))
    data = _coerce_payload(structured, raw_text)

    title = str(data.get("document_title") or "Документ").strip()
    notes = [str(note).strip() for note in data.get("context_notes") or [] if str(note).strip()]

    need_more_raw = data.get("need_more_info")
    if isinstance(need_more_raw, bool):
        need_more = need_more_raw
    elif isinstance(need_more_raw, str):
        lowered = need_more_raw.strip().lower()
        need_more = lowered in {"true", "yes", "1", "y", "да"}
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
            questions.append({"id": question_id, "text": text, "purpose": purpose})

    return DraftPlan(title=title or "Документ", questions=questions, notes=notes)


async def generate_document(
    openai_service,
    request_text: str,
    title: str,
    answers: list[dict[str, str]],
) -> DraftResult:
    answers_formatted = _format_answers(answers)
    generator_details = f"""
Предполагаемое название документа: {title}

Ответы юриста:
{answers_formatted or "(Ответы пока не получены)"}

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
    user_prompt = USER_TEMPLATE.format(
        request_heading="Запрос юриста и вводные данные:",
        request=request_text.strip(),
        details=generator_details,
    )
    response = await openai_service.ask_legal(
        system_prompt=DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
        user_text=user_prompt,
    )
    if not response.get("ok"):
        raise DocumentDraftingError(response.get("error") or "Не удалось получить ответ от модели")

    raw_text = response.get("text", "")
    structured = _extract_structured_payload(response.get("structured"))
    if structured:
        logger.debug("Using structured generator payload (keys: %s)", list(structured.keys()))
    data = _coerce_payload(structured, raw_text)

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


# ============================= DOCX builder (MD) ==============================

def build_docx_from_markdown(markdown: str, output_path: str) -> None:
    """
    Конвертирует подготовленный Markdown в DOCX, применяя базовые требования к юрдокам:
    - Бумага A4, поля 2.5 см, шрифт Times New Roman 12 pt, междустрочный интервал 1.5
    - Заголовки через стандартные стили, без псевдографики
    - Метаданные "Поле: Значение" — автоматом складываются в 2‑колоночную таблицу
    - Поддержка списков (-, *, •, —) и нумерации (1. / 1))
    - Поддержка Markdown‑таблиц
    - Нумерация страниц «Стр. X из Y» в нижнем колонтитуле
    """
    try:
        from docx import Document  # type: ignore
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
        from docx.oxml import OxmlElement  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
        from docx.shared import Cm, Pt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise DocumentDraftingError("Не удалось подготовить DOCX без пакета python-docx") from exc

    if not markdown or not markdown.strip():
        raise DocumentDraftingError("Пустой текст документа")

    document = Document()
    section = document.sections[0]

    # Формат страницы и поля (A4 + 2.5 см)
    with suppress(Exception):
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, attr, Cm(2.5))

    # ——— шрифты и стили
    def _ensure_font(style_name: str, *, size: int, bold: bool = False,
                     italic: bool = False, align: int | None = None) -> None:
        try:
            style = document.styles[style_name]
        except KeyError:
            return
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(size)
        font.bold = bold
        font.italic = italic
        # для кириллицы
        if style.element.rPr is not None:
            r_fonts = style.element.rPr.rFonts
            if r_fonts is not None:
                r_fonts.set(qn("w:eastAsia"), "Times New Roman")
        paragraph_format = getattr(style, "paragraph_format", None)
        if paragraph_format:
            paragraph_format.space_after = Pt(6)
            if align is not None:
                paragraph_format.alignment = align

    # Normal
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

    # ——— колонтитул: "Стр. X из Y"
    def _add_field(paragraph, instruction: str) -> None:
        run = paragraph.add_run()
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = instruction
        fld_separate = OxmlElement("w:fldChar")
        fld_separate.set(qn("w:fldCharType"), "separate")
        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        run._r.append(fld_begin)
        run._r.append(instr)
        run._r.append(fld_separate)
        run._r.append(fld_end)

    with suppress(Exception):
        footer = section.footer
        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(0)
        para.add_run("Стр. ")
        _add_field(para, "PAGE")
        para.add_run(" из ")
        _add_field(para, "NUMPAGES")

    # ——— inline разметка
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

    # ——— meta‑блок шапки
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

                fp = field_cell.paragraphs[0]
                fp.paragraph_format.space_before = Pt(0)
                fp.paragraph_format.space_after = Pt(0)
                fp.paragraph_format.first_line_indent = Cm(0)
                fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                fr = fp.add_run(field)
                fr.bold = True

                vp = value_cell.paragraphs[0]
                vp.paragraph_format.space_before = Pt(0)
                vp.paragraph_format.space_after = Pt(0)
                vp.paragraph_format.first_line_indent = Cm(0)
                vp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                _add_runs(vp, value)

            document.add_paragraph("")  # разделение от остального текста
        else:
            for field, value in metadata_buffer:
                paragraph = document.add_paragraph(style="Normal")
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.left_indent = Cm(0)
                paragraph.paragraph_format.space_after = Pt(6)
                _add_runs(paragraph, f"{field}: {value}")

        metadata_buffer = []

    # ——— основной разбор Markdown
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        raw_line = lines[idx]
        idx += 1
        line = raw_line.rstrip()

        # поддержка "жёсткого переноса" в Markdown (backslash в конце строки)
        if line.endswith("\\"):
            line = line[:-1].rstrip()

        if not line:
            flush_metadata()
            document.add_paragraph("")
            list_mode = None
            continue

        plain_line = line.strip()
        plain_lower = plain_line.lower()

        # эвристика для "Исковое заявление ..." в первой строке — как титул
        if plain_line and plain_lower.startswith("исковое заявление"):
            title_paragraph = document.add_paragraph(style="Title")
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_paragraph.paragraph_format.space_before = Pt(12)
            title_paragraph.paragraph_format.space_after = Pt(12)
            title_paragraph.paragraph_format.keep_with_next = True
            _add_runs(title_paragraph, plain_line)
            list_mode = None
            continue

        # сбор мета‑строк "Поле: Значение" до первого "обычного" абзаца/заголовка/списка
        if list_mode is None and plain_line:
            if not plain_line.startswith(("-", "*", "•", "—", "#")) and not _NUMBERED_ITEM_RE.match(plain_line):
                colon_match = _METADATA_LINE_RE.match(plain_line)
                if colon_match:
                    field = colon_match.group(1).strip()
                    value = colon_match.group(2).strip()
                    if value:
                        metadata_buffer.append((field, value))
                        continue

        # таблица в Markdown
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
            # вторая строка — разделитель?
            if len(parsed_rows) >= 2 and all(_TABLE_SEPARATOR_RE.fullmatch(cell.replace(" ", "")) for cell in parsed_rows[1]):
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
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
                        _add_runs(paragraph, cell_text)
                        if row_idx == 0:
                            for run in paragraph.runs:
                                run.bold = True
                document.add_paragraph("")
            list_mode = None
            continue

        flush_metadata()

        # заголовки
        if line.startswith("# "):
            content = line[2:].strip()
            title_paragraph = document.add_paragraph(style="Title")
            title_paragraph.paragraph_format.space_after = Pt(12)
            title_paragraph.paragraph_format.keep_with_next = True
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(title_paragraph, content)  # без насильного UPPERCASE — юридический стиль
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

        # списки
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

        # обычные абзацы
        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.space_after = Pt(6)
        if list_mode:
            paragraph.paragraph_format.left_indent = Cm(1.25)
            paragraph.paragraph_format.first_line_indent = Cm(0)
        else:
            # строка вида "Поле: значение" — без красной строки, слева
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


# ================================ UI helpers ==================================

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
