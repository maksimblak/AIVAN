"""
Модуль саммаризации документов
Создает структурированное резюме документа с ключевыми выводами для юристов
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

LANGUAGE_CONFIG: dict[str, dict[str, str]] = {
    "ru": {"prompt": "русский", "display": "Русский"},
    "en": {"prompt": "английский", "display": "English"},
}

DETAIL_LEVELS: dict[str, dict[str, str]] = {
    "detailed": {"label": "detailed", "ru": "Подробная", "en": "Detailed"},
    "brief": {"label": "brief", "ru": "Краткая", "en": "Brief"},
}

CHUNK_PROMPT = """
Ты — юридический аналитик. К тебе поступила часть договора или иного юридического документа.
Твоя задача — подготовить структурированное саммари этой части.

СТРОГО ВЕРНИ JSON без дополнительных комментариев.
Структура:
{
  "summary": "связный текст 3-5 предложений",
  "key_points": ["ключевые положения"],
  "deadlines": ["сроки и дедлайны"],
  "penalties": ["штрафы, неустойки, санкции"],
  "actions": ["рекомендуемые шаги/проверки"],
  "language": "ru|en"
}

Параметры:
- Уровень детализации: {detail_label}
- Желаемый язык ответа: {language_name} язык
- Номер части: {chunk_number} из {total_chunks}

Вот текст части (сохраняй фактические формулировки):
"""

AGGREGATION_PROMPT = """
Ты — помощник юриста. Ниже набор саммари частей документа в формате JSON.
Объедини их и верни итог в виде JSON той же структуры, но без дублирующихся пунктов.
Используй {language_name} язык. Уровень детализации: {detail_label}.

Сырые данные:
{chunk_payload}

СТРОГО ВЕРНИ JSON со следующими ключами:
{
  "summary": "целостный обзор",
  "key_points": ["ключевые пункты документа"],
  "deadlines": ["обязательные сроки"],
  "penalties": ["штрафы/санкции"],
  "actions": ["контрольный список"],
  "language": "ru|en"
}
"""

LANG_SECTIONS: dict[str, dict[str, str]] = {
    "ru": {
        "overview": "### Общий обзор",
        "key_points": "### Ключевые положения",
        "deadlines": "### Сроки и дедлайны",
        "penalties": "### Санкции и штрафы",
        "checklist": "### Чек-лист для контроля",
        "none": "Не указано",
    },
    "en": {
        "overview": "### Overview",
        "key_points": "### Key Clauses",
        "deadlines": "### Deadlines",
        "penalties": "### Penalties",
        "checklist": "### Control Checklist",
        "none": "Not specified",
    },
}

DATE_PATTERNS: tuple[str, ...] = (
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
)

PENALTY_KEYWORDS: tuple[str, ...] = (
    "штраф",
    "неустой",
    "пени",
    "penalty",
    "fine",
    "liquidated damages",
    "санкц",
)


class DocumentSummarizer(DocumentProcessor):
    """Создает структурированное саммари юридических документов"""

    def __init__(self, openai_service=None):
        super().__init__(name="DocumentSummarizer", max_file_size=50 * 1024 * 1024)
        self.openai_service = openai_service
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]

    async def process(
        self,
        file_path: str | Path,
        detail_level: str = "detailed",
        language: str = "ru",
        output_formats: list[str] | None = None,
        **_: Any,
    ) -> DocumentResult:
        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        normalized_detail = self._normalize_detail_level(detail_level)

        success, extracted = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {extracted}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(extracted)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        normalized_lang = self._normalize_language(language, cleaned_text)

        metadata = self._collect_metadata(cleaned_text, file_path, normalized_lang)
        deadlines_heuristic = self._extract_deadlines(cleaned_text)
        penalties_heuristic = self._extract_penalties(cleaned_text)

        try:
            summary_payload = await self._create_summary(
                cleaned_text,
                detail_level=normalized_detail,
                language=normalized_lang,
            )
        except ProcessingError:
            raise
        except Exception as exc:
            logger.exception("Не удалось подготовить саммари: %s", exc)
            raise ProcessingError(f"Ошибка саммари: {exc}", "SUMMARY_ERROR")

        summary_data = summary_payload["summary"]
        if not summary_data.get("deadlines") and deadlines_heuristic:
            summary_data["deadlines"] = deadlines_heuristic
        if not summary_data.get("penalties") and penalties_heuristic:
            summary_data["penalties"] = penalties_heuristic
        if not summary_data.get("actions"):
            summary_data["actions"] = self._build_checklist_from_metadata(
                deadlines_heuristic,
                penalties_heuristic,
            )

        summary_text = self._format_summary_text(summary_data, normalized_lang)

        result_payload = {
            "summary": {"content": summary_text, "structured": summary_data},
            "metadata": metadata,
            "detail_level": normalized_detail,
            "language": normalized_lang,
            "deadlines": summary_data.get("deadlines", []),
            "penalties": summary_data.get("penalties", []),
            "checklist": summary_data.get("actions", []),
            "processing_info": summary_payload.get("processing_info", {}),
        }

        if output_formats:
            result_payload["requested_formats"] = list(dict.fromkeys(output_formats))

        return DocumentResult.success_result(data=result_payload, message="Саммари подготовлено")

    async def _create_summary(self, text: str, detail_level: str, language: str) -> dict[str, Any]:
        chunks = TextProcessor.split_into_chunks(text, max_chunk_size=4000, overlap=400)
        chunk_results: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_result = await self._summarize_chunk(
                chunk,
                chunk_number=idx,
                total_chunks=len(chunks),
                detail_level=detail_level,
                language=language,
            )
            chunk_results.append(chunk_result)

        if len(chunk_results) == 1:
            structured = chunk_results[0]["summary"]
            return {
                "summary": structured,
                "processing_info": {
                    "chunks_processed": 1,
                    "strategy": "single",
                },
            }

        aggregated = await self._aggregate_chunks(chunk_results, detail_level, language)
        return {
            "summary": aggregated,
            "processing_info": {
                "chunks_processed": len(chunk_results),
                "strategy": "aggregate",
            },
        }

    async def _summarize_chunk(
        self,
        chunk_text: str,
        chunk_number: int,
        total_chunks: int,
        detail_level: str,
        language: str,
    ) -> dict[str, Any]:
        detail_label = DETAIL_LEVELS[detail_level]["ru" if language == "ru" else "en"]
        language_name = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["ru"])["prompt"]

        user_message = (
            CHUNK_PROMPT.format(
                detail_label=detail_label,
                language_name=language_name,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
            )
            + "\n"
            + chunk_text
        )

        response = await self.openai_service.ask_legal(
            system_prompt="Ты структурируешь юридические тексты.",
            user_message=user_message,
        )
        if not response.get("ok"):
            raise ProcessingError(
                f"Модель не смогла обработать часть {chunk_number}",
                "SUMMARY_MODEL_ERROR",
            )

        parsed = self._parse_summary_response(response.get("text", ""), language)
        parsed.setdefault("language", language)
        return {"summary": parsed, "raw_response": response.get("text", "")}

    async def _aggregate_chunks(
        self,
        chunk_results: list[dict[str, Any]],
        detail_level: str,
        language: str,
    ) -> dict[str, Any]:
        language_name = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["ru"])["prompt"]
        detail_label = DETAIL_LEVELS[detail_level]["ru" if language == "ru" else "en"]
        payload = json.dumps([item["summary"] for item in chunk_results], ensure_ascii=False)

        response = await self.openai_service.ask_legal(
            system_prompt="Ты объединяешь юридические саммари в единый отчет.",
            user_message=AGGREGATION_PROMPT.format(
                language_name=language_name,
                detail_label=detail_label,
                chunk_payload=payload,
            ),
        )
        if response.get("ok"):
            aggregated = self._parse_summary_response(response.get("text", ""), language)
        else:
            logger.warning("Агрегация через модель не удалась, объединяем эвристикой")
            aggregated = self._merge_chunks_locally([item["summary"] for item in chunk_results])

        aggregated.setdefault("language", language)
        return aggregated

    def _parse_summary_response(self, raw_text: str, language: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {
                "summary": "",
                "key_points": [],
                "deadlines": [],
                "penalties": [],
                "actions": [],
                "language": language,
            }

        json_payload = self._extract_json(text)
        if json_payload:
            try:
                data = json.loads(json_payload)
                return {
                    "summary": str(data.get("summary", "")).strip(),
                    "key_points": self._ensure_list_of_strings(data.get("key_points")),
                    "deadlines": self._ensure_list_of_strings(data.get("deadlines")),
                    "penalties": self._ensure_list_of_strings(data.get("penalties")),
                    "actions": self._ensure_list_of_strings(data.get("actions")),
                    "language": data.get("language", language),
                }
            except json.JSONDecodeError:
                logger.debug("Не удалось распарсить JSON саммари: %s", text)

        return {
            "summary": text,
            "key_points": [],
            "deadlines": [],
            "penalties": [],
            "actions": [],
            "language": language,
        }

    def _extract_json(self, text: str) -> str | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def _ensure_list_of_strings(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            result: list[str] = []
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        result.append(stripped)
            return result
        return []

    def _merge_chunks_locally(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        summary_parts: list[str] = []
        key_points: list[str] = []
        deadlines: list[str] = []
        penalties: list[str] = []
        actions: list[str] = []

        for item in summaries:
            summary_part = item.get("summary")
            if summary_part:
                summary_parts.append(str(summary_part).strip())
            key_points.extend(item.get("key_points", []))
            deadlines.extend(item.get("deadlines", []))
            penalties.extend(item.get("penalties", []))
            actions.extend(item.get("actions", []))

        def _dedup(items: list[str]) -> list[str]:
            seen: dict[str, None] = {}
            for entry in items:
                normalized = entry.strip()
                if normalized and normalized not in seen:
                    seen[normalized] = None
            return list(seen.keys())

        return {
            "summary": "\n\n".join(summary_parts),
            "key_points": _dedup(key_points),
            "deadlines": _dedup(deadlines),
            "penalties": _dedup(penalties),
            "actions": _dedup(actions),
        }

    def _format_summary_text(self, data: dict[str, Any], language: str) -> str:
        sections = LANG_SECTIONS.get(language, LANG_SECTIONS["ru"])
        lines: list[str] = [sections["overview"], data.get("summary") or sections["none"], ""]

        def append_block(title_key: str, values: list[str]) -> None:
            lines.append(sections[title_key])
            if values:
                for item in values:
                    lines.append(f"- {item}")
            else:
                lines.append(sections["none"])
            lines.append("")

        append_block("key_points", data.get("key_points", []))
        append_block("deadlines", data.get("deadlines", []))
        append_block("penalties", data.get("penalties", []))
        append_block("checklist", data.get("actions", []))

        return "\n".join(lines).strip()

    def _normalize_detail_level(self, raw_level: str) -> str:
        normalized = (raw_level or "").strip().lower()
        if normalized in DETAIL_LEVELS:
            return normalized
        if normalized in {"short", "mini", "brief"}:
            return "brief"
        return "detailed"

    def _normalize_language(self, raw_language: str, document_text: str | None = None) -> str:
        normalized = (raw_language or "").strip().lower()
        if normalized in LANGUAGE_CONFIG:
            return normalized
        if normalized in {"auto", "detect", ""} and document_text:
            return self._detect_language(document_text)
        if normalized.startswith("en"):
            return "en"
        if normalized.startswith("ru"):
            return "ru"
        if document_text:
            return self._detect_language(document_text)
        return "ru"

    def _detect_language(self, text: str) -> str:
        cyr = len(re.findall(r"[А-Яа-яЁё]", text))
        lat = len(re.findall(r"[A-Za-z]", text))
        if cyr == 0 and lat == 0:
            return "ru"
        return "ru" if cyr >= lat else "en"

    def _collect_metadata(self, text: str, file_path: str | Path, language: str) -> dict[str, Any]:
        meta = TextProcessor.extract_metadata(text)
        meta.update(
            {
                "language": language,
                "language_display": LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["ru"])["display"],
                "source_length_chars": len(text),
                "source_preview": text[:400],
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "file_path": str(file_path),
            }
        )
        return meta

    def _extract_deadlines(self, text: str, limit: int = 5) -> list[str]:
        results: list[str] = []
        for pattern in DATE_PATTERNS:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                cleaned = match.strip()
                if cleaned not in results:
                    results.append(cleaned)
                if len(results) >= limit:
                    return results
        return results

    def _extract_penalties(self, text: str, limit: int = 5) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        found: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in PENALTY_KEYWORDS):
                cleaned = sentence.strip()
                if cleaned and cleaned not in found:
                    found.append(cleaned)
                if len(found) >= limit:
                    break
        return found

    def _build_checklist_from_metadata(
        self,
        deadlines: list[str],
        penalties: list[str],
    ) -> list[str]:
        checklist: list[str] = []
        if deadlines:
            checklist.append("Проверить соблюдение всех указанных сроков")
        if penalties:
            checklist.append("Убедиться в корректности расчета неустоек и штрафов")
        checklist.append("Согласовать итоговое саммари с ответственным юристом")
        return checklist


__all__ = ["DocumentSummarizer"]
