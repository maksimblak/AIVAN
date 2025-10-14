"""
Модуль саммаризации документов
Создает структурированное резюме документа с ключевыми выводами для юристов
"""

from __future__ import annotations

import json
import logging
from src.core.settings import AppSettings

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

# ---------------------------------- Конфиг ----------------------------------

LANGUAGE_CONFIG: dict[str, dict[str, str]] = {
    "ru": {"prompt": "русский", "display": "Русский"},
    "en": {"prompt": "английский", "display": "English"},
}

DETAIL_LEVELS: dict[str, dict[str, str]] = {
    "detailed": {"label": "detailed", "ru": "Подробная", "en": "Detailed"},
    "brief": {"label": "brief", "ru": "Краткая", "en": "Brief"},
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
    "санкц",
    "penalty",
    "fine",
    "liquidated damages",
)

# LLM промпты
CHUNK_PROMPT = """
Ты — юридический аналитик. К тебе поступила часть договора или иного юридического документа.
Подготовь структурированное саммари ЭТОЙ ЧАСТИ.

СТРОГО ВЕРНИ ТОЛЬКО JSON без любых комментариев.
Структура:
{{
  "summary": "связный текст 3-5 предложений",
  "key_points": ["ключевые положения (без дублирования)"],
  "deadlines": ["сроки и дедлайны (если есть)"],
  "penalties": ["штрафы/неустойки/санкции (если есть)"],
  "actions": ["рекомендуемые шаги/проверки"],
  "language": "ru|en"
}}

Параметры:
- Уровень детализации: {detail_label}
- Язык ответа: {language_name}
- Номер части: {chunk_number} из {total_chunks}

Текст части (используй фактические формулировки и цитаты где уместно):
"""

AGGREGATION_PROMPT = """
Ты — помощник юриста. Ниже — список JSON-саммари частей документа.
Объедини их в единый итог, УДАЛИ ДУБЛИКАТЫ и СВЕДИ СХОЖИЕ ПУНКТЫ. Верни только JSON.

Требуемая структура:
{{
  "summary": "целостный обзор на 4-6 предложений",
  "key_points": ["ключевые положения без повторов"],
  "deadlines": ["ключевые сроки"],
  "penalties": ["штрафы/санкции"],
  "actions": ["контрольный чек-лист"],
  "language": "ru|en"
}}

Язык: {language_name}
Детализация: {detail_label}

Данные для объединения (JSON-список):
{chunk_payload}
"""

# ---------------------------------- UI-тексты ----------------------------------

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

# --------------------------------- Датаклассы ---------------------------------


@dataclass
class SummaryStruct:
    summary: str
    key_points: List[str]
    deadlines: List[str]
    penalties: List[str]
    actions: List[str]
    language: str


# --------------------------------- Саммаризатор ---------------------------------


class DocumentSummarizer(DocumentProcessor):
    """Создает структурированное саммари юридических документов"""

    def __init__(self, openai_service=None, settings: AppSettings | None = None):
        super().__init__(name="DocumentSummarizer", max_file_size=50 * 1024 * 1024)
        self.openai_service = openai_service
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]

        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings

        self.allow_ai = settings.get_bool("SUMMARY_ALLOW_AI", True)
        self.max_key_items = settings.get_int("SUMMARY_MAX_ITEMS", 12)
        # Увеличиваем дефолтный размер чанка до 10k символов, чтобы сокращать число запросов
        self.chunk_size = settings.get_int("SUMMARY_CHUNK_SIZE", 4000)
        self.chunk_overlap = settings.get_int("SUMMARY_CHUNK_OVERLAP", 400)

    # ------------------------------- Публичный API -------------------------------

    async def process(
        self,
        file_path: str | Path,
        detail_level: str = "detailed",
        language: str = "ru",
        output_formats: list[str] | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        **_: Any,
    ) -> DocumentResult:
        async def _notify(stage: str, percent: float, **payload: Any) -> None:
            if not progress_callback:
                return
            data: dict[str, Any] = {"stage": stage, "percent": float(percent)}
            for key, value in payload.items():
                if value is None:
                    continue
                data[key] = value
            try:
                await progress_callback(data)
            except Exception:
                logger.debug("Summary progress callback failed at %s", stage, exc_info=True)

        if not self.allow_ai or not self.openai_service:
            # оффлайн режим недопустим? — деградируем gracefully
            if not self.openai_service:
                logger.warning("OpenAI service is not initialized; using local summary heuristics")
        normalized_detail = self._normalize_detail_level(detail_level)

        success, extracted = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {extracted}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(extracted)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        normalized_lang = self._normalize_language(language, cleaned_text)
        metadata = self._collect_metadata(cleaned_text, file_path, normalized_lang)
        words_count = len(cleaned_text.split())

        await _notify("text_extracted", 12, words=words_count)
        await _notify(
            "metadata_collected",
            25,
            language=metadata.get("language_display") or metadata.get("language"),
            source_length=metadata.get("source_length_chars"),
        )

        # эвристики
        deadlines_heuristic = self._extract_deadlines(cleaned_text)
        penalties_heuristic = self._extract_penalties(cleaned_text)
        await _notify(
            "heuristics",
            40,
            deadlines=len(deadlines_heuristic),
            penalties=len(penalties_heuristic),
        )

        # основное саммари
        try:
            await _notify("summary_generation", 55)
            if self.allow_ai and self.openai_service:
                summary_payload = await self._create_summary_llm(
                    cleaned_text,
                    detail_level=normalized_detail,
                    language=normalized_lang,
                )
            else:
                summary_payload = await self._create_summary_local(
                    cleaned_text,
                    detail_level=normalized_detail,
                    language=normalized_lang,
                )
        except ProcessingError:
            raise
        except Exception as exc:
            logger.exception("Не удалось подготовить саммари: %s", exc)
            raise ProcessingError(f"Ошибка саммари: {exc}", "SUMMARY_ERROR")

        summary_data: Dict[str, Any] = summary_payload["summary"]
        await _notify(
            "summary_ready",
            75,
            key_points=len(summary_data.get("key_points") or []),
            actions=len(summary_data.get("actions") or []),
        )

        # подмешиваем эвристику, если модель не дала
        if not summary_data.get("deadlines") and deadlines_heuristic:
            summary_data["deadlines"] = deadlines_heuristic
        if not summary_data.get("penalties") and penalties_heuristic:
            summary_data["penalties"] = penalties_heuristic
        if not summary_data.get("actions"):
            summary_data["actions"] = self._build_checklist_from_metadata(
                summary_data.get("deadlines", []),
                summary_data.get("penalties", []),
            )

        # финальный вид для вывода
        summary_text = self._format_summary_text(summary_data, normalized_lang)
        await _notify("finalizing", 90, checklist=len(summary_data.get("actions") or []))

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

        await _notify("completed", 100, key_points=len(summary_data.get("key_points") or []), deadlines=len(summary_data.get("deadlines") or []), penalties=len(summary_data.get("penalties") or []))

        return DocumentResult.success_result(data=result_payload, message="Саммари подготовлено")

    # ------------------------ LLM / локальная агрегация ------------------------

    async def _create_summary_llm(self, text: str, detail_level: str, language: str) -> dict[str, Any]:
        chunks = TextProcessor.split_into_chunks(text, max_chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_results: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_result = await self._summarize_chunk_llm(
                chunk,
                chunk_number=idx,
                total_chunks=len(chunks),
                detail_level=detail_level,
                language=language,
            )
            chunk_results.append(chunk_result)

        if len(chunk_results) == 1:
            structured = chunk_results[0]["summary"]
            return {"summary": structured, "processing_info": {"chunks_processed": 1, "strategy": "single"}}

        aggregated = await self._aggregate_chunks_llm(chunk_results, detail_level, language)
        return {"summary": aggregated, "processing_info": {"chunks_processed": len(chunk_results), "strategy": "aggregate"}}

    async def _create_summary_local(self, text: str, detail_level: str, language: str) -> dict[str, Any]:
        """Полностью локальная, безопасная версия саммари (без обращения к API)."""
        struct = self._local_summarize(text, detail_level=detail_level, language=language)
        return {"summary": struct.__dict__, "processing_info": {"chunks_processed": 1, "strategy": "local"}}

    async def _summarize_chunk_llm(
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

        response = await self._ask_llm(system="Ты структурируешь юридические тексты.", user=user_message)
        parsed = self._parse_summary_response(response, language)
        parsed["language"] = parsed.get("language") or language
        parsed = self._trim_struct(parsed)
        return {"summary": parsed, "raw_response": response}

    async def _aggregate_chunks_llm(
        self,
        chunk_results: list[dict[str, Any]],
        detail_level: str,
        language: str,
    ) -> dict[str, Any]:
        language_name = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["ru"])["prompt"]
        detail_label = DETAIL_LEVELS[detail_level]["ru" if language == "ru" else "en"]
        payload = json.dumps([item["summary"] for item in chunk_results], ensure_ascii=False)

        response = await self._ask_llm(
            system="Ты объединяешь юридические саммари в единый отчёт.",
            user=AGGREGATION_PROMPT.format(
                language_name=language_name,
                detail_label=detail_label,
                chunk_payload=payload,
            ),
        )
        parsed = self._parse_summary_response(response, language)
        if not parsed.get("summary"):
            logger.warning("Агрегация через модель не удалась, объединяем локально")
            parsed = self._merge_chunks_locally([item["summary"] for item in chunk_results])
        parsed["language"] = parsed.get("language") or language
        parsed = self._trim_struct(parsed)
        return parsed

    async def _ask_llm(self, *, system: str, user: str) -> str:
        """Тонкая обёртка над openai_service.ask_legal с единым поведением."""
        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")
        resp = await self.openai_service.ask_legal(system_prompt=system, user_text=user)
        if not resp or not resp.get("ok"):
            raise ProcessingError("Модель не вернула результат", "SUMMARY_MODEL_ERROR")
        return resp.get("text", "") or ""

    # ----------------------------- Парсинг ответа -----------------------------

    def _parse_summary_response(self, raw_text: str, language: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return self._empty_struct(language)

        data = self._safe_json_loads(text)
        if isinstance(data, dict) and data:
            return {
                "summary": str(data.get("summary", "")).strip(),
                "key_points": self._ensure_list_of_strings(data.get("key_points")),
                "deadlines": self._ensure_list_of_strings(data.get("deadlines")),
                "penalties": self._ensure_list_of_strings(data.get("penalties")),
                "actions": self._ensure_list_of_strings(data.get("actions")),
                "language": data.get("language", language),
            }

        # если JSON не получился — кладём сырой текст в summary
        return {
            "summary": text[:2000],
            "key_points": [],
            "deadlines": [],
            "penalties": [],
            "actions": [],
            "language": language,
        }

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        """Извлекаем JSON из «болтливого» текста: сначала прямой loads, затем по скобкам."""
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            # вырежем код-блоки ```
            raw2 = re.sub(r"^```.*?```$", "", raw, flags=re.DOTALL | re.MULTILINE)
            i = raw2.find("{")
            j = raw2.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(raw2[i : j + 1])
        except Exception:
            return {}
        return {}

    # -------------------------- Локальное саммари (оффлайн) --------------------------

    def _local_summarize(self, text: str, *, detail_level: str, language: str) -> SummaryStruct:
        # 1) резюме — первые 3–6 предложений, длинные и информативные
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pick = 6 if detail_level == "detailed" else 3
        summary = " ".join(s.strip() for s in sentences[:pick] if s.strip())[:1500]

        # 2) ключевые положения — заголовки/пункты/строки с двоеточиями или маркерами
        bullets = []
        for line in text.splitlines():
            l = line.strip()
            if not l:
                continue
            if re.match(r"^(\d+[.)]|[-–—*•])\s+", l) or (":" in l and len(l) < 200):
                bullets.append(l)
            if len(bullets) >= self.max_key_items:
                break

        # 3) дедлайны/штрафы
        deadlines = self._extract_deadlines(text, limit=8 if detail_level == "detailed" else 5)
        penalties = self._extract_penalties(text, limit=8 if detail_level == "detailed" else 5)

        # 4) действия
        actions = []
        if deadlines:
            actions.append("Проверить соблюдение всех указанных сроков")
        if penalties:
            actions.append("Проверить корректность расчёта штрафов/неустоек")
        actions.append("Согласовать итоговое саммари с ответственным юристом")

        return SummaryStruct(
            summary=summary,
            key_points=self._dedup_list(bullets)[: self.max_key_items],
            deadlines=self._dedup_list(deadlines)[: self.max_key_items],
            penalties=self._dedup_list(penalties)[: self.max_key_items],
            actions=self._dedup_list(actions)[: self.max_key_items],
            language=language,
        )

    # ------------------------------ Аггрегация локально ------------------------------

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

        return {
            "summary": "\n\n".join(self._dedup_list(summary_parts))[:2000],
            "key_points": self._dedup_list(key_points)[: self.max_key_items],
            "deadlines": self._dedup_list(deadlines)[: self.max_key_items],
            "penalties": self._dedup_list(penalties)[: self.max_key_items],
            "actions": self._dedup_list(actions)[: self.max_key_items],
        }

    # ------------------------------ Форматирование ------------------------------

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

    # ------------------------------- Хелперы -------------------------------

    def _trim_struct(self, data: dict[str, Any]) -> dict[str, Any]:
        """Обрезаем списки до лимита и чистим пустяки."""
        def _clean_list(values: list[str]) -> list[str]:
            return [v.strip() for v in values if isinstance(v, str) and v.strip()]

        data["key_points"] = _clean_list(data.get("key_points", []))[: self.max_key_items]
        data["deadlines"] = _clean_list(data.get("deadlines", []))[: self.max_key_items]
        data["penalties"] = _clean_list(data.get("penalties", []))[: self.max_key_items]
        data["actions"] = _clean_list(data.get("actions", []))[: self.max_key_items]
        data["summary"] = (data.get("summary") or "").strip()[:2000]
        return data

    @staticmethod
    def _dedup_list(items: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for entry in items:
            norm = entry.strip()
            if norm and norm not in seen:
                seen[norm] = None
        return list(seen.keys())

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

    def _empty_struct(self, language: str) -> dict[str, Any]:
        return {"summary": "", "key_points": [], "deadlines": [], "penalties": [], "actions": [], "language": language}


__all__ = ["DocumentSummarizer"]
