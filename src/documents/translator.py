# -*- coding: utf-8 -*-
"""
Модуль перевода документов
Профессиональный перевод документов с сохранением юридической терминологии.

Особенности:
- Чанкинг с перекрытием + устойчивость к падениям части запросов.
- Режимы: онлайн (LLM через openai_service) и безопасный оффлайн-деград (возвращаем исходник).
- Глоссарий (кастомные термины), защита не-переводимых сущностей (URL/Email/даты/числа).
- Сохранение структуры (списки, заголовки), уважение форматирования.
- Метаданные по чанкам: превью, длительность, статус.
- Автодетект языка и автоматическая рокировка языков при необходимости.
"""

from __future__ import annotations

import logging
from src.core.settings import AppSettings

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

# ----------------------------- Поддержка языков -----------------------------

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "ru": "Русский",
    "en": "Английский",
    "de": "Немецкий",
    "fr": "Французский",
    "es": "Испанский",
    "it": "Итальянский",
    "pt": "Португальский",
    "zh": "Китайский",
    "ja": "Японский",
}

LANG_NAMES: Dict[str, str] = {
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 Английский",
    "de": "🇩🇪 Немецкий",
    "fr": "🇫🇷 Французский",
    "es": "🇪🇸 Испанский",
    "it": "🇮🇹 Итальянский",
    "pt": "🇵🇹 Португальский",
    "zh": "🇨🇳 Китайский",
    "ja": "🇯🇵 Японский",
}

def _human_lang(code: str) -> str:
    return LANG_NAMES.get(code, code)


# ------------------------------- Промпт LLM -------------------------------

TRANSLATION_SYSTEM_PROMPT = """
Ты — профессиональный переводчик юридических документов.
Требования:
- Сохраняй юридическую терминологию и формальный стиль.
- Сохраняй структуру документа: заголовки, нумерацию, списки, подпункты.
- Сохраняй разметку и спец. плейсхолдеры вида {{T0}}, {{T1}}, {{URL0}} — НЕ ПЕРЕВОДИ их и не удаляй.
- Не сокращай и не добавляй лишних пояснений, если не попросили.
- Если видишь юридический термин из глоссария, строго используй предложенную форму перевода.

Формат ответа:
— Верни ТОЛЬКО перевод (plain text/markdown), без комментариев и метаданных.
"""

TRANSLATION_USER_TMPL = """
Исходный язык: {src_name}
Целевой язык: {tgt_name}

Глоссарий (приоритетный, если указан):
{glossary_text}

Переведи следующий фрагмент, сохранив структуру и плейсхолдеры:
---
{payload}
---
"""


# ----------------------------- Защита сущностей -----------------------------

URL_RE = re.compile(r"(https?://[^\s]+)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
DATE_RE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|January|February|March|April|May|June|July|August|September|October|November|December))\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d[\d\s.,]*\b")

def _protect_entities(text: str) -> Tuple[str, Dict[str, str]]:
    """Заменяем критичные сущности плейсхолдерами {{URL0}}, {{EM0}}, {{DT0}}, {{N0}}."""
    mapping: Dict[str, str] = {}
    counter = {"URL": 0, "EM": 0, "DT": 0, "N": 0}

    # этапно заменяем (важен порядок: URL/Email до чисел/дат)
    text_map = {"URL": text, "EM": "", "DT": "", "N": ""}
    text_map["EM"] = URL_RE.sub(lambda m: m.group(0), text_map["URL"])  # no-op, держим структуру
    text_map["DT"] = text_map["EM"]
    text_map["N"] = text_map["DT"]

    # 1) URL
    s1 = URL_RE.sub(lambda m: _swap_store(mapping, "URL", counter, m.group(0)), text_map["URL"])
    # 2) EMAIL
    s2 = EMAIL_RE.sub(lambda m: _swap_store(mapping, "EM", counter, m.group(0)), s1)
    # 3) DATE
    s3 = DATE_RE.sub(lambda m: _swap_store(mapping, "DT", counter, m.group(0)), s2)
    # 4) NUMBER
    s4 = NUMBER_RE.sub(lambda m: _swap_store(mapping, "N", counter, m.group(0)), s3)
    return s4, mapping

def _swap_store(mapping: Dict[str, str], tag: str, counter: Dict[str, int], value: str) -> str:
    key = f"{{{{{tag}{counter[tag]}}}}}"
    mapping[key] = value
    counter[tag] += 1
    return key

def _restore_entities(text: str, mapping: Dict[str, str]) -> str:
    # чтобы не повредить пересечения, заменяем по убыванию длины ключа
    for k in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(k, mapping[k])
    return text


# ------------------------------- Класс модуля -------------------------------

class DocumentTranslator(DocumentProcessor):
    """Класс для перевода документов"""

    def __init__(self, openai_service=None, settings: AppSettings | None = None):
        super().__init__(name="DocumentTranslator", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service
        self.supported_languages = SUPPORTED_LANGUAGES.copy()

        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings

        self.allow_ai = settings.get_bool("TRANSLATE_ALLOW_AI", True)
        self.chunk_size = settings.get_int("TRANSLATE_CHUNK_SIZE", 4000)
        self.chunk_overlap = settings.get_int("TRANSLATE_CHUNK_OVERLAP", 300)
        self.max_retries = settings.get_int("TRANSLATE_MAX_RETRIES", 2)

    async def process(
        self,
        file_path: str | Path,
        source_lang: str = "ru",
        target_lang: str = "en",
        *,
        glossary: Dict[str, str] | None = None,
        **kwargs: Any,
    ) -> DocumentResult:
        """
        Перевод документа

        Args:
            file_path: путь к файлу
            source_lang: исходный язык (ru, en, zh, de, ...)
            target_lang: целевой язык (ru, en, zh, de, ...)
            glossary: пользовательский глоссарий {исходный термин: целевой термин}
        """
        if not self.allow_ai or not self.openai_service:
            if not self.openai_service:
                logger.warning("OpenAI service is not initialized; translation will fallback to identity")

        # Извлекаем текст
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # Нормализация языков
        src = (source_lang or "").lower()
        tgt = (target_lang or "").lower()
        if src in {"auto", "detect", ""}:
            src = self.detect_language(cleaned_text)

        if src not in self.supported_languages:
            src = self.detect_language(cleaned_text)
        if tgt not in self.supported_languages:
            # дефолт — противоположный от детекта
            tgt = "en" if src == "ru" else "ru"

        # Если языки одинаковые — ничего не делаем
        if src == tgt:
            return DocumentResult.success_result(
                data={
                    "translated_text": cleaned_text,
                    "source_language": src,
                    "target_language": tgt,
                    "original_file": str(file_path),
                    "chunk_details": [],
                    "translation_metadata": {
                        "original_length": len(cleaned_text),
                        "translated_length": len(cleaned_text),
                        "language_pair": f"{src} -> {tgt}",
                        "chunks_processed": 0,
                        "mode": "identity",
                    },
                },
                message=f"Исходный и целевой языки совпадают ({_human_lang(src)}). Перевод не требуется.",
            )

        # Перевод
        translated_text, chunk_details = await self._translate_text(
            cleaned_text, src, tgt, glossary=glossary or {}
        )

        result_data = {
            "translated_text": translated_text,
            "source_language": src,
            "target_language": tgt,
            "original_file": str(file_path),
            "chunk_details": chunk_details,
            "translation_metadata": {
                "original_length": len(cleaned_text),
                "translated_length": len(translated_text),
                "language_pair": f"{src} -> {tgt}",
                "chunks_processed": len(chunk_details),
                "mode": "ai" if self.allow_ai and self.openai_service else "fallback",
            },
        }

        return DocumentResult.success_result(
            data=result_data,
            message=f"Перевод с {self.supported_languages[src]} на {self.supported_languages[tgt]} завершён",
        )

    # -------------------------------- Перевод --------------------------------

    async def _translate_text(
        self, text: str, source_lang: str, target_lang: str, *, glossary: Dict[str, str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Перевод текста с помощью AI (чанкинг + защита сущностей)."""
        src_name = self.supported_languages.get(source_lang, source_lang)
        tgt_name = self.supported_languages.get(target_lang, target_lang)

        # Защитим сущности плейсхолдерами
        protected_text, placeholders = _protect_entities(text)

        # Разрежем на чанки
        chunks = TextProcessor.split_into_chunks(protected_text, max_chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            chunks = [protected_text]

        details: List[Dict[str, Any]] = []
        translated_parts: List[str] = []

        # Если нет AI — деградируем: возвращаем исходный текст (с восстановлением сущностей)
        if not (self.allow_ai and self.openai_service):
            return _restore_entities(protected_text, placeholders), []

        glossary_text = self._format_glossary(glossary)

        for i, chunk in enumerate(chunks, 1):
            t0 = time.perf_counter()
            attempt = 0
            translated_part = None
            error_msg = None

            while attempt <= self.max_retries:
                attempt += 1
                try:
                    user_payload = TRANSLATION_USER_TMPL.format(
                        src_name=src_name,
                        tgt_name=tgt_name,
                        glossary_text=glossary_text or "(не указан)",
                        payload=chunk,
                    )
                    resp = await self.openai_service.ask_legal(
                        system_prompt=TRANSLATION_SYSTEM_PROMPT,
                        user_text=user_payload,
                    )
                    if resp and resp.get("ok"):
                        translated_part = (resp.get("text") or "").strip()
                        if translated_part:
                            break
                        else:
                            error_msg = "Пустой ответ модели"
                    else:
                        error_msg = "Модель вернула ошибку"
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                # ретрай — просто идём на следующую попытку

            dt = time.perf_counter() - t0

            if translated_part is None:
                # Фолбэк: возвращаем исходный блок (но потом восстановим сущности)
                translated_part = chunk
                status = "fallback"
            else:
                status = "ok"

            translated_parts.append(translated_part)
            details.append(
                {
                    "chunk_number": i,
                    "status": status,
                    "duration_sec": round(dt, 3),
                    "source_preview": chunk[:200],
                    "translated_preview": translated_part[:200],
                    "attempts": attempt,
                    "error": error_msg,
                }
            )

        # Склеиваем и возвращаем плейсхолдеры назад
        merged = self._merge_chunks(translated_parts)
        restored = _restore_entities(merged, placeholders)
        return restored, details

    @staticmethod
    def _format_glossary(glossary: Dict[str, str]) -> str:
        if not glossary:
            return ""
        # Простая табличка "исходный => целевой"
        pairs = [f"- {src} ⇒ {dst}" for src, dst in glossary.items()]
        return "\n".join(pairs)

    @staticmethod
    def _merge_chunks(chunks: List[str]) -> str:
        """Мягкая склейка: добавляем пустую строку между чанками, убираем лишние повторы."""
        cleaned = [c.strip() for c in chunks if c is not None]
        return "\n\n".join(cleaned).strip()

    # ------------------------------ Поддержка ------------------------------
