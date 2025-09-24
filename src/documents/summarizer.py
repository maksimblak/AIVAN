"""
Модуль саммаризации документов
Создание краткой выжимки из больших текстовых документов с выделением ключевых элементов.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "ru": "Отвечай на русском языке, используй Markdown форматирование.",
    "en": "Respond in English, using Markdown formatting.",
}

DOCUMENT_SUMMARIZATION_PROMPT = """
Ты — эксперт по анализу и саммаризации юридических документов.

Твоя задача — создать структурированную выжимку из предоставленного документа.

СТРУКТУРА ВЫЖИМКИ:

1. **Краткое резюме** (2-3 предложения)
   - Суть документа
   - Основная цель

2. **Ключевые положения**
   - Основные пункты и условия
   - Права и обязанности сторон
   - Важные определения

3. **Дедлайны и временные рамки**
   - Все упомянутые даты
   - Сроки исполнения
   - Периоды действия

4. **Штрафы и санкции**
   - Неустойки
   - Пени
   - Ответственность за нарушения

5. **Финансовые условия**
   - Суммы платежей
   - Тарифы и расценки
   - Порядок расчетов

6. **Контрольные пункты**
   - Чек-лист для отслеживания исполнения
   - Ключевые моменты для контроля
   - Критические требования

ТРЕБОВАНИЯ К ВЫЖИМКЕ:
- Используй только информацию из документа
- Сохраняй точные формулировки для важных пунктов
- Выделяй риски и подводные камни
- Структурируй информацию для удобного восприятия
- Добавляй ссылки на пункты документа где возможно
""".strip()


class DocumentSummarizer(DocumentProcessor):
    """Класс для саммаризации документов"""

    def __init__(self, openai_service=None):
        super().__init__(
            name="DocumentSummarizer",
            max_file_size=50 * 1024 * 1024,  # 50MB
        )
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service

    async def process(
        self,
        file_path: Union[str, Path],
        detail_level: str = "detailed",
        language: str = "ru",
        **kwargs: Any,
    ) -> DocumentResult:
        """
        Основной метод саммаризации документа.

        Args:
            file_path: путь к файлу
            detail_level: уровень детализации ("brief" или "detailed")
            language: "ru" или "en"
            **kwargs: дополнительные параметры
        """

        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        # Проверка формата файла заранее
        ext = str(file_path).lower()
        if not any(ext.endswith(suf) for suf in self.supported_formats):
            raise ProcessingError(
                f"Неподдерживаемый формат файла: {file_path}", "UNSUPPORTED_FORMAT"
            )

        # Извлекаем текст из файла
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        # Очищаем и обрабатываем текст
        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # Получаем метаданные документа
        metadata = TextProcessor.extract_metadata(cleaned_text)

        language_code = (language or "ru").lower()
        if language_code not in LANGUAGE_INSTRUCTIONS:
            language_code = "ru"

        # Создаем саммаризацию
        summary_data = await self._create_summary(cleaned_text, detail_level, language_code)

        # Извлекаем структурированные данные
        structured_data = self._extract_structured_data(cleaned_text)

        # Формируем результат
        result_data: Dict[str, Any] = {
            "summary": summary_data,
            "structured_data": structured_data,
            "metadata": metadata,
            "original_file": str(file_path),
            "detail_level": detail_level,
            "language": language_code,
        }

        return DocumentResult.success_result(
            data=result_data, message="Саммаризация документа успешно завершена"
        )

    # ---------- LLM helpers ----------

    def _build_system_prompt(self, detail_level: str, language_code: str) -> str:
        if detail_level == "brief":
            detail_instruction = "Дай КРАТКУЮ выжимку (до 500 слов)."
        else:
            detail_instruction = "Дай ПОДРОБНУЮ выжимку (до 1500 слов)."

        return "\n".join(
            [
                DOCUMENT_SUMMARIZATION_PROMPT,
                LANGUAGE_INSTRUCTIONS[language_code],
                detail_instruction,
                "Отвечай строго по указанной структуре.",
            ]
        )

    async def _create_summary(
        self, text: str, detail_level: str, language: str
    ) -> Dict[str, Any]:
        """Создать саммаризацию с помощью OpenAI"""
        language_code = (language or "ru").lower()
        if language_code not in LANGUAGE_INSTRUCTIONS:
            language_code = "ru"

        system_prompt = self._build_system_prompt(detail_level, language_code)

        try:
            # Консервативный порог по символам (лучше по токенам, если появится токенайзер)
            if len(text) > 6000:
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=4000)
                summaries: List[str] = []

                for i, chunk in enumerate(chunks, start=1):
                    logger.info(f"Обрабатываю часть {i}/{len(chunks)}")
                    user_message = (
                        f"Это часть {i} из {len(chunks)} исходного документа. "
                        f"Сделай выжимку ТОЛЬКО по этой части, соблюдай структуру, "
                        f"сохраняй точные формулировки важных пунктов.\n\n"
                        f"<DOCUMENT_PART_{i}>\n{chunk}\n</DOCUMENT_PART_{i}>"
                    )

                    result = await self.openai_service.ask_legal(
                        system_prompt=system_prompt, user_message=user_message
                    )

                    if result.get("ok"):
                        summaries.append(result.get("text", ""))

                if summaries:
                    combined_text = "\n\n---\n\n".join(summaries)
                    final_user_message = (
                        "Объедини частичные выжимки ниже в единую, устрани дубли, "
                        "соблюдай исходную структуру и язык.\n\n"
                        f"<PARTIAL_SUMMARIES>\n{combined_text}\n</PARTIAL_SUMMARIES>"
                    )

                    final_result = await self.openai_service.ask_legal(
                        system_prompt=system_prompt, user_message=final_user_message
                    )

                    if final_result.get("ok"):
                        return {
                            "content": final_result.get("text", ""),
                            "method": "chunked_processing",
                            "chunks_processed": len(chunks),
                        }
            else:
                # Обрабатываем документ целиком
                user_message = (
                    "Проанализируй документ ниже и создай выжимку по инструкции в system.\n\n"
                    f"<DOCUMENT>\n{text}\n</DOCUMENT>"
                )

                result = await self.openai_service.ask_legal(
                    system_prompt=system_prompt, user_message=user_message
                )

                if result.get("ok"):
                    return {
                        "content": result.get("text", ""),
                        "method": "single_processing",
                        "chunks_processed": 1,
                    }

            raise ProcessingError("Не удалось создать саммаризацию", "OPENAI_ERROR")

        except Exception as e:
            logger.error(f"Ошибка при создании саммаризации: {e}")
            raise ProcessingError(f"Ошибка OpenAI: {str(e)}", "OPENAI_ERROR")

    # ---------- Structured data extraction ----------

    def _extract_structured_data(self, text: str) -> Dict[str, List[str]]:
        """Извлечь структурированные данные из текста"""
        structured_data = {
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "phone_numbers": self._extract_phone_numbers(text),
            "emails": self._extract_emails(text),
            "articles": self._extract_legal_articles(text),
            "penalties": self._extract_penalties(text),
        }
        return structured_data

    def _extract_dates(self, text: str) -> List[str]:
        """Извлечь даты из текста с нормализацией к ISO (где возможно) и сохранением порядка."""
        patterns = [
            r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",  # 31.12.2024
            r"\b\d{1,2}\.\d{1,2}\.\d{2}\b",  # 31.12.24
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # 31/12/2024
            r"\b\d{4}-\d{2}-\d{2}\b",  # 2024-12-31
            r"\b\d{1,2}\s+[А-Яа-яA-Za-z]+\s+\d{4}\b",  # 31 декабря 2024
        ]

        seen = set()
        found: List[str] = []
        for p in patterns:
            for m in re.findall(p, text):
                if m not in seen:
                    seen.add(m)
                    found.append(m)

        iso: List[str] = []
        months = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }

        for d in found:
            try:
                if re.match(r"\d{4}-\d{2}-\d{2}$", d):
                    iso.append(d)
                elif re.match(r"\d{1,2}\.\d{1,2}\.\d{4}$", d):
                    iso.append(datetime.strptime(d, "%d.%m.%Y").strftime("%Y-%m-%d"))
                elif re.match(r"\d{1,2}\.\d{1,2}\.\d{2}$", d):
                    iso.append(datetime.strptime(d, "%d.%m.%y").strftime("%Y-%m-%d"))
                elif re.match(r"\d{1,2}/\d{1,2}/\d{4}$", d):
                    iso.append(datetime.strptime(d, "%d/%m/%Y").strftime("%Y-%m-%d"))
                else:
                    m = re.match(r"(\d{1,2})\s+([А-Яа-яA-Za-z]+)\s+(\d{4})$", d)
                    if m:
                        day, mon, year = m.groups()
                        mon_num = months.get(mon.lower())
                        if mon_num:
                            iso.append(f"{year}-{mon_num:02d}-{int(day):02d}")
                        else:
                            iso.append(d)
                    else:
                        iso.append(d)
            except Exception:
                iso.append(d)

        # Уник с сохранением порядка
        return list(dict.fromkeys(iso))

    def _extract_amounts(self, text: str) -> List[str]:
        """Извлечь денежные суммы из текста с дедупликацией."""
        amount_patterns = [
            r"\b\d[\d\s]*\s*(?:руб|рубл|₽)\b",
            r"\b\d[\d\s]*\s*(?:тыс|млн|млрд)\s*(?:руб|рубл|₽)\b",
            r"\b\d[\d\s]*[.,]\d+\s*(?:руб|рубл|₽)\b",
        ]

        amounts: List[str] = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)

        # Нормализация пробелов и дедуп по порядку
        norm = [re.sub(r"\s+", " ", m).strip() for m in amounts]
        return list(dict.fromkeys(norm))

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Извлечь номера телефонов из текста (российские форматы) с дедупликацией."""
        phone_patterns = [
            r"\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}",
            r"\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
            r"8\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}",
            r"8\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
        ]

        phones: List[str] = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)

        return list(dict.fromkeys([re.sub(r"\s+", " ", p).strip() for p in phones]))

    def _extract_emails(self, text: str) -> List[str]:
        """Извлечь email адреса из текста с дедупликацией."""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        return list(dict.fromkeys(emails))

    def _extract_legal_articles(self, text: str) -> List[str]:
        """Извлечь ссылки на статьи законов с дедупликацией."""
        article_patterns = [
            r"\bст\.\s*\d+\b",
            r"\bстатья\s*\d+\b",
            r"\bстатье\s*\d+\b",
            r"\bстатьи\s*\d+\b",
            r"\bп\.\s*\d+\s*ст\.\s*\d+\b",
            r"\bпункт\s*\d+\s*статьи\s*\d+\b",
        ]

        articles: List[str] = []
        for pattern in article_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            articles.extend(matches)

        norm = [re.sub(r"\s+", " ", a).strip() for a in articles]
        return list(dict.fromkeys(norm))

    def _extract_penalties(self, text: str) -> List[str]:
        """Извлечь информацию о штрафах и неустойках (ограничиваем «окно» до 200 символов)."""
        penalty_patterns = [
            r"штраф.{0,200}?\d[\d\s.,]*\s*(?:руб|рубл|₽|%)",
            r"неустойк[аи].{0,200}?\d[\d\s.,]*\s*(?:руб|рубл|₽|%)",
            r"пен[ия].{0,200}?\d[\d\s.,]*\s*(?:руб|рубл|₽|%)",
            r"санкци[ия].{0,200}?\d[\d\s.,]*\s*(?:руб|рубл|₽|%)",
        ]

        hits: List[str] = []
        for p in penalty_patterns:
            hits.extend(re.findall(p, text, flags=re.IGNORECASE))

        norm = [re.sub(r"\s+", " ", h).strip() for h in hits]
        return list(dict.fromkeys(norm))

    # ---------- Checklist ----------

    async def create_checklist(self, summary_data: Dict[str, Any]) -> List[str]:
        """
        Создать контрольный чек-лист на основе саммаризации/результатов.

        Примечание: метод совместим и с полным result_data (где есть 'structured_data'),
        и с объектом summary (тогда будут только базовые пункты).
        """
        checklist_items: List[str] = [
            "Проверить правильность реквизитов всех сторон",
            "Убедиться в корректности предмета договора",
            "Проверить все указанные суммы и расчеты",
        ]

        # Пытаемся достать structured_data, если это полный result_data
        structured = summary_data.get("structured_data") if isinstance(summary_data, dict) else None
        if not structured and isinstance(summary_data, dict):
            # иногда передают результат верхнего уровня: {"summary": ..., "structured_data": ...}
            structured = summary_data.get("structured_data")

        if isinstance(structured, dict):
            if structured.get("dates"):
                checklist_items.append("Отслеживать все указанные в договоре даты")
            if structured.get("penalties"):
                checklist_items.append("Контролировать условия применения штрафов и неустоек")

        return checklist_items
