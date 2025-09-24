
"""
Модуль саммаризации документов
Создание краткой выжимки из больших текстовых документов с выделением ключевых элементов
"""

from __future__ import annotations
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

LANGUAGE_INSTRUCTIONS = {
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

Отвечай на русском языке, используй Markdown форматирование.
"""

class DocumentSummarizer(DocumentProcessor):
    """Класс для саммаризации документов"""

    def __init__(self, openai_service=None):
        super().__init__(
            name="DocumentSummarizer",
            max_file_size=50 * 1024 * 1024  # 50MB
        )
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        self.openai_service = openai_service

    async def process(self, file_path: Union[str, Path], detail_level: str = "detailed", language: str = "ru", **kwargs) -> DocumentResult:
        """
        Основной метод саммаризации документа

        Args:
            file_path: путь к файлу
            detail_level: уровень детализации ("brief" или "detailed")
            **kwargs: дополнительные параметры
        """

        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

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
        result_data = {
            "summary": summary_data,
            "structured_data": structured_data,
            "metadata": metadata,
            "original_file": str(file_path),
            "detail_level": detail_level,
            "language": language_code
        }

        return DocumentResult.success_result(
            data=result_data,
            message="Саммаризация документа успешно завершена"
        )

    async def _create_summary(self, text: str, detail_level: str, language: str) -> Dict[str, Any]:
        """Создать саммаризацию с помощью OpenAI"""

        language_code = (language or "ru").lower()
        if language_code not in LANGUAGE_INSTRUCTIONS:
            language_code = "ru"

        if detail_level == "brief":
            detail_instruction = "Дай КРАТКУЮ выжимку (до 500 слов)."
        else:
            detail_instruction = "Дай ПОДРОБНУЮ выжимку (до 1500 слов)."

        prompt = "
".join([
            DOCUMENT_SUMMARIZATION_PROMPT.strip(),
            LANGUAGE_INSTRUCTIONS[language_code],
            detail_instruction,
        ])

        try:
            # Если текст слишком длинный, разбиваем на части
            if len(text) > 8000:  # Оставляем запас для промпта
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=6000)
                summaries = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"Обрабатываю часть {i+1}/{len(chunks)}")
                    chunk_prompt = prompt + f"\n\nЧасть {i+1} из {len(chunks)}:\n\n{chunk}"

                    result = await self.openai_service.ask_legal(
                        system_prompt=prompt,
                        user_message=chunk_prompt
                    )

                    if result.get("ok"):
                        summaries.append(result.get("text", ""))

                if summaries:
                    # Объединяем саммаризации частей
                    combined_text = "\n\n---\n\n".join(summaries)
                    final_prompt = """
                    Объедини следующие саммаризации частей документа в единую структурированную выжимку:

                    """ + combined_text

                    final_result = await self.openai_service.ask_legal(
                        system_prompt=prompt,
                        user_message=final_prompt
                    )

                    if final_result.get("ok"):
                        return {
                            "content": final_result.get("text", ""),
                            "method": "chunked_processing",
                            "chunks_processed": len(chunks)
                        }
            else:
                # Обрабатываем документ целиком
                result = await self.openai_service.ask_legal(
                    system_prompt=prompt,
                    user_message=text
                )

                if result.get("ok"):
                    return {
                        "content": result.get("text", ""),
                        "method": "single_processing",
                        "chunks_processed": 1
                    }

            raise ProcessingError("Не удалось создать саммаризацию", "OPENAI_ERROR")

        except Exception as e:
            logger.error(f"Ошибка при создании саммаризации: {e}")
            raise ProcessingError(f"Ошибка OpenAI: {str(e)}", "OPENAI_ERROR")

    def _extract_structured_data(self, text: str) -> Dict[str, List[str]]:
        """Извлечь структурированные данные из текста"""

        structured_data = {
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "phone_numbers": self._extract_phone_numbers(text),
            "emails": self._extract_emails(text),
            "articles": self._extract_legal_articles(text),
            "penalties": self._extract_penalties(text)
        }

        return structured_data

    def _extract_dates(self, text: str) -> List[str]:
        """Извлечь даты из текста"""
        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # ДД.ММ.ГГГГ
            r'\d{1,2}\.\d{1,2}\.\d{2}',  # ДД.ММ.ГГ
            r'\d{1,2}/\d{1,2}/\d{4}',   # ДД/ММ/ГГГГ
            r'\d{4}-\d{2}-\d{2}',       # ГГГГ-ММ-ДД
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return list(set(dates))  # Убираем дубликаты

    def _extract_amounts(self, text: str) -> List[str]:
        """Извлечь денежные суммы из текста"""
        amount_patterns = [
            r'\d+\s*(?:руб|рубл|₽)',
            r'\d+\s*(?:тыс|млн|млрд)\s*(?:руб|рубл|₽)',
            r'\d+[.,]\d+\s*(?:руб|рубл|₽)',
            r'\d+\s+\d+\s*(?:руб|рубл|₽)',  # числа с пробелами
        ]

        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)

        return amounts

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Извлечь номера телефонов из текста"""
        phone_patterns = [
            r'\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}',
            r'\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}',
            r'8\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}',
            r'8\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}',
        ]

        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)

        return phones

    def _extract_emails(self, text: str) -> List[str]:
        """Извлечь email адреса из текста"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)

    def _extract_legal_articles(self, text: str) -> List[str]:
        """Извлечь ссылки на статьи законов"""
        article_patterns = [
            r'ст\.\s*\d+',
            r'статья\s*\d+',
            r'статье\s*\d+',
            r'статьи\s*\d+',
            r'п\.\s*\d+\s*ст\.\s*\d+',
            r'пункт\s*\d+\s*статьи\s*\d+',
        ]

        articles = []
        for pattern in article_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            articles.extend(matches)

        return articles

    def _extract_penalties(self, text: str) -> List[str]:
        """Извлечь информацию о штрафах и неустойках"""
        penalty_patterns = [
            r'штраф.*?\d+.*?(?:руб|рубл|₽)',
            r'неустойка.*?\d+.*?(?:руб|рубл|₽|%)',
            r'пеня.*?\d+.*?(?:руб|рубл|₽|%)',
            r'санкции.*?\d+.*?(?:руб|рубл|₽|%)',
        ]

        penalties = []
        for pattern in penalty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            penalties.extend([match.strip() for match in matches])

        return penalties

    async def create_checklist(self, summary_data: Dict[str, Any]) -> List[str]:
        """Создать контрольный чек-лист на основе саммаризации"""

        checklist_items = []

        # Базовые пункты контроля
        checklist_items.extend([
            "Проверить правильность реквизитов всех сторон",
            "Убедиться в корректности предмета договора",
            "Проверить все указанные суммы и расчеты"
        ])

        # Добавляем пункты на основе найденных дат
        if summary_data.get("structured_data", {}).get("dates"):
            checklist_items.append("Отслеживать все указанные в договоре даты")

        # Добавляем пункты на основе найденных штрафов
        if summary_data.get("structured_data", {}).get("penalties"):
            checklist_items.append("Контролировать условия применения штрафов и неустоек")

        return checklist_items