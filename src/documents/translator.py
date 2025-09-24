"""
Модуль перевода документов
Профессиональный перевод документов с сохранением юридической терминологии
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = """
Ты — профессиональный переводчик юридических документов.

ЗАДАЧА: Переведи документ с {source_lang} на {target_lang}, сохраняя:
- Юридическую терминологию
- Структуру документа
- Формальный стиль
- Точность правовых понятий

ОСОБЫЕ ТРЕБОВАНИЯ:
- Используй принятые переводы юридических терминов
- Сохраняй форматирование (списки, пункты, таблицы)
- Адаптируй к местному законодательству где необходимо
- Добавляй пояснения в скобках для сложных терминов

ЯЗЫКОВЫЕ ПАРЫ:
- Русский ↔ Английский
- Русский ↔ Китайский
- Русский ↔ Немецкий

Переводи только текст документа, сохраняя его структуру.
"""

class DocumentTranslator(DocumentProcessor):
    """Класс для перевода документов"""

    def __init__(self, openai_service=None):
        super().__init__(
            name="DocumentTranslator",
            max_file_size=50 * 1024 * 1024
        )
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        self.openai_service = openai_service

        self.supported_languages = {
            "ru": "русский",
            "en": "английский",
            "zh": "китайский",
            "de": "немецкий"
        }

    async def process(self, file_path: Union[str, Path], source_lang: str = "ru", target_lang: str = "en", **kwargs) -> DocumentResult:
        """
        Перевод документа

        Args:
            file_path: путь к файлу
            source_lang: исходный язык (ru, en, zh, de)
            target_lang: целевой язык (ru, en, zh, de)
        """

        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            raise ProcessingError("Неподдерживаемая языковая пара", "LANGUAGE_ERROR")

        # Извлекаем текст из файла
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # Переводим документ
        translated_text = await self._translate_text(
            cleaned_text,
            source_lang,
            target_lang
        )

        result_data = {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "original_file": str(file_path),
            "translation_metadata": {
                "original_length": len(cleaned_text),
                "translated_length": len(translated_text),
                "language_pair": f"{source_lang} -> {target_lang}"
            }
        }

        return DocumentResult.success_result(
            data=result_data,
            message=f"Перевод с {self.supported_languages[source_lang]} на {self.supported_languages[target_lang]} завершен"
        )

    async def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Перевод текста с помощью AI"""

        source_lang_name = self.supported_languages[source_lang]
        target_lang_name = self.supported_languages[target_lang]

        prompt = TRANSLATION_PROMPT.format(
            source_lang=source_lang_name,
            target_lang=target_lang_name
        )

        try:
            # Если текст слишком длинный, переводим частями
            if len(text) > 6000:
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=4000)
                translated_chunks = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"Переводим часть {i+1}/{len(chunks)}")

                    chunk_prompt = prompt + f"\n\nЧасть {i+1} из {len(chunks)} документа:\n\n{chunk}"

                    result = await self.openai_service.ask_legal(
                        system_prompt=prompt,
                        user_message=chunk
                    )

                    if result.get("ok"):
                        translated_chunks.append(result.get("text", ""))
                    else:
                        raise ProcessingError(f"Ошибка перевода части {i+1}", "TRANSLATION_ERROR")

                return "\n\n".join(translated_chunks)

            else:
                # Переводим документ целиком
                result = await self.openai_service.ask_legal(
                    system_prompt=prompt,
                    user_message=text
                )

                if result.get("ok"):
                    return result.get("text", "")
                else:
                    raise ProcessingError("Не удалось перевести документ", "TRANSLATION_ERROR")

        except Exception as e:
            logger.error(f"Ошибка перевода: {e}")
            raise ProcessingError(f"Ошибка перевода: {str(e)}", "TRANSLATION_ERROR")

    def get_supported_languages(self) -> Dict[str, str]:
        """Получить список поддерживаемых языков"""
        return self.supported_languages.copy()

    def detect_language(self, text: str) -> str:
        """Простое определение языка текста"""
        # Очень простая эвристика - можно улучшить
        if re.search(r'[а-яё]', text.lower()):
            return "ru"
        elif re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        elif re.search(r'[äöüß]', text.lower()):
            return "de"
        else:
            return "en"  # по умолчанию