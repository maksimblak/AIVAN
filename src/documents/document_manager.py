"""
Менеджер документов - центральный класс для управления всеми модулями работы с документами
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

from .base import DocumentStorage, ProcessingError, DocumentResult
from .summarizer import DocumentSummarizer
from .risk_analyzer import RiskAnalyzer
from .document_chat import DocumentChat
from .anonymizer import DocumentAnonymizer
from .translator import DocumentTranslator
from .ocr_converter import OCRConverter

logger = logging.getLogger(__name__)

class DocumentManager:
    """Центральный менеджер для работы с документами"""

    def __init__(self, openai_service=None, storage_path: str = "data/documents"):
        self.openai_service = openai_service
        self.storage = DocumentStorage(storage_path)

        # Инициализация модулей
        self.summarizer = DocumentSummarizer(openai_service)
        self.risk_analyzer = RiskAnalyzer(openai_service)
        self.document_chat = DocumentChat(openai_service)
        self.anonymizer = DocumentAnonymizer()
        self.translator = DocumentTranslator(openai_service)
        self.ocr_converter = OCRConverter()

    async def process_document(
        self,
        user_id: int,
        file_content: bytes,
        original_name: str,
        mime_type: str,
        operation: str,
        **kwargs
    ) -> DocumentResult:
        """
        Обработать документ указанным способом

        Args:
            user_id: ID пользователя
            file_content: содержимое файла
            original_name: оригинальное имя файла
            mime_type: MIME тип файла
            operation: тип операции ("summarize", "analyze_risks", "chat", "anonymize", "translate", "ocr")
            **kwargs: дополнительные параметры для операции
        """

        try:
            # Сохраняем файл
            document_info = await self.storage.save_document(
                user_id, file_content, original_name, mime_type
            )

            # Выбираем обработчик
            processor = self._get_processor(operation)
            if not processor:
                raise ProcessingError(f"Неизвестная операция: {operation}", "UNKNOWN_OPERATION")

            # Обрабатываем документ
            logger.info(f"Обрабатываем документ {original_name} операцией {operation} для пользователя {user_id}")
            result = await processor.safe_process(document_info.file_path, **kwargs)

            # Добавляем информацию о документе в результат
            if result.success:
                result.data["document_info"] = {
                    "original_name": document_info.original_name,
                    "file_size": document_info.size,
                    "upload_time": document_info.upload_time.isoformat(),
                    "user_id": user_id
                }

            # Очищаем временные файлы (опционально)
            self.storage.cleanup_old_files(user_id, max_age_hours=24)

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки документа {original_name}: {e}")
            raise ProcessingError(f"Ошибка обработки документа: {str(e)}", "PROCESSING_ERROR")

    def _get_processor(self, operation: str):
        """Получить процессор для указанной операции"""
        processors = {
            "summarize": self.summarizer,
            "analyze_risks": self.risk_analyzer,
            "chat": self.document_chat,
            "anonymize": self.anonymizer,
            "translate": self.translator,
            "ocr": self.ocr_converter
        }
        return processors.get(operation)

    async def chat_with_document(self, user_id: int, document_id: str, question: str) -> Dict[str, Any]:
        """Чат с загруженным документом"""
        try:
            return await self.document_chat.chat_with_document(document_id, question)
        except Exception as e:
            logger.error(f"Ошибка чата с документом {document_id}: {e}")
            raise ProcessingError(f"Ошибка чата: {str(e)}", "CHAT_ERROR")

    def get_supported_operations(self) -> Dict[str, Dict[str, Any]]:
        """Получить список поддерживаемых операций"""
        return {
            "summarize": {
                "name": "Саммаризация",
                "description": "Создание краткой выжимки документа с ключевыми положениями",
                "emoji": "📋",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["detail_level"]
            },
            "analyze_risks": {
                "name": "Анализ рисков",
                "description": "Выявление рисков и проблем в договорах и документах",
                "emoji": "⚠️",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["custom_criteria"]
            },
            "chat": {
                "name": "Чат с документом",
                "description": "Интерактивные вопросы-ответы по документу",
                "emoji": "💬",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": []
            },
            "anonymize": {
                "name": "Обезличивание",
                "description": "Удаление персональных данных из документа",
                "emoji": "🔒",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["anonymization_mode"]
            },
            "translate": {
                "name": "Перевод",
                "description": "Перевод документа на другие языки",
                "emoji": "🌍",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["source_lang", "target_lang"]
            },
            "ocr": {
                "name": "OCR распознавание",
                "description": "Распознавание текста из сканированных документов",
                "emoji": "👁️",
                "formats": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
                "parameters": ["output_format"]
            }
        }

    def get_operation_info(self, operation: str) -> Optional[Dict[str, Any]]:
        """Получить информацию об операции"""
        return self.get_supported_operations().get(operation)

    def format_result_for_telegram(self, result: DocumentResult, operation: str) -> str:
        """Форматировать результат для отправки в Telegram"""
        if not result.success:
            return f"❌ **Ошибка обработки**\n\n{result.message}"

        operation_info = self.get_operation_info(operation)
        emoji = operation_info.get("emoji", "📄") if operation_info else "📄"

        header = f"{emoji} **{operation_info.get('name', operation.title()) if operation_info else operation.title()}**\n"
        header += f"⏱️ Время обработки: {result.processing_time:.1f}с\n\n"

        if operation == "summarize":
            return self._format_summary_result(header, result.data)
        elif operation == "analyze_risks":
            return self._format_risk_analysis_result(header, result.data)
        elif operation == "chat":
            return self._format_chat_result(header, result.data)
        elif operation == "anonymize":
            return self._format_anonymize_result(header, result.data)
        elif operation == "translate":
            return self._format_translate_result(header, result.data)
        elif operation == "ocr":
            return self._format_ocr_result(header, result.data)
        else:
            return f"{header}✅ {result.message}"

    def _format_summary_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата саммаризации"""
        summary = data.get("summary", {})
        content = summary.get("content", "Содержимое недоступно")

        metadata = data.get("metadata", {})

        result = f"{header}**📊 Статистика документа:**\n"
        result += f"• Слов: {metadata.get('word_count', 'н/д')}\n"
        result += f"• Символов: {metadata.get('char_count', 'н/д')}\n\n"

        # Ограничиваем длину для Telegram
        if len(content) > 3000:
            content = content[:3000] + "...\n\n*(Полная версия выжимки доступна в файле)*"

        result += content

        return result

    def _format_risk_analysis_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата анализа рисков"""
        overall_risk = data.get("overall_risk_level", "неизвестен")

        risk_emojis = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }

        emoji = risk_emojis.get(overall_risk.lower(), "❓")

        result = f"{header}**{emoji} Общий уровень риска: {overall_risk.upper()}**\n\n"

        pattern_risks = data.get("pattern_risks", [])
        if pattern_risks:
            result += f"**⚠️ Обнаруженные риски: {len(pattern_risks)}**\n\n"

            for risk in pattern_risks[:3]:  # Показываем только первые 3
                level_emoji = risk_emojis.get(risk.get("risk_level", "medium"), "🟡")
                result += f"{level_emoji} {risk.get('description', 'Неописанный риск')}\n"

            if len(pattern_risks) > 3:
                result += f"\n*...и ещё {len(pattern_risks) - 3} рисков*\n"

        ai_analysis = data.get("ai_analysis", {})
        if ai_analysis.get("analysis"):
            analysis_text = ai_analysis["analysis"]
            if len(analysis_text) > 1500:
                analysis_text = analysis_text[:1500] + "...\n\n*(Полный анализ доступен в файле)*"
            result += f"\n**🤖 AI-анализ:**\n{analysis_text}"

        return result

    def _format_chat_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата чата"""
        question = data.get("question", "")
        answer = data.get("answer", "Ответ недоступен")

        result = f"**❓ Вопрос:** {question}\n\n"
        result += f"**💡 Ответ:**\n{answer}\n\n"

        fragments = data.get("relevant_fragments", [])
        if fragments:
            result += f"**📎 Релевантные фрагменты:**\n"
            for i, fragment in enumerate(fragments[:2], 1):
                result += f"{i}. *{fragment.get('text', '')[:200]}...*\n"

        return result

    def _format_anonymize_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата обезличивания"""
        report = data.get("anonymization_report", {})
        stats = report.get("statistics", {})

        result = f"{header}**🔒 Результаты обезличивания:**\n\n"

        total_items = sum(stats.values())
        result += f"Всего обработано элементов: **{total_items}**\n\n"

        if stats:
            result += "**По типам данных:**\n"
            type_names = {
                "names": "👤 ФИО",
                "phones": "📞 Телефоны",
                "emails": "📧 Email",
                "addresses": "🏠 Адреса",
                "documents": "📄 Номера документов",
                "bank_details": "🏦 Банковские реквизиты"
            }

            for data_type, count in stats.items():
                if count > 0:
                    name = type_names.get(data_type, data_type)
                    result += f"• {name}: {count}\n"

        result += f"\n✅ Документ готов к безопасной передаче"

        return result

    def _format_translate_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата перевода"""
        source_lang = data.get("source_language", "")
        target_lang = data.get("target_language", "")

        lang_names = {
            "ru": "🇷🇺 Русский",
            "en": "🇺🇸 Английский",
            "zh": "🇨🇳 Китайский",
            "de": "🇩🇪 Немецкий"
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        metadata = data.get("translation_metadata", {})

        result = f"{header}**🌍 Перевод завершен**\n"
        result += f"{source_name} → {target_name}\n\n"

        if metadata:
            result += f"📊 **Статистика:**\n"
            result += f"• Исходный текст: {metadata.get('original_length', 0)} символов\n"
            result += f"• Переведенный текст: {metadata.get('translated_length', 0)} символов\n\n"

        translated_text = data.get("translated_text", "")
        if len(translated_text) > 2000:
            preview = translated_text[:2000] + "..."
            result += f"**📄 Предварительный просмотр:**\n{preview}\n\n*(Полный перевод доступен в файле)*"
        else:
            result += f"**📄 Переведенный текст:**\n{translated_text}"

        return result

    def _format_ocr_result(self, header: str, data: Dict[str, Any]) -> str:
        """Форматирование результата OCR"""
        confidence = data.get("confidence_score", 0)
        quality = data.get("quality_analysis", {})

        confidence_emoji = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"

        result = f"{header}**👁️ OCR распознавание завершено**\n"
        result += f"{confidence_emoji} Уверенность: {confidence:.1f}%\n"
        result += f"📊 Качество: {quality.get('quality_level', 'неизвестно')}\n\n"

        processing_info = data.get("processing_info", {})
        if processing_info:
            result += f"**📈 Статистика:**\n"
            result += f"• Слов распознано: {processing_info.get('word_count', 0)}\n"
            result += f"• Символов: {processing_info.get('text_length', 0)}\n\n"

        recognized_text = data.get("recognized_text", "")
        if len(recognized_text) > 2000:
            preview = recognized_text[:2000] + "..."
            result += f"**📄 Распознанный текст:**\n{preview}\n\n*(Полный текст доступен в файле)*"
        else:
            result += f"**📄 Распознанный текст:**\n{recognized_text}"

        recommendations = quality.get("recommendations", [])
        if recommendations:
            result += f"\n\n**💡 Рекомендации:**\n"
            for rec in recommendations[:2]:
                result += f"• {rec}\n"

        return result

    async def cleanup_user_files(self, user_id: int, max_age_hours: int = 24):
        """Очистка файлов пользователя"""
        self.storage.cleanup_old_files(user_id, max_age_hours)
        self.document_chat.cleanup_old_documents(max_age_hours)