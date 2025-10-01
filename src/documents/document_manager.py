"""
Менеджер документов - центральный класс для управления всеми модулями работы с документами
"""

from __future__ import annotations

import importlib.util
import logging
from html import escape as html_escape
from pathlib import Path
from typing import Any

from .anonymizer import DocumentAnonymizer
from .base import DocumentResult, DocumentStorage, ProcessingError
from .document_chat import DocumentChat
from .ocr_converter import OCRConverter
from .risk_analyzer import RiskAnalyzer
from .summarizer import DocumentSummarizer
from .translator import DocumentTranslator

# Импорт утилит для безопасной HTML сборки
from src.core.safe_telegram import format_safe_html, split_html_for_telegram
from src.bot.ui_components import sanitize_telegram_html

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

        self._dependencies: dict[str, bool] = {
            'docx': self._module_available('docx'),
            'reportlab': self._module_available('reportlab.pdfgen'),
        }

        self.PROCESSOR_PARAM_WHITELIST: dict[str, set[str]] = {
            "summarize": {"detail_level", "language", "output_formats"},
            "analyze_risks": {"custom_criteria"},
            "chat": set(),
            "anonymize": {"anonymization_mode", "exclude_types"},
            "translate": {"source_lang", "target_lang", "output_formats"},
            "ocr": {"output_format"},
        }

    @staticmethod
    def _module_available(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    def _dependency_available(self, key: str) -> bool:
        return self._dependencies.get(key, True)

    def _dependency_notice(self, *, dependency: str, feature: str, format_name: str) -> dict[str, str]:
        message = f"{feature} requires {dependency} to be installed."
        logger.warning(message)
        return {"format": format_name, "error": message}


    async def process_document(
        self,
        user_id: int,
        file_content: bytes,
        original_name: str,
        mime_type: str,
        operation: str,
        **kwargs: Any,
    ) -> DocumentResult:
        """Обработать документ указанным способом."""

        try:
            document_info = await self.storage.save_document(
                user_id, file_content, original_name, mime_type
            )

            processor = self._get_processor(operation)
            if not processor:
                raise ProcessingError(f"Неизвестная операция: {operation}", "UNKNOWN_OPERATION")

            allowed_params = self.PROCESSOR_PARAM_WHITELIST.get(operation, set())
            safe_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                if key not in allowed_params or value is None:
                    continue
                if key == "exclude_types" and isinstance(value, list):
                    normalized = sorted({str(item).lower() for item in value if item})
                    safe_kwargs[key] = normalized
                else:
                    safe_kwargs[key] = value

            logger.info(
                "Обрабатываем документ %s операцией %s для пользователя %s",
                original_name,
                operation,
                user_id,
            )
            if safe_kwargs:
                logger.debug("Применяем параметры операции: %s", safe_kwargs)

            result = await processor.safe_process(document_info.file_path, **safe_kwargs)

            if result.success:
                result.data["document_info"] = {
                    "original_name": document_info.original_name,
                    "file_size": document_info.size,
                    "upload_time": document_info.upload_time.isoformat(),
                    "user_id": user_id,
                }
                if safe_kwargs:
                    result.data["applied_options"] = dict(safe_kwargs)

                exports = self._create_exports(operation, result.data, document_info, safe_kwargs)
                if exports:
                    result.data["exports"] = exports

            self.storage.cleanup_old_files(user_id, max_age_hours=24)
            return result

        except Exception as exc:
            logger.error("Ошибка обработки документа %s: %s", original_name, exc)
            raise ProcessingError(f"Ошибка обработки документа: {exc}", "PROCESSING_ERROR")

    def _get_processor(self, operation: str):
        processors = {
            "summarize": self.summarizer,
            "analyze_risks": self.risk_analyzer,
            "chat": self.document_chat,
            "anonymize": self.anonymizer,
            "translate": self.translator,
            "ocr": self.ocr_converter,
        }
        return processors.get(operation)

    def _append_export_note(self, base_text: str, data: dict[str, Any]) -> str:
        exports = data.get("exports") or []
        if not exports:
            return base_text

        lines = []
        for export in exports:
            fmt = str(export.get("format", "file")).upper()
            path_value = export.get("path", "")
            file_name = Path(path_value).name if path_value else path_value
            lines.append(f"• {html_escape(fmt)}: {html_escape(file_name)}")

        return base_text + "\n\n📎 Доступные файлы:\n" + "\n".join(lines)

    def _create_exports(
        self,
        operation: str,
        data: dict[str, Any],
        document_info,
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        exports: list[dict[str, Any]] = []
        try:
            if operation == "summarize":
                formats = options.get("output_formats") or ["docx", "pdf"]
                exports.extend(self._export_summary(document_info, data, formats))
            elif operation == "translate":
                formats = options.get("output_formats") or ["docx", "txt"]
                exports.extend(self._export_translation(document_info, data, formats))
            elif operation == "ocr":
                output_format = options.get("output_format", "txt")
                exports.extend(self._export_ocr(document_info, data, output_format))
            elif operation == "analyze_risks":
                exports.extend(self._export_risk_report(document_info, data))
        except Exception as export_error:
            logger.warning("Не удалось подготовить экспорт для %s: %s", operation, export_error)
        return exports

    def _export_summary(
        self, document_info, data: dict[str, Any], formats: list[str]
    ) -> list[dict[str, Any]]:
        summary_content = data.get("summary", {}).get("content")
        if not summary_content:
            return []

        exports: list[dict[str, Any]] = []
        base_name = f"{document_info.file_path.stem}_summary"
        export_dir = document_info.file_path.parent

        if "docx" in formats:
            if not self._dependency_available('docx'):
                exports.append(
                    self._dependency_notice(
                        dependency='python-docx',
                        feature='DOCX export',
                        format_name='docx',
                    )
                )
            else:
                from docx import Document  # type: ignore

                doc = Document()
                doc.add_heading("Сводка по документу", level=1)
                for block in summary_content.split("\n\n"):
                    doc.add_paragraph(block)
                docx_path = export_dir / f"{base_name}.docx"
                doc.save(docx_path)
                exports.append({"path": str(docx_path), "format": "docx", "label": "Резюме (DOCX)"})

        if "pdf" in formats:
            if not self._dependency_available('reportlab'):
                exports.append(
                    self._dependency_notice(
                        dependency='reportlab',
                        feature='PDF export',
                        format_name='pdf',
                    )
                )
            else:
                from reportlab.lib.pagesizes import A4  # type: ignore
                from reportlab.lib.units import mm  # type: ignore
                from reportlab.pdfgen import canvas  # type: ignore

                pdf_path = export_dir / f"{base_name}.pdf"
                canv = canvas.Canvas(str(pdf_path), pagesize=A4)
                width, height = A4
                text_obj = canv.beginText(20 * mm, height - 20 * mm)
                for line in summary_content.splitlines():
                    text_obj.textLine(line)
                    if text_obj.getY() < 20 * mm:
                        canv.drawText(text_obj)
                        canv.showPage()
                        text_obj = canv.beginText(20 * mm, height - 20 * mm)
                canv.drawText(text_obj)
                canv.save()
                exports.append({"path": str(pdf_path), "format": "pdf", "label": "Резюме (PDF)"})


        return exports

    def _export_translation(
        self, document_info, data: dict[str, Any], formats: list[str]
    ) -> list[dict[str, Any]]:
        translated_text = data.get("translated_text")
        if not translated_text:
            return []

        exports: list[dict[str, Any]] = []
        base_name = f"{document_info.file_path.stem}_translation"
        export_dir = document_info.file_path.parent

        if "docx" in formats:
            if not self._dependency_available('docx'):
                exports.append(
                    self._dependency_notice(
                        dependency='python-docx',
                        feature='Translation DOCX export',
                        format_name='docx',
                    )
                )
            else:
                from docx import Document  # type: ignore

                doc = Document()
                for block in translated_text.split("\n\n"):
                    doc.add_paragraph(block)
                docx_path = export_dir / f"{base_name}.docx"
                doc.save(docx_path)
                exports.append({"path": str(docx_path), "format": "docx", "label": "Перевод (DOCX)"})

        if "txt" in formats:
            txt_path = export_dir / f"{base_name}.txt"
            self._write_text_file(txt_path, translated_text)
            exports.append({"path": str(txt_path), "format": "txt", "label": "Перевод (TXT)"})

        return exports

    def _export_ocr(
        self, document_info, data: dict[str, Any], output_format: str
    ) -> list[dict[str, Any]]:
        recognized_text = data.get("recognized_text")
        if not recognized_text:
            return []

        exports: list[dict[str, Any]] = []
        base_name = f"{document_info.file_path.stem}_ocr"
        export_dir = document_info.file_path.parent
        fmt = output_format.lower()

        if fmt == "txt":
            txt_path = export_dir / f"{base_name}.txt"
            self._write_text_file(txt_path, recognized_text)
            exports.append({"path": str(txt_path), "format": "txt", "label": "Оцифровка (TXT)"})
        elif fmt == "docx":
            if not self._dependency_available('docx'):
                exports.append(
                    self._dependency_notice(
                        dependency='python-docx',
                        feature='OCR DOCX export',
                        format_name='docx',
                    )
                )
            else:
                from docx import Document  # type: ignore

                doc = Document()
                for block in recognized_text.split("\n\n"):
                    doc.add_paragraph(block)
                docx_path = export_dir / f"{base_name}.docx"
                doc.save(docx_path)
                exports.append({"path": str(docx_path), "format": "docx", "label": "Оцифровка (DOCX)"})

        elif fmt == "pdf":
            if not self._dependency_available('reportlab'):
                exports.append(
                    self._dependency_notice(
                        dependency='reportlab',
                        feature='OCR PDF export',
                        format_name='pdf',
                    )
                )
            else:
                from reportlab.lib.pagesizes import A4  # type: ignore
                from reportlab.lib.units import mm  # type: ignore
                from reportlab.pdfgen import canvas  # type: ignore

                pdf_path = export_dir / f"{base_name}.pdf"
                canv = canvas.Canvas(str(pdf_path), pagesize=A4)
                width, height = A4
                text_obj = canv.beginText(20 * mm, height - 20 * mm)
                for line in recognized_text.splitlines():
                    text_obj.textLine(line)
                    if text_obj.getY() < 20 * mm:
                        canv.drawText(text_obj)
                        canv.showPage()
                        text_obj = canv.beginText(20 * mm, height - 20 * mm)
                canv.drawText(text_obj)
                canv.save()
                exports.append({"path": str(pdf_path), "format": "pdf", "label": "Оцифровка (PDF)"})


        else:
            logger.warning("Неподдерживаемый формат экспорта OCR: %s", output_format)

        return exports

    def _export_risk_report(self, document_info, data: dict[str, Any]) -> list[dict[str, Any]]:
        overall = data.get("overall_risk_level")
        pattern_risks = data.get("pattern_risks", [])
        ai_analysis = data.get("ai_analysis", {}).get("analysis", "")
        highlighted = data.get("highlighted_text", "")
        if not overall and not pattern_risks and not ai_analysis:
            return []

        report_lines = [
            f"Общий уровень риска: {overall}",
            "",
            "Найденные риски:",
        ]
        for risk in pattern_risks:
            report_lines.append(
                f"- [{risk.get('risk_level', 'unknown')}] {risk.get('description', '')}"
            )
        if ai_analysis:
            report_lines.extend(["", "AI-анализ:", ai_analysis])
        if highlighted:
            report_lines.extend(["", "Контекст с подсветкой:", highlighted])

        txt_path = (
            document_info.file_path.parent / f"{document_info.file_path.stem}_risk_report.txt"
        )
        self._write_text_file(txt_path, "\n".join(report_lines))
        return [{"path": str(txt_path), "format": "txt", "label": "Отчёт о рисках"}]

    def _write_text_file(self, path: Path, content: str) -> None:
        path.write_text(content or "", encoding="utf-8")

    async def chat_with_document(
        self, user_id: int, document_id: str, question: str
    ) -> dict[str, Any]:
        try:
            return await self.document_chat.chat_with_document(document_id, question)
        except Exception as exc:
            logger.error("Ошибка чата с документом %s: %s", document_id, exc)
            raise ProcessingError(f"Ошибка чата: {exc}", "CHAT_ERROR")

    def get_supported_operations(self) -> dict[str, dict[str, Any]]:
        return {
            "summarize": {
                "name": "Саммаризация",
                "description": "Создание краткой выжимки документа с ключевыми положениями",
                "emoji": "📋",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["detail_level", "language"],
            },
            "analyze_risks": {
                "name": "Анализ рисков",
                "description": "Выявление рисков и проблем в договорах и документах",
                "emoji": "⚠️",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["custom_criteria"],
            },
            "chat": {
                "name": "Чат с документом",
                "description": "Интерактивные вопросы-ответы по документу",
                "emoji": "💬",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": [],
            },
            "anonymize": {
                "name": "Обезличивание",
                "description": "Удаление персональных данных из документа",
                "emoji": "🔐",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["anonymization_mode", "exclude_types"],
            },
            "translate": {
                "name": "Перевод",
                "description": "Перевод документа на другие языки",
                "emoji": "🌍",
                "formats": [".pdf", ".docx", ".doc", ".txt"],
                "parameters": ["source_lang", "target_lang"],
            },
            "ocr": {
                "name": "OCR распознавание",
                "description": "Распознавание текста из сканированных документов",
                "emoji": "🖭",
                "formats": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
                "parameters": ["output_format"],
            },
        }

    def get_operation_info(self, operation: str) -> dict[str, Any] | None:
        return self.get_supported_operations().get(operation)

    def build_telegram_chunks(self, html: str, max_len: int = 3900) -> list[str]:
        """Собирает безопасный HTML и режет на куски под лимиты Telegram."""
        safe_html = format_safe_html(html)               # нормализуем/балансируем теги
        chunks = split_html_for_telegram(safe_html, max_len=max_len)
        return chunks

    def format_result_for_telegram(self, result: DocumentResult, operation: str) -> str:
        if not result.success:
            raw_html = f"✖ <b>Ошибка обработки</b>\n\n{html_escape(str(result.message))}"
            return format_safe_html(raw_html)

        operation_info = self.get_operation_info(operation) or {}
        emoji = operation_info.get("emoji", "📄")
        name = operation_info.get("name", operation.title())

        header = f"{emoji} <b>{html_escape(name)}</b>\n"
        header += f"⏱️ Время обработки: {result.processing_time:.1f}с\n\n"

        # Получаем сырой HTML от соответствующего форматтера
        if operation == "summarize":
            raw_html = self._format_summary_result(header, result.data)
        elif operation == "analyze_risks":
            raw_html = self._format_risk_analysis_result(header, result.data)
        elif operation == "chat":
            raw_html = self._format_chat_result(header, result.data)
        elif operation == "anonymize":
            raw_html = self._format_anonymize_result(header, result.data)
        elif operation == "translate":
            raw_html = self._format_translate_result(header, result.data)
        elif operation == "ocr":
            raw_html = self._format_ocr_result(header, result.data)
        else:
            raw_html = f"{header}✔ {result.message}"

        # Конвертируем переводы строк и очищаем HTML перед отправкой
        normalized = raw_html.replace("\r\n", "\n")
        sanitized = sanitize_telegram_html(normalized)
        sanitized = sanitized.replace("<br><br>", "\n\n").replace("<br>", "\n")
        return sanitized

    def _format_summary_result(self, header: str, data: dict[str, Any]) -> str:
        summary = data.get("summary", {})
        content = summary.get("content", "Содержимое недоступно")
        metadata = data.get("metadata", {})
        detail_level = data.get("detail_level", "detailed")
        language = data.get("language", "ru")
        detail_text = {"detailed": "Подробная", "brief": "Краткая"}.get(detail_level, detail_level)
        language_text = {"ru": "Русский", "en": "Английский"}.get(language, language)

        result = f"{header}<b>📝 Саммаризация:</b>\n"
        result += f"Уровень детализации: {html_escape(detail_text)}\n"
        result += f"Язык: {html_escape(language_text)}\n\n{html_escape(content)}"

        if metadata:
            result += "\n\n<b>📊 Метаданные:</b>\n"
            for key, value in metadata.items():
                result += f"• {html_escape(str(key))}: {html_escape(str(value))}\n"

        return self._append_export_note(result, data)

    def _format_risk_analysis_result(self, header: str, data: dict[str, Any]) -> str:
        overall_risk = data.get("overall_risk_level", "неизвестен")
        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        emoji = risk_emojis.get(str(overall_risk).lower(), "✅")

        result = f"{header}<b>{emoji} Общий уровень риска: {html_escape(str(overall_risk).upper())}</b>\n\n"

        pattern_risks = data.get("pattern_risks", [])
        if pattern_risks:
            result += f"<b>⚠️ Обнаруженные риски: {len(pattern_risks)}</b>\n\n"
            for risk in pattern_risks[:3]:
                level_emoji = risk_emojis.get(str(risk.get('risk_level', 'medium')).lower(), '🟡')
                desc = html_escape(risk.get('description', 'Неописанный риск'))
                result += f"{level_emoji} {desc}\n"
            if len(pattern_risks) > 3:
                result += f"\n<i>...и ещё {len(pattern_risks) - 3} рисков</i>\n"

        ai_analysis = data.get("ai_analysis", {})
        if ai_analysis.get("analysis"):
            analysis_text = ai_analysis["analysis"]
            if len(analysis_text) > 1500:
                analysis_text = analysis_text[:1500] + "...\n\n(Полный анализ доступен в файле)"
            result += f"\n<b>🤖 AI-анализ:</b>\n{html_escape(analysis_text)}\n"

        legal_compliance = data.get("legal_compliance", {})
        violations = legal_compliance.get("violations") or []
        if violations:
            result += "\n<b>⚖️ Возможные нарушения:</b>\n"
            for violation in violations[:3]:
                reference = violation.get("reference")
                text = html_escape(violation.get("text", ""))
                if reference:
                    result += f"- {text} ({html_escape(reference)})\n"
                else:
                    result += f"- {text}\n"
            if len(violations) > 3:
                result += f"<i>...и ещё {len(violations) - 3} пунктов</i>\n"
        elif legal_compliance.get("status") == "completed":
            result += "\n<b>⚖️ Возможные нарушения:</b> не выявлены.\n"

        return self._append_export_note(result, data)

    def _format_chat_result(self, header: str, data: dict[str, Any]) -> str:
        question = data.get("question", "")
        answer = data.get("answer", "Ответ недоступен")

        result = f"{header}<b>✔ Вопрос:</b> {html_escape(question)}\n\n"
        result += f"<b>💡 Ответ:</b>\n{html_escape(answer)}\n\n"

        context_chunks = data.get("context_chunks", [])
        if context_chunks:
            result += "<b>🔍 Использованные фрагменты:</b>\n"
            for item in context_chunks:
                preview = item.get("excerpt", "")
                if len(preview) > 160:
                    preview = preview[:160] + "..."
                score = item.get("score")
                score_text = f" (релевантность {score:.2f})" if isinstance(score, (int, float)) else ""
                result += f"• Фрагмент {int(item.get('index', 0)) + 1}{score_text}: <i>{html_escape(preview)}</i>\n"
            result += "\n"

        fragments = data.get("relevant_fragments", [])
        if fragments:
            result += "<b>📓 Релевантные предложения:</b>\n"
            for i, fragment in enumerate(fragments[:2], 1):
                text = html_escape(fragment.get('text', '')[:200])
                result += f"{i}. <i>{text}...</i>\n"

        return self._append_export_note(result, data)

    def _format_anonymize_result(self, header: str, data: dict[str, Any]) -> str:
        report = data.get("anonymization_report", {})
        stats = report.get("statistics", {})
        result = f"{header}<b>🔒 Результаты обезличивания:</b>\n\n"

        total_items = sum(int(v) for v in stats.values()) if stats else 0
        result += f"Всего обработано элементов: <b>{total_items}</b>\n\n"

        if stats:
            result += "<b>По типам данных:</b>\n"
            type_names = {
                "names": "👤 ФИО",
                "phones": "📞 Телефоны",
                "emails": "📧 Email",
                "addresses": "🏠 Адреса",
                "documents": "📄 Номера документов",
                "bank_details": "🏦 Банковские реквизиты",
            }
            for data_type, count in stats.items():
                if int(count) > 0:
                    name = type_names.get(data_type, data_type)
                    result += f"• {html_escape(str(name))}: {int(count)}\n"

        result += "\n✅ Документ готов к безопасной передаче"
        return self._append_export_note(result, data)

    def _format_translate_result(self, header: str, data: dict[str, Any]) -> str:
        source_lang = data.get("source_language", "")
        target_lang = data.get("target_language", "")
        lang_names = {
            "ru": "🇷🇺 Русский",
            "en": "🇺🇸 Английский",
            "zh": "🇨🇳 Китайский",
            "de": "🇩🇪 Немецкий",
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        metadata = data.get("translation_metadata", {})

        result = f"{header}<b>🌍 Перевод завершен</b>\n"
        result += f"{html_escape(source_name)} → {html_escape(target_name)}\n\n"

        if metadata:
            result += "<b>📊 Статистика:</b>\n"
            result += f"• Исходный текст: {int(metadata.get('original_length', 0))} символов\n"
            result += f"• Переведенный текст: {int(metadata.get('translated_length', 0))} символов\n"
            if metadata.get('chunks_processed'):
                result += f"• Частей обработано: {int(metadata['chunks_processed'])}\n"
            result += "\n"

        translated_text = data.get("translated_text", "") or ""
        if len(translated_text) > 2000:
            preview = translated_text[:2000] + "..."
            result += f"<b>📄 Предварительный просмотр:</b>\n{html_escape(preview)}\n\n(Полный перевод доступен в файле)"
        else:
            result += f"<b>📄 Переведенный текст:</b>\n{html_escape(translated_text)}"

        return self._append_export_note(result, data)

    def _format_ocr_result(self, header: str, data: dict[str, Any]) -> str:
        confidence = data.get("confidence_score", 0)
        quality = data.get("quality_analysis", {})

        confidence_emoji = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"

        result = f"{header}<b>👁️ OCR распознавание завершено</b>\n"
        result += f"{confidence_emoji} Уверенность: {confidence:.1f}%\n"
        result += f"📊 Качество: {html_escape(quality.get('quality_level', 'неизвестно'))}\n\n"

        processing_info = data.get("processing_info", {})
        if processing_info:
            result += "<b>📈 Статистика:</b>\n"
            result += f"• Слов распознано: {processing_info.get('word_count', 0)}\n"
            result += f"• Символов: {processing_info.get('text_length', 0)}\n\n"

        recognized_text = data.get("recognized_text", "")
        if len(recognized_text) > 2000:
            preview = html_escape(recognized_text[:2000]) + "..."
            result += f"<b>📄 Распознанный текст:</b>\n{preview}\n\n<i>(Полный текст доступен в файле)</i>"
        else:
            result += f"<b>📄 Распознанный текст:</b>\n{html_escape(recognized_text)}"

        recommendations = quality.get("recommendations", [])
        if recommendations:
            result += "\n\n<b>💡 Рекомендации:</b>\n"
            for rec in recommendations[:2]:
                result += f"• {html_escape(rec)}\n"

        return self._append_export_note(result, data)

    async def cleanup_user_files(self, user_id: int, max_age_hours: int = 24):
        self.storage.cleanup_old_files(user_id, max_age_hours)
        self.document_chat.cleanup_old_documents(max_age_hours)
