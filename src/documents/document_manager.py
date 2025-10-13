"""
High level orchestration layer for document-related processors.
"""

from __future__ import annotations

import asyncio
import asyncio
import html
import json
import logging
import re
import tempfile
import uuid
from datetime import datetime
from dataclasses import asdict
from html import escape as html_escape
from pathlib import Path
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, List, Mapping

from src.core.excel_export import build_risk_excel
from src.core.settings import AppSettings

from .anonymizer import DocumentAnonymizer
from .base import DocumentInfo, DocumentResult, DocumentStorage, ProcessingError
from .document_chat import DocumentChat
from .document_drafter import DocumentDraftingError, build_docx_from_markdown
from .lawsuit_analyzer import LawsuitAnalyzer
from .ocr_converter import OCRConverter
from .risk_analyzer import RiskAnalyzer
from .storage_backends import ArtifactUploader, S3ArtifactUploader
from .summarizer import DocumentSummarizer
from .translator import DocumentTranslator
from .utils import TextProcessor, write_text_async

_LAWSUIT_SECTION_META: dict[str, tuple[str, str]] = {
    "demands": ("📝", "Требования"),
    "legal_basis": ("📚", "Правовое обоснование"),
    "evidence": ("📁", "Доказательства"),
    "strengths": ("✅", "Сильные стороны"),
    "risks": ("⚠️", "Риски и слабые места"),
    "missing_elements": ("❗", "Недостающие элементы"),
    "recommendations": ("💡", "Рекомендации"),
    "procedural_notes": ("📅", "Процессуальные заметки"),
}

logger = logging.getLogger(__name__)

_FILENAME_SANITIZE_RE = re.compile(r"[\\/:*?\"<>|\r\n]+")
_FILENAME_WHITESPACE_RE = re.compile(r"\s+")


class DocumentManager:
    """Facade that wires document processors, storage and formatting."""

    def __init__(self, openai_service=None, settings: AppSettings | None = None):
        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings
        self._openai_service = openai_service

        self.storage = self._init_storage(settings)
        self.summarizer = DocumentSummarizer(openai_service=openai_service, settings=settings)
        self.translator = DocumentTranslator(openai_service=openai_service, settings=settings)
        self.anonymizer = DocumentAnonymizer(settings=settings)
        self.risk_analyzer = RiskAnalyzer(openai_service=openai_service)
        self.chat = DocumentChat(openai_service=openai_service, settings=settings)
        self.ocr_converter = OCRConverter(settings=settings)
        self.lawsuit_analyzer = LawsuitAnalyzer(openai_service=openai_service)

        self._operations: Dict[str, Dict[str, Any]] = {
            "summarize": {
                "emoji": "📑",
                "name": "Краткая выжимка документа",
                "description": "Краткое содержание документа и ключевые выводы",
                "formats": ["TXT", "JSON"],
                "processor": self.summarizer,
            },
            "analyze_risks": {
                "emoji": "⚠️",
                "name": "Анализ рисков",
                "description": "Выявление потенциальных проблем и рекомендаций",
                "formats": ["TXT", "JSON"],
                "processor": self.risk_analyzer,
            },
            "lawsuit_analysis": {
                "emoji": "⚖️",
                "name": "Анализ искового заявления",
                "description": "Оценивает требования, правовую позицию и риски отказа",
                "formats": ["DOCX"],
                "processor": self.lawsuit_analyzer,
            },
            # "chat": {
            #     "emoji": "💬",
            #     "name": "Чат с документом",
            #     "description": "Задавайте произвольные вопросы по содержимому",
            #     "formats": ["interactive"],
            #     "processor": self.chat,
            # },
            "anonymize": {
                "emoji": "🕶️",
                "name": "Анонимизация",
                "description": "Удаление персональных данных из документа",
                "formats": ["TXT", "JSON"],
                "processor": self.anonymizer,
            },
            # "translate": {
            #     "emoji": "🌐",
            #     "name": "Перевод текста",
            #     "description": "Перевод текста на выбранный язык",
            #     "formats": ["TXT"],
            #     "processor": self.translator,
            # },
            "ocr": {
                "emoji": "📷",
                "name": "Распознание текста",
                "description": "Распознавание текста на изображениях и PDF",
                "formats": ["TXT"],
                "processor": self.ocr_converter,
            },
        }

        self._user_chat_sessions: Dict[int, Dict[str, Any]] = {}
        self._operations_cache_data: Dict[str, Dict[str, Any]] | None = None
        self._operations_cache_view: Mapping[str, Dict[str, Any]] | None = None

    # ------------------------------------------------------------------ API ---

    def _build_supported_operations(self) -> Dict[str, Dict[str, Any]]:
        operations: Dict[str, Dict[str, Any]] = {}
        for key, meta in self._operations.items():
            processor = meta.get("processor")
            upload_formats: List[str] = []
            if processor is not None:
                supported = getattr(processor, "supported_formats", None)
                if supported:
                    upload_formats = [fmt.lstrip(".").upper() for fmt in supported]

            descriptor: Dict[str, Any] = {
                "emoji": meta.get("emoji"),
                "name": meta.get("name"),
                "description": meta.get("description"),
                "formats": list(meta.get("formats", [])),
            }
            if upload_formats:
                descriptor["upload_formats"] = upload_formats
            operations[key] = descriptor
        return operations

    def get_supported_operations(self) -> Mapping[str, Dict[str, Any]]:
        if self._operations_cache_view is None:
            data = self._build_supported_operations()
            self._operations_cache_data = data
            self._operations_cache_view = MappingProxyType(data)
        return self._operations_cache_view

    def get_operation_info(self, operation: str) -> Dict[str, Any] | None:
        info = self.get_supported_operations().get(operation)
        if info is None:
            return None
        return dict(info)

    async def process_document(
        self,
        *,
        user_id: int,
        file_content: bytes,
        original_name: str,
        mime_type: str,
        operation: str,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        **options: Any,
    ) -> DocumentResult:
        meta = self._operations.get(operation)
        if not meta:
            return DocumentResult.error_result("Неизвестная операция", "UNKNOWN_OPERATION")

        try:
            doc_info = await self.storage.save_document(
                user_id=user_id,
                file_content=file_content,
                original_name=original_name,
                mime_type=mime_type,
            )
        except ProcessingError as exc:
            logger.warning("Failed to store document: %s", exc)
            return DocumentResult.error_result(exc.message, exc.error_code)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while storing document")
            return DocumentResult.error_result(str(exc), "STORAGE_ERROR")

        if operation == "chat":
            return await self._handle_chat_upload(user_id, doc_info, options)

        processor = meta.get("processor")
        try:
            safe_kwargs = dict(options)
            if progress_callback is not None:
                safe_kwargs["progress_callback"] = progress_callback
            result = await processor.safe_process(doc_info.file_path, **safe_kwargs)
        except ProcessingError as exc:
            self._safe_unlink(doc_info.file_path)
            logger.warning("Processor error (%s): %s", operation, exc)
            return DocumentResult.error_result(exc.message, exc.error_code)
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(doc_info.file_path)
            logger.exception("Unhandled error during %s", operation)
            return DocumentResult.error_result(str(exc), "PROCESSING_ERROR")

        result.data.setdefault("document_info", asdict(doc_info))
        if result.success:
            try:
                exports = await self._prepare_exports(operation, result, doc_info)
            except ProcessingError as exc:
                logger.warning("Failed to prepare exports for %s: %s", operation, exc)
                result = DocumentResult.error_result(exc.message, exc.error_code or "EXPORT_ERROR")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error while preparing exports for %s", operation)
                result = DocumentResult.error_result(
                    "Не удалось подготовить файл отчета", "EXPORT_ERROR"
                )
            else:
                if exports:
                    existing = list(result.data.get("exports") or [])
                    result.data["exports"] = existing + exports
        if not result.success:
            result.data.setdefault("exports", [])

        self._safe_unlink(doc_info.file_path)
        return result

    def format_result_for_telegram(self, result: DocumentResult, operation: str) -> str:
        formatter = {
            "summarize": self._format_summary_result,
            "translate": self._format_translation_result,
            "anonymize": self._format_anonymize_result,
            "analyze_risks": self._format_risk_result,
            "lawsuit_analysis": self._format_lawsuit_result,
            "chat": self._format_chat_loaded,
            "ocr": self._format_ocr_result,
        }.get(operation, self._format_generic_result)
        return formatter(result.data, result.message)

    def format_chat_answer_for_telegram(self, result: DocumentResult) -> str:
        data = result.data
        answer = html_escape(data.get("answer", ""))
        confidence = float(data.get("confidence", 0.0))
        citations = data.get("citations", [])
        lines = [
            "<b>Ответ по документу</b>",
            answer or "Нет ответа",
            "",
            f"Доверие модели: {confidence:.0%}",
        ]
        if citations:
            lines.append("<b>Цитаты:</b>")
            for item in citations[:5]:
                snippet = html_escape(str(item.get("snippet", "")))
                idx = item.get("chunk_index")
                lines.append(f"• [chunk #{idx}] {snippet}")
        return "\n".join(lines)

    async def answer_chat_question(self, user_id: int, question: str) -> DocumentResult:
        session = self._user_chat_sessions.get(user_id)
        if not session:
            raise ProcessingError("Сначала загрузите документ для чата", "CHAT_NOT_INITIALIZED")
        document_id = session["document_id"]
        return await self.chat.answer_question(document_id, question)

    def end_chat_session(self, user_id: int) -> bool:
        session = self._user_chat_sessions.pop(user_id, None)
        if session:
            self.chat.unload_document(session.get("document_id"))
            return True
        return False


    # -------------------------------------------------------------- helpers ---

    def _init_storage(self, settings: AppSettings) -> DocumentStorage:
        storage_root = settings.get_str("DOCUMENTS_STORAGE_PATH", None) or "data/documents"
        uploader = self._build_uploader(settings)
        return DocumentStorage(
            storage_path=storage_root,
            max_user_quota_mb=settings.document_storage_quota_mb,
            cleanup_max_age_hours=settings.document_cleanup_hours,
            cleanup_interval_seconds=settings.document_cleanup_interval_seconds,
            artifact_uploader=uploader,
        )

    def _build_uploader(self, settings: AppSettings) -> ArtifactUploader | None:
        bucket = settings.documents_s3_bucket
        if not bucket:
            return None
        return S3ArtifactUploader(
            bucket=bucket,
            prefix=settings.documents_s3_prefix,
            region_name=settings.documents_s3_region,
            endpoint_url=settings.documents_s3_endpoint,
            public_base_url=settings.documents_s3_public_url,
            acl=settings.documents_s3_acl,
        )

    async def _handle_chat_upload(
        self,
        user_id: int,
        doc_info: DocumentInfo,
        options: Dict[str, Any],
    ) -> DocumentResult:
        previous = self._user_chat_sessions.pop(user_id, None)
        if previous:
            self.chat.unload_document(previous.get("document_id"))
        try:
            document_id = await self.chat.load_document(doc_info.file_path)
        finally:
            self._safe_unlink(doc_info.file_path)
        session_payload = {
            "document_id": document_id,
            "info": asdict(doc_info),
        }
        self._user_chat_sessions[user_id] = session_payload
        data = {
            "document_id": document_id,
            "document_info": self.chat.get_document_info(document_id),
            "message": "Документ готов к чату. Используйте /askdoc для вопросов и /enddoc для завершения.",
        }
        return DocumentResult.success_result(data=data, message="Документ готов к чату")

    async def _prepare_exports(
        self,
        operation: str,
        result: DocumentResult,
        doc_info: DocumentInfo,
    ) -> List[Dict[str, Any]]:
        exports: List[Dict[str, Any]] = []
        base_name = Path(doc_info.original_name or "document").stem or "document"

        if operation == "summarize":
            summary = ((result.data.get("summary") or {}).get("content") or "").strip()
            structured = (result.data.get("summary") or {}).get("structured")
            if summary:
                path = await self._write_export(base_name, "summary", summary, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Summary"})
            if structured:
                json_payload = json.dumps(structured, ensure_ascii=False, indent=2)
                path = await self._write_export(base_name, "summary", json_payload, ".json")
                exports.append({"path": str(path), "format": "json", "label": "Summary JSON"})

        elif operation == "translate":
            translated = (result.data.get("translated_text") or "").strip()
            if translated:
                path = await self._write_export(base_name, "translation", translated, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Перевод"})

        elif operation == "anonymize":
            anonymized = (result.data.get("anonymized_text") or "").strip()
            report = result.data.get("anonymization_report")
            if anonymized:
                path = await self._write_export(base_name, "anonymized", anonymized, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Анонимизированный текст"})
            if report:
                json_payload = json.dumps(report, ensure_ascii=False, indent=2)
                path = await self._write_export(base_name, "anonymized_report", json_payload, ".json")
                exports.append({"path": str(path), "format": "json", "label": "Отчёт"})

        elif operation == "analyze_risks":
            highlighted = (result.data.get("highlighted_text") or "").strip()
            if highlighted:
                path = await self._write_export(base_name, "risk_highlight", highlighted, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Подсветка рисков"})
            json_payload = json.dumps(result.data, ensure_ascii=False, indent=2)
            path = await self._write_export(base_name, "risk_report", json_payload, ".json")
            exports.append({"path": str(path), "format": "json", "label": "Отчёт"})
            try:
                excel_path = await asyncio.to_thread(
                    build_risk_excel, result.data, file_stub=f"{base_name}_risks"
                )
                exports.append({"path": str(excel_path), "format": "xlsx", "label": "Отчёт (XLSX)"})
            except RuntimeError as exc:
                logger.warning("Excel export unavailable for risk analysis: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to build Excel risk report: %s", exc, exc_info=True)

        elif operation == "lawsuit_analysis":
            analysis = result.data.get("analysis") or {}
            markdown = self._build_lawsuit_markdown(analysis).strip()
            if not markdown:
                raise ProcessingError(
                    "Не удалось подготовить содержимое для DOCX-отчета",
                    "EMPTY_DOCX_CONTENT",
                )
            docx_path = self._build_human_friendly_temp_path(base_name, "анализ иска", ".docx")
            try:
                await asyncio.to_thread(build_docx_from_markdown, markdown, str(docx_path))
            except DocumentDraftingError as exc:
                raise ProcessingError(str(exc), "DOCX_EXPORT_ERROR") from exc
            except Exception as exc:  # noqa: BLE001
                raise ProcessingError(
                    "Ошибка при формировании DOCX-отчета",
                    "DOCX_EXPORT_ERROR",
                ) from exc
            exports.append({"path": str(docx_path), "format": "docx", "label": "Анализ (DOCX)"})

        elif operation == "ocr":
            recognized = (result.data.get("recognized_text") or "").strip()
            if recognized:
                path = await self._write_export(base_name, "ocr", recognized, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Распознанный текст"})

        return exports

    @staticmethod
    def _sanitize_filename_component(value: str, fallback: str | None = None) -> str:
        sanitized = _FILENAME_SANITIZE_RE.sub(" ", (value or ""))
        sanitized = _FILENAME_WHITESPACE_RE.sub(" ", sanitized).strip(" .-_")
        if not sanitized and fallback is not None:
            sanitized = fallback
        return sanitized

    def _build_human_friendly_temp_path(self, base: str, suffix: str, extension: str) -> Path:
        ext = extension if extension.startswith(".") else f".{extension}"
        base_clean = self._sanitize_filename_component(base, fallback="Документ")
        suffix_clean = self._sanitize_filename_component(suffix, fallback="")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name_parts = [base_clean]
        if suffix_clean:
            name_parts.append(suffix_clean)
        name_core = " - ".join(name_parts)
        filename = f"{name_core} ({timestamp}){ext}"
        temp_dir = Path(tempfile.gettempdir())
        candidate = temp_dir / filename
        counter = 1
        while candidate.exists():
            candidate = temp_dir / f"{name_core} ({timestamp}_{counter}){ext}"
            counter += 1
        return candidate

    async def _write_export(self, base: str, suffix: str, content: str, extension: str) -> Path:
        target = Path(tempfile.gettempdir()) / f"aivan_{base}_{suffix}_{uuid.uuid4().hex}{extension}"
        return await write_text_async(target, content)

    @staticmethod
    def _safe_unlink(path: str | Path) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    # ----------------------------------- formatters -----------------------------------

    def _format_summary_result(self, data: Dict[str, Any], message: str) -> str:
        summary_block = (data.get("summary") or {}).get("content") or ""
        structured = (data.get("summary") or {}).get("structured") or {}
        key_points = structured.get("key_points") or []
        deadlines = structured.get("deadlines") or []
        penalties = structured.get("penalties") or []
        checklist = structured.get("actions") or []

        lines = ["<b>Саммаризация документа</b>"]
        if summary_block:
            lines.append(html_escape(summary_block))
        if key_points:
            lines.append("")
            lines.append("<b>Ключевые пункты:</b>")
            for item in key_points[:8]:
                lines.append(f"• {html_escape(str(item))}")
        if deadlines:
            lines.append("")
            lines.append("<b>Сроки:</b>")
            for item in deadlines[:6]:
                lines.append(f"• {html_escape(str(item))}")
        if penalties:
            lines.append("")
            lines.append("<b>Ответственность:</b>")
            for item in penalties[:6]:
                lines.append(f"• {html_escape(str(item))}")
        if checklist:
            lines.append("")
            lines.append("<b>Рекомендуемые действия:</b>")
            for item in checklist[:6]:
                lines.append(f"• {html_escape(str(item))}")
        return "\n".join(lines)

    def _format_translation_result(self, data: Dict[str, Any], message: str) -> str:
        translated = data.get("translated_text") or ""
        src = data.get("source_language") or "?"
        tgt = data.get("target_language") or "?"
        lines = [
            "<b>Перевод документа</b>",
            f"Языковая пара: {html_escape(str(src))} → {html_escape(str(tgt))}",
            "",
            html_escape(translated)[:3500],
        ]
        return "\n".join(lines)

    def _format_anonymize_result(self, data: Dict[str, Any], message: str) -> str:
        anonymized = (data.get("anonymized_text") or "")[:3500]
        report = data.get("anonymization_report") or {}
        total = report.get("total_matches", 0)
        stats = report.get("counters") or {}
        lines = [
            "<b>Анонимизация</b>",
            f"Обнаружено фрагментов: {int(total)}",
            "",
            html_escape(anonymized),
        ]
        if stats:
            lines.append("")
            lines.append("<b>Категории:</b>")
            for key, value in list(stats.items())[:8]:
                lines.append(f"• {html_escape(str(key))}: {value}")
        return "\n".join(lines)

    def _format_risk_result(self, data: Dict[str, Any], message: str) -> str:
        overall = data.get("overall_risk_level") or "не определен"
        recommendations = data.get("recommendations") or []
        pattern_risks = data.get("pattern_risks") or []
        lines = [
            "<b>Анализ рисков</b>",
            f"Общий уровень риска: {html_escape(str(overall))}",
        ]
        if pattern_risks:
            lines.append("")
            lines.append("<b>Важные находки:</b>")
            for item in pattern_risks[:5]:
                desc = item.get("description") or ""
                level = item.get("risk_level") or ""
                lines.append(f"• {html_escape(str(level))}: {html_escape(desc)}")
        if recommendations:
            lines.append("")
            lines.append("<b>Рекомендации:</b>")
            for rec in recommendations[:6]:
                lines.append(f"• {html_escape(str(rec))}")
        return "\n".join(lines)

    def _format_lawsuit_result(self, data: Dict[str, Any], message: str) -> str:
        analysis: dict[str, Any] = data.get("analysis") or {}
        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Анализ искового заявления"

        divider = "──────────────────────────────"
        lines: list[str] = [
            f"📄 {title}",
            divider,
            "",
            "✨ Документ успешно создан!",
            "📎 Формат: DOCX",
        ]

        summary = str(analysis.get("summary") or "").strip()
        if summary:
            summary_clean = re.sub(r"\s+", " ", summary)
            if len(summary_clean) > 280:
                summary_clean = summary_clean[:277].rstrip() + "..."
            lines.extend(["", f"📝 {summary_clean}"])

        lines.extend(["", "💡 Проверьте содержимое и при необходимости внесите правки."])

        if data.get("truncated"):
            lines.extend(["", "⚠️ Анализ выполнен по усечённому тексту документа."])

        return "\n".join(lines).strip()

    def _format_chat_loaded(self, data: Dict[str, Any], message: str) -> str:
        info = data.get("document_info") or {}
        metadata = info.get("metadata") or {}
        doc_id = data.get("document_id")
        top_keywords: List[str] = []
        if doc_id and doc_id in self.chat.loaded_documents:
            try:
                source_text = self.chat.loaded_documents[doc_id]["text"]
                top_keywords = TextProcessor.top_keywords(source_text, limit=5)
            except Exception:
                top_keywords = []
        lines = [
            "<b>Документ готов к чату</b>",
            "Задайте вопрос текстом в этом диалоге. Для выхода используйте /cancel",
            "",
            f"Объем: {metadata.get('words', 0)} слов",
            f"Число предложений: {metadata.get('sentences', 0)}",
        ]
        if top_keywords:
            lines.append("")
            lines.append("<b>Ключевые слова:</b> " + ", ".join(top_keywords))
        return "\n".join(lines)

    def _format_ocr_result(self, data: Dict[str, Any], message: str) -> str:
        text = (data.get("recognized_text") or "")[:3500]
        confidence = float(data.get("confidence_score", 0.0))
        lines = [
            "<b>Распознанный текст</b>",
            f"Точность: {confidence:.1f}%",
            "",
            html_escape(text),
        ]
        return "\n".join(lines)

    def _format_generic_result(self, data: Dict[str, Any], message: str) -> str:
        return html_escape(message or "Готово")

    @staticmethod
    def _build_lawsuit_markdown(analysis: Dict[str, Any]) -> str:
        lines = ["# Анализ искового заявления", ""]

        summary = str(analysis.get("summary") or "").strip()
        if summary:
            lines.extend(["## ⚖️ Резюме", summary, ""])

        parties = analysis.get("parties") or {}
        party_lines: list[str] = []
        if parties.get("plaintiff"):
            party_lines.append(f"- Истец: {parties['plaintiff']}")
        if parties.get("defendant"):
            party_lines.append(f"- Ответчик: {parties['defendant']}")
        for item in parties.get("other") or []:
            text = str(item or "").strip()
            if text:
                party_lines.append(f"- Участник: {text}")
        if party_lines:
            lines.extend(["## 👥 Стороны", *party_lines, ""])

        def append_block(key: str, values: Any) -> None:
            cleaned = [str(value or "").strip() for value in (values or []) if str(value or "").strip()]
            if not cleaned:
                return
            icon, title = _LAWSUIT_SECTION_META[key]
            lines.append(f"## {icon} {title}")
            for entry in cleaned:
                lines.append(f"- {entry}")
            lines.append("")

        for section_key in (
            "demands",
            "legal_basis",
            "evidence",
            "strengths",
            "risks",
            "missing_elements",
            "recommendations",
            "procedural_notes",
        ):
            append_block(section_key, analysis.get(section_key))

        confidence = str(analysis.get("confidence") or "").strip()
        if confidence:
            lines.extend(["", f"_Уверенность анализа: {confidence}_"])

        return "\n".join(lines).strip()
