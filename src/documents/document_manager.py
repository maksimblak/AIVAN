"""
High level orchestration layer for document-related processors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, List, Mapping

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
_TYPE_ICON_MAP: Dict[str, str] = {
    "фио": "👤",
    "лицо": "👤",
    "адрес": "📍",
    "телефон": "☎️",
    "email": "📧",
    "e_mail": "📧",
    "электронная_почта": "📧",
    "инн": "🧾",
    "огрн": "🧾",
    "огрнип": "🧾",
    "паспорт": "🪪",
    "документ": "📄",
    "полис": "📄",
    "снилс": "🆔",
    "банковские_реквизиты": "🏦",
    "банк_реквизит": "🏦",
    "банковский_реквизит": "🏦",
    "банк": "🏦",
    "счет": "🏦",
    "бик": "🏦",
    "iban": "🏦",
    "swift": "🏦",
    "расчетный_счет": "🏦",
    "корреспондентский_счет": "🏦",
    "банковская_карта": "💳",
    "карта": "💳",
    "карта_банка": "💳",
    "организация": "🏢",
    "компания": "🏢",
    "юр_лицо": "🏢",
    "дата": "🗓️",
    "табельный_номер": "🆔",
    "регистрационный_номер": "🆔",
    "регистрационный_знак": "🚘",
    "птс": "🚗",
    "водительское_удостоверение": "🚗",
    "права": "🚗",
    "vin": "🚘",
    "номер": "🔢",
    "домен": "🌐",
    "url": "🔗",
    "ссылка": "🔗",
    "страница": "🔗",
    "адрес_сайта": "🔗",
}


def _resolve_type_icon(label: str) -> str:
    if not label:
        return "•"
    normalized = re.sub(r"[\s\-—–_/\.]+", "_", label.strip().lower())
    return _TYPE_ICON_MAP.get(normalized, "•")


def _is_blank_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return all(_is_blank_field(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return all(_is_blank_field(v) for v in value)
    return False


def _try_parse_lawsuit_json(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    def _attempt(text: str) -> dict[str, Any] | None:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    direct = _attempt(raw)
    if direct:
        return direct

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return _attempt(raw[start : end + 1])

    for line in raw.splitlines():
        candidate = _attempt(line)
        if candidate:
            return candidate

    return None


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
        self.anonymizer = DocumentAnonymizer(openai_service=openai_service, settings=settings)
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
            except Exception:  # noqa: BLE001
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
        rendered = formatter(result.data, result.message)
        return self._sanitize_telegram_html(rendered)

    @staticmethod
    def _sanitize_telegram_html(text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("\r", "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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
            summary_block = (result.data.get("summary") or {}).get("content") or ""
            structured = (result.data.get("summary") or {}).get("structured") or {}
            docx_path = await self._build_docx_summary(base_name, summary_block, structured)
            exports.append({"path": str(docx_path), "format": "docx", "label": "Выжимка (DOCX)"})

        elif operation == "translate":
            translated = (result.data.get("translated_text") or "").strip()
            if translated:
                path = await self._write_export(base_name, "translation", translated, ".txt")
                exports.append({"path": str(path), "format": "txt", "label": "Перевод"})

        elif operation == "anonymize":
            docx_ready = result.data.get("anonymized_docx")
            if docx_ready:
                exports.append(
                    {
                        "path": str(docx_ready),
                        "format": "docx",
                        "label": "Анонимизированный документ",
                    }
                )
            else:
                anonymized = (result.data.get("anonymized_text") or "").strip()
                if anonymized:
                    docx_path = await self._build_docx_anonymized(base_name, anonymized)
                    exports.append(
                        {
                            "path": str(docx_path),
                            "format": "docx",
                            "label": "Анонимизированный документ",
                        }
                    )

        elif operation == "analyze_risks":
            docx_path = await self._build_docx_risk(base_name, result.data or {})
            exports.append(
                {"path": str(docx_path), "format": "docx", "label": "Анализ рисков (DOCX)"}
            )

        elif operation == "lawsuit_analysis":
            analysis = result.data.get("analysis") or {}
            markdown = (result.data.get("markdown") or "").strip()
            if not markdown:
                markdown = self._build_lawsuit_markdown(analysis).strip()
            if not markdown:
                markdown = str(result.data.get("fallback_markdown") or "").strip()
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
                docx_path = await self._build_docx_ocr(base_name, recognized)
                exports.append(
                    {"path": str(docx_path), "format": "docx", "label": "Распознанный текст (DOCX)"}
                )

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
        target = (
            Path(tempfile.gettempdir()) / f"aivan_{base}_{suffix}_{uuid.uuid4().hex}{extension}"
        )
        return await write_text_async(target, content)

    async def _build_docx_from_markdown_safe(
        self, base_name: str, suffix: str, markdown: str
    ) -> Path:
        docx_path = self._build_human_friendly_temp_path(base_name, suffix, ".docx")
        content = (markdown or "").strip()
        if not content:
            content = f"# {suffix or 'Документ'}\n\n(данные отсутствуют)"
        try:
            await asyncio.to_thread(build_docx_from_markdown, content, str(docx_path))
        except DocumentDraftingError as exc:
            raise ProcessingError(str(exc), "DOCX_EXPORT_ERROR") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Ошибка при формировании DOCX-файла", "DOCX_EXPORT_ERROR"
            ) from exc
        return docx_path

    @staticmethod
    def _safe_unlink(path: str | Path) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    async def _build_docx_anonymized(self, base_name: str, anonymized_text: str) -> Path:
        markdown = self._anonymized_to_markdown(anonymized_text)
        return await self._build_docx_from_markdown_safe(base_name, "анонимизация", markdown)

    async def _build_docx_summary(
        self,
        base_name: str,
        summary_text: str,
        structured: Mapping[str, Any] | Dict[str, Any] | None,
    ) -> Path:
        sections: list[str] = ["# Краткая выжимка"]
        summary_clean = (summary_text or "").strip()
        if summary_clean:
            sections.append(summary_clean)

        def _add_list(title: str, items: Any) -> None:
            values = [str(item).strip() for item in (items or []) if str(item).strip()]
            if not values:
                return
            sections.append(f"## {title}")
            sections.extend(f"- {value}" for value in values)

        data = dict(structured or {})
        _add_list("Ключевые пункты", data.get("key_points"))
        _add_list("Сроки", data.get("deadlines"))
        _add_list("Ответственность", data.get("penalties"))
        _add_list("Рекомендуемые действия", data.get("actions"))

        markdown = "\n\n".join(sections)
        return await self._build_docx_from_markdown_safe(base_name, "выжимка", markdown)

    async def _build_docx_risk(self, base_name: str, risk_data: Mapping[str, Any]) -> Path:
        data = dict(risk_data or {})
        sections: list[str] = ["# Анализ рисков"]
        overall = str(data.get("overall_risk_level") or "не определён")
        sections.append(f"**Общий уровень риска:** {overall}")

        summary = (data.get("ai_analysis") or {}).get("summary") or ""
        if summary:
            sections.append("")
            sections.append(summary.strip())

        def _normalise(text: str, limit: int | None = None) -> str:
            base = re.sub(r"\s+", " ", text or "").strip()
            if limit and len(base) > limit:
                return base[: limit - 3].rstrip() + "..."
            return base

        def _risk_icon(level: str) -> str:
            mapping = {
                "critical": "❗",
                "high": "⚠️",
                "medium": "🔶",
                "low": "🔹",
            }
            return mapping.get(level.lower(), "•")

        def _format_risk_block(entry: Mapping[str, Any]) -> list[str]:
            level = str(entry.get("risk_level") or entry.get("level") or "").strip().lower()
            icon = _risk_icon(level)
            desc = _normalise(
                str(entry.get("description") or entry.get("note") or entry.get("text") or ""), 320
            )
            snippet = _normalise(str(entry.get("clause_text") or ""), 420)
            hint = _normalise(str(entry.get("strategy_hint") or ""), 320)
            law_refs = [
                _normalise(str(ref), 160)
                for ref in (entry.get("law_refs") or [])
                if str(ref).strip()
            ]
            header_level = level.upper() if level else "N/A"
            header = (
                f"- **{icon} [{header_level}] {desc}**"
                if desc
                else f"- **{icon} [{header_level}] Риск без описания**"
            )
            block = [header]
            if snippet:
                block.append(f"  - Фрагмент: {snippet}")
            if hint:
                block.append(f"  - Что сделать: {hint}")
            if law_refs:
                block.append(f"  - Нормы: {', '.join(law_refs[:4])}")
            return block

        def _append_risks(title: str, items: Any) -> None:
            seen: set[tuple[str, str]] = set()
            blocks: list[str] = []
            for entry in items or []:
                desc_key = _normalise(
                    str(entry.get("description") or entry.get("note") or ""), None
                ).lower()
                snippet_key = _normalise(str(entry.get("clause_text") or ""), None).lower()
                key = (desc_key, snippet_key)
                if desc_key and key in seen:
                    continue
                seen.add(key)
                blocks.extend(_format_risk_block(entry))
            if blocks:
                sections.append(f"## {title}")
                sections.extend(blocks)

        _append_risks("Проверка по шаблонам", data.get("pattern_risks"))
        ai_risks = ((data.get("ai_analysis") or {}).get("risks")) or []
        _append_risks("Выявленные риски", ai_risks)

        recommendations = [
            str(item).strip() for item in data.get("recommendations") or [] if str(item).strip()
        ]
        if recommendations:
            sections.append("## Рекомендации")
            sections.extend(f"- {text}" for text in recommendations)

        compliance = (data.get("legal_compliance") or {}).get("violations") or []
        _append_risks("Нарушения", compliance)

        markdown = "\n\n".join(sections)
        return await self._build_docx_from_markdown_safe(base_name, "анализ рисков", markdown)

    async def _build_docx_ocr(self, base_name: str, recognized_text: str) -> Path:
        text = (recognized_text or "").strip()
        markdown = "# Распознанный текст\n\n" + (text if text else "(данные отсутствуют)")
        return await self._build_docx_from_markdown_safe(base_name, "распознавание", markdown)

    @staticmethod
    def _anonymized_to_markdown(text: str) -> str:
        if not text.strip():
            return ""
        paragraphs = text.split("\n\n")
        pieces = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            if stripped.startswith("• ") or stripped.startswith("- "):
                lines = [line.strip() for line in stripped.split("\n") if line.strip()]
                pieces.extend(lines)
            else:
                pieces.append(stripped)
        return "\n\n".join(pieces)

    # ----------------------------------- formatters -----------------------------------

    def _format_summary_result(self, data: Dict[str, Any], message: str) -> str:
        summary_payload = data.get("summary") or {}
        summary_block = summary_payload.get("content") or ""
        structured = summary_payload.get("structured") or {}

        key_points = structured.get("key_points") or []
        deadlines = structured.get("deadlines") or []
        penalties = structured.get("penalties") or []
        checklist = structured.get("actions") or []

        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Краткая выжимка документа"

        divider_html = "<code>" + ("─" * 30) + "</code>"
        title_html = html_escape(title)

        lines: list[str] = [
            f"<b>📄 {title_html}</b>",
            divider_html,
            "",
            "<b>✨ Краткая выжимка готова!</b>",
            "📎 <b>Формат:</b> DOCX",
        ]

        structured_summary = str(structured.get("summary") or "").strip()
        preview_source = structured_summary or summary_block
        preview_flat = re.sub(r"\s+", " ", str(preview_source)).strip()
        if preview_flat:
            if len(preview_flat) > 280:
                preview_flat = preview_flat[:277].rstrip() + "..."
            lines.extend(["", f"<b>📝 Кратко:</b> {html_escape(preview_flat)}"])

        def append_section(
            title: str, icon: str, items: list[Any], limit: int = 5, *, clip: int = 220
        ) -> None:
            if not items:
                return
            lines.append("")
            lines.append(f"<b>{icon} {title}</b>")
            for entry in items[:limit]:
                text = str(entry or "").strip()
                if not text:
                    continue
                flattened = re.sub(r"\s+", " ", text)
                if clip and len(flattened) > clip:
                    flattened = flattened[: clip - 3].rstrip() + "..."
                lines.append(f"• {html_escape(flattened)}")

        append_section("Основное", "📌", key_points, limit=6)
        append_section("Сроки", "⏰", deadlines, limit=5)
        append_section("Ответственность", "⚖️", penalties, limit=5)
        append_section("Рекомендуемые действия", "✅", checklist, limit=6)

        lines.extend(["", "<i>💡 Проверьте содержимое и при необходимости внесите правки.</i>"])

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
        """Форматирование результата анонимизации с улучшенным UI."""
        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Анонимизация"

        report = data.get("anonymization_report") or {}
        counters = report.get("statistics") or report.get("counters") or {}
        total_masked = report.get("processed_items")
        if total_masked is None:
            total_masked = report.get("total_matches")
        if total_masked is None and counters:
            try:
                total_masked = sum(int(v) for v in counters.values())
            except Exception:  # noqa: BLE001
                total_masked = None

        try:
            total_int = int(total_masked) if total_masked is not None else None
        except (TypeError, ValueError):
            total_int = None

        lines: list[str] = [
            f"🕶️ <b>{title}</b>",
            "",
            "✨ Документ обезличен",
        ]

        if total_int is not None and total_int > 0:
            lines.append(f"🛡️ Замен: <b>{total_int}</b>")

        # Собираем уникальные типы замен
        replacements_map = data.get("anonymization_map") or {}
        processed_items = report.get("processed_items") or []
        type_labels = report.get("type_labels") or {}

        # Группируем по типам
        types_count: dict[str, int] = {}
        if processed_items:
            for item in processed_items:
                label = str(item.get("label") or "").strip()
                if not label:
                    item_type = str(item.get("type") or "").strip()
                    label = str(type_labels.get(item_type, "") or "").strip()
                if label:
                    types_count[label] = types_count.get(label, 0) + 1

        # Показываем статистику по типам
        if types_count:
            type_parts = []
            for type_name, count in sorted(types_count.items(), key=lambda x: -x[1])[:5]:
                icon = _resolve_type_icon(type_name)
                type_parts.append(f"{icon} {type_name}: {count}")

            if type_parts:
                lines.append("")
                lines.append("<b>📊 Типы данных:</b>")
                for part in type_parts:
                    lines.append(part)

        # Топ-3 примера замен (компактно)
        if replacements_map and processed_items:
            seen_pairs: set[tuple[str, str]] = set()
            display_rows: list[str] = []
            for item in processed_items[:10]:  # Проверяем больше элементов
                original = str(item.get("value") or "").strip()
                if not original:
                    continue
                replacement_raw = replacements_map.get(original)
                if replacement_raw is None:
                    continue

                original_clean = re.sub(r"\s+", " ", original).strip()
                if len(original_clean) > 40:
                    original_clean = original_clean[:37].rstrip() + "..."

                replacement_display = replacement_raw.strip()
                if not replacement_display:
                    replacement_display = "[удалено]"
                if len(replacement_display) > 30:
                    replacement_display = replacement_display[:27].rstrip() + "..."

                key = (original.strip().lower(), replacement_display.lower())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                display_rows.append(
                    f"{html_escape(original_clean)} → {html_escape(replacement_display)}"
                )
                if len(display_rows) >= 3:
                    break

            if display_rows:
                lines.extend(["", "<b>🔄 Примеры:</b>"])
                for row in display_rows:
                    lines.append(f"• {row}")

        # Превью (короче)
        preview_source = str(data.get("anonymized_text") or "")
        preview_clean = re.sub(r"\s+", " ", preview_source).strip()
        if preview_clean:
            if len(preview_clean) > 200:
                preview_clean = preview_clean[:197].rstrip() + "..."
            lines.extend(["", f"<i>📝 {html_escape(preview_clean)}</i>"])

        lines.extend(["", "📎 Документ: <b>DOCX</b>"])

        return "\n".join(lines)

    def _format_risk_result(self, data: Dict[str, Any], message: str) -> str:
        """Форматирование результата анализа рисков с улучшенным UI."""
        recommendations = data.get("recommendations") or []
        pattern_risks = data.get("pattern_risks") or []
        ai_analysis = data.get("ai_analysis") or {}
        ai_risks = ai_analysis.get("risks") or []
        ai_summary = str(ai_analysis.get("summary") or "").strip()

        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Анализ рисков"

        title_html = html_escape(title)

        level_meta = {
            "critical": ("🔴", "критический"),
            "high": ("🟠", "высокий"),
            "medium": ("🟡", "средний"),
            "low": ("⚪", "низкий"),
        }
        severity_order = ["critical", "high", "medium", "low"]

        # Объединяем все риски и дедуплицируем
        all_risks_raw: list[Mapping[str, Any]] = []
        all_risks_raw.extend(pattern_risks)
        all_risks_raw.extend(ai_risks)

        # Дедупликация по описанию и фрагменту
        seen_keys: set[tuple[str, str]] = set()
        all_risks: list[Mapping[str, Any]] = []
        for risk in all_risks_raw:
            desc = str(risk.get("description") or risk.get("note") or "").strip().lower()
            snippet = str(risk.get("clause_text") or "").strip().lower()
            key = (re.sub(r"\s+", " ", desc), re.sub(r"\s+", " ", snippet))
            if key not in seen_keys and desc:
                seen_keys.add(key)
                all_risks.append(risk)

        overall_level = str(data.get("overall_risk_level") or "").lower().strip()
        overall_icon, overall_label = level_meta.get(overall_level, ("⚪", "нет данных"))
        overall_label_display = overall_label.capitalize() if overall_label else "Не определён"

        # Header
        lines: list[str] = [
            f"<b>⚖️ {title_html}</b>",
            "",
            f"✨ <b>Анализ выполнен</b> • {overall_icon} {overall_label_display}",
        ]

        if all_risks:
            # Подсчет по уровням
            counts: dict[str, int] = {lvl: 0 for lvl in severity_order}
            for item in all_risks:
                level = str(item.get("risk_level") or item.get("level") or "").lower().strip()
                if level in counts:
                    counts[level] += 1

            # Компактная статистика
            stat_parts = []
            for level in severity_order:
                count = counts.get(level, 0)
                if count > 0:
                    icon, label = level_meta[level]
                    stat_parts.append(f"{icon} {count}")

            if stat_parts:
                lines.append(f"📊 <b>Обнаружено:</b> {' | '.join(stat_parts)}")

            # Сортируем риски по важности
            def _severity_rank(item: Mapping[str, Any]) -> tuple[int, str]:
                level = str(item.get("risk_level") or item.get("level") or "").lower().strip()
                rank = (
                    severity_order.index(level) if level in severity_order else len(severity_order)
                )
                return rank, str(item.get("description") or item.get("note") or "")

            sorted_risks = sorted(all_risks, key=_severity_rank)

            # Показываем топ-4 риска
            lines.append("")
            lines.append("<b>🎯 Ключевые риски:</b>")
            for idx, risk in enumerate(sorted_risks[:4], 1):
                level = str(risk.get("risk_level") or risk.get("level") or "").lower().strip()
                icon, label = level_meta.get(level, ("⚪", "-"))
                desc = str(risk.get("description") or risk.get("note") or "").strip()
                if len(desc) > 180:
                    desc = desc[:177].rstrip() + "..."

                hint = str(risk.get("strategy_hint") or "").strip()
                if len(hint) > 150:
                    hint = hint[:147].rstrip() + "..."

                lines.append(f"{idx}. {icon} <b>{html_escape(desc)}</b>")
                if hint:
                    lines.append(f"   <i>→ {html_escape(hint)}</i>")

                # Пустая строка между пунктами (кроме последнего)
                if idx < len(sorted_risks[:4]):
                    lines.append("")
        else:
            lines.append("")
            lines.append("✅ <b>Существенных рисков не обнаружено</b>")

        # Рекомендации (компактно)
        if recommendations:
            unique_recs = list(
                dict.fromkeys(str(r).strip() for r in recommendations if str(r).strip())
            )[:3]
            if unique_recs:
                lines.append("")
                lines.append("<b>💡 Рекомендации:</b>")
                for idx, rec in enumerate(unique_recs, 1):
                    if len(rec) > 200:
                        rec = rec[:197].rstrip() + "..."
                    lines.append(f"• {html_escape(rec)}")

                    # Пустая строка между рекомендациями (кроме последней)
                    if idx < len(unique_recs):
                        lines.append("")

        # Краткое резюме (если есть)
        if ai_summary:
            summary_compact = re.sub(r"\s+", " ", ai_summary).strip()
            if len(summary_compact) > 200:
                summary_compact = summary_compact[:197].rstrip() + "..."
            lines.extend(["", f"<i>📝 {html_escape(summary_compact)}</i>"])

        lines.extend(["", "📎 Полный отчёт: <b>DOCX</b>"])

        return "\n".join(lines)

    @staticmethod
    def _format_risk_badge(
        item: Mapping[str, Any], level_meta: Mapping[str, tuple[str, str]]
    ) -> str:
        level = str(item.get("risk_level") or item.get("level") or "").lower().strip()
        icon, label = level_meta.get(level, ("⚪", level or "-"))
        desc = str(item.get("description") or item.get("note") or "").strip()
        if len(desc) > 160:
            desc = desc[:157].rstrip() + "..."
        desc_html = html_escape(desc)
        hint = str(item.get("strategy_hint") or "").strip()
        hint_html = html_escape(hint) if hint else ""
        badge = f"{icon} <b>{label.capitalize()}</b>"
        if desc_html:
            badge += f" — {desc_html}"
        if hint_html:
            badge += f"\n    <i>{hint_html}</i>"
        return badge

    def _format_lawsuit_result(self, data: Dict[str, Any], message: str) -> str:
        analysis: dict[str, Any] = data.get("analysis") or {}
        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Анализ искового заявления"

        divider = "──────────────────────────────"
        title_html = html_escape(title)
        divider_html = f"<code>{divider}</code>"
        lines: list[str] = [
            f"<b>📄 {title_html}</b>",
            divider_html,
            "",
            "<b>✨ Документ успешно проанализирован!</b>",
            "📎 <b>Формат:</b> DOCX",
        ]

        def _append_paragraph_block(title: str, text: str) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            lines.append("")
            lines.append(f"<b>{title}</b>")
            for paragraph in re.split(r"\n{2,}", cleaned):
                paragraph = paragraph.strip()
                if paragraph:
                    lines.append(html_escape(paragraph))

        def _append_list_block(
            title: str, items: list[str], *, icon: str = "•", limit: int | None = None
        ) -> None:
            cleaned_items = [str(item or "").strip() for item in items if str(item or "").strip()]
            if not cleaned_items:
                return
            overflow = 0
            if limit is not None and len(cleaned_items) > limit:
                overflow = len(cleaned_items) - limit
                cleaned_items = cleaned_items[:limit]
            lines.append("")
            lines.append(f"<b>{title}</b>")
            for entry in cleaned_items:
                lines.append(f"{icon} {html_escape(entry)}")
            if overflow:
                lines.append(f"… ещё {overflow} пункт(ов)")

        summary = str(analysis.get("summary") or "").strip()
        if summary:
            _append_paragraph_block("📝 Кратко:", summary)

        parties = analysis.get("parties") or {}
        party_items: list[str] = []
        if parties.get("plaintiff"):
            party_items.append(f"Истец: {parties['plaintiff']}")
        if parties.get("defendant"):
            party_items.append(f"Ответчик: {parties['defendant']}")
        for item in parties.get("other") or []:
            text = str(item or "").strip()
            if text:
                party_items.append(f"Участник: {text}")
        if party_items:
            _append_list_block("👥 Стороны", party_items)

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
            icon, title = _LAWSUIT_SECTION_META[section_key]
            section_items = analysis.get(section_key) or []
            _append_list_block(f"{icon} {title}", section_items, limit=8)

        overall_assessment = str(analysis.get("overall_assessment") or "").strip()
        if overall_assessment:
            _append_paragraph_block("📊 Общая оценка", overall_assessment)

        risk_highlights = analysis.get("risk_highlights") or []
        _append_list_block("🚨 Риски и ошибки", risk_highlights, limit=6)

        strategy = analysis.get("strategy") or {}
        strategy_lines: list[str] = []
        if strategy.get("success_probability"):
            strategy_lines.append(f"Вероятность успеха: {strategy['success_probability']}")
        strategy_lines.extend(strategy.get("actions") or [])
        if strategy_lines:
            _append_list_block("🧭 Судебная стратегия", strategy_lines, limit=6)

        case_law_items = analysis.get("case_law") or []
        formatted_cases: list[str] = []
        for entry in case_law_items:
            if not isinstance(entry, Mapping):
                continue
            court = str(entry.get("court") or "").strip()
            year = str(entry.get("year") or "").strip()
            summary_text = str(entry.get("summary") or "").strip()
            link = str(entry.get("link") or "").strip()
            parts: list[str] = []
            if court and year:
                parts.append(html_escape(f"{court} ({year})"))
            elif court:
                parts.append(html_escape(court))
            elif year:
                parts.append(html_escape(year))
            if summary_text:
                parts.append(html_escape(summary_text))
            if link:
                parts.append(f"Ссылка: {html_escape(link)}")
            combined = " — ".join(parts)
            if combined:
                formatted_cases.append(combined)
        _append_list_block("📖 Судебная практика", formatted_cases, limit=6)

        improvement_steps = analysis.get("improvement_steps") or []
        _append_list_block("🔧 Рекомендации по улучшению", improvement_steps, limit=6)

        confidence = str(analysis.get("confidence") or "").strip()
        if confidence:
            lines.append("")
            lines.append(f"<i>Уверенность анализа: {html_escape(confidence)}</i>")

        if data.get("structured_payload_repaired"):
            lines.append("")
            lines.append(
                "<i>⚠️ Ответ модели был усечён; структура восстановлена автоматически, проверьте данные вручную.</i>"
            )

        if data.get("fallback_used"):
            lines.append("")
            lines.append(
                "<i>⚠️ Структурированный ответ не получен, добавлен текстовый ответ модели.</i>"
            )

        meta_notes = [
            html_escape(str(note or "").strip())
            for note in (analysis.get("meta_notes") or [])
            if str(note or "").strip()
        ]
        if meta_notes:
            lines.append("")
            for note in meta_notes:
                lines.append(f"<i>{note}</i>")

        lines.append("")
        lines.append("<i>💡 Проверьте содержимое и при необходимости внесите правки.</i>")

        if data.get("truncated"):
            lines.extend(["", "<i>⚠️ Анализ выполнен по усечённому тексту документа.</i>"])

        report = data.get("anonymization_report") or {}
        engine = report.get("engine")
        notes = report.get("notes") or []
        if engine == "openai":
            lines.append("")
            lines.append("<i>✔️ Анонимизация выполнена.</i>")
        elif engine == "fallback":
            lines.append("")
            lines.append("<i>⚠️ Использован резервный режим анонимизации.</i>")
        if engine == "pattern":
            stats = report.get("statistics") or {}
            total_masked = sum(stats.values())
            lines.append("")
            lines.append(f"<i>🔐 Обезличено фрагментов: {total_masked}</i>")
        for note in notes:
            note_text = str(note or "").strip()
            if note_text:
                lines.append("")
                lines.append(f"<i>{html_escape(note_text)}</i>")

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
        """Форматирование результата распознавания текста с улучшенным UI."""
        text = (data.get("recognized_text") or "").strip()
        confidence = float(data.get("confidence_score", 0.0))

        doc_info = data.get("document_info") or {}
        original_name = str(doc_info.get("original_name") or "").strip()
        title = Path(original_name).stem if original_name else ""
        title = title.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = "Распознавание текста"

        title_html = html_escape(title)

        # Определяем качество распознавания
        if confidence >= 90:
            quality_icon = "🟢"
            quality_label = "Отличное"
        elif confidence >= 75:
            quality_icon = "🟡"
            quality_label = "Хорошее"
        elif confidence >= 50:
            quality_icon = "🟠"
            quality_label = "Среднее"
        else:
            quality_icon = "🔴"
            quality_label = "Низкое"

        lines: list[str] = [
            f"<b>📸 {title_html}</b>",
            "",
            f"✨ <b>Текст распознан</b> • {quality_icon} {quality_label}",
            f"📊 <b>Точность:</b> {confidence:.1f}%",
        ]

        # Показываем превью текста (первые 400 символов)
        if text:
            text_preview = text[:400]
            if len(text) > 400:
                text_preview += "..."

            # Подсчитываем базовую статистику
            words_count = len(text.split())
            lines_count = len([line for line in text.split("\n") if line.strip()])

            lines.extend(
                [
                    "",
                    f"📝 <b>Объём:</b> {words_count} слов • {lines_count} строк",
                    "",
                    "<b>Превью:</b>",
                    f"<i>{html_escape(text_preview)}</i>",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "⚠️ <i>Текст не распознан или документ пуст</i>",
                ]
            )

        lines.extend(["", "📎 Полный текст: <b>DOCX</b>"])

        return "\n".join(lines)

    def _format_generic_result(self, data: Dict[str, Any], message: str) -> str:
        return html_escape(message or "Готово")

    @staticmethod
    def _build_lawsuit_markdown(analysis: Dict[str, Any]) -> str:
        analysis = dict(analysis or {})

        fallback_raw = str(analysis.get("fallback_raw_text") or "").strip()
        parsed_from_fallback = _try_parse_lawsuit_json(fallback_raw) if fallback_raw else None
        if parsed_from_fallback:
            parsed_from_fallback = dict(parsed_from_fallback)

            for key in ("summary", "overall_assessment", "confidence"):
                if _is_blank_field(analysis.get(key)) and not _is_blank_field(
                    parsed_from_fallback.get(key)
                ):
                    analysis[key] = parsed_from_fallback.get(key)

            for key in (
                "demands",
                "legal_basis",
                "evidence",
                "strengths",
                "risks",
                "missing_elements",
                "recommendations",
                "procedural_notes",
                "risk_highlights",
                "case_law",
                "improvement_steps",
            ):
                if _is_blank_field(analysis.get(key)) and not _is_blank_field(
                    parsed_from_fallback.get(key)
                ):
                    analysis[key] = parsed_from_fallback.get(key)

            parties = dict(analysis.get("parties") or {})
            fallback_parties = parsed_from_fallback.get("parties") or {}
            for part_key in ("plaintiff", "defendant", "other"):
                if _is_blank_field(parties.get(part_key)) and not _is_blank_field(
                    fallback_parties.get(part_key)
                ):
                    parties[part_key] = fallback_parties.get(part_key)
            analysis["parties"] = parties

            strategy = dict(analysis.get("strategy") or {})
            fallback_strategy = parsed_from_fallback.get("strategy") or {}
            if _is_blank_field(strategy.get("success_probability")) and not _is_blank_field(
                fallback_strategy.get("success_probability")
            ):
                strategy["success_probability"] = fallback_strategy.get("success_probability")
            if _is_blank_field(strategy.get("actions")) and not _is_blank_field(
                fallback_strategy.get("actions")
            ):
                strategy["actions"] = fallback_strategy.get("actions")
            analysis["strategy"] = strategy

            meta_notes = [
                str(note or "").strip()
                for note in (analysis.get("meta_notes") or [])
                if str(note or "").strip()
            ]
            meta_notes.append(
                "Использован текст модели для восстановления структурированного отчёта."
            )
            analysis["meta_notes"] = meta_notes
            analysis["fallback_raw_text"] = ""

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
            cleaned = [
                str(value or "").strip() for value in (values or []) if str(value or "").strip()
            ]
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

        if analysis.get("structured_payload_repaired"):
            lines.extend(
                [
                    "",
                    "_Ответ модели был усечён; структура восстановлена автоматически — перепроверьте содержание._",
                ]
            )

        meta_notes = [
            str(note or "").strip()
            for note in (analysis.get("meta_notes") or [])
            if str(note or "").strip()
        ]
        if meta_notes:
            lines.extend(["", "## Примечания", *meta_notes])

        fallback_raw = str(analysis.get("fallback_raw_text") or "").strip()
        if fallback_raw:
            lines.extend(["", "## Ответ модели (fallback)", fallback_raw])

        return "\n".join(lines).strip()
