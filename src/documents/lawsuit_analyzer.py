from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[\s\S]*\}")

MAX_INPUT_CHARS = 15_000

LAWSUIT_ANALYSIS_SYSTEM_PROMPT = """
Ты — опытный процессуалист. Анализируешь исковое заявление и оцениваешь его готовность к подаче.
Найди цели иска, сильные и слабые стороны правовой позиции, риски и недостающие элементы.

Строго верни JSON со структурой:
{
  "summary": "краткое резюме 3-4 предложения",
  "parties": {
    "plaintiff": "кто подает иск (если указан)",
    "defendant": "к кому предъявлен",
    "other": ["иные участники или оставь список пустым"]
  },
  "demands": ["перечень требований иска"],
  "legal_basis": ["нормы права, на которые опирается заявитель"],
  "evidence": ["какие доказательства указаны"],
  "strengths": ["сильные стороны позиции или аргументации"],
  "risks": ["риски отказа, слабые места, процессуальные проблемы"],
  "missing_elements": ["чего может не хватать (документы, факты, формулировки)"],
  "recommendations": ["что доработать перед подачей"],
  "procedural_notes": ["важные процессуальные нюансы (подсудность, госпошлина, сроки)"],
  "confidence": "high|medium|low"
}

Если каких-то данных нет — оставь поле пустым или список/строку пустой. Никакого текста вне JSON.
""".strip()

LAWSUIT_ANALYSIS_USER_PROMPT = """
Ниже текст искового заявления. Проанализируй его по схеме из системной инструкции.

Если документ усечён: {truncated_hint}

=== ТЕКСТ ДОКУМЕНТА ===
{document_excerpt}
""".strip()


def _extract_first_json(payload: str) -> dict[str, Any]:
    """Попытаться выделить первый JSON-объект из ответа модели."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = _JSON_RE.search(payload)
        if not match:
            raise ProcessingError("Ответ модели не похож на JSON", "PARSE_ERROR")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ProcessingError("Не удалось разобрать JSON из ответа модели", "PARSE_ERROR") from exc


def _clean_list(items: Any) -> list[str]:
    cleaned: list[str] = []
    if isinstance(items, (list, tuple)):
        for item in items:
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
    elif isinstance(items, str):
        text = items.strip()
        if text:
            cleaned.append(text)
    return cleaned


class LawsuitAnalyzer(DocumentProcessor):
    """Анализирует исковые заявления: требования, доказательства, риски и рекомендации."""

    def __init__(self, openai_service=None):
        super().__init__(name="LawsuitAnalyzer", max_file_size=50 * 1024 * 1024)
        self.openai_service = openai_service
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]

    async def process(
        self,
        file_path: str | Path,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        **_: Any,
    ) -> DocumentResult:
        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        async def _notify(stage: str, percent: float, **payload: Any) -> None:
            if not progress_callback:
                return
            data: dict[str, Any] = {"stage": stage, "percent": float(percent)}
            for key, value in payload.items():
                if value is not None:
                    data[key] = value
            try:
                await progress_callback(data)
            except Exception:
                logger.debug("LawsuitAnalyzer progress callback failed at %s", stage, exc_info=True)

        success, extracted = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {extracted}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(extracted)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        await _notify("text_extracted", 20.0, words=len(cleaned_text.split()))

        truncated = len(cleaned_text) > MAX_INPUT_CHARS
        excerpt = cleaned_text[:MAX_INPUT_CHARS]
        if truncated:
            excerpt += "\n\n[Текст усечён для анализа из-за ограничения по объёму]"

        user_prompt = LAWSUIT_ANALYSIS_USER_PROMPT.format(
            truncated_hint="Документ передан частично, отметь недостающие элементы." if truncated else "Документ передан полностью.",
            document_excerpt=excerpt,
        )

        await _notify("model_request", 45.0)

        response = await self.openai_service.ask_legal(
            system_prompt=LAWSUIT_ANALYSIS_SYSTEM_PROMPT,
            user_text=user_prompt,
        )
        if not response.get("ok"):
            raise ProcessingError(response.get("error") or "Не удалось получить ответ от модели", "OPENAI_ERROR")

        structured_payload = response.get("structured")
        raw_text = (response.get("text") or "").strip()
        if isinstance(structured_payload, Mapping) and structured_payload:
            payload = dict(structured_payload)
        else:
            if not raw_text:
                raise ProcessingError("Пустой ответ модели", "OPENAI_EMPTY")
            payload = _extract_first_json(raw_text)

        analysis = {
            "summary": str(payload.get("summary") or "").strip(),
            "parties": {
                "plaintiff": str((payload.get("parties") or {}).get("plaintiff") or "").strip(),
                "defendant": str((payload.get("parties") or {}).get("defendant") or "").strip(),
                "other": _clean_list((payload.get("parties") or {}).get("other")),
            },
            "demands": _clean_list(payload.get("demands")),
            "legal_basis": _clean_list(payload.get("legal_basis")),
            "evidence": _clean_list(payload.get("evidence")),
            "strengths": _clean_list(payload.get("strengths")),
            "risks": _clean_list(payload.get("risks")),
            "missing_elements": _clean_list(payload.get("missing_elements")),
            "recommendations": _clean_list(payload.get("recommendations")),
            "procedural_notes": _clean_list(payload.get("procedural_notes")),
            "confidence": str(payload.get("confidence") or "").strip(),
        }

        confidence_label = analysis.get("confidence")
        if confidence_label:
            await _notify("analysis_ready", 85.0, note=f"Уверенность: {confidence_label}")
        else:
            await _notify("analysis_ready", 85.0)

        markdown_report = self._build_markdown_report(analysis)

        await _notify("completed", 100.0)

        return DocumentResult.success_result(
            data={
                "analysis": analysis,
                "markdown": markdown_report,
                "raw_response": raw_text,
                "truncated": truncated,
            },
            message="Анализ искового заявления готов",
        )

    @staticmethod
    def _build_markdown_report(analysis: dict[str, Any]) -> str:
        lines = ["# Анализ искового заявления", ""]

        summary = analysis.get("summary")
        if summary:
            lines.extend(["## Резюме", summary.strip(), ""])

        parties = analysis.get("parties") or {}
        party_lines = []
        plaintiff = parties.get("plaintiff")
        defendant = parties.get("defendant")
        if plaintiff:
            party_lines.append(f"- Истец: {plaintiff.strip()}")
        if defendant:
            party_lines.append(f"- Ответчик: {defendant.strip()}")
        others = parties.get("other") or []
        for item in others:
            party_lines.append(f"- Участник: {item.strip()}")
        if party_lines:
            lines.extend(["## Стороны", *party_lines, ""])

        def append_block(title: str, items: list[str]) -> None:
            if not items:
                return
            lines.append(f"## {title}")
            for item in items:
                lines.append(f"- {item.strip()}")
            lines.append("")

        append_block("Требования", analysis.get("demands") or [])
        append_block("Правовое обоснование", analysis.get("legal_basis") or [])
        append_block("Доказательства", analysis.get("evidence") or [])
        append_block("Сильные стороны", analysis.get("strengths") or [])
        append_block("Риски и слабые места", analysis.get("risks") or [])
        append_block("Недостающие элементы", analysis.get("missing_elements") or [])
        append_block("Рекомендации", analysis.get("recommendations") or [])
        append_block("Процессуальные заметки", analysis.get("procedural_notes") or [])

        confidence = analysis.get("confidence")
        if confidence:
            lines.extend(["", f"_Уровень уверенности анализа: {confidence}_"])

        return "\n".join(lines).strip()
