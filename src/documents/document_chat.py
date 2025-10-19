"""
Модуль "Чат с документом"
Интерактивное взаимодействие с загруженными документами через вопросы-ответы.

Возможности:
- Гибридный ретривер: ключевые слова + эвристика по совпадениям.
- Чанки с глобальными оффсетами (start/end) для точных цитат.
- Строгий JSON-ответ от LLM и устойчивый парсинг результата.
- Локальный ответ на основе найденных фрагментов, если LLM недоступен.
- Подсветка и возврат фрагментов по span в исходном тексте.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Sequence

from src.core.settings import AppSettings

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

QUESTION_STOPWORDS = {
    "что",
    "это",
    "где",
    "когда",
    "если",
    "как",
    "или",
    "какой",
    "какая",
    "какие",
    "and",
    "the",
    "for",
    "from",
    "with",
    "into",
    "onto",
    "there",
    "here",
}

DOCUMENT_CHAT_SYSTEM = """
Ты — ассистент, отвечающий только на основе данных из контекста.
Верни ТОЛЬКО JSON следующего вида (никаких комментариев снаружи):

{
  "answer": "короткий и точный ответ на языке пользователя",
  "citations": [
    {"chunk_index": 0, "span": {"start": 123, "end": 180}},
    {"chunk_index": 2, "span": {"start": 10, "end": 90}}
  ],
  "confidence": 0.0_to_1.0
}

Правила:
- Если в контексте нет информации, честно укажи это в "answer" и верни пустой список citations.
- Цитируй только реальные позиции из переданного контекста (по индексам символов в тексте соответствующего фрагмента).
- Не выдумывай законы/факты вне контекста. Не ссылайся на внешние источники.
"""

DOCUMENT_CHAT_USER_TMPL = """
Вопрос пользователя:
{question}

Контекст (части документа):
{context}
"""


@dataclass
class Chunk:
    index: int
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


class DocumentChat(DocumentProcessor):
    """Класс для интерактивного чата с документами."""

    def __init__(self, openai_service=None, settings: AppSettings | None = None):
        super().__init__(name="DocumentChat", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service
        self.loaded_documents: dict[str, dict[str, Any]] = {}

        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings

        self.allow_ai = settings.get_bool("CHAT_ALLOW_AI", True)
        self.max_context_chunks = max(1, settings.get_int("CHAT_MAX_CONTEXT_CHUNKS", 3))
        self.chunk_size = settings.get_int("CHAT_CHUNK_SIZE", 3000)
        self.chunk_overlap = settings.get_int("CHAT_CHUNK_OVERLAP", 400)

    # ------------------------------- Загрузка/хранилище -------------------------------

    async def load_document(self, file_path: str | Path, document_id: str | None = None, progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> str:
        """Загрузить документ для чата и вернуть идентификатор сессии."""

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
                logger.debug("DocChat progress callback failed at %s", stage, exc_info=True)

        if not document_id:
            document_id = f"doc_{int(datetime.now().timestamp())}"

        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        await _notify("text_extracted", 20, words=len(cleaned_text.split()))

        chunk_texts = TextProcessor.split_into_chunks(
            cleaned_text, max_chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        await _notify("chunking", 45, chunks=len(chunk_texts))
        chunks_indexed = self._index_chunks(cleaned_text, chunk_texts)
        await _notify("indexing", 75, chunks=len(chunk_texts))

        self.loaded_documents[document_id] = {
            "text": cleaned_text,
            "file_path": str(file_path),
            "loaded_at": datetime.now(),
            "metadata": TextProcessor.extract_metadata(cleaned_text),
            "chunks": chunk_texts,
            "chunks_indexed": chunks_indexed,
        }

        logger.info("Документ %s загружен с ID: %s", file_path, document_id)
        await _notify("completed", 100, chunks=len(chunk_texts))
        return document_id

    async def process(self, file_path: str | Path, progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None, **kwargs) -> DocumentResult:
        """Совместимость с интерфейсом DocumentProcessor — загрузка документа."""
        try:
            document_id = await self.load_document(file_path, progress_callback=progress_callback)
            return DocumentResult.success_result(
                data={
                    "document_id": document_id,
                    "document_info": self.get_document_info(document_id),
                    "message": "Документ загружен и готов для чата",
                },
                message="Документ успешно загружен для интерактивного чата",
            )
        except ProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                f"Ошибка загрузки документа: {exc}", "LOAD_ERROR"
            ) from exc

    def get_document_info(self, document_id: str) -> dict[str, Any]:
        doc = self.loaded_documents.get(document_id)
        if not doc:
            return {}
        return {
            "document_id": document_id,
            "file_path": doc["file_path"],
            "loaded_at": doc["loaded_at"].isoformat(),
            "metadata": doc["metadata"],
            "chunks_count": len(doc["chunks"]),
        }

    def unload_document(self, document_id: str) -> None:
        self.loaded_documents.pop(document_id, None)

    # ------------------------------- Основной API -------------------------------

    async def answer_question(self, document_id: str, question: str) -> DocumentResult:
        question = (question or "").strip()
        if not question:
            raise ProcessingError("Вопрос не должен быть пустым", "EMPTY_QUESTION")

        document = self.loaded_documents.get(document_id)
        if not document:
            raise ProcessingError("Документ не найден. Загрузите его заново.", "DOCUMENT_NOT_FOUND")

        relevant_chunks = self._select_relevant_chunks(document, question)
        context_text = self._build_context(relevant_chunks)

        ai_payload: dict[str, Any] | None = None
        if self.allow_ai and self.openai_service:
            try:
                user_prompt = DOCUMENT_CHAT_USER_TMPL.format(question=question, context=context_text)
                response = await self.openai_service.ask_legal(
                    system_prompt=DOCUMENT_CHAT_SYSTEM,
                    user_text=user_prompt,
                )
                ai_payload = self._parse_llm_payload(response or {}, relevant_chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Document chat LLM call failed: %s", exc, exc_info=True)
                ai_payload = None

        if not ai_payload:
            ai_payload = self._local_answer(question, relevant_chunks)

        answer_text = ai_payload.get("answer", "Не удалось сформировать ответ")
        citations = ai_payload.get("citations", [])
        confidence = float(ai_payload.get("confidence", 0.0))

        data = {
            "document_id": document_id,
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence,
            "used_chunks": [chunk.to_dict() for chunk in relevant_chunks],
        }
        return DocumentResult.success_result(
            data=data,
            message="Ответ сформирован",
        )

    # ------------------------------- Внутренние методы -------------------------------

    def _index_chunks(self, full_text: str, chunk_texts: Sequence[str]) -> List[Chunk]:
        indexed: List[Chunk] = []
        cursor = 0
        for idx, chunk_text in enumerate(chunk_texts):
            snippet = chunk_text.strip()
            if not snippet:
                continue
            start = full_text.find(snippet, cursor)
            if start == -1:
                snippet_short = snippet[: min(len(snippet), 120)]
                start = full_text.find(snippet_short, cursor)
            if start == -1:
                start = full_text.find(snippet)
            if start == -1:
                start = cursor
            end = start + len(snippet)
            cursor = max(end, cursor)
            indexed.append(Chunk(index=idx, start=start, end=end, text=snippet))
        if not indexed and chunk_texts:
            indexed.append(Chunk(index=0, start=0, end=len(chunk_texts[0]), text=chunk_texts[0]))
        return indexed

    def _select_relevant_chunks(self, document: dict[str, Any], question: str) -> List[Chunk]:
        chunks: List[Chunk] = list(document.get("chunks_indexed", []))
        if not chunks:
            return []
        tokens = self._tokenize(question)
        if not tokens:
            return chunks[: self.max_context_chunks]

        scores: List[tuple[float, Chunk]] = []
        for chunk in chunks:
            score = self._score_chunk(chunk, tokens)
            if score > 0:
                scores.append((score, chunk))
        if not scores:
            return chunks[: self.max_context_chunks]

        scores.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for _, chunk in scores[: self.max_context_chunks]]
        return selected

    def _score_chunk(self, chunk: Chunk, tokens: set[str]) -> float:
        text_lower = chunk.text.lower()
        score = 0.0
        for token in tokens:
            occurrences = text_lower.count(token)
            if occurrences:
                score += 1.0 + occurrences * 0.2
        keyword_overlap = tokens.intersection(TextProcessor.top_keywords(chunk.text, limit=8, min_length=3))
        score += len(keyword_overlap) * 0.5
        # Penalise extremely long chunks so that shorter answers have a chance
        length_penalty = math.log1p(len(chunk.text))
        score /= max(length_penalty, 1.0)
        return score

    def _build_context(self, chunks: Sequence[Chunk]) -> str:
        parts: List[str] = []
        for chunk in chunks:
            parts.append(f"[chunk #{chunk.index}]\n{chunk.text}")
        return "\n\n".join(parts)

    def _parse_llm_payload(
        self,
        response: dict[str, Any],
        fallback_chunks: Sequence[Chunk],
    ) -> dict[str, Any] | None:
        if not isinstance(response, dict):
            return None
        text = (response.get("text") or "").strip()
        if not text:
            return None
        payload = self._safe_json_load(text)
        if not payload:
            return None

        answer = str(payload.get("answer") or "").strip()
        citations_raw = payload.get("citations") or []
        confidence = payload.get("confidence", 0.0)

        citations: List[dict[str, Any]] = []
        chunks_by_index = {chunk.index: chunk for chunk in fallback_chunks}
        for item in citations_raw:
            if not isinstance(item, dict):
                continue
            chunk_idx = item.get("chunk_index")
            if chunk_idx not in chunks_by_index:
                continue
            span = item.get("span") or {}
            start = int(span.get("start", 0))
            end = int(span.get("end", start + 200))
            chunk = chunks_by_index[chunk_idx]
            snippet = chunk.text[start:end].strip()
            citations.append(
                {
                    "chunk_index": chunk_idx,
                    "span": {"start": start, "end": end},
                    "snippet": snippet,
                }
            )

        if not answer and fallback_chunks:
            answer = fallback_chunks[0].text[:400]

        return {
            "answer": answer,
            "citations": citations,
            "confidence": float(confidence or 0.0),
        }

    def _local_answer(self, question: str, chunks: Sequence[Chunk]) -> dict[str, Any]:
        if not chunks:
            return {
                "answer": "Не удалось подобрать ответ по документу",
                "citations": [],
                "confidence": 0.0,
            }
        question_tokens = self._tokenize(question)
        best_chunk = max(chunks, key=lambda chunk: self._score_chunk(chunk, question_tokens))
        snippet_length = 400
        snippet = best_chunk.text[:snippet_length].strip()
        return {
            "answer": snippet,
            "citations": [
                {
                    "chunk_index": best_chunk.index,
                    "span": {"start": 0, "end": min(len(best_chunk.text), snippet_length)},
                    "snippet": snippet,
                }
            ],
            "confidence": 0.3,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = set()
        for raw in re.findall(r"\w+", text.lower()):
            if len(raw) < 2:
                continue
            if raw in QUESTION_STOPWORDS:
                continue
            tokens.add(raw)
        return tokens

    @staticmethod
    def _safe_json_load(raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Попробуем найти JSON внутри текста (иногда модель добавляет пояснения)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

