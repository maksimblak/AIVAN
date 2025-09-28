"""
Модуль "Чат с документом"
Интерактивное взаимодействие с загруженными документами через вопросы-ответы.

Новые фичи:
- Гибридный ретривер: BM25-lite + TF-IDF + keyword-оверлап.
- Чанки с глобальными оффсетами (start/end) для точных цитат.
- Строгий JSON-ответ от LLM: {"answer","citations":[{"chunk_index","span":{"start","end"}}], "confidence"}.
- Безопасный парсинг JSON + graceful оффлайн-режим (без API).
- Подсветка и возврат фрагментов по спанам, а не через .replace().
- Эвристический локальный ответ, если LLM отключён/упал.

ENV:
- CHAT_ALLOW_AI=true|false  — разрешать ли вызов LLM.
- CHAT_MAX_CONTEXT_CHUNKS=3 — сколько топ-чанков давать в контекст.
- CHAT_CHUNK_SIZE=3000       — размер чанка при загрузке.
- CHAT_CHUNK_OVERLAP=400     — перекрытие чанков.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

QUESTION_STOPWORDS = {
    "что", "это", "где", "когда", "если", "как", "или",
    "and", "the", "for", "from", "with", "into", "onto", "there", "here",
}

DOCUMENT_CHAT_SYSTEM = """
Ты — ассистент, отвечающий ТОЛЬКО на основе данных из контекста.
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
- Цитируй только существующие позиции из переданного контекста (по индексам символов в тексте соответствующего фрагмента).
- Не выдумывай законы/факты вне контекста. Не ссылаться на внешние источники.
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
    start: int  # глобальный оффсет в исходном документе
    end: int    # глобальный оффсет в исходном документе
    text: str


class DocumentChat(DocumentProcessor):
    """Класс для интерактивного чата с документами"""

    def __init__(self, openai_service=None):
        super().__init__(name="DocumentChat", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service
        self.loaded_documents: dict[str, dict[str, Any]] = {}

        # настройки из окружения
        self.allow_ai = (os.getenv("CHAT_ALLOW_AI", "true").lower() in {"1", "true", "yes", "on"})
        self.max_context_chunks = int(os.getenv("CHAT_MAX_CONTEXT_CHUNKS", "3"))
        self.chunk_size = int(os.getenv("CHAT_CHUNK_SIZE", "3000"))
        self.chunk_overlap = int(os.getenv("CHAT_CHUNK_OVERLAP", "400"))

    # ------------------------------- Загрузка/хранилище -------------------------------

    async def load_document(self, file_path: str | Path, document_id: str | None = None) -> str:
        """
        Загрузить документ для чата

        Returns:
            document_id для использования в чате
        """
        if not document_id:
            document_id = f"doc_{int(datetime.now().timestamp())}"

        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        chunk_texts = TextProcessor.split_into_chunks(
            cleaned_text, max_chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        chunks_indexed = self._index_chunks(cleaned_text, chunk_texts)

        self.loaded_documents[document_id] = {
            "text": cleaned_text,
            "file_path": str(file_path),
            "loaded_at": datetime.now(),
            "metadata": TextProcessor.extract_metadata(cleaned_text),
            "chunks": chunk_texts,
            "chunks_indexed": chunks_indexed,  # список Chunk с глобальными оффсетами
        }

        logger.info("Документ %s загружен с ID: %s", file_path, document_id)
        return document_id

    async def process(self, file_path: str | Path, **kwargs) -> DocumentResult:
        """Соответствие интерфейсу DocumentProcessor — грузим документ и возвращаем ID"""
        try:
            document_id = await self.load_document(file_path)
            return DocumentResult.success_result(
                data={
                    "document_id": document_id,
                    "document_info": self.get_document_info(document_id),
                    "message": "Документ загружен и готов для чата",
                },
                message="Документ успешно загружен для интерактивного чата",
            )
        except Exception as e:
            raise ProcessingError(f"Ошибка загрузки документа: {str(e)}", "LOAD_ERROR")

    def get_document_info(self, document_id: str) -> dict[str, Any]:
        if document_id not in self.loaded_documents:
            return {}
        doc = self.loaded_documents[document_id]
        return {
            "document_id": document_id,
            "file_path": doc["file_path"],
            "loaded_at": doc["loaded_at"].isoformat(),
            "metadata": doc["metadata"],
            "chunks_count": len(doc["chunks"]),
        }

    def get_loaded_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "document_id": doc_id,
                "file_path": data["file_path"],
                "loaded_at": data["loaded_at"].isoformat(),
                "word_count": data["metadata"]["word_count"],
            }
            for doc_id, data in self.loaded_documents.items()
        ]

    def remove_document(self, document_id: str) -> bool:
        if document_id in self.loaded_documents:
            del self.loaded_documents[document_id]
            logger.info("Документ %s удален из памяти", document_id)
            return True
        return False

    def cleanup_old_documents(self, max_age_hours: int = 24):
        now = datetime.now()
        victims = [
            doc_id
            for doc_id, data in self.loaded_documents.items()
            if (now - data["loaded_at"]).total_seconds() / 3600.0 > max_age_hours
        ]
        for doc_id in victims:
            self.remove_document(doc_id)
        if victims:
            logger.info("Очищено %d старых документов", len(victims))

    # --------------------------------- Чат ---------------------------------

    async def chat_with_document(self, document_id: str, question: str) -> dict[str, Any]:
        if document_id not in self.loaded_documents:
            raise ProcessingError("Документ не найден. Сначала загрузите документ.", "DOCUMENT_NOT_FOUND")

        doc = self.loaded_documents[document_id]
        full_text: str = doc["text"]
        chunks_indexed: List[Chunk] = doc.get("chunks_indexed", [])
        if not chunks_indexed:
            # бэкап, если чего-то не сохранили
            chunk_texts = doc.get("chunks", []) or TextProcessor.split_into_chunks(full_text, max_chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            chunks_indexed = self._index_chunks(full_text, chunk_texts)
            doc["chunks_indexed"] = chunks_indexed

        # выбор контекста
        selected = self._select_relevant_chunks(chunks_indexed, question, max_chunks=self.max_context_chunks)
        context_chunks = [
            {"index": ch.index, "score": ch_score, "excerpt": ch.text[:1800]}
            for ch, ch_score in selected
        ]
        if context_chunks:
            context_text = "\n\n".join(
                f"[Фрагмент {c['index']}]\n{c['excerpt']}" for c in context_chunks
            )
        else:
            context_text = full_text[:6000]  # last resort

        # LLM или оффлайн
        if self.allow_ai and self.openai_service:
            response_text = await self._ask_llm(
                system=DOCUMENT_CHAT_SYSTEM,
                user=DOCUMENT_CHAT_USER_TMPL.format(question=question, context=context_text),
            )
            payload = self._safe_json_loads(response_text)
            answer, citations, confidence = self._normalize_llm_payload(payload, selected)
        else:
            answer, citations, confidence = self._local_answer(full_text, selected, question)

        relevant_fragments = self._citations_to_fragments(full_text, chunks_indexed, citations)

        return {
            "answer": answer,
            "question": question,
            "confidence": confidence,
            "relevant_fragments": relevant_fragments,  # [{"text","start","end","chunk_index"}...]
            "context_chunks": context_chunks,
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
        }

    # ---------------------------- Ретривер/поиск ----------------------------

    def _select_relevant_chunks(
        self, chunks: List[Chunk], question: str, max_chunks: int = 3
    ) -> List[Tuple[Chunk, float]]:
        """Гибрид: BM25-lite + TF-IDF косинус + keyword-оверлап. Возвращает [(chunk, score), ...]."""
        if not chunks:
            return []

        # токены вопроса
        q_tokens = self._preprocess_tokens(question)
        if not q_tokens:
            q_tokens = self._preprocess_tokens(question, allow_short=True)
        if not q_tokens:
            return [(chunks[i], 0.0) for i in range(min(max_chunks, len(chunks)))]

        # подготовим токены чанков и частоты
        doc_tokens: List[List[str]] = []
        df: Dict[str, int] = {}
        for ch in chunks:
            toks = self._preprocess_tokens(ch.text)
            doc_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        N = len(chunks)

        # IDF для TF-IDF
        import math
        idf: Dict[str, float] = {t: math.log((N / (df.get(t, 0) + 1))) + 1.0 for t in df}

        def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            from collections import Counter
            c = Counter(tokens)
            total = float(len(tokens)) or 1.0
            return {t: (c[t] / total) * idf.get(t, 1.0) for t in c}

        q_vec = tfidf_vec(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        # BM25-lite параметры
        k1, b = 1.5, 0.75
        avgdl = sum(len(toks) for toks in doc_tokens) / max(1, N)

        def bm25_score(tokens: List[str]) -> float:
            from collections import Counter
            c = Counter(tokens)
            score = 0.0
            dl = len(tokens) or 1
            for t in q_tokens:
                df_t = df.get(t, 0)
                if df_t == 0:
                    continue
                idf_bm = math.log(1 + (N - df_t + 0.5) / (df_t + 0.5))
                tf = c.get(t, 0)
                score += idf_bm * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
            return score

        def cosine(vec_doc: Dict[str, float]) -> float:
            dot = sum(vec_doc.get(t, 0.0) * q_vec.get(t, 0.0) for t in set(vec_doc) | set(q_vec))
            d_norm = math.sqrt(sum(v * v for v in vec_doc.values())) or 1.0
            return dot / (d_norm * q_norm)

        # скорим
        scored: List[Tuple[Chunk, float]] = []
        for i, ch in enumerate(chunks):
            toks = doc_tokens[i]
            if not toks:
                continue
            # TF-IDF косинус
            vec = tfidf_vec(toks)
            cossim = cosine(vec)
            # BM25-lite
            bm = bm25_score(toks)
            # keyword overlap (ослабленный)
            overlap = len(set(toks) & set(q_tokens)) / max(1, len(set(q_tokens)))
            # гибридный скор
            score = 0.55 * bm + 0.35 * cossim + 0.10 * overlap
            if score > 0:
                scored.append((ch, float(score)))

        if not scored:
            return [(chunks[i], 0.0) for i in range(min(max_chunks, len(chunks)))]

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:max_chunks]

    @staticmethod
    def _preprocess_tokens(text: str, allow_short: bool = False) -> List[str]:
        tokens = re.findall(r"[\w\-]+", (text or "").lower())
        if not allow_short:
            tokens = [t for t in tokens if len(t) > 2]
        tokens = [t for t in tokens if t not in QUESTION_STOPWORDS]
        return tokens

    # ------------------------------ LLM обвязка ------------------------------

    async def _ask_llm(self, *, system: str, user: str) -> str:
        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")
        resp = await self.openai_service.ask_legal(system_prompt=system, user_text=user)
        if not resp or not resp.get("ok"):
            raise ProcessingError("Не удалось получить ответ от модели", "AI_ERROR")
        return resp.get("text", "") or ""

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            pass
        # вырезать код-блоки, найти JSON по скобкам
        try:
            raw2 = re.sub(r"^```.*?```$", "", raw, flags=re.DOTALL | re.MULTILINE)
            i = raw2.find("{")
            j = raw2.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(raw2[i : j + 1])
        except Exception:
            return {}
        return {}

    def _normalize_llm_payload(
        self, payload: Any, selected: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Приводим LLM-ответ к (answer, citations, confidence). Цитаты валидируем по длине чанка."""
        answer = ""
        citations: List[Dict[str, Any]] = []
        confidence = 0.5

        if isinstance(payload, dict):
            answer = str(payload.get("answer") or "").strip()
            try:
                confidence = float(payload.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
            for cit in payload.get("citations", []) or []:
                try:
                    idx = int(cit.get("chunk_index"))
                    span = cit.get("span") or {}
                    s, e = int(span.get("start", -1)), int(span.get("end", -1))
                    # валидируем с выбранными чанками
                    ch_map = {c.index: c for c, _ in selected}
                    if idx in ch_map and 0 <= s < e <= len(ch_map[idx].text):
                        citations.append({"chunk_index": idx, "span": {"start": s, "end": e}})
                except Exception:
                    continue

        if not answer:
            answer = "В предоставленном контексте нет информации для точного ответа."
            citations = []
            confidence = 0.3

        return answer, citations, max(0.0, min(1.0, confidence))

    # ------------------------------- Локальный ответ -------------------------------

    def _local_answer(
        self, full_text: str, selected: List[Tuple[Chunk, float]], question: str
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Простой оффлайн-ответ: собираем по 1–2 предложения из лучших чанков."""
        pieces: List[str] = []
        citations: List[Dict[str, Any]] = []
        for ch, _ in selected[:2]:
            sent = self._best_sentences(ch.text, question, max_sentences=2)
            if sent:
                pieces.append(sent)
                # грубая цитата — первые 120 символов матчей
                s_local = max(0, ch.text.lower().find(self._preselect_keyword(question)))
                e_local = min(len(ch.text), s_local + 120) if s_local >= 0 else min(120, len(ch.text))
                if e_local > s_local >= 0:
                    citations.append({"chunk_index": ch.index, "span": {"start": s_local, "end": e_local}})
        answer = " ".join(pieces) or "Информации для ответа в документе не найдено."
        return answer, citations, 0.4 if pieces else 0.2

    @staticmethod
    def _preselect_keyword(q: str) -> str:
        toks = re.findall(r"[\w\-]+", (q or "").lower())
        toks = [t for t in toks if t not in QUESTION_STOPWORDS and len(t) > 3]
        return toks[0] if toks else (re.findall(r"[\w\-]+", (q or "").lower())[:1] or [""])[0]

    def _best_sentences(self, text: str, question: str, max_sentences: int = 2) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text)
        q_toks = set(self._preprocess_tokens(question, allow_short=True))
        scored: List[Tuple[float, str]] = []
        for s in sents:
            stoks = set(self._preprocess_tokens(s, allow_short=True))
            if not stoks:
                continue
            overlap = len(q_toks & stoks) / max(1, len(q_toks))
            if overlap > 0:
                scored.append((overlap, s.strip()))
        scored.sort(key=lambda t: t[0], reverse=True)
        return " ".join([s for _, s in scored[:max_sentences]])

    # ------------------------------- Цитаты/фрагменты -------------------------------

    def _citations_to_fragments(
        self, full_text: str, chunks_indexed: List[Chunk], citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        frags: List[Dict[str, Any]] = []
        ch_map = {c.index: c for c in chunks_indexed}
        for cit in citations or []:
            idx = int(cit.get("chunk_index", -1))
            span = cit.get("span") or {}
            s_local, e_local = int(span.get("start", -1)), int(span.get("end", -1))
            ch = ch_map.get(idx)
            if not ch or not (0 <= s_local < e_local <= len(ch.text)):
                continue
            # глобальные координаты
            g_start = ch.start + s_local
            g_end = ch.start + e_local
            frags.append(
                {
                    "chunk_index": idx,
                    "start": g_start,
                    "end": g_end,
                    "text": full_text[g_start:g_end],
                    "chunk_preview": ch.text[:120],
                }
            )
        # дедуп по диапазонам
        seen = set()
        uniq = []
        for f in frags:
            key = (f["chunk_index"], f["start"], f["end"])
            if key not in seen:
                seen.add(key)
                uniq.append(f)
        return uniq[:5]

    # ------------------------------- Индексация чанков -------------------------------

    @staticmethod
    def _index_chunks(full_text: str, chunk_texts: List[str]) -> List[Chunk]:
        """Сопоставляет каждому чанку глобальные оффсеты (start/end) в исходном тексте."""
        chunks: List[Chunk] = []
        pos = 0
        for i, ct in enumerate(chunk_texts):
            if not ct:
                continue
            # ищем ct, начиная с pos; если не нашли (из-за нормализаций), используем эвристику
            j = full_text.find(ct, pos)
            if j == -1:
                # эвристика: берем первые ~50 символов чанка для поиска якоря
                anchor = ct[:50]
                k = full_text.find(anchor, pos)
                if k != -1:
                    j = k
                else:
                    # крайний случай — используем текущий pos как начало
                    j = pos
            start = j
            end = min(len(full_text), start + len(ct))
            chunks.append(Chunk(index=i, start=start, end=end, text=ct))
            pos = end
        return chunks
