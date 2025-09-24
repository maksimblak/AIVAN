"""
Модуль "Чат с документом"
Интерактивное взаимодействие с загруженными документами через вопросы-ответы
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

DOCUMENT_CHAT_PROMPT = """
Ты — помощник для работы с документами. Пользователь загрузил документ и задает вопросы по его содержанию.

ТВОИ ВОЗМОЖНОСТИ:
- Отвечать на вопросы только на основе предоставленного документа
- Цитировать релевантные фрагменты с указанием их местоположения
- Объяснять правовые термины в контексте документа
- Находить связи между разными частями документа

ПРАВИЛА:
- Используй ТОЛЬКО информацию из документа
- При ответе всегда указывай, откуда взята информация
- Если информации в документе нет, честно об этом скажи
- Выделяй цитаты из документа курсивом
- Давай точные и конкретные ответы

Документ для анализа:
{document_text}

Отвечай на вопросы пользователя на основе этого документа.
"""

class DocumentChat(DocumentProcessor):
    """Класс для интерактивного чата с документами"""

    def __init__(self, openai_service=None):
        super().__init__(
            name="DocumentChat",
            max_file_size=50 * 1024 * 1024  # 50MB
        )
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        self.openai_service = openai_service
        self.loaded_documents: Dict[str, Dict[str, Any]] = {}

    async def load_document(self, file_path: Union[str, Path], document_id: Optional[str] = None) -> str:
        """
        Загрузить документ для чата

        Returns:
            document_id для использования в чате
        """
        if not document_id:
            document_id = f"doc_{int(datetime.now().timestamp())}"

        # Извлекаем текст из файла
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        # Очищаем и обрабатываем текст
        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # Сохраняем документ в памяти
        self.loaded_documents[document_id] = {
            "text": cleaned_text,
            "file_path": str(file_path),
            "loaded_at": datetime.now(),
            "metadata": TextProcessor.extract_metadata(cleaned_text),
            "chunks": TextProcessor.split_into_chunks(cleaned_text, max_chunk_size=3000)
        }

        logger.info(f"Документ {file_path} загружен с ID: {document_id}")
        return document_id

    async def chat_with_document(self, document_id: str, question: str) -> Dict[str, Any]:
        """
        Задать вопрос по документу

        Args:
            document_id: ID загруженного документа
            question: вопрос пользователя

        Returns:
            Ответ с релевантными цитатами и ссылками
        """
        if document_id not in self.loaded_documents:
            raise ProcessingError("Документ не найден. Сначала загрузите документ.", "DOCUMENT_NOT_FOUND")

        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        document_data = self.loaded_documents[document_id]
        document_text = document_data["text"]

        try:
            # Формируем промпт с документом
            prompt = DOCUMENT_CHAT_PROMPT.format(document_text=document_text[:6000])  # Ограничиваем длину

            # Задаем вопрос
            result = await self.openai_service.ask_legal(
                system_prompt=prompt,
                user_message=question
            )

            if not result.get("ok"):
                raise ProcessingError("Не удалось получить ответ от AI", "AI_ERROR")

            answer = result.get("text", "")

            # Находим релевантные фрагменты
            relevant_fragments = self._find_relevant_fragments(document_text, question, answer)

            return {
                "answer": answer,
                "question": question,
                "relevant_fragments": relevant_fragments,
                "document_id": document_id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Ошибка чата с документом: {e}")
            raise ProcessingError(f"Ошибка обработки вопроса: {str(e)}", "CHAT_ERROR")

    async def process(self, file_path: Union[str, Path], **kwargs) -> DocumentResult:
        """
        Основной метод для загрузки документа для чата

        Этот метод реализован для соответствия интерфейсу DocumentProcessor
        """
        try:
            document_id = await self.load_document(file_path)

            return DocumentResult.success_result(
                data={
                    "document_id": document_id,
                    "document_info": self.get_document_info(document_id),
                    "message": "Документ загружен и готов для чата"
                },
                message="Документ успешно загружен для интерактивного чата"
            )

        except Exception as e:
            raise ProcessingError(f"Ошибка загрузки документа: {str(e)}", "LOAD_ERROR")

    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Получить информацию о загруженном документе"""
        if document_id not in self.loaded_documents:
            return {}

        doc_data = self.loaded_documents[document_id]
        return {
            "document_id": document_id,
            "file_path": doc_data["file_path"],
            "loaded_at": doc_data["loaded_at"].isoformat(),
            "metadata": doc_data["metadata"],
            "chunks_count": len(doc_data["chunks"])
        }

    def _find_relevant_fragments(self, document_text: str, question: str, answer: str) -> List[Dict[str, Any]]:
        """Найти релевантные фрагменты документа"""
        fragments = []

        # Простой поиск ключевых слов из вопроса в документе
        question_words = set(question.lower().split())
        sentences = document_text.split('.')

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = set(sentence.lower().split())

            # Проверяем пересечение слов
            overlap = len(question_words & sentence_words)
            if overlap >= 2:  # Минимум 2 совпадения
                fragments.append({
                    "text": sentence,
                    "position": i,
                    "relevance_score": overlap / len(question_words),
                    "context": self._get_sentence_context(sentences, i)
                })

        # Сортируем по релевантности
        fragments.sort(key=lambda x: x["relevance_score"], reverse=True)

        return fragments[:5]  # Возвращаем топ 5 фрагментов

    def _get_sentence_context(self, sentences: List[str], index: int, context_size: int = 1) -> str:
        """Получить контекст вокруг предложения"""
        start = max(0, index - context_size)
        end = min(len(sentences), index + context_size + 1)

        context_sentences = sentences[start:end]
        return '. '.join(s.strip() for s in context_sentences if s.strip())

    def get_loaded_documents(self) -> List[Dict[str, Any]]:
        """Получить список всех загруженных документов"""
        return [
            {
                "document_id": doc_id,
                "file_path": doc_data["file_path"],
                "loaded_at": doc_data["loaded_at"].isoformat(),
                "word_count": doc_data["metadata"]["word_count"]
            }
            for doc_id, doc_data in self.loaded_documents.items()
        ]

    def remove_document(self, document_id: str) -> bool:
        """Удалить документ из памяти"""
        if document_id in self.loaded_documents:
            del self.loaded_documents[document_id]
            logger.info(f"Документ {document_id} удален из памяти")
            return True
        return False

    def cleanup_old_documents(self, max_age_hours: int = 24):
        """Очистка старых документов из памяти"""
        current_time = datetime.now()
        to_remove = []

        for doc_id, doc_data in self.loaded_documents.items():
            age = (current_time - doc_data["loaded_at"]).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(doc_id)

        for doc_id in to_remove:
            self.remove_document(doc_id)

        logger.info(f"Очищено {len(to_remove)} старых документов")