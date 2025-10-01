"""
Утилиты для работы с документами различных форматов
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Глобальный лок для синхронизации файловых операций
_FILE_IO_LOCK = asyncio.Lock()


# Утилиты для безопасного асинхронного файлового I/O
async def read_text_async(path: str | Path, encoding: str = "utf-8", errors: str = "ignore") -> str:
    """Асинхронно читает текстовый файл с общим локом"""
    p = Path(path)
    async with _FILE_IO_LOCK:
        return await asyncio.to_thread(p.read_text, encoding=encoding, errors=errors)


async def write_text_async(path: str | Path, data: str, encoding: str = "utf-8") -> None:
    """Асинхронно записывает текстовый файл с общим локом"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with _FILE_IO_LOCK:
        await asyncio.to_thread(p.write_text, data, encoding=encoding)


async def read_bytes_async(path: str | Path) -> bytes:
    """Асинхронно читает бинарный файл с общим локом"""
    p = Path(path)
    async with _FILE_IO_LOCK:
        return await asyncio.to_thread(p.read_bytes)


async def write_bytes_async(path: str | Path, data: bytes) -> None:
    """Асинхронно записывает бинарный файл с общим локом"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with _FILE_IO_LOCK:
        await asyncio.to_thread(p.write_bytes, data)


class FileFormatHandler:
    """Базовый класс для обработки различных форматов файлов"""

    @staticmethod
    def detect_file_type(file_path: str | Path) -> tuple[str, str]:
        """Определить тип файла по расширению и MIME-типу"""
        path = Path(file_path)
        extension = path.suffix.lower()

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        return extension, mime_type

    @staticmethod
    async def extract_text_from_file(file_path: str | Path) -> tuple[bool, str]:
        """Извлечь текст из файла в зависимости от его типа"""
        path = Path(file_path)
        extension, _ = FileFormatHandler.detect_file_type(path)

        try:
            if extension == ".txt":
                return await FileFormatHandler._extract_from_txt(path)
            elif extension == ".pdf":
                return await FileFormatHandler._extract_from_pdf(path)
            elif extension == ".docx":
                return await FileFormatHandler._extract_from_docx(path)
            elif extension == ".doc":
                return await FileFormatHandler._extract_from_doc(path)
            else:
                return False, f"Неподдерживаемый формат файла: {extension}"
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {file_path}: {e}")
            return False, f"Ошибка обработки файла: {str(e)}"

    @staticmethod
    async def _extract_from_txt(file_path: Path) -> tuple[bool, str]:
        """Извлечь текст из TXT файла"""
        try:
            # Пробуем разные кодировки
            encodings = ["utf-8", "cp1251", "latin-1"]

            for encoding in encodings:
                try:
                    content = await read_text_async(file_path, encoding=encoding, errors="strict")
                    return True, content
                except (UnicodeDecodeError, UnicodeError):
                    continue

            return False, "Не удалось определить кодировку файла"

        except Exception as e:
            return False, f"Ошибка чтения TXT файла: {str(e)}"

    @staticmethod
    async def _extract_from_pdf(file_path: Path) -> tuple[bool, str]:
        """Извлечь текст из PDF файла"""
        try:
            # Попробуем импортировать PyPDF2 или альтернативы
            try:
                import PyPDF2

                def _extract_with_pypdf2():
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text

                text = await asyncio.to_thread(_extract_with_pypdf2)

                if text.strip():
                    return True, text
                else:
                    return False, "PDF файл не содержит извлекаемого текста (возможно, это скан)"

            except ImportError:
                # Если PyPDF2 не установлен, попробуем другие библиотеки
                try:
                    import pdfplumber

                    def _extract_with_pdfplumber():
                        with pdfplumber.open(file_path) as pdf:
                            text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                        return text

                    text = await asyncio.to_thread(_extract_with_pdfplumber)

                    if text.strip():
                        return True, text
                    else:
                        return False, "PDF файл не содержит извлекаемого текста"

                except ImportError:
                    return (
                        False,
                        "Для работы с PDF файлами требуется установить PyPDF2 или pdfplumber",
                    )

        except Exception as e:
            return False, f"Ошибка чтения PDF файла: {str(e)}"

    @staticmethod
    async def _extract_from_docx(file_path: Path) -> tuple[bool, str]:
        """Извлечь текст из DOCX файла"""
        try:
            import docx  # type: ignore
        except ImportError:
            return False, "Для работы с DOCX файлами требуется установить python-docx"

        def _load_docx() -> str:
            document = docx.Document(file_path)
            parts: list[str] = []
            for paragraph in document.paragraphs:
                parts.append(paragraph.text)
            for table in document.tables:
                for row in table.rows:
                    parts.append("\t".join(cell.text for cell in row.cells))
            return "\n".join(parts)

        try:
            text_content = await asyncio.to_thread(_load_docx)
            if not text_content.strip():
                return False, "DOCX файл не содержит извлекаемого текста"
            return True, text_content
        except Exception as e:
            return False, f"Ошибка чтения DOCX файла: {e}"

    @staticmethod
    async def _extract_from_doc(file_path: Path) -> tuple[bool, str]:
        """Извлечь текст из DOC файла"""
        try:
            import textract  # type: ignore
        except ImportError:
            return (
                False,
                "Для обработки DOC файлов требуется установить textract (poetry install --extras full).",
            )

        loop = asyncio.get_event_loop()
        try:
            text_bytes = await loop.run_in_executor(None, textract.process, str(file_path))
            text = text_bytes.decode("utf-8", errors="ignore")
            if not text.strip():
                return False, "DOC файл не содержит извлекаемого текста"
            return True, text
        except Exception as e:
            return False, f"Ошибка чтения DOC файла: {e}"


class TextProcessor:
    """Класс для предварительной обработки текста"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка и нормализация текста"""
        if not text:
            return ""

        # Убираем лишние пробелы и переносы
        lines = text.splitlines()
        cleaned = [line.rstrip() for line in lines]
        return "\n".join(cleaned)

    @staticmethod
    def split_into_chunks(text: str, max_chunk_size: int = 4000, overlap: int = 200) -> list[str]:
        """Разделить текст на части для обработки"""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Найдем конец куска
            end = start + max_chunk_size

            if end >= len(text):
                # Последний кусок
                chunks.append(text[start:])
                break

            # Попробуем найти границу предложения
            chunk_text = text[start:end]

            # Ищем последнюю точку, восклицательный или вопросительный знак
            sentence_ends = []
            for i, char in enumerate(reversed(chunk_text)):
                if char in ".!?":
                    sentence_ends.append(len(chunk_text) - i)
                    break

            if sentence_ends:
                actual_end = start + sentence_ends[0]
                chunks.append(text[start:actual_end])
                start = actual_end - overlap  # Перекрытие для контекста
            else:
                # Если не нашли границу предложения, режем по словам
                words = chunk_text.split()
                if len(words) > 1:
                    chunk_text = " ".join(words[:-1])  # Убираем последнее слово
                    chunks.append(chunk_text)
                    start = start + len(chunk_text) - overlap
                else:
                    # Принудительно режем
                    chunks.append(chunk_text)
                    start = end - overlap

        return chunks

    @staticmethod
    def extract_metadata(text: str) -> dict[str, Any]:
        """Извлечь метаданные из текста"""
        metadata = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split("\n")),
            "has_tables": "\t" in text,
            "has_numbers": any(char.isdigit() for char in text),
            "has_dates": False,  # Можно добавить регулярные выражения для поиска дат
            "has_emails": "@" in text,
            "has_phones": False,  # Можно добавить регулярные выражения для поиска телефонов
        }

        return metadata


def format_file_size(size_bytes: int) -> str:
    """Форматировать размер файла в читаемом виде"""
    if size_bytes == 0:
        return "0 Б"

    size_names = ["Б", "КБ", "МБ", "ГБ"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def is_text_file(file_path: str | Path) -> bool:
    """Проверить, является ли файл текстовым"""
    extension = Path(file_path).suffix.lower()
    text_extensions = [
        ".txt",
        ".md",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".py",
        ".java",
        ".cpp",
        ".h",
    ]
    return extension in text_extensions


def is_document_file(file_path: str | Path) -> bool:
    """Проверить, является ли файл документом"""
    extension = Path(file_path).suffix.lower()
    doc_extensions = [".pdf", ".doc", ".docx", ".odt", ".rtf"]
    return extension in doc_extensions


def is_image_file(file_path: str | Path) -> bool:
    """Проверить, является ли файл изображением"""
    extension = Path(file_path).suffix.lower()
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]
    return extension in image_extensions
