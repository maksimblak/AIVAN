"""
Утилиты для работы с документами различных форматов
"""

from __future__ import annotations
import asyncio
import io
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
import mimetypes

logger = logging.getLogger(__name__)

class FileFormatHandler:
    """Базовый класс для обработки различных форматов файлов"""

    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> tuple[str, str]:
        """Определить тип файла по расширению и MIME-типу"""
        path = Path(file_path)
        extension = path.suffix.lower()

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        return extension, mime_type

    @staticmethod
    async def extract_text_from_file(file_path: Union[str, Path]) -> tuple[bool, str]:
        """Извлечь текст из файла в зависимости от его типа"""
        path = Path(file_path)
        extension, _ = FileFormatHandler.detect_file_type(path)

        try:
            if extension == ".txt":
                return await FileFormatHandler._extract_from_txt(path)
            elif extension == ".pdf":
                return await FileFormatHandler._extract_from_pdf(path)
            elif extension in [".docx", ".doc"]:
                return await FileFormatHandler._extract_from_docx(path)
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
            encodings = ['utf-8', 'cp1251', 'latin-1']

            for encoding in encodings:
                try:
                    async with asyncio.Lock():
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                    return True, content
                except UnicodeDecodeError:
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

                async with asyncio.Lock():
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"

                if text.strip():
                    return True, text
                else:
                    return False, "PDF файл не содержит извлекаемого текста (возможно, это скан)"

            except ImportError:
                # Если PyPDF2 не установлен, попробуем другие библиотеки
                try:
                    import pdfplumber

                    async with asyncio.Lock():
                        with pdfplumber.open(file_path) as pdf:
                            text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"

                    if text.strip():
                        return True, text
                    else:
                        return False, "PDF файл не содержит извлекаемого текста"

                except ImportError:
                    return False, "Для работы с PDF файлами требуется установить PyPDF2 или pdfplumber"

        except Exception as e:
            return False, f"Ошибка чтения PDF файла: {str(e)}"

    @staticmethod
    async def _extract_from_docx(file_path: Path) -> tuple[bool, str]:
        """Извлечь текст из DOCX файла"""
        try:
            try:
                import docx

                async with asyncio.Lock():
                    doc = docx.Document(file_path)
                    text = ""

                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"

                    # Извлекаем текст из таблиц
                    for table in doc.tables:
                        for row in table.rows:
                            for cell in row.cells:
                                text += cell.text + "\t"
                            text += "\n"

                return True, text

            except ImportError:
                return False, "Для работы с DOCX файлами требуется установить python-docx"

        except Exception as e:
            return False, f"Ошибка чтения DOCX файла: {str(e)}"

class TextProcessor:
    """Класс для предварительной обработки текста"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка и нормализация текста"""
        if not text:
            return ""

        # Убираем лишние пробелы и переносы
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            cleaned_line = ' '.join(line.split())  # Убираем множественные пробелы
            if cleaned_line:  # Пропускаем пустые строки
                cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)

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
                if char in '.!?':
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
                    chunk_text = ' '.join(words[:-1])  # Убираем последнее слово
                    chunks.append(chunk_text)
                    start = start + len(chunk_text) - overlap
                else:
                    # Принудительно режем
                    chunks.append(chunk_text)
                    start = end - overlap

        return chunks

    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """Извлечь метаданные из текста"""
        metadata = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split('\n')),
            "has_tables": '\t' in text,
            "has_numbers": any(char.isdigit() for char in text),
            "has_dates": False,  # Можно добавить регулярные выражения для поиска дат
            "has_emails": '@' in text,
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

def is_text_file(file_path: Union[str, Path]) -> bool:
    """Проверить, является ли файл текстовым"""
    extension = Path(file_path).suffix.lower()
    text_extensions = ['.txt', '.md', '.json', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.h']
    return extension in text_extensions

def is_document_file(file_path: Union[str, Path]) -> bool:
    """Проверить, является ли файл документом"""
    extension = Path(file_path).suffix.lower()
    doc_extensions = ['.pdf', '.doc', '.docx', '.odt', '.rtf']
    return extension in doc_extensions

def is_image_file(file_path: Union[str, Path]) -> bool:
    """Проверить, является ли файл изображением"""
    extension = Path(file_path).suffix.lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
    return extension in image_extensions