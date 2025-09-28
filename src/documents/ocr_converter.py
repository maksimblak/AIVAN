"""
Модуль OCR и конвертации
Распознавание сканированных документов и преобразование в редактируемый текст
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import is_image_file

logger = logging.getLogger(__name__)


class OCRConverter(DocumentProcessor):
    """Класс для OCR распознавания и конвертации документов"""

    def __init__(self):
        super().__init__(
            name="OCRConverter", max_file_size=100 * 1024 * 1024  # 100MB для изображений
        )
        self.supported_formats = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]

    async def process(
        self, file_path: str | Path, output_format: str = "txt", **kwargs
    ) -> DocumentResult:
        """
        OCR распознавание документа

        Args:
            file_path: путь к файлу
            output_format: формат вывода ("txt", "docx", "pdf")
        """

        path = Path(file_path)
        file_extension = path.suffix.lower()

        try:
            if is_image_file(path):
                # OCR изображения
                text, confidence = await self._ocr_image(path)
            elif file_extension == ".pdf":
                # Проверяем, является ли PDF сканом
                text, confidence = await self._ocr_pdf(path)
            else:
                raise ProcessingError(
                    f"Неподдерживаемый формат для OCR: {file_extension}", "FORMAT_ERROR"
                )

            if not text.strip():
                raise ProcessingError("Не удалось распознать текст в документе", "OCR_NO_TEXT")

            # Обработка и очистка распознанного текста
            cleaned_text = self._clean_ocr_text(text)

            # Анализ качества распознавания
            quality_analysis = self._analyze_ocr_quality(text, confidence)

            result_data = {
                "recognized_text": cleaned_text,
                "confidence_score": confidence,
                "quality_analysis": quality_analysis,
                "original_file": str(file_path),
                "output_format": output_format,
                "processing_info": {
                    "file_type": "image" if is_image_file(path) else "pdf",
                    "text_length": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                },
            }

            return DocumentResult.success_result(
                data=result_data,
                message=f"OCR распознавание завершено с уверенностью {confidence:.1f}%",
            )

        except Exception as e:
            raise ProcessingError(f"Ошибка OCR: {str(e)}", "OCR_ERROR")

    async def _ocr_image(self, image_path: Path) -> tuple[str, float]:
        """OCR распознавание изображения"""
        # Сначала пробуем PaddleOCR (основной движок)
        try:
            result = await self._paddleocr_image(image_path)
            if result[0]:  # Если текст получен
                logger.info("Успешно использован PaddleOCR")
                return result
        except Exception as e:
            logger.warning(f"PaddleOCR не удался: {e}. Переключаемся на Tesseract.")


        # Fallback на Tesseract
        try:
            result = await self._tesseract_image(image_path)
            if result[0]:  # Если текст получен
                logger.info("Успешно использован Tesseract (fallback)")
                return result
        except Exception as e:
            logger.error(f"Tesseract не удался: {e}")

        # Если оба не сработали
        logger.error("Не удалось выполнить OCR ни с PaddleOCR, ни с Tesseract")
        return self._mock_ocr_result(image_path)

    async def _paddleocr_image(self, image_path: Path) -> tuple[str, float]:
        """OCR распознавание изображения с помощью PaddleOCR"""
        try:
            from paddleocr import PaddleOCR

            # Инициализируем PaddleOCR для русского и английского языков
            ocr = PaddleOCR(use_angle_cls=True, lang='ru')  # ru для русского языка

            # Выполняем OCR
            result = ocr.ocr(str(image_path), cls=True)

            if not result or not result[0]:
                return "", 0.0

            # Собираем текст и вычисляем среднюю уверенность
            texts = []
            confidences = []

            for line in result[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if text and text.strip():
                        texts.append(text.strip())
                        confidences.append(confidence * 100)  # Переводим в проценты

            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return full_text, avg_confidence

        except ImportError:
            raise Exception("PaddleOCR не установлен")
        except Exception as e:
            raise Exception(f"Ошибка PaddleOCR: {str(e)}")

    async def _tesseract_image(self, image_path: Path) -> tuple[str, float]:
        """OCR распознавание изображения с помощью Tesseract (fallback)"""
        try:
            import pytesseract
            from PIL import Image

            # Загружаем изображение
            image = Image.open(image_path)

            # Распознаем текст с получением данных о уверенности
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, lang="rus+eng"
            )

            # Собираем текст и вычисляем среднюю уверенность
            words = []
            confidences = []

            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i].strip()
                confidence = int(ocr_data["conf"][i])

                if word and confidence > 0:
                    words.append(word)
                    confidences.append(confidence)

            text = " ".join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_confidence

        except ImportError:
            raise Exception("pytesseract не установлен")
        except Exception as e:
            raise Exception(f"Ошибка Tesseract: {str(e)}")

    async def _ocr_pdf(self, pdf_path: Path) -> tuple[str, float]:
        """Улучшенное распознавание PDF с множественными стратегиями"""
        logger.info(f"Начало обработки PDF: {pdf_path}")

        # Стратегия 1: Извлечение встроенного текста (быстро)
        extracted_text, text_confidence = await self._extract_pdf_text(pdf_path)

        # Проверяем качество извлеченного текста
        text_quality = self._analyze_extracted_text_quality(extracted_text)
        logger.info(f"Качество извлеченного текста: {text_quality}")

        # Если текст хорошего качества, возвращаем его
        if text_quality >= 0.7 and len(extracted_text.strip()) > 50:
            logger.info("Использован встроенный текст PDF")
            return extracted_text, text_confidence

        # Стратегия 2: Гибридный подход (текст + OCR)
        if text_quality > 0.3:
            logger.info("Применяем гибридный подход (текст + OCR)")
            ocr_text, ocr_confidence = await self._hybrid_pdf_processing(pdf_path, extracted_text)
            return ocr_text, ocr_confidence

        # Стратегия 3: Полный OCR (медленно, но точно)
        logger.info("Применяем полный OCR")
        return await self._full_pdf_ocr(pdf_path)

    async def _extract_pdf_text(self, pdf_path: Path) -> tuple[str, float]:
        """Извлечение встроенного текста из PDF с PyMuPDF"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)

            doc.close()

            if text_parts:
                return "\n\n".join(text_parts), 100.0
            else:
                return "", 0.0

        except ImportError:
            logger.error("PyMuPDF не установлен")
            return "", 0.0
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {e}")
            return "", 0.0

    def _analyze_extracted_text_quality(self, text: str) -> float:
        """Анализ качества извлеченного текста"""
        if not text or len(text.strip()) < 10:
            return 0.0

        # Подсчитываем различные характеристики
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        suspicious_chars = sum(1 for c in text if c in '����□■●○♦♠♣♥')

        # Соотношения
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        suspicious_ratio = suspicious_chars / total_chars if total_chars > 0 else 0

        # Проверка на наличие осмысленных слов
        words = text.split()
        meaningful_words = sum(1 for word in words if len(word) > 2 and word.isalpha())
        word_ratio = meaningful_words / len(words) if words else 0

        # Итоговая оценка качества
        quality_score = (
            alpha_ratio * 0.4 +  # Доля букв
            word_ratio * 0.4 +   # Доля осмысленных слов
            (1 - suspicious_ratio) * 0.2  # Отсутствие подозрительных символов
        )

        return min(1.0, quality_score)

    async def _hybrid_pdf_processing(self, pdf_path: Path, extracted_text: str) -> tuple[str, float]:
        """Гибридная обработка: комбинирование извлеченного текста и OCR"""
        try:
            # Получаем OCR текст
            ocr_text, ocr_confidence = await self._full_pdf_ocr(pdf_path)

            if not ocr_text:
                return extracted_text, 50.0

            # Простая эвристика для объединения
            if len(ocr_text) > len(extracted_text) * 1.5:
                # OCR дал больше текста - вероятно лучше
                logger.info("OCR дал больше текста, используем его")
                return ocr_text, ocr_confidence
            else:
                # Комбинируем тексты
                combined = f"{extracted_text}\n\n--- OCR ДОПОЛНЕНИЕ ---\n\n{ocr_text}"
                avg_confidence = (50.0 + ocr_confidence) / 2
                logger.info("Объединяем извлеченный текст и OCR")
                return combined, avg_confidence

        except Exception as e:
            logger.error(f"Ошибка гибридной обработки: {e}")
            return extracted_text, 30.0

    async def _full_pdf_ocr(self, pdf_path: Path) -> tuple[str, float]:
        """Полный OCR обработка PDF с ocrmypdf"""
        try:
            import ocrmypdf
            import tempfile
            import asyncio

            # Создаем временный файл для результата OCR
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_output:
                tmp_output_path = Path(tmp_output.name)

            try:
                logger.info(f"Запуск OCR обработки с ocrmypdf")

                # Запускаем ocrmypdf в отдельном потоке
                def run_ocrmypdf():
                    ocrmypdf.ocr(
                        input_file=str(pdf_path),
                        output_file=str(tmp_output_path),
                        language=['rus', 'eng'],
                        force_ocr=True,
                        skip_text=False,
                        redo_ocr=True,
                        optimize=1,
                        jpeg_quality=95,
                        png_quality=95
                    )

                # Выполняем OCR асинхронно
                await asyncio.to_thread(run_ocrmypdf)

                # Извлекаем текст из обработанного PDF
                ocr_text, confidence = await self._extract_pdf_text(tmp_output_path)

                logger.info(f"OCR завершен, получено {len(ocr_text)} символов")
                return ocr_text, confidence if confidence > 0 else 85.0

            finally:
                # Удаляем временный файл
                if tmp_output_path.exists():
                    tmp_output_path.unlink()

        except ImportError:
            logger.error("ocrmypdf не установлен")
            return await self._fallback_pdf_ocr(pdf_path)
        except Exception as e:
            logger.error(f"Ошибка OCR с ocrmypdf: {e}")
            return await self._fallback_pdf_ocr(pdf_path)

    async def _fallback_pdf_ocr(self, pdf_path: Path) -> tuple[str, float]:
        """Fallback OCR через PyMuPDF + PaddleOCR/Tesseract"""
        try:
            import fitz
            import tempfile
            import os

            doc = fitz.open(str(pdf_path))
            all_text = []
            all_confidences = []

            for page_num in range(len(doc)):
                logger.info(f"OCR страницы {page_num + 1}/{len(doc)}")
                page = doc[page_num]

                # Конвертируем страницу в изображение
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Увеличиваем разрешение

                # Сохраняем как временное изображение
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    pix.save(tmp_file.name)
                    tmp_path = Path(tmp_file.name)

                try:
                    # Сначала пробуем PaddleOCR
                    try:
                        page_text, page_confidence = await self._paddleocr_image(tmp_path)
                        if page_text:
                            all_text.append(page_text)
                            all_confidences.append(page_confidence)
                            continue
                    except Exception as e:
                        logger.warning(f"PaddleOCR не удался для страницы {page_num + 1}: {e}")

                    # Fallback на Tesseract
                    page_text, page_confidence = await self._tesseract_image(tmp_path)
                    if page_text:
                        all_text.append(page_text)
                        all_confidences.append(page_confidence)

                finally:
                    # Удаляем временный файл
                    if tmp_path.exists():
                        os.unlink(tmp_path)

            doc.close()

            text = "\n\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

            return text, avg_confidence

        except Exception as e:
            logger.error(f"Ошибка fallback OCR: {e}")
            return self._mock_ocr_result(pdf_path)

    def _mock_ocr_result(self, file_path: Path) -> tuple[str, float]:
        """Заглушка для OCR результата"""
        return (
            f"[OCR ЗАГЛУШКА] Содержимое файла {file_path.name} не распознано. Требуется установка tesseract-ocr и pytesseract.",
            50.0,
        )

    def _clean_ocr_text(self, text: str) -> str:
        """Очистка текста после OCR"""
        if not text:
            return ""

        # Убираем лишние пробелы и переносы
        cleaned = " ".join(text.split())

        # Применяем корректировки осторожно с проверкой контекста
        cleaned = self._apply_ocr_corrections(cleaned)

        return cleaned

    def _apply_ocr_corrections(self, text: str) -> str:
        """Применяет умные исправления частых ошибок OCR с учетом контекста"""
        import re

        # Словарь исправлений с паттернами контекста
        corrections = [
            # Цифры vs буквы в числовых контекстах
            (r'\b(\d+)О(\d+)\b', r'\10\2'),  # 2О23 -> 2023
            (r'\bЗ(\d+)\b', r'3\1'),        # З0 -> 30
            (r'\b(\d+)З\b', r'\13'),        # 20З -> 203

            # Русские буквы в английских словах
            (r'\bрrо\b', 'pro'),
            (r'\bсоm\b', 'com'),
            (r'\bоrg\b', 'org'),

            # Английские буквы в русских словах
            (r'\bгоd\b', 'год'),
            (r'\bнaлог\b', 'налог'),
            (r'\bрублеи\b', 'рублей'),
            (r'\bзaкон\b', 'закон'),
            (r'\bпрaво\b', 'право'),

            # Частые OCR ошибки в юридических текстах
            (r'\bстaтья\b', 'статья'),
            (r'\bдoговор\b', 'договор'),
            (r'\bуслoвие\b', 'условие'),
            (r'\bтребoвание\b', 'требование'),
            (r'\bзaявление\b', 'заявление'),
            (r'\bрeшение\b', 'решение'),

            # Исправление неправильных кавычек и символов
            (r'«([^»]*)»', r'"\1"'),         # « » -> " "
            (r'„([^"]*)"', r'"\1"'),         # „ " -> " "
            (r'([0-9])О([0-9])', r'\g<1>0\g<2>'),  # цифра О цифра -> цифра 0 цифра

            # Исправление точек в числах
            (r'(\d+)\.(\d{3})\b(?!\d)', r'\1\2'),  # 1.000 -> 1000 (если не десятичная дробь)

            # Исправление дефисов и тире
            (r'(\w)\s*-\s*(\w)', r'\1-\2'),  # убираем пробелы вокруг дефисов в словах
        ]

        corrected = text

        # Применяем исправления по порядку
        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        # Исправляем множественные пробелы
        corrected = re.sub(r'\s+', ' ', corrected)

        # Исправляем пробелы перед знаками препинания
        corrected = re.sub(r'\s+([,.!?;:])', r'\1', corrected)

        # Исправляем отсутствие пробелов после знаков препинания
        corrected = re.sub(r'([,.!?;:])([А-Яа-яA-Za-z])', r'\1 \2', corrected)

        return corrected.strip()

    def _analyze_ocr_quality(self, text: str, confidence: float) -> dict[str, Any]:
        """Анализ качества OCR распознавания"""

        quality_level = "низкое"
        if confidence >= 90:
            quality_level = "отличное"
        elif confidence >= 80:
            quality_level = "хорошее"
        elif confidence >= 70:
            quality_level = "удовлетворительное"
        elif confidence >= 60:
            quality_level = "среднее"

        # Анализ текста
        word_count = len(text.split()) if text else 0
        char_count = len(text) if text else 0

        # Проверка на наличие подозрительных символов (ошибки OCR)
        suspicious_chars = sum(1 for char in text if char in "°§¤¦№")

        recommendations = []
        if confidence < 70:
            recommendations.append("Рекомендуется проверить качество исходного изображения")
        if suspicious_chars > 0:
            recommendations.append("Обнаружены подозрительные символы - требуется ручная проверка")
        if word_count < 10:
            recommendations.append("Распознано мало текста - проверьте настройки OCR")

        return {
            "confidence": confidence,
            "quality_level": quality_level,
            "word_count": word_count,
            "char_count": char_count,
            "suspicious_chars": suspicious_chars,
            "recommendations": recommendations,
        }

    def get_required_dependencies(self) -> list[str]:
        """Получить список требуемых зависимостей для OCR"""
        return [
            "ocrmypdf>=16.11.0",
            "pymupdf>=1.26.4",
            "paddlepaddle>=2.6.0",
            "paddleocr>=2.8.0",
            "pytesseract>=0.3.10",
            "pillow>=9.0.0",
            "tesseract-ocr (системная утилита)",
        ]
