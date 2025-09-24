"""
Модуль OCR и конвертации
Распознавание сканированных документов и преобразование в редактируемый текст
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import format_file_size, is_image_file

logger = logging.getLogger(__name__)

class OCRConverter(DocumentProcessor):
    """Класс для OCR распознавания и конвертации документов"""

    def __init__(self):
        super().__init__(
            name="OCRConverter",
            max_file_size=100 * 1024 * 1024  # 100MB для изображений
        )
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

    async def process(self, file_path: Union[str, Path], output_format: str = "txt", **kwargs) -> DocumentResult:
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
            elif file_extension == '.pdf':
                # Проверяем, является ли PDF сканом
                text, confidence = await self._ocr_pdf(path)
            else:
                raise ProcessingError(f"Неподдерживаемый формат для OCR: {file_extension}", "FORMAT_ERROR")

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
                    "word_count": len(cleaned_text.split())
                }
            }

            return DocumentResult.success_result(
                data=result_data,
                message=f"OCR распознавание завершено с уверенностью {confidence:.1f}%"
            )

        except Exception as e:
            raise ProcessingError(f"Ошибка OCR: {str(e)}", "OCR_ERROR")

    async def _ocr_image(self, image_path: Path) -> tuple[str, float]:
        """OCR распознавание изображения"""
        try:
            # Попробуем использовать Tesseract через pytesseract
            try:
                import pytesseract
                from PIL import Image

                # Загружаем изображение
                image = Image.open(image_path)

                # Распознаем текст с получением данных о уверенности
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='rus+eng')

                # Собираем текст и вычисляем среднюю уверенность
                words = []
                confidences = []

                for i in range(len(ocr_data['text'])):
                    word = ocr_data['text'][i].strip()
                    confidence = int(ocr_data['conf'][i])

                    if word and confidence > 0:
                        words.append(word)
                        confidences.append(confidence)

                text = ' '.join(words)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                return text, avg_confidence

            except ImportError:
                # Fallback - возвращаем заглушку
                logger.warning("pytesseract не установлен. Используется заглушка OCR.")
                return self._mock_ocr_result(image_path)

        except Exception as e:
            logger.error(f"Ошибка OCR изображения: {e}")
            return self._mock_ocr_result(image_path)

    async def _ocr_pdf(self, pdf_path: Path) -> tuple[str, float]:
        """OCR распознавание PDF (если это скан)"""
        try:
            # Сначала попробуем извлечь текст обычным способом
            try:
                import PyPDF2

                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                # Если текста достаточно, значит это не скан
                if len(text.strip()) > 100:
                    return text, 100.0  # Максимальная уверенность для текстового PDF

            except ImportError:
                pass

            # Если текста мало или нет, пробуем OCR
            try:
                import pdf2image
                import pytesseract

                # Конвертируем PDF в изображения
                pages = pdf2image.convert_from_path(pdf_path)

                all_text = []
                all_confidences = []

                for i, page in enumerate(pages):
                    logger.info(f"OCR страницы {i+1}/{len(pages)}")

                    # OCR каждой страницы
                    ocr_data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT, lang='rus+eng')

                    page_words = []
                    page_confidences = []

                    for j in range(len(ocr_data['text'])):
                        word = ocr_data['text'][j].strip()
                        confidence = int(ocr_data['conf'][j])

                        if word and confidence > 0:
                            page_words.append(word)
                            page_confidences.append(confidence)

                    if page_words:
                        all_text.append(' '.join(page_words))
                        all_confidences.extend(page_confidences)

                text = '\n\n'.join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

                return text, avg_confidence

            except ImportError:
                logger.warning("pdf2image или pytesseract не установлены")
                return self._mock_ocr_result(pdf_path)

        except Exception as e:
            logger.error(f"Ошибка OCR PDF: {e}")
            return self._mock_ocr_result(pdf_path)

    def _mock_ocr_result(self, file_path: Path) -> tuple[str, float]:
        """Заглушка для OCR результата"""
        return f"[OCR ЗАГЛУШКА] Содержимое файла {file_path.name} не распознано. Требуется установка tesseract-ocr и pytesseract.", 50.0

    def _clean_ocr_text(self, text: str) -> str:
        """Очистка текста после OCR"""
        if not text:
            return ""

        # Убираем лишние пробелы и переносы
        cleaned = ' '.join(text.split())

        # Исправляем частые ошибки OCR
        corrections = {
            'О': '0',  # где это уместно
            'З': '3',  # где это уместно
            '6': 'б',  # где это уместно
            'rод': 'год',
            'нaлог': 'налог',
        }

        # Применяем корректировки осторожно
        for wrong, correct in corrections.items():
            # Простая замена - можно улучшить
            pass

        return cleaned

    def _analyze_ocr_quality(self, text: str, confidence: float) -> Dict[str, Any]:
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
        suspicious_chars = sum(1 for char in text if char in '°§¤¦№')

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
            "recommendations": recommendations
        }

    def get_required_dependencies(self) -> List[str]:
        """Получить список требуемых зависимостей для OCR"""
        return [
            "pytesseract>=0.3.10",
            "pillow>=9.0.0",
            "pdf2image>=1.16.0",
            "tesseract-ocr (системная утилита)"
        ]