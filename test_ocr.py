#!/usr/bin/env python3
"""
Тест OCR функциональности с PaddleOCR и Tesseract
"""

import asyncio
import logging
from pathlib import Path
from src.documents.ocr_converter import OCRConverter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_ocr():
    """Тест OCR конвертера"""
    converter = OCRConverter()

    print("=== Тест OCR конвертера ===")
    print(f"Поддерживаемые форматы: {converter.supported_formats}")
    print(f"Зависимости: {converter.get_required_dependencies()}")

    # Проверяем доступность зависимостей
    print("\n=== Проверка зависимостей ===")

    try:
        import paddleocr
        print("[OK] PaddleOCR доступен")
    except ImportError:
        print("[NO] PaddleOCR не установлен")

    try:
        import pytesseract
        print("[OK] pytesseract доступен")
    except ImportError:
        print("[NO] pytesseract не установлен")

    try:
        from PIL import Image
        print("[OK] Pillow доступен")
    except ImportError:
        print("[NO] Pillow не установлен")

    try:
        import pdf2image
        print("[OK] pdf2image доступен")
    except ImportError:
        print("[NO] pdf2image не установлен")

    # Если у вас есть тестовое изображение, раскомментируйте:
    # test_image_path = Path("test_image.jpg")
    # if test_image_path.exists():
    #     print(f"\n=== Тест OCR изображения: {test_image_path} ===")
    #     try:
    #         result = await converter.process(test_image_path)
    #         if result.success:
    #             print(f"✓ OCR успешен")
    #             print(f"Текст: {result.data['recognized_text'][:200]}...")
    #             print(f"Уверенность: {result.data['confidence_score']:.1f}%")
    #         else:
    #             print(f"✗ OCR не удался: {result.error}")
    #     except Exception as e:
    #         print(f"✗ Ошибка при тестировании: {e}")
    # else:
    #     print(f"\n⚠ Тестовое изображение {test_image_path} не найдено")

    print("\n=== Тест завершен ===")

if __name__ == "__main__":
    asyncio.run(test_ocr())