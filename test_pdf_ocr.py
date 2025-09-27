#!/usr/bin/env python3
"""
Тест PDF-OCR функциональности
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

async def test_pdf_ocr():
    """Тест PDF OCR конвертера"""
    converter = OCRConverter()

    print("=== Тест PDF-OCR ===")
    print(f"Зависимости: {converter.get_required_dependencies()}")

    # Проверяем доступность зависимостей для PDF-OCR
    print("\n=== Проверка PDF-OCR зависимостей ===")

    try:
        import fitz  # PyMuPDF
        print("[OK] PyMuPDF (fitz) доступен")
    except ImportError:
        print("[NO] PyMuPDF не установлен")

    try:
        import ocrmypdf
        print("[OK] ocrmypdf доступен")
    except ImportError:
        print("[NO] ocrmypdf не установлен")

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

    # Создаем тестовый PDF с текстом для проверки
    print("\n=== Создание тестового PDF ===")
    test_pdf_path = Path("test_document.pdf")

    if not test_pdf_path.exists():
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(test_pdf_path), pagesize=letter)
            c.drawString(100, 750, "Тестовый документ для OCR")
            c.drawString(100, 730, "Этот текст должен быть распознан системой.")
            c.drawString(100, 710, "Документ содержит русский и English текст.")
            c.drawString(100, 690, "Дата: 2025-09-27")
            c.drawString(100, 670, "Номер документа: 12345")
            c.save()
            print(f"[OK] Создан тестовый PDF: {test_pdf_path}")
        except ImportError:
            print("[NO] reportlab не доступен, не можем создать тестовый PDF")
            return

    # Тестируем PDF-OCR
    if test_pdf_path.exists():
        print(f"\n=== Тест PDF-OCR: {test_pdf_path} ===")
        try:
            result = await converter.process(test_pdf_path)
            if result.success:
                print("[OK] PDF-OCR успешен")
                text_preview = result.data['recognized_text'][:200].replace('\n', ' ')
                print(f"Распознанный текст: {text_preview}...")
                print(f"Уверенность: {result.data['confidence_score']:.1f}%")
                print(f"Анализ качества: {result.data['quality_analysis']['quality_level']}")

                # Показываем рекомендации если есть
                recommendations = result.data['quality_analysis'].get('recommendations', [])
                if recommendations:
                    print("Рекомендации:")
                    for rec in recommendations:
                        print(f"  - {rec}")
            else:
                print(f"[NO] PDF-OCR не удался: {result.error}")
        except Exception as e:
            print(f"[ERROR] Ошибка при тестировании PDF-OCR: {e}")
    else:
        print("[NO] Тестовый PDF не найден")

    print("\n=== Тест завершен ===")

if __name__ == "__main__":
    asyncio.run(test_pdf_ocr())