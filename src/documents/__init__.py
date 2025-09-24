"""
Модули для работы с документами - система документооборота AIVAN
Включает: саммаризацию, анализ рисков, чат с документами, обезличивание, перевод, OCR
"""

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .summarizer import DocumentSummarizer
from .risk_analyzer import RiskAnalyzer
from .document_chat import DocumentChat
from .anonymizer import DocumentAnonymizer
from .translator import DocumentTranslator
from .ocr_converter import OCRConverter

__all__ = [
    "DocumentManager"
    "DocumentProcessor",
    "DocumentResult",
    "ProcessingError",
    "DocumentSummarizer",
    "RiskAnalyzer",
    "DocumentChat",
    "DocumentAnonymizer",
    "DocumentTranslator",
    "OCRConverter",
]