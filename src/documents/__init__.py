"""
Модули для работы с документами - система документооборота AIVAN
Включает: саммаризацию, анализ рисков, чат с документами, обезличивание, перевод, распознание текста
"""

from .anonymizer import DocumentAnonymizer
from .base import DocumentProcessor, DocumentResult, ProcessingError
from .document_chat import DocumentChat
from .document_manager import DocumentManager
from .lawsuit_analyzer import LawsuitAnalyzer
from .ocr_converter import OCRConverter
from .risk_analyzer import RiskAnalyzer
from .storage_backends import ArtifactUploader, NoopArtifactUploader, S3ArtifactUploader
from .summarizer import DocumentSummarizer
from .translator import DocumentTranslator

__all__ = [
    "DocumentManager",
    "DocumentProcessor",
    "DocumentResult",
    "ProcessingError",
    "DocumentSummarizer",
    "RiskAnalyzer",
    "LawsuitAnalyzer",
    "DocumentChat",
    "DocumentAnonymizer",
    "DocumentTranslator",
    "OCRConverter",
    "ArtifactUploader",
    "NoopArtifactUploader",
    "S3ArtifactUploader",
]
