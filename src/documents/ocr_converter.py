"""
Модуль OCR и конвертации
Распознавание сканированных документов и преобразование в редактируемый текст.
Пайплайн: PaddleOCR → (опционально) OpenAI Vision → mock.
Для PDF: сначала встроенный текст, затем постраничный OCR с конкурентной обработкой.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import is_image_file

logger = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————————
# Конфигурация по ENV с безопасными дефолтами
# ————————————————————————————————————————————————————————————————————————————

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except Exception:
        return default


# ————————————————————————————————————————————————————————————————————————————
# Основной класс
# ————————————————————————————————————————————————————————————————————————————

@dataclass
class PageResult:
    page: int
    text: str
    confidence: float


class OCRConverter(DocumentProcessor):
    """OCR распознавание и конвертация документов."""

    def __init__(self) -> None:
        super().__init__(name="OCRConverter", max_file_size=100 * 1024 * 1024)  # 100 MB
        self.supported_formats = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]

        # Настройки из окружения
        self.paddle_lang_base: str = (os.getenv("OCR_LANG") or "ru").strip()
        self.use_openai: bool = _env_bool("OCR_ALLOW_OPENAI", True)
        self.openai_model: str = (os.getenv("OPENAI_OCR_MODEL") or "gpt-4o-mini").strip()
        self.openai_timeout: float = _env_float("OPENAI_TIMEOUT", 60.0)
        self.max_pdf_concurrency: int = max(1, _env_int("OCR_MAX_CONCURRENCY", 3))
        self.preprocess_max_side: int = max(512, _env_int("OCR_MAX_SIDE", 2600))

        # Простенький in-memory кэш результатов (на время жизни процесса)
        # ключ: (sha256 файла, "paddle"|"openai") -> (text, confidence)
        self._cache: Dict[Tuple[str, str], Tuple[str, float]] = {}

    # ——————————————————————————————————————
    # Публичный метод обработки
    # ——————————————————————————————————————

    async def process(
        self, file_path: str | Path, output_format: str = "txt", **kwargs: Any
    ) -> DocumentResult:
        path = Path(file_path)
        file_extension = path.suffix.lower()

        try:
            if is_image_file(path):
                text, confidence = await self._ocr_image(path)
                file_type = "image"
                pages_info: List[Dict[str, Any]] = []
            elif file_extension == ".pdf":
                text, confidence, pages_info = await self._ocr_pdf(path)
                file_type = "pdf"
            else:
                raise ProcessingError(
                    f"Неподдерживаемый формат для OCR: {file_extension}", code="FORMAT_ERROR"
                )

            if not text or not text.strip():
                raise ProcessingError("Не удалось распознать текст в документе", code="OCR_NO_TEXT")

            cleaned_text = self._clean_ocr_text(text)
            quality_analysis = self._analyze_ocr_quality(cleaned_text, confidence)

            result_data: Dict[str, Any] = {
                "recognized_text": cleaned_text,
                "confidence_score": float(confidence),
                "quality_analysis": quality_analysis,
                "original_file": str(path),
                "output_format": output_format,
                "processing_info": {
                    "file_type": file_type,
                    "text_length": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                    "pages_processed": len(pages_info) if pages_info else (1 if file_type == "image" else 0),
                },
            }
            if pages_info:
                result_data["pages"] = pages_info  # [{page, text, confidence}...]

            return DocumentResult.success_result(
                data=result_data,
                message=f"OCR распознавание завершено с уверенностью {confidence:.1f}%",
            )

        except ProcessingError:
            raise
        except Exception as e:
            logger.exception("Ошибка OCR")
            raise ProcessingError(f"Ошибка OCR: {e}", code="OCR_ERROR") from e

    # ——————————————————————————————————————
    # Изображения
    # ——————————————————————————————————————

    async def _ocr_image(self, image_path: Path) -> Tuple[str, float]:
        """
        OCR изображения: предобработка → PaddleOCR → (опционально) OpenAI Vision → mock.
        Используется кэш по контент-хэшу.
        """
        # Контент-хэш для кэширования
        sha = await asyncio.to_thread(lambda: hashlib.sha256(image_path.read_bytes()).hexdigest())

        # 0) Предобработка (временный файл)
        preprocessed: Path | None = None
        try:
            preprocessed = await self._preprocess_image(image_path)
            img_for_ocr = preprocessed or image_path

            # 1) PaddleOCR (кэш)
            cached = self._cache.get((sha, "paddle"))
            if cached:
                return cached

            text, conf = await self._paddleocr_image(img_for_ocr)
            if text:
                self._cache[(sha, "paddle")] = (text, conf)
                return text, conf

            # 2) OpenAI Vision (кэш)
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if self.use_openai and api_key:
                cached = self._cache.get((sha, "openai"))
                if cached:
                    return cached

                text, conf = await self._openai_ocr_image(img_for_ocr, api_key=api_key)
                if text:
                    self._cache[(sha, "openai")] = (text, conf)
                    return text, conf

            # 3) Заглушка
            return self._mock_ocr_result(image_path)

        finally:
            if preprocessed and preprocessed.exists():
                try:
                    preprocessed.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _preprocess_image(self, image_path: Path) -> Path | None:
        """
        Лёгкая предобработка: EXIF-rotate, даунскейл, grayscale, автоконтраст, median filter.
        Возвращает путь к временному PNG или None, если предобработка не нужна/не вышло.
        """
        try:
            from PIL import Image, ImageOps, ImageFilter

            def _do() -> Path:
                im = Image.open(image_path)
                im = ImageOps.exif_transpose(im)

                if max(im.size) > self.preprocess_max_side:
                    im.thumbnail((self.preprocess_max_side, self.preprocess_max_side))

                im = ImageOps.grayscale(im)
                im = ImageOps.autocontrast(im)
                im = im.filter(ImageFilter.MedianFilter(3))

                # необязательный deskew через OpenCV (если установлен)
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore

                    arr = np.array(im)
                    th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    coords = cv2.findNonZero(255 - th)
                    if coords is not None:
                        rect = cv2.minAreaRect(coords)
                        angle = rect[-1]
                        angle = -(90 + angle) if angle < -45 else -angle
                        (h, w) = arr.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        im = Image.fromarray(arr)
                except Exception:
                    pass

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                try:
                    im.save(tmp.name, format="PNG", optimize=True)
                    return Path(tmp.name)
                finally:
                    tmp.close()

            return await asyncio.to_thread(_do)

        except Exception as e:
            logger.debug("Предобработка изображения не выполнена: %s", e)
            return None

    async def _paddleocr_image(self, image_path: Path) -> Tuple[str, float]:
        """
        OCR через PaddleOCR с адаптивным языком: пробуем базовый язык и EN, берём лучший результат.
        """
        try:
            from paddleocr import PaddleOCR  # импорт внутри
        except Exception as e:
            raise RuntimeError("PaddleOCR не установлен") from e

        langs_try: List[str] = [self.paddle_lang_base]
        if self.paddle_lang_base.lower() != "en":
            langs_try.append("en")

        best_text, best_conf = "", 0.0

        for lang in langs_try:
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang)
                result = await asyncio.to_thread(ocr.ocr, str(image_path), True)
                text, conf = self._collect_paddle_result(result)
                logger.info("PaddleOCR lang=%s → conf=%.1f, len=%d", lang, conf, len(text))
                if conf > best_conf:
                    best_text, best_conf = text, conf
                if best_conf >= 85.0:  # достаточно хорошо
                    break
            except Exception as e:
                logger.warning("PaddleOCR (lang=%s) ошибка: %s", lang, e)

        return best_text, best_conf

    @staticmethod
    def _collect_paddle_result(result: Any) -> Tuple[str, float]:
        if not result or not result[0]:
            return "", 0.0
        texts: List[str] = []
        confs: List[float] = []
        for line in result[0]:
            if line and len(line) >= 2:
                _, (text, score) = line[0], line[1]
                if text and str(text).strip():
                    texts.append(str(text).strip())
                    try:
                        confs.append(float(score) * 100.0)
                    except Exception:
                        pass
        text = " ".join(texts).strip()
        conf = (sum(confs) / len(confs)) if confs else 0.0
        return text, conf

    async def _openai_ocr_image(self, image_path: Path, api_key: str) -> Tuple[str, float]:
        """
        OCR через OpenAI Vision (Chat Completions API).
        Возвращает (plain_text, confidence_heuristic).
        """
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        img_bytes = await asyncio.to_thread(image_path.read_bytes)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        payload = {
            "model": self.openai_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an OCR engine. Extract ALL visible text from the image "
                        "in logical reading order. Output plain UTF-8 text only, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract plain UTF-8 text from this image."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        data = await self._openai_post(payload, headers=headers, timeout=self.openai_timeout)
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        confidence = 90.0 if text else 0.0
        return text, float(confidence)

    async def _openai_post(self, json: dict, headers: dict, timeout: float, max_retries: int = 3) -> dict:
        """
        HTTP POST с ретраями/бэкоффом для OpenAI.
        """
        backoff = 1.5
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json)
                    r.raise_for_status()
                    return r.json()
                except httpx.HTTPStatusError as e:
                    last_err = e
                    code = e.response.status_code
                    if code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(backoff)
                        backoff *= 1.7
                        continue
                    raise
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    last_err = e
                    await asyncio.sleep(backoff)
                    backoff *= 1.7
            raise last_err or RuntimeError("OpenAI request failed")

    # ——————————————————————————————————————
    # PDF
    # ——————————————————————————————————————

    async def _ocr_pdf(self, pdf_path: Path) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        1) Пытаемся извлечь встроенный текст.
        2) Если качества недостаточно — рендерим все страницы в PNG и OCR-им их конкурентно.
        Возвращаем (combined_text, avg_confidence, pages_info[]).
        """
        logger.info("Начало обработки PDF: %s", pdf_path)

        # 1) Встроенный текст
        embedded_text, embedded_conf = await self._extract_pdf_text(pdf_path)
        quality = self._analyze_extracted_text_quality(embedded_text)
        logger.info("Качество встроенного текста: %.2f", quality)

        if quality >= 0.7 and len(embedded_text.strip()) > 50:
            logger.info("Использован встроенный текст PDF")
            return embedded_text, embedded_conf, []

        # 2) Рендер всех страниц → PNG
        page_images = await self._render_pdf_pages_to_images(pdf_path)
        try:
            # 3) Конкурентный OCR страниц
            pages: List[PageResult] = await self._ocr_images_concurrently(page_images)

            if pages:
                pages_sorted = sorted(pages, key=lambda p: p.page)
                combined = "\n\n".join(p.text for p in pages_sorted if p.text)
                avg_conf = sum(p.confidence for p in pages_sorted) / len(pages_sorted)

                # Гибрид с встроенным текстом, если он был
                if embedded_text.strip():
                    if len(combined) > len(embedded_text) * 1.5:
                        logger.info("OCR дал существенно больше текста — берём OCR")
                        return combined, float(avg_conf), [
                            {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                        ]
                    merged = f"{embedded_text}\n\n--- OCR ДОПОЛНЕНИЕ ---\n\n{combined}"
                    return merged, float((embedded_conf + avg_conf) / 2.0), [
                        {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                    ]

                return combined, float(avg_conf), [
                    {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                ]

            return "", 0.0, []

        finally:
            # Удаляем временные PNG
            for p in page_images:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _extract_pdf_text(self, pdf_path: Path) -> Tuple[str, float]:
        """Извлечение встроенного текста из PDF с PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF

            def _read() -> Tuple[str, float]:
                doc = fitz.open(str(pdf_path))
                try:
                    parts: List[str] = []
                    for i in range(len(doc)):
                        page = doc[i]
                        txt = page.get_text()
                        if txt and txt.strip():
                            parts.append(txt)
                    if parts:
                        return "\n\n".join(parts), 100.0
                    return "", 0.0
                finally:
                    doc.close()

            return await asyncio.to_thread(_read)

        except ImportError:
            logger.error("PyMuPDF (fitz) не установлен")
            return "", 0.0
        except Exception as e:
            logger.error("Ошибка извлечения текста из PDF: %s", e)
            return "", 0.0

    async def _render_pdf_pages_to_images(self, pdf_path: Path) -> List[Path]:
        """Рендерит все страницы PDF в временные PNG и возвращает список путей."""
        try:
            import fitz  # PyMuPDF

            def _render_all() -> List[Path]:
                doc = fitz.open(str(pdf_path))
                try:
                    paths: List[Path] = []
                    for i in range(len(doc)):
                        page = doc[i]
                        pm = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # x2 масштаб
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        pm.save(tmp.name)
                        paths.append(Path(tmp.name))
                        tmp.close()
                    return paths
                finally:
                    doc.close()

            return await asyncio.to_thread(_render_all)

        except Exception as e:
            logger.error("Не удалось отрендерить PDF в изображения: %s", e)
            return []

    async def _ocr_images_concurrently(self, image_paths: List[Path]) -> List[PageResult]:
        """OCR для списка изображений с ограничением конкуренции."""
        sem = asyncio.Semaphore(self.max_pdf_concurrency)
        results: List[PageResult] = []

        async def worker(idx: int, p: Path):
            async with sem:
                try:
                    text, conf = await self._ocr_image(p)
                    if text:
                        results.append(PageResult(page=idx + 1, text=text, confidence=float(conf)))
                except Exception as e:
                    logger.warning("Ошибка OCR страницы %s: %s", idx + 1, e)

        await asyncio.gather(*(worker(i, p) for i, p in enumerate(image_paths)))
        return results

    # ——————————————————————————————————————
    # Очистка/качество
    # ——————————————————————————————————————

    def _mock_ocr_result(self, file_path: Path) -> Tuple[str, float]:
        return (
            f"[OCR ЗАГЛУШКА] Содержимое файла {file_path.name} не распознано локальными движками.",
            50.0,
        )

    def _clean_ocr_text(self, text: str) -> str:
        """Очистка текста: схлопывание пробелов + аккуратные паттерн-фиксы."""
        if not text:
            return ""
        cleaned = " ".join(text.split())
        cleaned = self._apply_ocr_corrections(cleaned)
        return cleaned

    def _apply_ocr_corrections(self, text: str) -> str:
        """Умные исправления частых OCR-ошибок (минимум ложных срабатываний)."""
        import re

        rules: List[Tuple[str, str]] = [
            # 1) цифры/буквы в числовом окружении
            (r"(?<=\d)[ОO](?=\d)", "0"),
            (r"(?<=\d)З(?=\d)", "3"),

            # 2) частые рус/англ подмены
            (r"\bнaлог\b", "налог"),
            (r"\bзaкон\b", "закон"),
            (r"\bпрaво\b", "право"),
            (r"\bрублеи\b", "рублей"),
            (r"\bрro\b", "pro"),
            (r"\bсоm\b", "com"),
            (r"\bоrg\b", "org"),

            # 3) кавычки
            (r"«([^»]*)»", r'"\1"'),
            (r"„([^\"\n]*)\"", r'"\1"'),

            # 4) пробелы/дефисы/пунктуация
            (r"(\w)\s*-\s*(\w)", r"\1-\2"),
            (r"\s+([,.;:!?])", r"\1"),
            (r"([,.;:!?])([А-Яа-яA-Za-z])", r"\1 \2"),
        ]

        out = text
        for pat, repl in rules:
            out = re.sub(pat, repl, out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _analyze_extracted_text_quality(self, text: str) -> float:
        """Оценка пригодности встроенного PDF-текста."""
        if not text or len(text.strip()) < 10:
            return 0.0

        total = len(text)
        alpha = sum(1 for c in text if c.isalpha())
        suspicious_set = set("�□■●○◆◇♦♠♣♥")
        suspicious = sum(1 for c in text if c in suspicious_set)

        words = text.split()
        meaningful = sum(1 for w in words if len(w) > 2 and any(ch.isalpha() for ch in w))

        alpha_ratio = alpha / total if total else 0.0
        susp_ratio = suspicious / total if total else 0.0
        word_ratio = meaningful / len(words) if words else 0.0

        score = alpha_ratio * 0.4 + word_ratio * 0.4 + (1.0 - susp_ratio) * 0.2
        return float(min(1.0, max(0.0, score)))

    def _analyze_ocr_quality(self, text: str, confidence: float) -> Dict[str, Any]:
        """Человеко-понятная оценка качества OCR результата."""
        level = "низкое"
        if confidence >= 90:
            level = "отличное"
        elif confidence >= 80:
            level = "хорошее"
        elif confidence >= 70:
            level = "удовлетворительное"
        elif confidence >= 60:
            level = "среднее"

        word_count = len(text.split()) if text else 0
        char_count = len(text) if text else 0
        suspicious_chars = sum(1 for ch in text if ch in "°§¤¦№�")

        recommendations: List[str] = []
        if confidence < 70:
            recommendations.append("Рекомендуется проверить качество исходного изображения")
        if suspicious_chars > 0:
            recommendations.append("Обнаружены подозрительные символы — требуется ручная проверка")
        if word_count < 10:
            recommendations.append("Распознано мало текста — проверьте настройки OCR")

        return {
            "confidence": float(confidence),
            "quality_level": level,
            "word_count": int(word_count),
            "char_count": int(char_count),
            "suspicious_chars": int(suspicious_chars),
            "recommendations": recommendations,
        }

    # ——————————————————————————————————————
    # Зависимости
    # ——————————————————————————————————————

    def get_required_dependencies(self) -> List[str]:
        """Список требований (без Tesseract/ocrmypdf)."""
        return [
            "pymupdf>=1.22.0",       # fitz — извлечение текста и рендер страниц
            "paddlepaddle>=2.6.0",   # бэкенд для PaddleOCR
            "paddleocr>=2.8.0",      # PaddleOCR
            "pillow>=9.0.0",         # предобработка изображений
            "httpx>=0.24.0",         # OpenAI fallback
            # "opencv-python-headless>=4.7.0", # (опционально) для deskew
        ]
