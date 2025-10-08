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
from src.core.settings import AppSettings

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import httpx

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import is_image_file

logger = logging.getLogger(__name__)


# ————————————————————————————————————————————————————————————————————————————
# Конфигурация по ENV с безопасными дефолтами
# ————————————————————————————————————————————————————————————————————————————


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

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__(name="OCRConverter", max_file_size=100 * 1024 * 1024)  # 100 MB
        self.supported_formats = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]

        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings
        self.paddle_lang_base: str = (settings.get_str("OCR_LANG", "ru") or "ru").strip()
        self.use_openai: bool = settings.get_bool("OCR_ALLOW_OPENAI", True)
        self.openai_model: str = (settings.get_str("OPENAI_OCR_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
        self.openai_timeout: float = settings.get_float("OPENAI_TIMEOUT", 60.0)
        self.max_pdf_concurrency: int = max(1, settings.get_int("OCR_MAX_CONCURRENCY", 3))
        self.preprocess_max_side: int = max(512, settings.get_int("OCR_MAX_SIDE", 2600))
        self.second_pass_conf: float = settings.get_float("OCR_SECOND_PASS_CONF", 90.0)
        self.second_pass_qlt: float = settings.get_float("OCR_SECOND_PASS_QLT", 0.65)
        self.detector_limit: int = settings.get_int("OCR_DET_LIMIT", 1280)
        self._openai_api_key: str = settings.openai_api_key

        # In-memory кэш: (sha256, engine) -> (text, conf)
        self._cache: Dict[Tuple[str, str], Tuple[str, float]] = {}

    # ——————————————————————————————————————
    # Публичный метод обработки
    # ——————————————————————————————————————

    async def process(
        self, file_path: str | Path, output_format: str = "txt", progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None, **kwargs: Any
    ) -> DocumentResult:
        async def _notify(stage: str, percent: float, **payload: Any) -> None:
            if not progress_callback:
                return
            data: dict[str, Any] = {'stage': stage, 'percent': float(percent)}
            for key, value in payload.items():
                if value is None:
                    continue
                data[key] = value
            try:
                await progress_callback(data)
            except Exception:
                logger.debug('OCR progress callback failed at %s', stage, exc_info=True)

        path = Path(file_path)
        file_extension = path.suffix.lower()

        await _notify('preparing', 10, file_type=file_extension)

        try:
            if is_image_file(path):
                text, confidence = await self._ocr_image(path)
                file_type = "image"
                pages_info: List[Dict[str, Any]] = []
            elif file_extension == ".pdf":
                text, confidence, pages_info = await self._ocr_pdf(path, progress_callback=progress_callback)
                file_type = "pdf"
            else:
                raise ProcessingError(
                    f"Неподдерживаемый формат для OCR: {file_extension}", "FORMAT_ERROR"
                )

            if not text or not text.strip():
                raise ProcessingError("Не удалось распознать текст в документе", "OCR_NO_TEXT")

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
            raise ProcessingError(f"Ошибка OCR: {e}", "OCR_ERROR") from e

    # ——————————————————————————————————————
    # Изображения
    # ——————————————————————————————————————

    async def _ocr_image(self, image_path: Path) -> Tuple[str, float]:
        """
        OCR изображения: предобработка → PaddleOCR → (опционально) OpenAI Vision → mock.
        Используется кэш по контент-хэшу и метрика читабельности.
        """
        sha = await asyncio.to_thread(lambda: hashlib.sha256(image_path.read_bytes()).hexdigest())

        preprocessed: Path | None = None
        try:
            preprocessed = await self._preprocess_image(image_path)
            img_for_ocr = preprocessed or image_path

            # Paddle (кэш)
            cached = self._cache.get((sha, "paddle"))
            if cached:
                paddle_text, paddle_conf = cached
            else:
                paddle_text, paddle_conf = await self._paddleocr_image(img_for_ocr)
                if paddle_text:
                    self._cache[(sha, "paddle")] = (paddle_text, paddle_conf)

            best_text, best_conf = paddle_text, paddle_conf
            best_score = self._readability_score(paddle_text) if paddle_text else 0.0

            # OpenAI — если пусто или «каша»
            api_key = self._openai_api_key
            if self.use_openai and api_key and self._should_try_openai(paddle_text, paddle_conf):
                cached = self._cache.get((sha, "openai"))
                if cached:
                    oa_text, oa_conf = cached
                else:
                    oa_text, oa_conf = await self._openai_ocr_image(img_for_ocr, api_key=api_key)
                    if oa_text:
                        self._cache[(sha, "openai")] = (oa_text, oa_conf)

                oa_score = self._readability_score(oa_text) if oa_text else 0.0
                if oa_score > best_score or (oa_score == best_score and len(oa_text) > len(best_text)):
                    best_text, best_conf, best_score = oa_text, oa_conf, oa_score

            if best_text:
                # Композитная уверенность: движок × читабельность
                best_conf = self._refine_confidence(best_text, best_conf)
                return best_text, best_conf

            return self._mock_ocr_result(image_path)

        finally:
            if preprocessed and preprocessed.exists():
                try:
                    preprocessed.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _preprocess_image(self, image_path: Path) -> Path | None:
        """
        Предобработка: EXIF-rotate, масштабирование, grayscale, контраст, шумоподавление,
        лёгкая резкость, (опционально) CLAHE + deskew через OpenCV.
        Возвращает путь к временному PNG или None.
        """
        try:
            from PIL import Image, ImageOps, ImageFilter, ImageEnhance

            def _do() -> Path:
                im = Image.open(image_path)
                im = ImageOps.exif_transpose(im)

                # мягкий апскейл, если фото маленькое
                max_side = max(im.size)
                target = max(1200, min(self.preprocess_max_side, 1600))
                if max_side < target:
                    scale = target / float(max_side)
                    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)

                # даунскейл, если очень большое
                if max(im.size) > self.preprocess_max_side:
                    im.thumbnail((self.preprocess_max_side, self.preprocess_max_side), Image.LANCZOS)

                im = ImageOps.grayscale(im)
                im = ImageOps.autocontrast(im, cutoff=1)
                im = ImageEnhance.Sharpness(im).enhance(1.3)
                im = im.filter(ImageFilter.MedianFilter(size=3))

                # опционально: CLAHE + deskew
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore
                    arr = np.array(im)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    arr = clahe.apply(arr)

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

    async def _paddleocr_image(self, image_path: Path) -> tuple[str, float]:
        """
        PaddleOCR с адаптивным языком: пробуем базовый и EN, берём лучший.
        Если PaddleOCR недоступен — возвращаем ("", 0.0), чтобы сработал fallback.
        """
        try:
            from paddleocr import PaddleOCR  # импорт внутри
        except Exception as e:
            logger.warning("PaddleOCR import failed: %s", e)
            return "", 0.0

        try:
            det_limit = self.detector_limit
        except Exception:
            det_limit = self.detector_limit

        langs_try: list[str] = [self.paddle_lang_base]
        if self.paddle_lang_base.lower() != "en":
            langs_try.append("en")

        best_text, best_conf = "", 0.0
        for lang in langs_try:
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang, det_limit_side_len=det_limit)
                result = await asyncio.to_thread(ocr.ocr, str(image_path), True)
                text, conf = self._collect_paddle_result(result)
                logger.info("PaddleOCR lang=%s → conf=%.1f, len=%d", lang, conf, len(text))
                if conf > best_conf:
                    best_text, best_conf = text, conf
                if best_conf >= 85.0:
                    break
            except Exception as e:
                logger.warning("PaddleOCR (lang=%s) error: %s", lang, e)

        return best_text, best_conf

    @staticmethod
    def _collect_paddle_result(result: Any) -> Tuple[str, float]:
        """
        Собираем строки, отбрасываем шум (score<0.25),
        длино-взвешенное среднее по confidence (в процентах).
        """
        if not result or not result[0]:
            return "", 0.0

        lines: List[Tuple[str, float]] = []
        for line in result[0]:
            if line and len(line) >= 2:
                _, (text, score) = line[0], line[1]
                if text:
                    s = str(text).strip()
                    try:
                        sc = float(score)
                    except Exception:
                        sc = 0.0
                    if s and sc >= 0.25:
                        lines.append((s, sc))

        if not lines:
            return "", 0.0

        text = "\n".join(s for s, _ in lines).strip()
        weights = [max(1, len(s)) for s, _ in lines]
        scores = [sc * 100.0 for _, sc in lines]
        total_w = sum(weights)
        conf = sum(w * sc for w, sc in zip(weights, scores)) / total_w if total_w else 0.0
        return text, float(conf)

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
            for _ in range(max_retries):
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

    async def _ocr_pdf(self, pdf_path: Path, progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        1) Пытаемся извлечь встроенный текст.
        2) Если качества недостаточно — рендерим все страницы в PNG и OCR-им их конкурентно.
        Возвращаем (combined_text, avg_confidence, pages_info[]).
        """

        async def _emit(stage: str, percent: float, **payload: Any) -> None:
            if not progress_callback:
                return
            data: dict[str, Any] = {'stage': stage, 'percent': float(percent)}
            for key, value in payload.items():
                if value is None:
                    continue
                data[key] = value
            try:
                await progress_callback(data)
            except Exception:
                logger.debug('OCR PDF progress callback failed at %s', stage, exc_info=True)

        logger.info("Начало обработки PDF: %s", pdf_path)

        # 1) Встроенный текст
        embedded_text, embedded_conf = await self._extract_pdf_text(pdf_path)
        quality = self._analyze_extracted_text_quality(embedded_text)
        logger.info("Качество встроенного текста: %.2f", quality)

        if quality >= 0.7 and len(embedded_text.strip()) > 50:
            logger.info("Использован встроенный текст PDF")
            final_emb_conf = self._refine_confidence(embedded_text, embedded_conf)
            await _emit("ocr_running", 65, pages_total=1, pages_done=1, mode="embedded")
            await _emit("completed", 100, confidence=float(final_emb_conf), pages_total=1, mode="embedded")
            return embedded_text, float(final_emb_conf), []

        # 2) Рендер всех страниц → PNG
        page_images = await self._render_pdf_pages_to_images(pdf_path)
        try:
            # 3) Конкурентный OCR страниц
            pages: List[PageResult] = await self._ocr_images_concurrently(page_images, progress_callback=progress_callback)

            if pages:
                pages_sorted = sorted(pages, key=lambda p: p.page)
                combined = "\n\n".join(p.text for p in pages_sorted if p.text)
                avg_conf = sum(p.confidence for p in pages_sorted) / len(pages_sorted)

                # Гибрид с встроенным текстом, если он был
                if embedded_text.strip():
                    if len(combined) > len(embedded_text) * 1.5:
                        logger.info("OCR дал существенно больше текста — берём OCR")
                        final_conf = self._refine_confidence(combined, avg_conf)
                        return combined, float(final_conf), [
                            {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                        ]
                    merged = f"{embedded_text}\n\n--- OCR ДОПОЛНЕНИЕ ---\n\n{combined}"
                    final_merged_conf = self._refine_confidence(merged, (embedded_conf + avg_conf) / 2.0)
                    return merged, float(final_merged_conf), [
                        {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                    ]

                final_conf = self._refine_confidence(combined, avg_conf)
                return combined, float(final_conf), [
                    {"page": p.page, "text": p.text, "confidence": float(p.confidence)} for p in pages_sorted
                ]

            return "", 0.0, []

        finally:
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

    async def _ocr_images_concurrently(self, image_paths: List[Path], progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> List[PageResult]:
        """OCR для списка изображений с ограничением конкуренции."""
        total = len(image_paths)
        if total == 0:
            return []

        sem = asyncio.Semaphore(self.max_pdf_concurrency)
        results: List[PageResult] = []
        lock = asyncio.Lock()
        progress = {"done": 0}

        async def worker(idx: int, p: Path):
            async with sem:
                try:
                    text, conf = await self._ocr_image(p)
                    if text:
                        results.append(PageResult(page=idx + 1, text=text, confidence=float(conf)))
                except Exception as e:
                    logger.warning("Ошибка OCR страницы %s: %s", idx + 1, e)
                finally:
                    async with lock:
                        progress["done"] += 1
                        if progress_callback:
                            percent = 45 + (progress["done"] / total) * 40
                            data = {
                                "stage": "ocr_page",
                                "percent": percent,
                                "pages_total": total,
                                "pages_done": progress["done"],
                            }
                            try:
                                await progress_callback(data)
                            except Exception:
                                logger.debug("OCR page progress callback failed", exc_info=True)

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
    # Читабельность / композитная уверенность / 2-й проход
    # ——————————————————————————————————————

    def _readability_score(self, text: str) -> float:
        """
        Эвристика «насколько похоже на нормальный текст» (0..1).
        Учитывает долю букв, долю кириллицы, среднюю длину слова и штраф за «смешанные» слова.
        """
        import re

        t = (text or "").strip()
        if not t:
            return 0.0

        total = len(t)
        alpha = sum(1 for c in t if c.isalpha())
        if total == 0:
            return 0.0
        alpha_ratio = alpha / total

        words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", t)
        if not words:
            return alpha_ratio * 0.3

        cyr = sum(1 for w in words if re.fullmatch(r"[А-Яа-яЁё]{2,}", w))
        lat = sum(1 for w in words if re.fullmatch(r"[A-Za-z]{2,}", w))
        mixed = len(words) - cyr - lat

        avg_len = sum(len(w) for w in words) / len(words)
        cyr_ratio = cyr / len(words)
        mixed_ratio = mixed / len(words)

        score = (
            0.35 * alpha_ratio
            + 0.35 * cyr_ratio
            + 0.20 * min(1.0, avg_len / 7.0)
            - 0.20 * mixed_ratio
        )
        return max(0.0, min(1.0, score))

    def _refine_confidence(self, text: str, engine_conf: float) -> float:
        """
        Композитная уверенность: смесь читабельности и уверенности движка.
        final = 100 * (0.6 * readability + 0.4 * engine_conf/100), clamp[0..99.5].
        Для очень хороших текстов слегка дотягиваем до 95+.
        """
        r = self._readability_score(text)  # 0..1
        final = 100.0 * (0.6 * r + 0.4 * (engine_conf / 100.0))
        if r >= 0.85 and engine_conf >= 88:
            final = max(final, 95.0)
        return float(min(99.5, max(0.0, final)))

    def _should_try_openai(self, paddle_text: str, paddle_conf: float) -> bool:
        """
        Решаем, имеет ли смысл пробовать OpenAI, даже если Paddle что-то распознал.
        """
        score = self._readability_score(paddle_text)
        if paddle_conf < self.second_pass_conf:
            return True
        if score < self.second_pass_qlt:
            return True
        if len(paddle_text) < 80 and score < 0.5:
            return True
        return False

    # ——————————————————————————————————————
    # Зависимости
    # ——————————————————————————————————————
