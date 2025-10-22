from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import suppress, asynccontextmanager
from html import escape as html_escape
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import FSInputFile, Message, InlineKeyboardButton, InlineKeyboardMarkup

from core.bot_app.promt import JUDICIAL_PRACTICE_SEARCH_PROMPT
from core.bot_app.status_manager import ProgressStatus
from core.bot_app.stream_manager import StreamManager, StreamingCallback
from core.bot_app.typing_indicator import send_typing_once, typing_action
from core.bot_app.ui_components import Emoji
from src.core.attachments import QuestionAttachment
from src.core.exceptions import (
    ErrorContext,
    NetworkException,
    OpenAIException,
    SystemException,
    ValidationException,
)
from src.core.safe_telegram import format_safe_html, send_html_text
from src.core.bot_app import context as simple_context
from src.core.bot_app.common import get_user_session, get_safe_db_method, ensure_valid_user_id
from src.core.bot_app.feedback import (
    ensure_rating_snapshot,
    handle_pending_feedback,
    send_rating_request,
)
from src.core.garant_api import GarantAPIError
from src.core.validation import InputValidator
from src.core.excel_export import build_practice_excel

__all__ = [
    "ResponseTimer",
    "process_question",
    "process_question_with_attachments",
    "register_question_handlers",
]

QUESTION_ATTACHMENT_MAX_BYTES = 4 * 1024 * 1024  # 4MB per attachment
LONG_TEXT_HINT_THRESHOLD = 700  # heuristic порог для подсказки про длинные тексты
TELEGRAM_HTML_SAFE_LIMIT = 3500


@asynccontextmanager
async def _noop_async_context():
    yield


def _split_html_for_telegram(
    html: str,
    limit: int = TELEGRAM_HTML_SAFE_LIMIT,
    reserve: int = 0,
) -> list[str]:
    """Split sanitized HTML into chunks that fit Telegram limits."""
    text = (html or "").strip()
    if not text:
        return []

    if reserve > 0:
        limit = max(limit - reserve, 1)

    def _smart_cut(segment: str, max_len: int) -> tuple[str, str]:
        cut = min(max_len, len(segment))
        if cut <= 0:
            return "", segment

        lt = segment.rfind("<", 0, cut)
        gt = segment.rfind(">", 0, cut)
        if lt > gt:
            cut = lt
        for pattern in ("\n\n", "\n", " • ", " — ", ". ", "; ", ": ", ", ", " "):
            pos = segment.rfind(pattern, 0, cut)
            if pos >= 0 and pos >= cut - 200:
                cut = pos + len(pattern)
                break

        if cut <= 0:
            cut = min(max_len, len(segment))
        head = segment[:cut].rstrip()
        tail = segment[cut:]
        if not head and tail:
            head = tail[:max_len]
            tail = tail[max_len:]
        return head, tail

    chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(buffer)
            buffer = ""

    for paragraph in text.split("\n\n"):
        segment = paragraph.strip()
        if not segment:
            continue

        candidate = (buffer + ("\n\n" if buffer else "") + segment) if buffer else segment
        if len(candidate) <= limit:
            buffer = candidate
            continue

        flush_buffer()

        remaining = segment
        while len(remaining) > limit:
            head, remaining = _smart_cut(remaining, limit)
            if not head:
                raw_slice = remaining[:limit]
                fallback = raw_slice.rstrip() or raw_slice
                chunks.append(fallback)
                remaining = remaining[len(raw_slice):]
                continue
            chunks.append(head)
        buffer = remaining

    flush_buffer()

    return chunks


def _ensure_double_newlines(html: str) -> str:
    """Ensure single newlines are promoted to double newlines for better spacing."""
    if not html:
        return ""

    normalized = html.replace("\r\n", "\n")
    normalized = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _back_to_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Назад в меню", callback_data="back_to_main")]
        ]
    )

logger = logging.getLogger("ai-ivan.simple.questions")


class ResponseTimer:
    def __init__(self) -> None:
        self._t0: float | None = None
        self.duration = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def stop(self) -> None:
        if self._t0 is not None:
            self.duration = max(0.0, time.monotonic() - self._t0)

    def get_duration_text(self) -> str:
        seconds = int(self.duration)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


GARANT_KIND_LABELS = {
    "301": "Суды общей юрисдикции",
    "302": "Арбитражные суды",
    "303": "Суды по уголовным делам",
}


def _prepare_garant_excel_fragments(
    search_results: Sequence[Any] | None,
    sutyazhnik_results: Sequence[Any] | None,
    *,
    document_base_url: str | None,
    max_items: int = 10,
) -> list[Any]:
    if not max_items or max_items <= 0:
        return []

    fragments: list[Any] = []
    seen: set[tuple[Any, ...]] = set()

    def _add_fragment(
        key: tuple[Any, ...],
        title: str,
        *,
        excerpt: str,
        court: str | None = None,
        url: str | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
        date: str | None = None,
        region: str | None = None,
        relevance: float | str | None = None,
        summary: str | None = None,
        decision: str | None = None,
        norms: str | None = None,
        applicability: str | None = None,
    ) -> None:
        if len(fragments) >= max_items or key in seen:
            return
        seen.add(key)
        metadata: dict[str, Any] = {
            "title": title,
            "name": title,
            "court": court or "",
            "url": url or "",
            "link": url or "",
            "date": date or "",
            "decision_date": date or "",
            "region": region or "",
            "score": relevance if relevance is not None else "",
            "summary": summary or excerpt,
            "decision_summary": decision or "",
            "norms_summary": norms or "",
            "applicability": applicability or "",
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        match = SimpleNamespace(metadata=metadata, score=None)
        fragments.append(SimpleNamespace(match=match, header=title, excerpt=excerpt))

    search_results = list(search_results or [])
    sutyazhnik_results = list(sutyazhnik_results or [])

    # 1) Приоритет — судебные решения из Сутяжника.
    for priority_kind in ("301", "302", "303"):
        kind_label = GARANT_KIND_LABELS.get(priority_kind, "Суды")
        for item in sutyazhnik_results:
            kind_value = str(getattr(item, "kind", "") or "")
            if kind_value != priority_kind:
                continue
            for ref in getattr(item, "courts", []) or []:
                title = str(getattr(ref, "name", "") or "").strip()
                if not title:
                    continue
                url = (
                    ref.absolute_url(document_base_url)
                    if hasattr(ref, "absolute_url")
                    else getattr(ref, "url", None)
                )
                topic = getattr(ref, "topic", None)
                norm_names = [
                    str(getattr(norm, "name", "") or "").strip()
                    for norm in getattr(item, "norms", []) or []
                    if str(getattr(norm, "name", "") or "").strip()
                ]
                norms_text = "\n".join(norm_names)
                key = ("sutyazhnik_court", kind_value, topic, url, title)
                metadata_extra = {
                    "source": "sutyazhnik",
                    "kind": kind_value,
                    "topic": topic,
                    "norm_names": norm_names,
                }
                excerpt = f"{kind_label}: {title}"
                _add_fragment(
                    key,
                    title=title,
                    excerpt=excerpt,
                    court=kind_label,
                    url=url,
                    metadata_extra=metadata_extra,
                    date=None,
                    region=None,
                    relevance=None,
                    summary=f"Решение категории «{kind_label}»: {title}",
                    decision=f"Решение суда ({kind_label}). Ознакомьтесь с выводами по ссылке.",
                    norms=norms_text,
                    applicability=f"Анализируйте применимость решения ({kind_label}) к вашему делу.",
                )
                if len(fragments) >= max_items:
                    return fragments

    # 2) Нормативные акты, если решений недостаточно.
    for item in sutyazhnik_results:
        kind_value = str(getattr(item, "kind", "") or "")
        kind_label = GARANT_KIND_LABELS.get(kind_value, "Нормативные акты")
        for ref in getattr(item, "norms", []) or []:
            title = str(getattr(ref, "name", "") or "").strip()
            if not title:
                continue
            url = (
                ref.absolute_url(document_base_url)
                if hasattr(ref, "absolute_url")
                else getattr(ref, "url", None)
            )
            topic = getattr(ref, "topic", None)
            key = ("sutyazhnik_norm", kind_value, topic, url, title)
            metadata_extra = {
                "source": "sutyazhnik_norm",
                "kind": kind_value,
                "topic": topic,
            }
            excerpt = f"Нормативный акт ({kind_label})"
            _add_fragment(
                key,
                title=title,
                excerpt=excerpt,
                court=kind_label,
                url=url,
                metadata_extra=metadata_extra,
                date=None,
                region=None,
                relevance=None,
                summary=f"Нормативный акт для категории «{kind_label}».",
                decision="Используйте норму при аргументации позиции.",
                norms=title,
                applicability="Цитируйте при подготовке ответов и процессуальных документов.",
            )
            if len(fragments) >= max_items:
                return fragments

    # 3) Документы из обычного поиска ГАРАНТа.
    for result in search_results:
        document = getattr(result, "document", None)
        if not document:
            continue
        title = str(getattr(document, "name", "") or "").strip()
        if not title:
            continue
        url = (
            document.absolute_url(document_base_url)
            if hasattr(document, "absolute_url")
            else getattr(document, "url", None)
        )
        topic = getattr(document, "topic", None)
        snippets = list(getattr(result, "snippets", []) or [])
        if snippets:
            snippet = snippets[0]
            path = snippet.formatted_path() if hasattr(snippet, "formatted_path") else ""
            entry = getattr(snippet, "entry", None)
            relevance_value: float | str | None = getattr(snippet, "relevance", None)
            if isinstance(relevance_value, str):
                try:
                    relevance_value = float(relevance_value)
                except ValueError:
                    pass
            excerpt_parts: list[str] = []
            if entry is not None:
                excerpt_parts.append(f"Блок {entry}")
            if path:
                excerpt_parts.append(path)
            excerpt = " — ".join(excerpt_parts) if excerpt_parts else "Документ из поиска ГАРАНТ"
            metadata_extra = {
                "source": "search",
                "topic": topic,
                "entry": entry,
            }
        else:
            excerpt = "Документ из поиска ГАРАНТ"
            metadata_extra = {
                "source": "search",
                "topic": topic,
            }
            relevance_value = None
        key = ("search", topic, url, title)
        _add_fragment(
            key,
            title=title,
            excerpt=excerpt,
            url=url,
            metadata_extra=metadata_extra,
            summary=excerpt or f"Документ из поиска ГАРАНТ: {title}",
            decision="Требуется изучить текст решения по ссылке.",
            norms="",
            applicability="Используйте документ для расширения практики по теме.",
            date=None,
            region=None,
            relevance=relevance_value,
        )
        if len(fragments) >= max_items:
            break

    return fragments


async def _collect_question_attachments(message: Message) -> list[QuestionAttachment]:
    bot = message.bot
    if bot is None:
        raise ValueError("Не удалось получить контекст бота для загрузки файла")

    attachments: list[QuestionAttachment] = []

    if message.document:
        document = message.document
        if document.file_size and document.file_size > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError("Файл слишком большой. Максимальный размер вложения — 4 МБ.")

        file_info = await bot.get_file(document.file_id)
        file_stream = await bot.download_file(file_info.file_path)
        try:
            data = await asyncio.to_thread(file_stream.read)
        finally:
            close_method = getattr(file_stream, "close", None)
            if callable(close_method):
                close_method()

        attachments.append(
            QuestionAttachment(
                filename=document.file_name or "document",
                mime_type=document.mime_type or "application/octet-stream",
                data=data,
            )
        )

    if message.photo:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError("Изображение слишком большое. Максимальный размер вложения — 4 МБ.")

        file_info = await bot.get_file(photo.file_id)
        file_stream = await bot.download_file(file_info.file_path)
        try:
            data = await asyncio.to_thread(file_stream.read)
        finally:
            close_method = getattr(file_stream, "close", None)
            if callable(close_method):
                close_method()

        filename = f"photo_{photo.file_unique_id}.jpg"
        attachments.append(
            QuestionAttachment(
                filename=filename,
                mime_type="image/jpeg",
                data=data,
            )
        )

    if not attachments:
        raise ValueError("Не удалось определить вложение. Поддерживаются документы и изображения.")

    return attachments


async def _validate_question_or_reply(
    message: Message,
    text: str,
    user_id: int,
) -> str | None:
    validation = InputValidator.validate_question(text, user_id)
    if not validation.is_valid:
        errors = "\n".join(validation.errors or [])
        await message.answer(
            f"{Emoji.WARNING} <b>Некорректный вопрос</b>\n\n{html_escape(errors)}",
            parse_mode=ParseMode.HTML,
        )
        return None

    if validation.warnings:
        warnings = "\n".join(validation.warnings or [])
        await message.answer(
            f"{Emoji.INFO} <b>Предупреждение</b>\n\n{html_escape(warnings)}",
            parse_mode=ParseMode.HTML,
        )

    return validation.cleaned_data or text


async def _rate_limit_guard(user_id: int, message: Message) -> bool:
    limiter = simple_context.rate_limiter
    if limiter is None:
        return True

    allowed = await limiter.allow(user_id)
    if allowed:
        return True

    await message.answer(
        f"{Emoji.WARNING} <b>Слишком много запросов</b>\n\n"
        "Попробуйте повторить вопрос чуть позже.",
        parse_mode=ParseMode.HTML,
    )
    return False


async def _delete_status_message(bot, chat_id: int, message_id: int, attempts: int = 3) -> bool:
    if not bot or not message_id:
        return False

    for attempt in range(attempts):
        try:
            await bot.delete_message(chat_id, message_id)
            return True
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest as exc:
            text = str(exc).lower()
            if "message to delete not found" in text or "message can't be deleted" in text:
                return False
            await asyncio.sleep(0.2 * (attempt + 1))
        except Exception:
            await asyncio.sleep(0.2 * (attempt + 1))
    return False


async def _start_status_indicator(message: Message) -> ProgressStatus | None:
    if not simple_context.USE_ANIMATION or not message.bot:
        return None

    status = ProgressStatus(
        message.bot,
        message.chat.id,
        show_checklist=True,
        show_context_toggle=False,
        min_edit_interval=0.9,
        auto_advance_stages=True,
    )
    await status.start(auto_cycle=True, interval=2.7)
    return status


async def _stop_status_indicator(status: ProgressStatus | None, ok: bool) -> None:
    if status is None:
        return

    try:
        if ok:
            await status.complete()
            if status.message_id:
                await _delete_status_message(status.bot, status.chat_id, status.message_id)
        else:
            await status.fail("Ошибка обработки запроса")
    except Exception as exc:  # pragma: no cover - только лог
        logger.debug("Failed to finalize status indicator: %s", exc)


async def process_question_with_attachments(message: Message) -> None:
    caption = (message.caption or "").strip()
    if not caption:
        warning_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>Добавьте текст вопроса</b>",
            "Напишите короткое описание ситуации в подписи к файлу и отправьте снова.",
        ])
        await message.answer(warning_msg, parse_mode=ParseMode.HTML)
        return

    try:
        attachments = await _collect_question_attachments(message)
    except ValueError as exc:
        error_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>Не удалось обработать вложение</b>",
            html_escape(str(exc)),
        ])
        await message.answer(error_msg, parse_mode=ParseMode.HTML)
        return

    if not attachments:
        await process_question(message, text_override=caption)
        return

    await process_question(message, text_override=caption, attachments=attachments)


async def process_question(
    message: Message,
    *,
    text_override: str | None = None,
    attachments: Sequence[QuestionAttachment] | None = None,
) -> str | None:
    """Главный обработчик юридических вопросов."""
    if not message.from_user:
        return None

    user_session = get_user_session(message.from_user.id)
    attachments_list = list(attachments or [])

    action = "upload_document" if attachments_list else "typing"
    await send_typing_once(message.bot, message.chat.id, action)

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="process_question")
    except ValidationException as exc:
        context = ErrorContext(
            chat_id=message.chat.id if message.chat else None,
            function_name="process_question",
        )
        error_handler = simple_context.error_handler
        if error_handler is not None:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in process_question: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Ошибка идентификатора пользователя</b>\n\nПопробуйте перезапустить диалог.",
            parse_mode=ParseMode.HTML,
        )
        return None

    chat_id = message.chat.id
    message_id = message.message_id

    error_handler = simple_context.error_handler
    db = simple_context.db
    access_service = simple_context.access_service
    openai_service = simple_context.openai_service

    error_context = ErrorContext(
        user_id=user_id, chat_id=chat_id, message_id=message_id, function_name="process_question"
    )

    question_text = ((text_override if text_override is not None else (message.text or ""))).strip()
    quota_msg_to_send: str | None = None

    if not hasattr(user_session, "pending_feedback_request_id"):
        user_session.pending_feedback_request_id = None
    if user_session.pending_feedback_request_id is not None:
        await handle_pending_feedback(message, user_session, question_text)
        return None

    if question_text.startswith("/"):
        return None

    if error_handler is None:
        raise SystemException("Error handler not initialized", error_context)

    cleaned = await _validate_question_or_reply(message, question_text, user_id)
    if not cleaned:
        return None
    question_text = cleaned

    practice_mode_active = bool(getattr(user_session, "practice_search_mode", False))
    practice_mode_used = practice_mode_active
    practice_excel_fragments: list[Any] = []
    garant_search_results: list[Any] = []
    garant_sutyazhnik_results: list[Any] = []
    system_prompt_override: str | None = None
    rag_context = ""
    garant_context = ""
    garant_sutyazhnik_context = ""

    if practice_mode_active:
        system_prompt_override = JUDICIAL_PRACTICE_SEARCH_PROMPT
        rag_service = getattr(simple_context, "judicial_rag", None)
        if rag_service is not None:
            try:
                rag_context, _ = await rag_service.build_context(question_text)
            except Exception as rag_exc:  # noqa: BLE001
                logger.warning("Failed to build judicial practice context: %s", rag_exc)
        garant_client = getattr(simple_context, "garant_client", None)
        if getattr(garant_client, "enabled", False):
            try:
                garant_search_results = await garant_client.search_with_snippets(question_text)
                garant_context = garant_client.format_results(garant_search_results)
            except GarantAPIError as garant_exc:
                logger.warning("Garant API search failed: %s", garant_exc)
            except Exception as garant_exc:  # noqa: BLE001
                logger.error("Unexpected Garant API failure: %s", garant_exc, exc_info=True)
            if getattr(garant_client, "sutyazhnik_enabled", False):
                try:
                    garant_sutyazhnik_results = await garant_client.sutyazhnik_search(question_text)
                    garant_sutyazhnik_context = garant_client.format_sutyazhnik_results(garant_sutyazhnik_results)
                except GarantAPIError as garant_exc:
                    logger.warning("Garant Sutyazhnik search failed: %s", garant_exc)
                except Exception as garant_exc:  # noqa: BLE001
                    logger.error("Unexpected Garant Sutyazhnik failure: %s", garant_exc, exc_info=True)
            practice_excel_fragments = _prepare_garant_excel_fragments(
                garant_search_results,
                garant_sutyazhnik_results,
                document_base_url=getattr(garant_client, "document_base_url", None),
                max_items=10,
            )
        setattr(user_session, "practice_search_mode", False)

    request_blocks = [question_text]

    if rag_context:
        request_blocks.append("[Контекст судебной практики]\n" + rag_context)

    if garant_context:
        request_blocks.append(garant_context)
    if garant_sutyazhnik_context:
        request_blocks.append(garant_sutyazhnik_context)

    if attachments_list:
        attachment_lines = []
        for idx, item in enumerate(attachments_list, start=1):
            size_kb = max(1, item.size // 1024)
            attachment_lines.append(f"{idx}. {item.filename} ({item.mime_type}, {size_kb} KB)")
        request_blocks.append("[Вложения]\n" + "\n".join(attachment_lines))

    request_text = "\n\n".join(request_blocks)

    logger.info("Processing question from user %s: %s", user_id, question_text[:100])
    timer = ResponseTimer()
    timer.start()
    use_streaming = bool(simple_context.USE_STREAMING and not attachments_list)

    status: ProgressStatus | None = None

    ok_flag = False
    request_error_type = None
    stream_manager: StreamManager | None = None
    had_stream_content = False
    stream_final_text: str | None = None
    final_answer_text: str | None = None
    result: dict[str, Any] = {}
    request_start_time = time.time()
    request_record_id: int | None = None

    try:
        if not await _rate_limit_guard(user_id, message):
            return None

        quota_text = ""
        quota_is_trial: bool = False
        if access_service is not None:
            decision = await access_service.check_and_consume(user_id)
            if not decision.allowed:
                if decision.has_subscription and decision.subscription_plan:
                    plan_info = get_safe_db_method("get_plan_pricing", default_return=None)
                    plan_name = decision.subscription_plan
                    if plan_info:
                        try:
                            plan_data = plan_info(plan_name)
                            plan_name = plan_data.plan.name
                        except Exception:
                            pass
                    await message.answer(
                        f"{Emoji.WARNING} <b>Лимит запросов исчерпан</b>\n\n"
                        f"Тариф: <b>{plan_name}</b>\n"
                        "Продлите подписку или обновите тариф, чтобы продолжить работу.",
                        parse_mode=ParseMode.HTML,
                    )
                    return None

                await message.answer(
                    f"{Emoji.WARNING} <b>Квота запросов исчерпана</b>\n\n"
                    "Оформите подписку командой /buy, чтобы продолжить.",
                    parse_mode=ParseMode.HTML,
                )
                return None

            quota_text = decision.message or ""
            quota_is_trial = bool(decision.is_trial)

        status = await _start_status_indicator(message)

        typing_context_manager = (
            typing_action(message.bot, message.chat.id, action)
            if status is None and message.bot
            else _noop_async_context()
        )

        back_button_sent = False

        models = simple_context.derived().models if callable(simple_context.derived) else None
        model_to_use = (models or {}).get("primary") if models else None

        if openai_service is None:
            raise SystemException("OpenAI service is not available", error_context)

        async with typing_context_manager:
            try:
                stream_manager = StreamManager(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    reply_to_message_id=message.message_id,
                    send_interval=1.8,
                ) if use_streaming else None

                callback = StreamingCallback(stream_manager) if stream_manager else None

                result = await openai_service.answer_question(
                    request_text,
                    system_prompt=system_prompt_override,
                    attachments=attachments_list or None,
                    stream_callback=callback,
                    model=model_to_use,
                    user_id=user_id,
                )
                final_answer_text = result.get("text")

                if stream_manager:
                    had_stream_content = bool((stream_manager.pending_text or "").strip())
                    stream_final_text = stream_manager.pending_text or ""
                    await stream_manager.stop()
                    if stream_manager.message and message.bot:
                        with suppress(Exception):
                            await message.bot.delete_message(
                                message.chat.id, stream_manager.message.message_id
                            )

            except (NetworkException, OpenAIException) as exc:
                ok_flag = False
                request_error_type = type(exc).__name__
                raise
            else:
                ok_flag = True
                if practice_mode_used and practice_excel_fragments:
                    summary_payload = result.get("practice_summary")
                    if not isinstance(summary_payload, dict):
                        summary_payload = {} if summary_payload is None else {"summary_html": str(summary_payload)}
                    existing_fragments = list(summary_payload.get("fragments") or [])
                    combined_fragments = existing_fragments + practice_excel_fragments

                    deduped_fragments: list[Any] = []
                    seen_fragment_keys: set[tuple[Any, ...]] = set()
                    for fragment in combined_fragments:
                        metadata: Mapping[str, Any] | dict[str, Any] = {}
                        match = getattr(fragment, "match", None)
                        if match is not None:
                            metadata = getattr(match, "metadata", {}) or {}
                        if not isinstance(metadata, Mapping):
                            metadata = {}
                        key = (
                            metadata.get("title"),
                            metadata.get("url"),
                            metadata.get("topic"),
                        )
                        if key in seen_fragment_keys:
                            continue
                        seen_fragment_keys.add(key)
                        deduped_fragments.append(fragment)
                        if len(deduped_fragments) >= 10:
                            break

                    summary_payload["fragments"] = deduped_fragments
                    summary_payload.setdefault("summary_html", result.get("text") or "")
                    result["practice_summary"] = summary_payload

        timer.stop()

        if status:
            await status.update_stage(percent=92)
            await status.complete()

        raw_response_text = result.get("text") or stream_final_text or ""
        if raw_response_text:
            clean_html = format_safe_html(raw_response_text)
            chunks = _split_html_for_telegram(clean_html, TELEGRAM_HTML_SAFE_LIMIT)

            if not chunks:
                logger.info("Skipping empty Telegram payload after formatting")
            elif len(chunks) == 1:
                formatted_chunk = _ensure_double_newlines(chunks[0])
                try:
                    await message.answer(
                        formatted_chunk,
                        parse_mode=ParseMode.HTML,
                        reply_markup=_back_to_main_keyboard(),
                    )
                    back_button_sent = True
                except TelegramBadRequest as exc:
                    logger.warning(
                        "Telegram rejected HTML chunk (len=%s), using safe sender fallback: %s",
                        len(formatted_chunk),
                        exc,
                    )
                    await send_html_text(
                        message.bot,
                        chat_id=message.chat.id,
                        raw_text=raw_response_text,
                        reply_to_message_id=message.message_id,
                    )
                except Exception as send_exc:
                    logger.warning("Failed to send HTML chunk, fallback to safe sender: %s", send_exc)
                    await send_html_text(
                        message.bot,
                        chat_id=message.chat.id,
                        raw_text=raw_response_text,
                        reply_to_message_id=message.message_id,
                    )
            else:
                adjusted_chunks = chunks
                seen_lengths: set[int] = set()
                while True:
                    total = len(adjusted_chunks)
                    if total in seen_lengths:
                        break
                    seen_lengths.add(total)
                    prefix_len = len(f"<b>Часть {total}/{total}</b>\n\n")
                    recomputed = _split_html_for_telegram(
                        clean_html,
                        TELEGRAM_HTML_SAFE_LIMIT,
                        reserve=prefix_len,
                    )
                    if not recomputed:
                        adjusted_chunks = []
                        break
                    if len(recomputed) == total:
                        adjusted_chunks = recomputed
                        break
                    adjusted_chunks = recomputed

                if not adjusted_chunks:
                    logger.info("Skipping multipart Telegram payload after prefix adjustment")
                else:
                    total = len(adjusted_chunks)
                    idx = 0
                    try:
                        for idx, chunk in enumerate(adjusted_chunks, start=1):
                            formatted_chunk = _ensure_double_newlines(chunk)
                            prefix = f"<b>Часть {idx}/{total}</b>\n\n"
                            await message.answer(
                                prefix + formatted_chunk,
                                parse_mode=ParseMode.HTML,
                                reply_markup=_back_to_main_keyboard() if idx == total else None,
                            )
                        back_button_sent = True
                    except TelegramBadRequest as exc:
                        logger.warning(
                            "Telegram rejected multipart HTML at part %s/%s: %s. Falling back to safe sender.",
                            idx,
                            total,
                            exc,
                        )
                        await send_html_text(
                            message.bot,
                            chat_id=message.chat.id,
                            raw_text=raw_response_text,
                            reply_to_message_id=message.message_id,
                        )
                    except Exception as send_exc:
                        logger.warning(
                            "Failed to send multipart HTML (part %s/%s). Fallback to safe sender: %s",
                            idx,
                            total,
                            send_exc,
                        )
                        await send_html_text(
                            message.bot,
                            chat_id=message.chat.id,
                            raw_text=raw_response_text,
                            reply_to_message_id=message.message_id,
                        )

        if use_streaming and had_stream_content and stream_manager is not None:
            combined_stream_text = stream_final_text or ""
            if combined_stream_text:
                await stream_manager.finalize(combined_stream_text)

        if result.get("extra_content"):
            with suppress(Exception):
                await send_html_text(
                    message.bot,
                    chat_id=message.chat.id,
                    raw_text=result["extra_content"],
                    reply_to_message_id=message.message_id,
                )

        if result.get("attachments"):
            for attachment in result["attachments"]:
                try:
                    caption = attachment.get("caption")
                    file_path = attachment.get("path")
                    if not file_path:
                        continue
                    await message.answer_document(
                        FSInputFile(file_path),
                        caption=caption,
                        parse_mode=ParseMode.HTML if caption else None,
                    )
                except Exception as exc:
                    logger.warning("Failed to send attachment %s: %s", attachment, exc)

        if result.get("practice_summary"):
            summary_payload = result["practice_summary"]
            try:
                summary_html = summary_payload.get("summary_html", "")
                rag_fragments = summary_payload.get("fragments") or []
                structured_payload = summary_payload.get("structured", {})
                excel_source = final_answer_text or raw_response_text or ""
                practice_excel_path = await asyncio.to_thread(
                    build_practice_excel,
                    summary_html=excel_source,
                    fragments=rag_fragments,
                    structured=structured_payload,
                    file_stub="practice_report",
                )
                await message.answer_document(
                    FSInputFile(str(practice_excel_path)),
                    caption="📊 Отчёт по судебной практике (XLSX)",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Failed to build practice Excel", exc_info=True)
            finally:
                with suppress(Exception):
                    if "practice_excel_path" in locals() and practice_excel_path:
                        practice_excel_path.unlink(missing_ok=True)

        # Показываем контекстную подсказку о возможностях бота
        hint_text: str | None = None
        try:
            from src.core.bot_app.hints import get_contextual_hint

            candidate_contexts: list[str] = []
            if attachments_list:
                candidate_contexts.append("document_uploaded")
            if practice_mode_active:
                candidate_contexts.append("after_search")
            if len(question_text) >= LONG_TEXT_HINT_THRESHOLD:
                candidate_contexts.append("long_text")
            candidate_contexts.append("text_question")

            seen: set[str] = set()
            for ctx_name in candidate_contexts:
                if ctx_name in seen:
                    continue
                seen.add(ctx_name)
                hint_text = await get_contextual_hint(db, user_id, context=ctx_name)
                if hint_text:
                    break

        except Exception as hint_error:
            logger.debug("Failed to send hint: %s", hint_error)

        quota_text = quota_text or result.get("quota_message", "")
        if quota_text:
            with suppress(Exception):
                await message.answer(quota_text, parse_mode=ParseMode.HTML)
        if quota_msg_to_send:
            with suppress(Exception):
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)

        if not back_button_sent:
            with suppress(Exception):
                await message.answer(
                    "Выберите следующее действие:",
                    reply_markup=_back_to_main_keyboard(),
                )
            back_button_sent = True

        if hasattr(user_session, "add_question_stats"):
            with suppress(Exception):
                user_session.add_question_stats(timer.duration)

        if db is not None and hasattr(db, "record_request"):
            with suppress(Exception):
                request_time_ms = int((time.time() - request_start_time) * 1000)
                record_request_fn = get_safe_db_method("record_request", default_return=None)
                if record_request_fn:
                    request_record_id = await record_request_fn(
                        user_id=user_id,
                        request_type="legal_question",
                        tokens_used=0,
                        response_time_ms=request_time_ms,
                        success=True,
                        error_type=None,
                    )
        if request_record_id:
            if final_answer_text and message.from_user:
                with suppress(Exception):
                    await ensure_rating_snapshot(
                        request_record_id,
                        message.from_user,
                        final_answer_text or "",
                    )
            with suppress(Exception):
                await send_rating_request(
                    message,
                    request_record_id,
                    answer_snapshot=final_answer_text or "",
                )

        if hint_text:
            with suppress(Exception):
                await message.answer(hint_text, parse_mode=ParseMode.HTML)

        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)
        return (isinstance(result, dict) and (result.get("text") or "")) or ""

    except Exception as exc:
        if error_handler is not None:
            try:
                custom_exc = await error_handler.handle_exception(exc, error_context)
                user_message = getattr(
                    custom_exc, "user_message", "Произошла системная ошибка. Попробуйте позже."
                )
            except Exception:
                logger.exception("Error handler failed for user %s", user_id)
                user_message = "Произошла системная ошибка. Попробуйте позже."
        else:
            logger.exception("Error processing question for user %s (no error handler)", user_id)
            user_message = "Произошла ошибка. Попробуйте позже."

        if db is not None and hasattr(db, "record_request"):
            with suppress(Exception):
                request_time_ms = (
                    int((time.time() - request_start_time) * 1000)
                    if "request_start_time" in locals()
                    else 0
                )
                err_type = request_error_type if "request_error_type" in locals() else type(exc).__name__
                record_request_fn = get_safe_db_method("record_request", default_return=None)
                if record_request_fn:
                    await record_request_fn(
                        user_id=user_id,
                        request_type="legal_question",
                        tokens_used=0,
                        response_time_ms=request_time_ms,
                        success=False,
                        error_type=str(err_type),
                    )

        with suppress(Exception):
            await message.answer(
                "❌ <b>Ошибка обработки запроса</b>\n\n"
                f"{user_message}\n\n"
                "💡 <b>Рекомендации:</b>\n"
                "• Переформулируйте вопрос\n"
                "• Попробуйте через несколько минут\n"
                "• Обратитесь в поддержку, если проблема повторяется",
                parse_mode=ParseMode.HTML,
            )
        raise
    finally:
        timer.stop()
        await _stop_status_indicator(status, ok=ok_flag)


def register_question_handlers(dp: Dispatcher) -> None:
    dp.message.register(process_question_with_attachments, F.photo | F.document)
    dp.message.register(process_question, F.text & ~F.text.startswith("/"))

