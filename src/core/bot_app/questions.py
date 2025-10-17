from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from html import escape as html_escape
from typing import Any, Sequence

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import FSInputFile, Message

from src.bot.promt import JUDICIAL_PRACTICE_SEARCH_PROMPT
from src.bot.status_manager import ProgressStatus
from src.bot.stream_manager import StreamManager, StreamingCallback
from src.bot.typing_indicator import send_typing_once, typing_action
from src.bot.ui_components import Emoji
from src.core.attachments import QuestionAttachment
from src.core.exceptions import (
    ErrorContext,
    NetworkException,
    OpenAIException,
    SystemException,
    ValidationException,
)
from src.core.safe_telegram import send_html_text
from src.core.bot_app import context as simple_context
from src.core.bot_app.common import get_user_session, get_safe_db_method, ensure_valid_user_id
from src.core.bot_app.feedback import (
    ensure_rating_snapshot,
    handle_pending_feedback,
    send_rating_request,
)
from src.core.garant_api import GarantAPIError
from src.core.validation import InputValidator, ValidationSeverity
from src.core.excel_export import build_practice_excel

__all__ = [
    "ResponseTimer",
    "process_question",
    "process_question_with_attachments",
    "register_question_handlers",
]

QUESTION_ATTACHMENT_MAX_BYTES = 4 * 1024 * 1024  # 4MB per attachment
LONG_TEXT_HINT_THRESHOLD = 700  # heuristic РїРѕСЂРѕРі РґР»СЏ РїРѕРґСЃРєР°Р·РєРё РїСЂРѕ РґР»РёРЅРЅС‹Рµ С‚РµРєСЃС‚С‹

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


async def _collect_question_attachments(message: Message) -> list[QuestionAttachment]:
    bot = message.bot
    if bot is None:
        raise ValueError("РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ РєРѕРЅС‚РµРєСЃС‚ Р±РѕС‚Р° РґР»СЏ Р·Р°РіСЂСѓР·РєРё С„Р°Р№Р»Р°")

    attachments: list[QuestionAttachment] = []

    if message.document:
        document = message.document
        if document.file_size and document.file_size > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError("Р¤Р°Р№Р» СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕР№. РњР°РєСЃРёРјР°Р»СЊРЅС‹Р№ СЂР°Р·РјРµСЂ РІР»РѕР¶РµРЅРёСЏ вЂ” 4 РњР‘.")

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
            raise ValueError("РР·РѕР±СЂР°Р¶РµРЅРёРµ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕРµ. РњР°РєСЃРёРјР°Р»СЊРЅС‹Р№ СЂР°Р·РјРµСЂ РІР»РѕР¶РµРЅРёСЏ вЂ” 4 РњР‘.")

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
        raise ValueError("РќРµ СѓРґР°Р»РѕСЃСЊ РѕРїСЂРµРґРµР»РёС‚СЊ РІР»РѕР¶РµРЅРёРµ. РџРѕРґРґРµСЂР¶РёРІР°СЋС‚СЃСЏ РґРѕРєСѓРјРµРЅС‚С‹ Рё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ.")

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
            f"{Emoji.WARNING} <b>РќРµРєРѕСЂСЂРµРєС‚РЅС‹Р№ РІРѕРїСЂРѕСЃ</b>\n\n{html_escape(errors)}",
            parse_mode=ParseMode.HTML,
        )
        return None

    if validation.warnings:
        warnings = "\n".join(validation.warnings or [])
        await message.answer(
            f"{Emoji.INFO} <b>РџСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ</b>\n\n{html_escape(warnings)}",
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
        f"{Emoji.WARNING} <b>РЎР»РёС€РєРѕРј РјРЅРѕРіРѕ Р·Р°РїСЂРѕСЃРѕРІ</b>\n\n"
        "РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕРІС‚РѕСЂРёС‚СЊ РІРѕРїСЂРѕСЃ С‡СѓС‚СЊ РїРѕР·Р¶Рµ.",
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
            await status.fail("РћС€РёР±РєР° РѕР±СЂР°Р±РѕС‚РєРё Р·Р°РїСЂРѕСЃР°")
    except Exception as exc:  # pragma: no cover - С‚РѕР»СЊРєРѕ Р»РѕРі
        logger.debug("Failed to finalize status indicator: %s", exc)


async def process_question_with_attachments(message: Message) -> None:
    caption = (message.caption or "").strip()
    if not caption:
        warning_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>Р”РѕР±Р°РІСЊС‚Рµ С‚РµРєСЃС‚ РІРѕРїСЂРѕСЃР°</b>",
            "РќР°РїРёС€РёС‚Рµ РєРѕСЂРѕС‚РєРѕРµ РѕРїРёСЃР°РЅРёРµ СЃРёС‚СѓР°С†РёРё РІ РїРѕРґРїРёСЃРё Рє С„Р°Р№Р»Сѓ Рё РѕС‚РїСЂР°РІСЊС‚Рµ СЃРЅРѕРІР°.",
        ])
        await message.answer(warning_msg, parse_mode=ParseMode.HTML)
        return

    try:
        attachments = await _collect_question_attachments(message)
    except ValueError as exc:
        error_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±СЂР°Р±РѕС‚Р°С‚СЊ РІР»РѕР¶РµРЅРёРµ</b>",
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
    """Р“Р»Р°РІРЅС‹Р№ РѕР±СЂР°Р±РѕС‚С‡РёРє СЋСЂРёРґРёС‡РµСЃРєРёС… РІРѕРїСЂРѕСЃРѕРІ."""
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
            f"{Emoji.WARNING} <b>РћС€РёР±РєР° РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂР° РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ</b>\n\nРџРѕРїСЂРѕР±СѓР№С‚Рµ РїРµСЂРµР·Р°РїСѓСЃС‚РёС‚СЊ РґРёР°Р»РѕРі.",
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
    system_prompt_override: str | None = None
    rag_context = ""
    garant_context = ""

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
                garant_results = await garant_client.search_with_snippets(question_text)
                garant_context = garant_client.format_results(garant_results)
            except GarantAPIError as garant_exc:
                logger.warning("Garant API search failed: %s", garant_exc)
            except Exception as garant_exc:  # noqa: BLE001
                logger.error("Unexpected Garant API failure: %s", garant_exc, exc_info=True)
        setattr(user_session, "practice_search_mode", False)

    request_blocks = [question_text]

    if rag_context:
        request_blocks.append("[РљРѕРЅС‚РµРєСЃС‚ СЃСѓРґРµР±РЅРѕР№ РїСЂР°РєС‚РёРєРё]\n" + rag_context)

    if garant_context:
        request_blocks.append(garant_context)

    if attachments_list:
        attachment_lines = []
        for idx, item in enumerate(attachments_list, start=1):
            size_kb = max(1, item.size // 1024)
            attachment_lines.append(f"{idx}. {item.filename} ({item.mime_type}, {size_kb} KB)")
        request_blocks.append("[Р’Р»РѕР¶РµРЅРёСЏ]\n" + "\n".join(attachment_lines))

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
                        f"{Emoji.WARNING} <b>Р›РёРјРёС‚ Р·Р°РїСЂРѕСЃРѕРІ РёСЃС‡РµСЂРїР°РЅ</b>\n\n"
                        f"РўР°СЂРёС„: <b>{plan_name}</b>\n"
                        "РџСЂРѕРґР»РёС‚Рµ РїРѕРґРїРёСЃРєСѓ РёР»Рё РѕР±РЅРѕРІРёС‚Рµ С‚Р°СЂРёС„, С‡С‚РѕР±С‹ РїСЂРѕРґРѕР»Р¶РёС‚СЊ СЂР°Р±РѕС‚Сѓ.",
                        parse_mode=ParseMode.HTML,
                    )
                    return None

                await message.answer(
                    f"{Emoji.WARNING} <b>РљРІРѕС‚Р° Р·Р°РїСЂРѕСЃРѕРІ РёСЃС‡РµСЂРїР°РЅР°</b>\n\n"
                    "РћС„РѕСЂРјРёС‚Рµ РїРѕРґРїРёСЃРєСѓ РєРѕРјР°РЅРґРѕР№ /buy, С‡С‚РѕР±С‹ РїСЂРѕРґРѕР»Р¶РёС‚СЊ.",
                    parse_mode=ParseMode.HTML,
                )
                return None

            quota_text = decision.message or ""
            quota_is_trial = bool(decision.is_trial)

        status = await _start_status_indicator(message)

        models = simple_context.derived().models if callable(simple_context.derived) else None
        model_to_use = (models or {}).get("primary") if models else None

        if openai_service is None:
            raise SystemException("OpenAI service is not available", error_context)

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

        timer.stop()

        if status:
            await status.update_stage(percent=92)
            await status.complete()

        if result.get("mode") == "html":
            await send_html_text(
                message.bot,
                chat_id=message.chat.id,
                html=result.get("text", ""),
                reply_to=message.message_id,
            )
        else:
            text_to_send = result.get("text") or stream_final_text or ""
            if text_to_send:
                with suppress(Exception):
                    await message.answer(text_to_send, parse_mode=ParseMode.HTML)

        if use_streaming and had_stream_content and stream_manager is not None:
            combined_stream_text = stream_final_text or ""
            if combined_stream_text:
                await stream_manager.finalize(combined_stream_text)

        if result.get("extra_content"):
            with suppress(Exception):
                await send_html_text(
                    message.bot,
                    chat_id=message.chat.id,
                    html=result["extra_content"],
                    reply_to=message.message_id,
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
                excel_source = final_answer_text or text_to_send or ""
                practice_excel_path = await asyncio.to_thread(
                    build_practice_excel,
                    summary_html=excel_source,
                    fragments=rag_fragments,
                    structured=structured_payload,
                    file_stub="practice_report",
                )
                await message.answer_document(
                    FSInputFile(str(practice_excel_path)),
                    caption="рџ“Љ РћС‚С‡С‘С‚ РїРѕ СЃСѓРґРµР±РЅРѕР№ РїСЂР°РєС‚РёРєРµ (XLSX)",
                    parse_mode=ParseMode.HTML,
                )
            except Exception as excel_error:  # noqa: BLE001
                logger.warning("Failed to build practice Excel", exc_info=True)
            finally:
                with suppress(Exception):
                    if "practice_excel_path" in locals() and practice_excel_path:
                        practice_excel_path.unlink(missing_ok=True)

        # РџРѕРєР°Р·С‹РІР°РµРј РєРѕРЅС‚РµРєСЃС‚РЅСѓСЋ РїРѕРґСЃРєР°Р·РєСѓ Рѕ РІРѕР·РјРѕР¶РЅРѕСЃС‚СЏС… Р±РѕС‚Р°
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
            hint_text: str | None = None
            for ctx_name in candidate_contexts:
                if ctx_name in seen:
                    continue
                seen.add(ctx_name)
                hint_text = await get_contextual_hint(db, user_id, context=ctx_name)
                if hint_text:
                    break

            if hint_text:
                with suppress(Exception):
                    await message.answer(hint_text, parse_mode=ParseMode.HTML)
        except Exception as hint_error:
            logger.debug("Failed to send hint: %s", hint_error)

        quota_text = quota_text or result.get("quota_message", "")
        if quota_text:
            with suppress(Exception):
                await message.answer(quota_text, parse_mode=ParseMode.HTML)
        if quota_msg_to_send:
            with suppress(Exception):
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)

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

        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)
        return (isinstance(result, dict) and (result.get("text") or "")) or ""

    except Exception as exc:
        if error_handler is not None:
            try:
                custom_exc = await error_handler.handle_exception(exc, error_context)
                user_message = getattr(
                    custom_exc, "user_message", "РџСЂРѕРёР·РѕС€Р»Р° СЃРёСЃС‚РµРјРЅР°СЏ РѕС€РёР±РєР°. РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ."
                )
            except Exception:
                logger.exception("Error handler failed for user %s", user_id)
                user_message = "РџСЂРѕРёР·РѕС€Р»Р° СЃРёСЃС‚РµРјРЅР°СЏ РѕС€РёР±РєР°. РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ."
        else:
            logger.exception("Error processing question for user %s (no error handler)", user_id)
            user_message = "РџСЂРѕРёР·РѕС€Р»Р° РѕС€РёР±РєР°. РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ."

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
                "вќЊ <b>РћС€РёР±РєР° РѕР±СЂР°Р±РѕС‚РєРё Р·Р°РїСЂРѕСЃР°</b>\n\n"
                f"{user_message}\n\n"
                "рџ’Ў <b>Р РµРєРѕРјРµРЅРґР°С†РёРё:</b>\n"
                "вЂў РџРµСЂРµС„РѕСЂРјСѓР»РёСЂСѓР№С‚Рµ РІРѕРїСЂРѕСЃ\n"
                "вЂў РџРѕРїСЂРѕР±СѓР№С‚Рµ С‡РµСЂРµР· РЅРµСЃРєРѕР»СЊРєРѕ РјРёРЅСѓС‚\n"
                "вЂў РћР±СЂР°С‚РёС‚РµСЃСЊ РІ РїРѕРґРґРµСЂР¶РєСѓ, РµСЃР»Рё РїСЂРѕР±Р»РµРјР° РїРѕРІС‚РѕСЂСЏРµС‚СЃСЏ",
                parse_mode=ParseMode.HTML,
            )
        raise
    finally:
        timer.stop()
        await _stop_status_indicator(status, ok=ok_flag)


def register_question_handlers(dp: Dispatcher) -> None:
    dp.message.register(process_question_with_attachments, F.photo | F.document)
    dp.message.register(process_question, F.text & ~F.text.startswith("/"))

