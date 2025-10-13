from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import time
from contextlib import suppress
from html import escape as html_escape
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence, TYPE_CHECKING

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.status_manager import ProgressStatus
from src.bot.typing_indicator import send_typing_once, typing_action
from src.bot.ui_components import Emoji
from src.core.bot_app import context as simple_context
from src.core.bot_app.formatting import _format_progress_extras, _split_plain_text
from src.core.bot_app.menus import cmd_start
from src.core.bot_app.voice import download_voice_to_temp
from src.documents.base import ProcessingError
from src.documents.document_drafter import (
    DocumentDraftingError,
    build_docx_from_markdown,
    format_plan_summary,
    generate_document,
    plan_document,
)

if TYPE_CHECKING:
    from src.core.audio_service import AudioService
    from src.core.openai_service import OpenAIService
    from src.documents.document_manager import DocumentManager

__all__ = [
    "DocumentProcessingStates",
    "DocumentDraftStates",
    "register_document_handlers",
]

logger = logging.getLogger("ai-ivan.simple.documents")

settings = simple_context.settings

GENERIC_INTERNAL_ERROR_HTML = "<i>Произошла внутренняя ошибка. Попробуйте позже.</i>"
GENERIC_INTERNAL_ERROR_TEXT = "Произошла внутренняя ошибка. Попробуйте позже."

_NUMBERED_ANSWER_RE = re.compile(r"^\s*(\d+)[\).:-]\s*(.*)")
_BULLET_ANSWER_RE = re.compile(r"^\s*[-\u2022]\s*(.*)")
_HEADING_PATTERN_RE = re.compile(
    r"^\s*(?![-\u2022])(?!\d+[\).:-])([A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451\u0030-\u0039][^:]{0,80}):\s*(.*)$"
)


class DocumentProcessingStates(StatesGroup):
    waiting_for_document = State()
    processing_document = State()


class DocumentDraftStates(StatesGroup):
    waiting_for_request = State()
    asking_details = State()
    generating = State()


def _get_document_manager() -> DocumentManager | None:
    return simple_context.document_manager


def _get_openai_service() -> OpenAIService | None:
    return simple_context.openai_service


def _get_audio_service() -> AudioService | None:
    return simple_context.audio_service


def _build_ocr_reply_markup(output_format: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{Emoji.BACK} Назад", callback_data="back_to_menu"),
                InlineKeyboardButton(
                    text=f"{Emoji.DOCUMENT} Следующий файл", callback_data=f"ocr_upload_more:{output_format}"
                ),
            ]
        ]
    )


_BASE_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "start": ("Подготавливаем задачу", "⚙️"),
    "downloading": ("Загружаем документ", "📥"),
    "uploaded": ("Файл получен", "📁"),
    "processing": ("Обрабатываем документ", "🧠"),
    "finalizing": ("Форматируем результат", "📄"),
    "completed": ("Готово", "✅"),
    "failed": ("Ошибка обработки", "❌"),
}

_STAGE_LABEL_OVERRIDES: dict[str, dict[str, tuple[str, str]]] = {
    "summarize": {
        "processing": ("Готовим выжимку", "📝"),
        "finalizing": ("Формируем итог", "✨"),
    },
    "analyze_risks": {
        "processing": ("Ищем риски", "⚠️"),
        "pattern_scan": ("Мыслительный анализ", "🔍"),
        "ai_analysis": ("Проверяем ИИ-моделью", "🤖"),
        "compliance_check": ("Проверяем соблюдение норм", "⚖️"),
        "aggregation": ("Подготавливаем отчёт", "📊"),
        "highlighting": ("Выделяем замечания", "📌"),
    },
    "lawsuit_analysis": {
        "processing": ("Анализируем иск", "⚖️"),
        "model_request": ("Сверяем с прецедентами", "📚"),
        "analysis_ready": ("Формируем рекомендации", "📝"),
    },
    "anonymize": {
        "processing": ("Скрываем персональные данные", "🕶️"),
        "finalizing": ("Готовим отчёт об обезличивании", "📋"),
    },
    "translate": {
        "processing": ("Переводим документ", "🌐"),
        "finalizing": ("Собираем перевод", "🗂️"),
    },
    "ocr": {
        "processing": ("Распознаём текст", "🔍"),
        "finalizing": ("Подготавливаем результат", "📄"),
        "ocr_page": ("Распознаём страницы", "📄"),
    },
    "chat": {
        "processing": ("Обрабатываем запрос", "💬"),
        "finalizing": ("Формируем ответ", "✅"),
        "chunking": ("Разбиваем документ", "🧩"),
        "indexing": ("Индексируем содержимое", "🗂️"),
    },
}


_LAWSUIT_STAGE_ORDER = [
    "start",
    "downloading",
    "uploaded",
    "processing",
    "model_request",
    "analysis_ready",
    "finalizing",
    "completed",
]


def _schedule_message_deletion(bot, chat_id: int, message_id: int, delay: float = 5.0) -> None:
    async def _deleter() -> None:
        await asyncio.sleep(delay)
        with suppress(Exception):
            await bot.delete_message(chat_id, message_id)

    asyncio.create_task(_deleter())


def _get_stage_labels(operation: str) -> dict[str, tuple[str, str]]:
    labels = _BASE_STAGE_LABELS.copy()
    labels.update(_STAGE_LABEL_OVERRIDES.get(operation, {}))
    return labels


def _build_completion_payload(op: str, result_obj) -> dict[str, Any]:
    data = getattr(result_obj, "data", None) or {}
    payload: dict[str, Any] = {}
    if op == "analyze_risks":
        pattern = len(data.get("pattern_risks", []) or [])
        ai_risks = len(((data.get("ai_analysis") or {}).get("risks")) or [])
        payload["risks_found"] = pattern + ai_risks
        payload["violations"] = len(((data.get("legal_compliance") or {}).get("violations")) or [])
        payload["overall"] = data.get("overall_risk_level")
    elif op == "summarize":
        summary_struct = ((data.get("summary") or {}).get("structured")) or {}
        payload["words"] = len(((summary_struct.get("summary")) or "").split())
        payload["chunks_total"] = len(summary_struct.get("key_points") or [])
    elif op == "anonymize":
        report = data.get("anonymization_report") or {}
        masked = report.get("processed_items")
        if masked is None:
            stats = report.get("statistics") or {}
            masked = sum(int(v) for v in stats.values()) if stats else 0
        payload["masked"] = int(masked or 0)
    elif op == "translate":
        meta = data.get("translation_metadata") or {}
        payload["language_pair"] = meta.get("language_pair")
        payload["chunks_total"] = meta.get("chunks_processed")
        payload["mode"] = meta.get("mode")
    elif op == "ocr":
        payload["confidence"] = data.get("confidence_score")
        processing = data.get("processing_info") or {}
        payload["pages_total"] = processing.get("pages_processed") or len(data.get("pages", []) or [])
        payload["mode"] = processing.get("file_type")
    elif op == "chat":
        info = data.get("document_info") or {}
        payload["chunks_total"] = info.get("chunks_count")
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def _make_progress_updater(
    message: Message,
    status_msg: Message,
    *,
    file_name: str,
    operation_name: str,
    file_size_kb: int,
    stage_labels: dict[str, tuple[str, str]],
) -> tuple[Callable[[dict[str, Any]], Awaitable[None]], dict[str, Any]]:
    progress_state: dict[str, Any] = {"percent": 0, "stage": "start", "started_at": time.monotonic()}

    async def send_progress(update: dict[str, Any]) -> None:
        nonlocal progress_state, status_msg
        if not status_msg or not status_msg.message_id:
            return
        stage = str(update.get("stage") or progress_state["stage"] or "processing")
        percent_val = update.get("percent")
        if percent_val is None:
            percent = progress_state["percent"]
        else:
            percent = max(0, min(100, int(round(float(percent_val)))))
        if percent < progress_state["percent"] and stage != "failed":
            percent = progress_state["percent"]

        progress_state["stage"] = stage
        progress_state["percent"] = percent

        label, icon = stage_labels.get(stage, stage_labels.get("processing", ("Обработка", "⚙️")))
        extras_line = _format_progress_extras(update)
        elapsed = time.monotonic() - progress_state["started_at"]
        elapsed_text = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        total_segments = 10
        progress_ratio = max(0.0, min(1.0, float(percent) / 100.0))
        filled_segments = int(round(progress_ratio * total_segments))
        filled_segments = max(0, min(total_segments, filled_segments))
        progress_bar = "█" * filled_segments + "░" * (total_segments - filled_segments)

        header = f"{icon} <b>{label}</b>"
        border = "┌" + "─" * 32
        body_lines = [
            f"│ Прогресс: {progress_bar} {percent}%",
            f"│ 📄 Файл: <b>{html_escape(file_name)}</b>",
            f"│ 🛠️ Операция: {html_escape(operation_name)}",
            f"│ 📦 Размер: {file_size_kb} КБ",
            f"│ ⏱️ Время: {elapsed_text}",
        ]
        if extras_line:
            body_lines.append(f"│ {extras_line}")
        footer = "└" + "─" * 32

        lines = [header, border, *body_lines, footer]

        try:
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
                text="\n".join(lines),
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.debug("Progress edit failed: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.debug("Unexpected progress update error: %s", exc)

    return send_progress, progress_state


async def handle_doc_draft_start(callback: CallbackQuery, state: FSMContext) -> None:
    """Запуск режима подготовки нового документа."""
    if not callback.from_user:
        await callback.answer("❌ Не удалось определить пользователя")
        return

    try:
        await send_typing_once(callback.bot, callback.message.chat.id, "typing")

        await state.clear()
        await state.set_state(DocumentDraftStates.waiting_for_request)

        intro_text = (
            f"✨ <b>Создание юридического документа</b>\n"
            f"<code>{'━' * 35}</code>\n\n"
            f"📋 <b>Как это работает:</b>\n\n"
            f"<b>1️⃣ Опишите задачу</b>\n"
            f"   └ Расскажите, какой документ нужен\n\n"
            f"<b>2️⃣ Отвечайте на вопросы</b>\n"
            f"   └ Я уточню детали для точности\n\n"
            f"<b>3️⃣ Получите DOCX</b>\n"
            f"   └ Готовый документ за минуту\n\n"
            f"<code>{'━' * 35}</code>\n\n"
            f"💡 <i>Совет: Опишите ситуацию максимально подробно — "
            f"это поможет создать точный документ с первого раза</i>\n\n"
            f"<b>Примеры запросов:</b>\n"
            f"• Исковое заявление о взыскании долга\n"
            f"• Договор оказания юридических услуг\n"
            f"• Жалоба в Роспотребнадзор\n\n"
            f"👇 <b>Опишите, что нужно создать:</b>"
        )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=f"{Emoji.BACK} Отмена", callback_data="doc_draft_cancel")]]
        )
        await callback.message.answer(intro_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось запустить конструктор документа: %s", exc, exc_info=True)
        await callback.answer("Не удалось запустить конструктор", show_alert=True)


async def handle_doc_draft_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    """Отмена процесса создания документа."""
    await state.clear()
    with suppress(Exception):
        await callback.message.answer(
            f"🚫 <b>Создание документа отменено</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"💡 Вы можете начать заново в любой момент",
            parse_mode=ParseMode.HTML,
        )
    with suppress(Exception):
        await callback.answer("Отменено")


async def handle_doc_draft_request(
    message: Message,
    state: FSMContext,
    *,
    text_override: str | None = None,
) -> None:
    """Обработка исходного запроса юриста."""
    source_text = text_override if text_override is not None else message.text
    request_text = (source_text or "").strip()
    if not request_text:
        await message.answer(
            f"⚠️ <b>Пустой запрос</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"📝 Пожалуйста, опишите какой документ нужен\n\n"
            f"<i>Например:</i>\n"
            f"• Договор аренды квартиры\n"
            f"• Исковое заявление о возврате товара\n"
            f"• Претензия в управляющую компанию",
            parse_mode=ParseMode.HTML,
        )
        return

    openai_service = _get_openai_service()
    if openai_service is None:
        await message.answer(
            f"❌ <b>Сервис недоступен</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"⚠️ Генерация документов временно недоступна\n"
            f"🔄 Попробуйте позже или обратитесь к администратору",
            parse_mode=ParseMode.HTML,
        )
        await state.clear()
        return

    await send_typing_once(message.bot, message.chat.id, "typing")

    progress = ProgressStatus(
        message.bot,
        message.chat.id,
        steps=[
            {"label": "🔍 Определяю тип документа"},
            {"label": "📝 Формирую план вопросов"},
            {"label": "✨ Подготавливаю структуру"},
        ],
        show_context_toggle=False,
        show_checklist=True,
        auto_advance_stages=True,
        percent_thresholds=[0, 50, 90],
    )

    await progress.start(auto_cycle=True, interval=1.5)

    try:
        plan = await plan_document(openai_service, request_text)
        await progress.complete()
        await asyncio.sleep(0.3)
    except DocumentDraftingError as err:
        await progress.fail(note=str(err))
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка планирования документа: %s", exc, exc_info=True)
        await progress.fail(note="Попробуйте еще раз")
        await state.clear()
        return
    else:
        with suppress(Exception):
            if progress.message_id:
                await message.bot.delete_message(message.chat.id, progress.message_id)

    await state.update_data(
        draft_request=request_text,
        draft_plan={"title": plan.title, "questions": plan.questions, "notes": plan.notes},
        draft_answers=[],
        current_question_index=0,
    )

    summary = format_plan_summary(plan)
    for chunk in _split_plain_text(summary):
        await message.answer(chunk, parse_mode=ParseMode.HTML)

    if plan.questions:
        await state.set_state(DocumentDraftStates.asking_details)
        await _send_questions_prompt(
            message,
            plan.questions,
            title="Вопросы для подготовки документа",
        )
    else:
        await state.set_state(DocumentDraftStates.generating)
        await message.answer(
            f"✅ <b>Информации достаточно!</b>\n"
            f"<code>{'▰' * 20}</code>\n\n"
            f"🚀 Приступаю к формированию документа\n"
            f"⏱ Это займет около минуты",
            parse_mode=ParseMode.HTML,
        )
        await _finalize_draft(message, state)


async def handle_doc_draft_answer(
    message: Message,
    state: FSMContext,
    *,
    text_override: str | None = None,
) -> None:
    """Обработка ответов юриста на уточняющие вопросы."""
    await send_typing_once(message.bot, message.chat.id, "typing")

    data = await state.get_data()
    plan = data.get("draft_plan") or {}
    questions = plan.get("questions") or []
    index = data.get("current_question_index", 0)

    if index >= len(questions):
        await message.answer(
            f"✅ <b>Ответы получены</b>\n"
            f"<code>{'▰' * 20}</code>\n\n"
            f"🚀 Приступаю к формированию документа",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(DocumentDraftStates.generating)
        await _finalize_draft(message, state)
        return

    source_text = text_override if text_override is not None else message.text
    answer_text = (source_text or "").strip()
    if not answer_text:
        await message.answer(
            f"⚠️ <b>Пустой ответ</b>\n\n"
            f"📝 Пожалуйста, введите ваш ответ на вопрос",
            parse_mode=ParseMode.HTML,
        )
        return

    answers = data.get("draft_answers") or []
    remaining_questions = questions[index:]
    question_headings = [str(q.get("text", "") or "") for q in remaining_questions]
    bulk_answers = _extract_answer_chunks(
        answer_text,
        expected_count=len(remaining_questions),
        question_headings=question_headings,
    )

    if bulk_answers:
        used_count = 0
        for offset, chunk in enumerate(bulk_answers):
            if offset >= len(remaining_questions):
                break
            question = remaining_questions[offset]
            answers.append({"question": question.get("text", ""), "answer": chunk})
            used_count += 1

        if used_count > 0:
            if used_count < len(bulk_answers):
                extra = "\n".join(bulk_answers[used_count:]).strip()
                if extra and answers:
                    answers[-1]["answer"] = f"{answers[-1]['answer']}\n{extra}"
            index += used_count

            await state.update_data(draft_answers=answers, current_question_index=index)
            if index < len(questions):
                missing_numbers = ", ".join(str(i) for i in range(index + 1, len(questions) + 1))
                await message.answer(
                    f"⚠️ <b>Неполные ответы</b>\n"
                    f"<code>{'─' * 30}</code>\n\n"
                    f"✅ Получено ответов: <b>{index}</b>\n"
                    f"❌ Осталось вопросов: <b>{len(questions) - index}</b>\n"
                    f"📝 Номера вопросов: {missing_numbers}\n\n"
                    f"<b>Как дополнить:</b>\n"
                    f"• Отправьте недостающие ответы одним сообщением\n"
                    f"• Отделяйте пустой строкой или нумеруйте",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await state.set_state(DocumentDraftStates.generating)
                await _finalize_draft(message, state)
            return

    if index < len(questions):
        current_question = questions[index]
        answers.append({"question": current_question.get("text", ""), "answer": answer_text})
        index += 1

        await state.update_data(draft_answers=answers, current_question_index=index)

        if index < len(questions):
            next_question = questions[index]
            next_text = html_escape(next_question.get("text", ""))
            purpose = next_question.get("purpose")

            lines = [
                f"{Emoji.SUCCESS} <b>Ответ принят</b>",
                f"<code>{'▰' * 20}</code>",
                "",
                f"{Emoji.QUESTION} <b>Вопрос {index + 1} из {len(questions)}</b>",
                next_text,
            ]
            if purpose:
                lines.append(f"<i>💡 {html_escape(str(purpose))}</i>")
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
        else:
            await state.set_state(DocumentDraftStates.generating)
            await _finalize_draft(message, state)
        return

    await message.answer(
        f"{Emoji.WARNING} Не удалось обработать ответ. Попробуйте ещё раз.",
        parse_mode=ParseMode.HTML,
    )


async def _extract_doc_voice_text(message: Message) -> str | None:
    """Распознать голосовое сообщение в сценарии составления документа."""
    if not message.voice:
        return None

    audio_service = _get_audio_service()
    if audio_service is None:
        await message.answer(f"{Emoji.WARNING} Голосовой режим недоступен, отправьте текстовое сообщение.")
        return None

    try:
        voice_enabled = settings().voice_mode_enabled
    except RuntimeError:
        voice_enabled = settings().voice_mode_enabled

    if not voice_enabled:
        await message.answer(f"{Emoji.WARNING} Голосовой режим сейчас выключен. Пришлите ответ текстом.")
        return None

    if not message.bot:
        await message.answer(f"{Emoji.WARNING} Не удалось получить доступ к боту. Ответьте текстом.")
        return None

    temp_voice_path: Path | None = None
    try:
        await audio_service.ensure_short_enough(message.voice.duration)

        async with typing_action(message.bot, message.chat.id, "record_voice"):
            temp_voice_path = await download_voice_to_temp(message)
            transcript = await audio_service.transcribe(temp_voice_path)
    except ValueError as duration_error:
        logger.warning("Document draft voice input too long: %s", duration_error)
        await message.answer(
            f"{Emoji.WARNING} Голосовое сообщение слишком длинное. Максимальная длительность — {audio_service.max_duration_seconds} секунд."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to transcribe voice for document draft: %s", exc)
        await message.answer(
            f"{Emoji.ERROR} Не получилось распознать голос. Отправьте ответ текстом, пожалуйста."
        )
        return None
    finally:
        with suppress(Exception):
            if temp_voice_path:
                temp_voice_path.unlink()

    preview = html_escape(transcript[:500])
    if len(transcript) > 500:
        preview += "…"
    await message.answer(
        f"{Emoji.MICROPHONE} Распознанный текст:\n<i>{preview}</i>",
        parse_mode=ParseMode.HTML,
    )
    return transcript


async def handle_doc_draft_request_voice(message: Message, state: FSMContext) -> None:
    """Обработать голосовой запрос на составление документа."""
    transcript = await _extract_doc_voice_text(message)
    if transcript is None:
        return
    await handle_doc_draft_request(message, state, text_override=transcript)


async def handle_doc_draft_answer_voice(message: Message, state: FSMContext) -> None:
    """Обработать голосовой ответ в сценарии уточняющих вопросов."""
    transcript = await _extract_doc_voice_text(message)
    if transcript is None:
        return
    await handle_doc_draft_answer(message, state, text_override=transcript)


def _extract_answer_chunks(
    answer_text: str,
    *,
    expected_count: int | None = None,
    question_headings: Sequence[str] | None = None,
) -> list[str] | None:
    """Разбить объединённый ответ на отдельные части."""
    text = (answer_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2007", " ")
        .replace("\u2060", "")
        .replace("\ufeff", "")
    )
    text = text.strip()
    if not text:
        return None

    lines = text.split("\n")
    numbered_pattern = _NUMBERED_ANSWER_RE
    answers: list[str] = []
    current: list[str] | None = None
    has_numbers = False

    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            has_numbers = True
            if current is not None:
                chunk = "\n".join(current).strip()
                if chunk:
                    answers.append(chunk)
            current = [match.group(2)]
        else:
            if current is not None:
                current.append(line)
    if has_numbers:
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                answers.append(chunk)
        if len(answers) > 1:
            return answers

    bullet_pattern = _BULLET_ANSWER_RE
    first_nonempty = next((line for line in lines if line.strip()), "")
    if bullet_pattern.match(first_nonempty):
        answers = []
        current = None
        for line in lines:
            match = bullet_pattern.match(line)
            if match:
                if current:
                    chunk = "\n".join(current).strip()
                    if chunk:
                        answers.append(chunk)
                current = [match.group(1)]
            else:
                if current:
                    current.append(line)
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                answers.append(chunk)
        if len(answers) > 1:
            return answers

    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?:\n[ \t\u00A0\u2007\u202F\u2060]*){2,}", text)
        if chunk.strip()
    ]
    if len(chunks) > 1:
        if question_headings:
            normalized = [
                re.sub(r"\s+", " ", (heading or "")).strip().lower()
                for heading in question_headings
            ]

            def _normalize_line(line: str) -> str:
                return re.sub(r"\s+", " ", (line or "")).strip().lower()

            ordered: list[str] = []
            idx = 0
            for chunk in chunks:
                first_line = chunk.split("\n", 1)[0]
                first_norm = _normalize_line(first_line)
                if idx < len(normalized) and normalized[idx] and first_norm == normalized[idx]:
                    ordered.append(chunk)
                    idx += 1
                else:
                    ordered.append(chunk)
            return ordered
        return chunks

    if expected_count is not None and expected_count < 2:
        return None

    heading_pattern = _HEADING_PATTERN_RE
    candidates: list[str] = []
    current_chunk: list[str] = []
    heading_boundaries = 0

    for line in lines:
        match = heading_pattern.match(line)
        is_heading = False
        if match:
            heading_text = match.group(1).strip()
            if (
                heading_text
                and len(heading_text) <= 80
                and len(heading_text.split()) <= 8
                and not re.search(r"[.!?]", heading_text)
            ):
                is_heading = True

        if is_heading and current_chunk:
            chunk = "\n".join(current_chunk).strip()
            if chunk:
                candidates.append(chunk)
            current_chunk = [line]
            heading_boundaries += 1
        else:
            current_chunk.append(line)
    if current_chunk:
        chunk = "\n".join(current_chunk).strip()
        if chunk:
            candidates.append(chunk)

    if heading_boundaries >= 1 and len(candidates) > 1:
        return candidates

    return None


async def _send_questions_prompt(
    message: Message,
    questions: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    if not questions:
        return

    await send_typing_once(message.bot, message.chat.id, "typing")

    question_blocks: list[str] = []
    for idx, question in enumerate(questions, 1):
        text = html_escape(question.get("text", ""))
        purpose = question.get("purpose")

        block_lines = [f"<b>{idx}. {text}</b>"]
        if purpose:
            block_lines.append(f"<i>   💡 {html_escape(purpose)}</i>")
        question_blocks.append("\n".join(block_lines))

    if not question_blocks:
        return

    max_len = 3500

    def _start_chunk(suffix: str = "") -> tuple[list[str], int, int]:
        header_lines = [
            f"📋 <b>{title}{suffix}</b>",
            f"<code>{'─' * 35}</code>",
            "",
        ]
        length = 0
        count = 0
        for line in header_lines:
            if count:
                length += 1
            length += len(line)
            count += 1
        return header_lines, length, count

    def _append_lines(
        buffer: list[str], current_len: int, line_count: int, new_lines: Sequence[str]
    ) -> tuple[int, int]:
        for line in new_lines:
            if line_count:
                current_len += 1
            buffer.append(line)
            current_len += len(line)
            line_count += 1
        return current_len, line_count

    def _estimate_len(current_len: int, line_count: int, new_lines: Sequence[str]) -> int:
        length = current_len
        count = line_count
        for line in new_lines:
            if count:
                length += 1
            length += len(line)
            count += 1
        return length

    chunk_lines, chunk_len, line_count = _start_chunk()
    base_line_count = line_count

    for block in question_blocks:
        while True:
            candidate_len = _estimate_len(chunk_len, line_count, (block, ""))
            if candidate_len <= max_len:
                chunk_len, line_count = _append_lines(chunk_lines, chunk_len, line_count, (block, ""))
                break

            if line_count > base_line_count:
                await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)
                chunk_lines, chunk_len, line_count = _start_chunk(" (продолжение)")
                base_line_count = line_count
                continue

            await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)
            chunk_lines, chunk_len, line_count = _start_chunk(" (продолжение)")
            base_line_count = line_count
            chunk_len, line_count = _append_lines(chunk_lines, chunk_len, line_count, (block, ""))
            break

    if line_count > base_line_count:
        await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)


_TITLE_SANITIZE_RE = re.compile(r"[\\/:*?\"<>|\r\n]+")
_TITLE_WHITESPACE_RE = re.compile(r"\s+")

_CAPTION_MAX_LENGTH = 1024
_SUMMARY_PREVIEW_MAX_ITEMS = 3
_SUMMARY_PREVIEW_ITEM_MAX_LEN = 500


def _prepare_document_titles(raw_title: str | None) -> tuple[str, str, str]:
    base = (raw_title or "").strip()
    if not base:
        base = "Документ"
    if base.endswith(")") and "(" in base:
        simplified = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
        if simplified:
            base = simplified
    display_title = _TITLE_WHITESPACE_RE.sub(" ", base).strip()
    if not display_title:
        display_title = "Документ"

    file_stub = _TITLE_SANITIZE_RE.sub("_", display_title).strip("._ ")
    if not file_stub:
        file_stub = "Документ"
    max_len = 80
    if len(file_stub) > max_len:
        file_stub = file_stub[:max_len].rstrip("._ ")
        if not file_stub:
            file_stub = "Документ"
    filename = f"{file_stub}.docx"
    return display_title, f"{Emoji.DOCUMENT} {display_title}", filename


async def _finalize_draft(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    request_text = data.get("draft_request", "")
    plan = data.get("draft_plan") or {}
    answers = data.get("draft_answers") or []
    title = plan.get("title", "Документ")

    openai_service = _get_openai_service()
    if openai_service is None:
        await message.answer(f"{Emoji.ERROR} Сервис генерации документов временно недоступен. Попробуйте позже.")
        await state.clear()
        return

    progress: ProgressStatus | None = None
    try:
        progress = ProgressStatus(
            message.bot,
            message.chat.id,
            steps=[
                {"label": "Готовим черновик"},
                {"label": "Проверяем структуру"},
                {"label": "Формируем DOCX"},
                {"label": "Отправляем файл"},
            ],
            show_context_toggle=False,
            show_checklist=True,
            auto_advance_stages=True,
            min_edit_interval=0.5,
            percent_thresholds=[0, 55, 80, 95],
        )
        await progress.start(auto_cycle=True, interval=1.4)
        await progress.update_stage(percent=5, step=1)
    except Exception as progress_err:  # pragma: no cover
        logger.debug("Failed to start document drafting progress: %s", progress_err)
        progress = None

    try:
        async with typing_action(message.bot, message.chat.id, "upload_document"):
            result = await generate_document(openai_service, request_text, title, answers)
    except DocumentDraftingError as err:
        if progress:
            await progress.fail(note=str(err))
        await message.answer(f"{Emoji.ERROR} Не удалось подготовить документ: {err}")
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка генерации документа: %s", exc, exc_info=True)
        if progress:
            await progress.fail(note="Сбой при генерации документа")
        await message.answer(f"{Emoji.ERROR} Произошла ошибка при генерации документа")
        await state.clear()
        return

    if progress:
        await progress.update_stage(percent=65, step=2)

    if result.status != "ok":
        if progress:
            note = "Нужны дополнительные уточнения" if result.follow_up_questions else "Не удалось завершить документ"
            await progress.fail(note=note)
        if result.follow_up_questions:
            extra_questions = [
                {"id": f"f{i+1}", "text": item, "purpose": "Дополнительное уточнение"}
                for i, item in enumerate(result.follow_up_questions)
            ]
            await state.update_data(
                draft_plan={
                    "title": result.title or title,
                    "questions": extra_questions,
                    "notes": plan.get("notes", []),
                },
                current_question_index=0,
                draft_answers=answers,
            )
            await state.set_state(DocumentDraftStates.asking_details)
            await message.answer(f"{Emoji.WARNING} Нужно несколько уточнений, чтобы завершить документ.")
            await _send_questions_prompt(
                message,
                extra_questions,
                title="Дополнительные вопросы",
            )
            return

        issues_text = "\n".join(result.issues) or "Модель не смогла подготовить документ."
        await message.answer(f"{Emoji.WARNING} Документ не готов. Причина:\n{issues_text}")
        await state.clear()
        return

    summary_sections: list[str] = []

    def _format_summary_block(items: Sequence[str]) -> str:
        blocks: list[str] = []
        for raw in items:
            text = str(raw or "").strip()
            if not text:
                continue

            heading: str | None = None
            details: str | None = None
            if ":" in text:
                heading_candidate, details_candidate = text.split(":", 1)
                heading_candidate = heading_candidate.strip()
                details_candidate = details_candidate.strip()
                if heading_candidate:
                    heading = html_escape(heading_candidate)
                if details_candidate:
                    parts = [part.strip() for part in details_candidate.split(";") if part.strip()]
                    if len(parts) > 1:
                        details = "<br>".join(f"— {html_escape(part)}" for part in parts)
                    else:
                        details = html_escape(details_candidate)
            if heading is None and details is None:
                blocks.append(f"• {html_escape(text)}")
            elif heading and details:
                blocks.append(f"• <b>{heading}</b><br>{details}")
            elif heading:
                blocks.append(f"• <b>{heading}</b>")
            elif details:
                blocks.append(f"• {details}")
        return "\n\n".join(blocks)

    validated_items = result.validated or []
    issues_items = result.issues or []

    if validated_items:
        validated_block = _format_summary_block(validated_items)
        if validated_block:
            summary_sections.append(f"{Emoji.SUCCESS} <b>Проверено</b>\n{validated_block}")

    if issues_items:
        issues_block = _format_summary_block(issues_items)
        if issues_block:
            summary_sections.append(f"{Emoji.WARNING} <b>На что обратить внимание</b>\n{issues_block}")

    summary_block = "\n\n".join(summary_sections) if summary_sections else ""

    def _truncate_item_text(raw_text: str) -> str:
        text = re.sub(r"\s+", " ", raw_text).strip()
        if len(text) <= _SUMMARY_PREVIEW_ITEM_MAX_LEN:
            return text
        snippet = text[: _SUMMARY_PREVIEW_ITEM_MAX_LEN].rstrip()
        snippet = re.sub(r"\s+\S*$", "", snippet)
        if not snippet:
            snippet = text[: _SUMMARY_PREVIEW_ITEM_MAX_LEN]
        return snippet.rstrip() + "…"

    def _prepare_preview_items(items: Sequence[str]) -> list[str]:
        cleaned = [str(item or "").strip() for item in items if str(item or "").strip()]
        preview: list[str] = []
        for raw_text in cleaned[: _SUMMARY_PREVIEW_MAX_ITEMS]:
            preview.append(_truncate_item_text(raw_text))
        if len(cleaned) > _SUMMARY_PREVIEW_MAX_ITEMS:
            preview.append("…")
        return preview

    if progress:
        await progress.update_stage(percent=85, step=3)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        await asyncio.to_thread(build_docx_from_markdown, result.markdown, str(tmp_path))
        display_title, caption, filename = _prepare_document_titles(result.title or title)

        header_parts = [
            f"{Emoji.DOCUMENT} <b>{display_title}</b>",
            f"<code>{'─' * 30}</code>",
        ]
        footer_parts = [
            f"{Emoji.MAGIC} Анализ успешно произведён!",
            "📎 Формат: DOCX",
            f"<i>{Emoji.IDEA} Проверьте содержимое и при необходимости внесите правки</i>",
        ]

        caption_parts = header_parts.copy()
        if summary_block:
            caption_parts.append(summary_block)
        caption_parts.extend(footer_parts)
        final_caption = "\n\n".join(caption_parts)

        if len(final_caption) > _CAPTION_MAX_LENGTH and summary_block:
            preview_sections: list[str] = []
            validated_preview = _prepare_preview_items(validated_items)
            if validated_preview:
                preview_sections.append(
                    f"{Emoji.SUCCESS} <b>Проверено</b>\n{_format_summary_block(validated_preview)}"
                )
            issues_preview = _prepare_preview_items(issues_items)
            if issues_preview:
                preview_sections.append(
                    f"{Emoji.WARNING} <b>На что обратить внимание</b>\n{_format_summary_block(issues_preview)}"
                )
            if preview_sections:
                caption_parts = header_parts + ["\n\n".join(preview_sections)] + footer_parts
            else:
                caption_parts = header_parts + footer_parts
            final_caption = "\n\n".join(caption_parts)
            if len(final_caption) > _CAPTION_MAX_LENGTH:
                final_caption = "\n\n".join(header_parts + footer_parts)

        if progress:
            await progress.update_stage(percent=95, step=4)

        await message.answer_document(
            FSInputFile(str(tmp_path), filename=filename),
            caption=final_caption,
            parse_mode=ParseMode.HTML,
        )
        if progress:
            await progress.complete(note="Документ готов")
            await asyncio.sleep(0.3)
            with suppress(Exception):
                if progress.message_id:
                    await message.bot.delete_message(message.chat.id, progress.message_id)
    except DocumentDraftingError as err:
        if progress:
            await progress.fail(note=str(err))
        await message.answer(
            f"❌ <b>Ошибка формирования DOCX</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"⚠️ {err}",
            parse_mode=ParseMode.HTML,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    await state.clear()


async def handle_document_processing(callback: CallbackQuery) -> None:
    """Обработка кнопки работы с документами."""
    document_manager = _get_document_manager()
    if document_manager is None:
        await callback.answer("Сервис обработки документов недоступен", show_alert=True)
        return

    try:
        operations = document_manager.get_supported_operations()

        buttons = [
            [
                InlineKeyboardButton(
                    text="⚖️ Анализ искового заявления",
                    callback_data="doc_operation_lawsuit_analysis",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{Emoji.MAGIC} Создание юридического документа",
                    callback_data="doc_draft_start",
                )
            ],
        ]

        secondary_buttons = []
        for op_key, op_info in operations.items():
            if op_key in {"translate", "chat", "lawsuit_analysis"}:
                continue
            emoji = op_info.get("emoji", "📄")
            name = op_info.get("name", op_key)
            secondary_buttons.append(
                InlineKeyboardButton(text=f"{emoji} {name}", callback_data=f"doc_operation_{op_key}")
            )

        for i in range(0, len(secondary_buttons), 2):
            buttons.append(secondary_buttons[i:i + 2])

        buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_menu")])

        message_text = (
            "🗂️ <b>Работа с документами</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🤖 <i>Автоматическая обработка и анализ\n"
            "   ваших документов с помощью ИИ</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Доступные операции:</b>\n\n"
            "📄 <b>Краткая выжимка</b>\n"
            "   └ Превращает объёмные документы\n"
            "      в короткие выжимки\n\n"
            "⚠️ <b>Риск-анализ</b>\n"
            "   └ Находит опасные формулировки\n"
            "      и проблемные места в договорах\n\n"
            "⚖️ <b>Анализ искового заявления</b>\n"
            "   └ Оценивает правовую позицию,\n"
            "      риски и рекомендации\n\n"
            "🔒 <b>Обезличивание</b>\n"
            "   └ Скрывает персональные данные\n"
            "      и конфиденциальные сведения\n\n"
            "🔍 <b>Распознавание текста</b>\n"
            "   └ Извлекает текст со сканов\n"
            "      и фотографий\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "👇 Выберите нужную операцию:"
        )

        await callback.message.answer(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )
        await callback.answer()

    except Exception as exc:  # noqa: BLE001
        await callback.answer(f"Ошибка: {exc}")
        logger.error("Ошибка в handle_document_processing: %s", exc, exc_info=True)


async def handle_document_operation(callback: CallbackQuery, state: FSMContext) -> None:
    """Обработка выбора операции с документом."""
    document_manager = _get_document_manager()
    if document_manager is None:
        await callback.answer("Сервис обработки документов недоступен", show_alert=True)
        return

    try:
        await send_typing_once(callback.bot, callback.message.chat.id, "typing")

        data = callback.data or ""
        if not data.startswith("doc_operation_"):
            await callback.answer("Операция не найдена", show_alert=True)
            return

        operation = data.replace("doc_operation_", "", 1)
        await state.clear()
        await state.update_data(document_operation=operation, operation_options={})

        await state.set_state(DocumentProcessingStates.waiting_for_document)

        operation_info = document_manager.get_operation_info(operation) or {}
        description = operation_info.get("description") or ""
        formats = ", ".join(operation_info.get("supported_formats") or ["PDF", "DOCX", "TXT"])

        detailed_descriptions = {
            "summarize": (
                "📝 <b>Краткое содержание</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Что делает:</b>\n\n"
                "📚 <b>Выжимка</b>\n"
                "   └ Ключевые выводы и факты\n\n"
                "🧭 <b>Структура</b>\n"
                "   └ Логичная последовательность\n\n"
                "💡 <b>Советы</b>\n"
                "   └ На что обратить внимание\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Идеально для:</b>\n"
                "   • Больших документов\n"
                "   • Решений судов\n"
                "   • Консультаций и заключений"
            ),
            "analyze_risks": (
                "⚠️ <b>Risk-анализ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🛡️ <b>Что проверяем:</b>\n\n"
                "📜 <b>Критичные условия</b>\n"
                "   └ Нестандартные формулировки\n\n"
                "⚖️ <b>Правовые риски</b>\n"
                "   └ Нарушения законодательства\n\n"
                "💰 <b>Финансовые риски</b>\n"
                "   └ Штрафы и санкции\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Результат:</b>\n"
                "   ✓ Оценка риска\n"
                "   ✓ Перечень нарушений\n"
                "   ✓ Рекомендации по исправлению"
            ),
            "lawsuit_analysis": (
                "⚖️ <b>Анализ искового заявления</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🧠 <b>Что получаете:</b>\n\n"
                "📌 <b>Аргументация</b>\n"
                "   └ Сильные и слабые стороны\n\n"
                "📚 <b>Практика</b>\n"
                "   └ Поиск релевантных решений\n\n"
                "🛠️ <b>Рекомендации</b>\n"
                "   └ Что улучшить в позиции\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Идеально для:</b>\n"
                "   • Подготовки к суду\n"
                "   • Анализа претензий"
            ),
            "chat": (
                "💬 <b>Чат с документом</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🤖 <b>Как работает:</b>\n"
                "   • Отвечает на вопросы по содержанию\n"
                "   • Помогает быстро ориентироваться\n"
                "   • Уточняет детали из документа"
            ),
            "anonymize": (
                "🕶️ <b>Обезличивание</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Что делает:</b>\n\n"
                "🧾 <b>Поиск данных</b>\n"
                "   └ Персональные и идентификационные\n\n"
                "🔒 <b>Замена</b>\n"
                "   └ Удаляет или заменяет данные\n\n"
                "📋 <b>Отчёт</b>\n"
                "   └ Какие данные скрыты"
            ),
            "translate": (
                "🌍 <b>Перевод документов</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Функции:</b>\n\n"
                "📄 <b>Перевод текста</b>\n"
                "   └ С сохранением структуры\n\n"
                "🧾 <b>Терминология</b>\n"
                "   └ Юридические и технические термины\n\n"
                "📐 <b>Форматирование</b>\n"
                "   └ Сохраняем исходную разметку"
            ),
            "ocr": (
                "🔍 <b>Распознавание текста</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🖼️ <b>Что умеет:</b>\n\n"
                "📷 <b>Извлечение текста</b>\n"
                "   └ Со сканов и фото\n\n"
                "✍️ <b>Типы текста</b>\n"
                "   └ Рукописный и печатный\n\n"
                "🔄 <b>Форматы</b>\n"
                "   └ DOCX, TXT, PDF"
            ),
        }

        detailed_description = detailed_descriptions.get(operation, f"{html_escape(description)}")

        message_text = (
            f"{detailed_description}\n\n"
            f"📄 <b>Поддерживаемые форматы:</b> {html_escape(formats)}\n\n"
            "📎 <b>Загрузите документ для обработки</b>"
        )

        await callback.message.answer(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="◀️ Назад к операциям", callback_data="document_processing")]
                ]
            ),
        )
        await callback.answer()

        await state.set_state(DocumentProcessingStates.waiting_for_document)

    except Exception as exc:  # noqa: BLE001
        await callback.answer(f"Ошибка: {exc}")
        logger.error("Ошибка в handle_document_operation: %s", exc, exc_info=True)


async def handle_back_to_menu(callback: CallbackQuery, state: FSMContext) -> None:
    """Возврат в главное меню."""
    try:
        document_manager = _get_document_manager()
        if document_manager is not None and callback.from_user:
            document_manager.end_chat_session(callback.from_user.id)

        await state.clear()
        await cmd_start(callback.message)
        await callback.answer()
    except Exception as exc:  # noqa: BLE001
        await callback.answer(f"Ошибка: {exc}")
        logger.error("Ошибка в handle_back_to_menu: %s", exc, exc_info=True)


async def handle_ocr_upload_more(callback: CallbackQuery, state: FSMContext) -> None:
    """Prepare state for another OCR upload after a result message."""
    output_format = "txt"
    data = callback.data or ""
    if ":" in data:
        _, payload = data.split(":", 1)
        if payload:
            output_format = payload
    try:
        with suppress(Exception):
            await callback.message.edit_reply_markup()

        await state.clear()
        await state.update_data(
            document_operation="ocr",
            operation_options={"output_format": output_format},
        )
        await state.set_state(DocumentProcessingStates.waiting_for_document)

        await callback.message.answer(
            f"{Emoji.DOCUMENT} Отправьте следующий файл или фото для режима \"распознание текста\".",
            parse_mode=ParseMode.HTML,
        )
        await callback.answer("Готов к загрузке нового документа")
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_ocr_upload_more: %s", exc, exc_info=True)
        await callback.answer("Не удалось подготовить повторную загрузку", show_alert=True)


async def handle_document_upload(message: Message, state: FSMContext) -> None:
    """Обработка загружённого документа."""
    document_manager = _get_document_manager()
    if document_manager is None:
        await message.answer(f"{Emoji.ERROR} Сервис обработки документов недоступен.")
        return

    try:
        if not message.document:
            await message.answer("❌ Ошибка: документ не найден")
            return

        async with typing_action(message.bot, message.chat.id, "upload_document"):
            data = await state.get_data()
            operation = data.get("document_operation")
            options = dict(data.get("operation_options") or {})
            output_format = str(options.get("output_format", "txt"))

            if not operation:
                await message.answer("❌ Операция не выбрана. Начните заново с /start")
                await state.clear()
                return

            await state.set_state(DocumentProcessingStates.processing_document)

            file_name = message.document.file_name or "unknown"
            file_size = message.document.file_size or 0
            mime_type = message.document.mime_type or "application/octet-stream"

            max_size = 50 * 1024 * 1024
            if file_size > max_size:
                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} Файл слишком большой. Максимальный размер: {max_size // (1024*1024)} МБ",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                await state.clear()
                return

            operation_info = document_manager.get_operation_info(operation) or {}
            operation_name = operation_info.get("name", operation)
            file_size_kb = max(1, file_size // 1024)

            stage_labels = _get_stage_labels(operation)

            status_msg: Message | None = None
            progress_state: dict[str, Any] = {"percent": 0, "stage": "start", "started_at": time.monotonic()}
            extras_last_text: str | None = None
            progress_status: ProgressStatus | None = None

            if operation == "lawsuit_analysis":
                stage_order = [key for key in _LAWSUIT_STAGE_ORDER if key in stage_labels]
                if not stage_order:
                    stage_order = list(stage_labels.keys())
                stage_index_map = {key: idx for idx, key in enumerate(stage_order, start=1)}
                progress_steps: list[dict[str, str]] = []
                for key in stage_order:
                    title, icon = stage_labels.get(key, (key, ""))
                    progress_steps.append({"label": f"{icon} {title}".strip()})
                if not progress_steps:
                    progress_steps = [{"label": "Анализ"}]
                progress_status = ProgressStatus(
                    message.bot,
                    message.chat.id,
                    steps=progress_steps,
                    show_context_toggle=False,
                    show_checklist=True,
                    auto_advance_stages=False,
                    min_edit_interval=0.5,
                    display_total_seconds=180,
                )
                await progress_status.start(auto_cycle=True, interval=1.0)

                async def send_progress(update: dict[str, Any]) -> None:
                    nonlocal progress_state, extras_last_text, progress_status
                    stage = str(update.get("stage") or progress_state["stage"] or "processing")
                    if stage not in stage_index_map and stage not in {"completed", "failed"}:
                        stage = progress_state["stage"]
                    percent_val = update.get("percent")
                    if percent_val is None:
                        percent = progress_state["percent"]
                    else:
                        percent = max(0, min(100, int(round(float(percent_val)))))
                        if percent < progress_state["percent"] and stage != "failed":
                            percent = progress_state["percent"]
                    progress_state["stage"] = stage
                    progress_state["percent"] = percent

                    note_text = str(update.get("note") or "").strip() or None
                    extras_line = _format_progress_extras(update)

                    if progress_status:
                        if stage == "completed":
                            extra_note = note_text or (extras_line or None)
                            await progress_status.complete(note=extra_note)
                            msg_id = progress_status.message_id
                            chat_id = progress_status.chat_id
                            bot = progress_status.bot
                            if msg_id:
                                with suppress(Exception):
                                    await bot.delete_message(chat_id, msg_id)
                            progress_status = None
                        elif stage == "failed":
                            fail_note = note_text or (extras_line or update.get("error"))
                            await progress_status.fail(note=fail_note)
                        else:
                            step = stage_index_map.get(stage)
                            await progress_status.update_stage(percent=percent, step=step)

                    if (
                        extras_line
                        and stage not in {"completed", "failed"}
                        and extras_line != extras_last_text
                        and not update.get("note")
                    ):
                        extras_last_text = extras_line
                        await message.answer(extras_line, parse_mode=ParseMode.HTML)
            else:
                status_msg = await message.answer("⏳ Подготавливаем обработку…", parse_mode=ParseMode.HTML)
                send_progress, progress_state = _make_progress_updater(
                    message,
                    status_msg,
                    file_name=file_name,
                    operation_name=operation_name,
                    file_size_kb=file_size_kb,
                    stage_labels=stage_labels,
                )
            await send_progress({"stage": "start", "percent": 5})

            try:
                await send_progress({"stage": "downloading", "percent": 18})
                file_info = await message.bot.get_file(message.document.file_id)
                file_path = file_info.file_path

                if not file_path:
                    raise ProcessingError("Не удалось получить путь к файлу", "FILE_ERROR")

                file_content = await message.bot.download_file(file_path)
                try:
                    file_bytes = await asyncio.to_thread(file_content.read)
                finally:
                    close_method = getattr(file_content, "close", None)
                    if callable(close_method):
                        close_method()
                await send_progress({"stage": "uploaded", "percent": 32})

                await send_progress({"stage": "processing", "percent": 45, "note": "Подготовка анонимизации"})
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    progress_callback=send_progress,
                    **options,
                )
                await send_progress({"stage": "finalizing", "percent": 90})

                if result.success:
                    formatted_result = document_manager.format_result_for_telegram(result, operation)
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    for idx, chunk in enumerate(_split_plain_text(formatted_result, limit=3500)):
                        await message.answer(
                            chunk,
                            reply_markup=reply_markup if idx == 0 else None,
                        )

                    exports = result.data.get("exports") or []
                    for export in exports:
                        export_path = export.get("path")
                        if not export_path:
                            error_msg = export.get("error")
                            if error_msg:
                                await message.answer(f"{Emoji.WARNING} {error_msg}")
                            continue
                        label = export.get("label") or export.get("name")
                        export_file_name = Path(export_path).name
                        format_tag = str(export.get("format", "file")).upper()
                        parts = [f"📄 {format_tag}"]
                        if label:
                            parts.append(str(label))
                        parts.append(export_file_name)
                        caption = " • ".join(part for part in parts if part)
                        try:
                            await message.answer_document(FSInputFile(export_path), caption=caption)
                        except Exception as send_error:  # noqa: BLE001
                            logger.error("Не удалось отправить файл %s: %s", export_path, send_error, exc_info=True)
                            await message.answer(f"Не удалось отправить файл {export_file_name}")
                        finally:
                            with suppress(Exception):
                                Path(export_path).unlink(missing_ok=True)

                    completion_payload = _build_completion_payload(operation, result)
                    await send_progress({'stage': 'completed', 'percent': 100, **completion_payload})
                    if 'status_msg' in locals() and status_msg:
                        with suppress(Exception):
                            await asyncio.sleep(0.6)
                            await status_msg.delete()

                    logger.info("Successfully processed document %s for user %s", file_name, message.from_user.id)
                else:
                    await send_progress(
                        {'stage': 'failed', 'percent': progress_state['percent'], 'note': result.message}
                    )
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(
                        f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{html_escape(str(result.message))}",
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )
                    if 'status_msg' in locals() and status_msg:
                        with suppress(Exception):
                            await status_msg.delete()

            except Exception as exc:  # noqa: BLE001
                with suppress(Exception):
                    await send_progress(
                        {"stage": "failed", "percent": progress_state.get("percent", 0), "note": GENERIC_INTERNAL_ERROR_HTML}
                    )
                if 'status_msg' in locals() and status_msg:
                    with suppress(Exception):
                        await status_msg.delete()

                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                logger.error("Error processing document %s: %s", file_name, exc, exc_info=True)

            finally:
                await state.clear()

    except Exception as exc:  # noqa: BLE001
        reply_markup = None
        if 'operation' in locals() and locals().get('operation') == "ocr":
            reply_markup = _build_ocr_reply_markup(locals().get('output_format', 'txt'))
        await message.answer(
            f"{Emoji.ERROR} <b>Произошла ошибка</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
        logger.error("Error in handle_document_upload: %s", exc, exc_info=True)
        await state.clear()


async def handle_photo_upload(message: Message, state: FSMContext) -> None:
    """Обработка загруженной фотографии для OCR."""
    document_manager = _get_document_manager()
    if document_manager is None:
        await message.answer(f"{Emoji.ERROR} Сервис обработки документов недоступен.")
        return

    try:
        if not message.photo:
            await message.answer("❌ Ошибка: фотография не найдена")
            return

        async with typing_action(message.bot, message.chat.id, "upload_photo"):
            data = await state.get_data()
            operation = data.get("document_operation")
            options = dict(data.get("operation_options") or {})
            output_format = str(options.get("output_format", "txt"))

            if not operation:
                await message.answer("❌ Операция не выбрана. Начните заново с /start")
                await state.clear()
                return

            await state.set_state(DocumentProcessingStates.processing_document)

            photo = message.photo[-1]
            file_name = f"photo_{photo.file_id}.jpg"
            file_size = photo.file_size or 0
            mime_type = "image/jpeg"

            max_size = 20 * 1024 * 1024
            if file_size > max_size:
                await message.answer(
                    f"❌ Фотография слишком большая. Максимальный размер: {max_size // (1024*1024)} МБ"
                )
                await state.clear()
                return

            operation_info = document_manager.get_operation_info(operation) or {}
            operation_name = operation_info.get("name", operation)

            file_size_kb = max(1, file_size // 1024)
            stage_labels = _get_stage_labels(operation)

            status_msg = await message.answer(
                f"📷 Обрабатываем фотографию для режима \"распознание текста\"...\n\n"
                f"⏳ Операция: {html_escape(operation_name)}\n"
                f"📏 Размер: {file_size_kb} КБ",
                parse_mode=ParseMode.HTML,
            )

            send_progress, progress_state = _make_progress_updater(
                message,
                status_msg,
                file_name=file_name,
                operation_name=operation_name,
                file_size_kb=file_size_kb,
                stage_labels=stage_labels,
            )

            try:
                await send_progress({"stage": "start", "percent": 5})

                file_info = await message.bot.get_file(photo.file_id)
                file_path = file_info.file_path

                if not file_path:
                    raise ProcessingError("Не удалось получить путь к фотографии", "FILE_ERROR")

                file_content = await message.bot.download_file(file_path)
                try:
                    file_bytes = await asyncio.to_thread(file_content.read)
                finally:
                    close_method = getattr(file_content, "close", None)
                    if callable(close_method):
                        close_method()
                await send_progress({"stage": "uploaded", "percent": 32})

                await send_progress({"stage": "processing", "percent": 45, "note": "Подготовка анонимизации"})
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    progress_callback=send_progress,
                    **options,
                )
                await send_progress({"stage": "finalizing", "percent": 90})

                if result.success:
                    formatted_result = document_manager.format_result_for_telegram(result, operation)

                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    for idx, chunk in enumerate(_split_plain_text(formatted_result, limit=3500)):
                        await message.answer(
                            chunk,
                            reply_markup=reply_markup if idx == 0 else None,
                        )

                    exports = result.data.get("exports") or []
                    for export in exports:
                        export_path = export.get("path")
                        if not export_path:
                            error_msg = export.get("error")
                            if error_msg:
                                await message.answer(f"{Emoji.WARNING} {error_msg}")
                            continue
                        label = export.get("label") or export.get("name")
                        export_file_name = Path(export_path).name
                        format_tag = str(export.get("format", "file")).upper()
                        parts = [f"📄 {format_tag}"]
                        if label:
                            parts.append(str(label))
                        parts.append(export_file_name)
                        caption = " • ".join(part for part in parts if part)
                        try:
                            await message.answer_document(FSInputFile(export_path), caption=caption)
                        except Exception as send_error:  # noqa: BLE001
                            logger.error(
                                "Не удалось отправить файл %s: %s", export_path, send_error, exc_info=True
                            )
                            await message.answer(f"Не удалось отправить файл {export_file_name}")
                        finally:
                            with suppress(Exception):
                                Path(export_path).unlink(missing_ok=True)

                    completion_payload = _build_completion_payload(operation, result)
                    await send_progress({"stage": "completed", "percent": 100, **completion_payload})
                    if 'status_msg' in locals() and status_msg:
                        with suppress(Exception):
                            await asyncio.sleep(0.6)
                            await status_msg.delete()

                    logger.info("Successfully processed photo %s for user %s", file_name, message.from_user.id)
                else:
                    await send_progress(
                        {"stage": "failed", "percent": progress_state["percent"], "note": result.message}
                    )
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(
                        f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{html_escape(str(result.message))}",
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )

            except Exception as exc:  # noqa: BLE001
                with suppress(Exception):
                    await send_progress(
                        {"stage": "failed", "percent": progress_state["percent"], "note": GENERIC_INTERNAL_ERROR_TEXT}
                    )
                    if 'status_msg' in locals() and status_msg:
                        await status_msg.delete()

                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                logger.error("Error processing photo %s: %s", file_name, exc, exc_info=True)

            finally:
                await state.clear()

    except Exception as exc:  # noqa: BLE001
        await message.answer("❌ Произошла внутренняя ошибка. Попробуйте позже.")
        logger.error("Error in handle_photo_upload: %s", exc, exc_info=True)
        await state.clear()


async def cmd_askdoc(message: Message) -> None:
    """Start chat-based document question session."""
    document_manager = _get_document_manager()
    if document_manager is None or not message.from_user:
        await message.answer(
            f"{Emoji.WARNING} Сессия документа не найдена. Загрузите документ с режимом \"Чат\"."
        )
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(f"{Emoji.WARNING} Укажите вопрос после команды, например: /askdoc Какой срок?")
        return

    question = parts[1].strip()
    try:
        async with typing_action(message.bot, message.chat.id, "typing"):
            result = await document_manager.answer_chat_question(message.from_user.id, question)
    except ProcessingError as exc:
        await message.answer(f"{Emoji.WARNING} {html_escape(exc.message)}", parse_mode=ParseMode.HTML)
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Document chat failed: %s", exc, exc_info=True)
        await message.answer(
            f"{Emoji.ERROR} Не удалось получить ответ. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    formatted = document_manager.format_chat_answer_for_telegram(result)
    await message.answer(formatted, parse_mode=ParseMode.HTML)


async def cmd_enddoc(message: Message) -> None:
    """Finish chat-based document session."""
    document_manager = _get_document_manager()
    if document_manager is None or not message.from_user:
        await message.answer(f"{Emoji.WARNING} Активная сессия не найдена.")
        return

    closed = document_manager.end_chat_session(message.from_user.id)
    if closed:
        await message.answer(f"{Emoji.SUCCESS} Чат с документом завершён.")
    else:
        await message.answer(f"{Emoji.WARNING} Активная сессия не найдена.")


def register_document_handlers(dp: Dispatcher) -> None:
    """Register all document-related handlers."""
    dp.callback_query.register(handle_document_processing, F.data == "document_processing")
    dp.callback_query.register(handle_document_operation, F.data.startswith("doc_operation_"))
    dp.callback_query.register(handle_doc_draft_start, F.data == "doc_draft_start")
    dp.callback_query.register(handle_doc_draft_cancel, F.data == "doc_draft_cancel")
    dp.callback_query.register(handle_ocr_upload_more, F.data.startswith("ocr_upload_more:"))
    dp.callback_query.register(handle_back_to_menu, F.data == "back_to_menu")

    dp.message.register(handle_doc_draft_request, DocumentDraftStates.waiting_for_request, F.text)
    dp.message.register(handle_doc_draft_request_voice, DocumentDraftStates.waiting_for_request, F.voice)
    dp.message.register(handle_doc_draft_answer, DocumentDraftStates.asking_details, F.text)
    dp.message.register(handle_doc_draft_answer_voice, DocumentDraftStates.asking_details, F.voice)
    dp.message.register(handle_document_upload, DocumentProcessingStates.waiting_for_document, F.document)
    dp.message.register(handle_photo_upload, DocumentProcessingStates.waiting_for_document, F.photo)
    dp.message.register(cmd_askdoc, Command("askdoc"))
    dp.message.register(cmd_enddoc, Command("enddoc"))
