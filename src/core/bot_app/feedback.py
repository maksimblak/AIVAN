from __future__ import annotations

import logging
from typing import Optional

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    User,
)

from src.bot.ui_components import Emoji
from src.core.exceptions import ValidationException
from src.core.bot_app import context as simple_context
from src.core.bot_app.common import ensure_valid_user_id, get_user_session, get_safe_db_method

__all__ = [
    "format_user_display",
    "ensure_rating_snapshot",
    "send_rating_request",
    "handle_pending_feedback",
    "cmd_ratings_stats",
    "register_feedback_handlers",
]

logger = logging.getLogger("ai-ivan.simple.feedback")


def format_user_display(user: User | None) -> str:
    if user is None:
        return ""
    parts: list[str] = []
    if user.username:
        parts.append(f"@{user.username}")
    name = " ".join(filter(None, [user.first_name, user.last_name])).strip()
    if name and name not in parts:
        parts.append(name)
    if not parts and user.first_name:
        parts.append(user.first_name)
    if not parts:
        parts.append(str(user.id))
    return " ".join(parts)


async def ensure_rating_snapshot(request_id: int, telegram_user: User | None, answer_text: str) -> None:
    if not answer_text.strip() or telegram_user is None:
        return

    db = simple_context.db
    if db is None:
        return

    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        return

    username = format_user_display(telegram_user)
    await add_rating_fn(
        request_id,
        telegram_user.id,
        0,
        None,
        username=username,
        answer_text=answer_text,
    )


def _create_rating_keyboard(request_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=f"rate_like_{request_id}"),
                InlineKeyboardButton(text="👎 Улучшить", callback_data=f"rate_dislike_{request_id}"),
            ]
        ]
    )


async def send_rating_request(
    message: Message,
    request_id: int,
    *,
    answer_snapshot: Optional[str] = None,
) -> None:
    if not message.from_user:
        return

    user_session = get_user_session(message.from_user.id)
    if answer_snapshot:
        user_session.last_answer_snapshot = answer_snapshot.strip()
    else:
        user_session.last_answer_snapshot = None

    keyboard = _create_rating_keyboard(request_id)
    await message.answer(
        f"{Emoji.STAR} <b>Оцените ответ</b>\n\n"
        "Это поможет нам становиться лучше.",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def handle_pending_feedback(message: Message, user_session, text_override: str | None = None):
    """Обработка текстового комментария после оценки."""
    if not message.from_user:
        return

    feedback_source = text_override if text_override is not None else (message.text or "")
    if not feedback_source or not user_session.pending_feedback_request_id:
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="handle_pending_feedback")
    except ValidationException as exc:
        logger.warning("Ignore feedback: invalid user id (%s)", exc)
        user_session.pending_feedback_request_id = None
        return

    request_id = user_session.pending_feedback_request_id
    feedback_text = feedback_source.strip()

    user_session.pending_feedback_request_id = None

    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        await message.answer("❌ Сервис отзывов временно недоступен")
        return

    get_rating_fn = get_safe_db_method("get_rating", default_return=None)
    existing_rating = await get_rating_fn(request_id, user_id) if get_rating_fn else None

    rating_value = -1
    answer_snapshot = ""
    if existing_rating:
        if existing_rating.rating not in (None, 0):
            rating_value = existing_rating.rating
        if getattr(existing_rating, "answer_text", None):
            answer_snapshot = existing_rating.answer_text or ""

    if not answer_snapshot:
        session_snapshot = getattr(user_session, "last_answer_snapshot", None)
        if session_snapshot:
            answer_snapshot = session_snapshot

    username_display = format_user_display(message.from_user)

    success = await add_rating_fn(
        request_id,
        user_id,
        rating_value,
        feedback_text,
        username=username_display,
        answer_text=answer_snapshot,
    )
    if success:
        await message.answer(
            "Спасибо за подробный отзыв!\n\nВаш комментарий поможет нам сделать ответы лучше.",
            parse_mode=ParseMode.HTML,
        )
        logger.info(
            "Received feedback for request %s from user %s: %s",
            request_id,
            user_id,
            feedback_text,
        )
        user_session.last_answer_snapshot = None
    else:
        await message.answer("❌ Не удалось сохранить отзыв")


async def handle_rating_callback(callback: CallbackQuery):
    """Обработка кнопок рейтинга и запрос текстового комментария."""
    if not callback.data or not callback.from_user:
        await callback.answer("Некорректные данные")
        return

    try:
        user_id = ensure_valid_user_id(callback.from_user.id, context="handle_rating_callback")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in rating callback: %s", exc)
        await callback.answer("Некорректный пользователь", show_alert=True)
        return

    user_session = get_user_session(user_id)

    try:
        parts = callback.data.split("_")
        if len(parts) != 3:
            await callback.answer("Некорректный формат данных")
            return
        action = parts[1]
        if action not in {"like", "dislike"}:
            await callback.answer("Неизвестное действие")
            return
        request_id = int(parts[2])
    except (ValueError, IndexError):
        await callback.answer("Некорректный формат данных")
        return

    get_rating_fn = get_safe_db_method("get_rating", default_return=None)
    existing_rating = await get_rating_fn(request_id, user_id) if get_rating_fn else None

    if existing_rating and existing_rating.rating not in (None, 0):
        await callback.answer("По этому ответу уже собрана обратная связь")
        return

    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        await callback.answer("Сервис рейтингов временно недоступен")
        return

    rating_value = 1 if action == "like" else -1
    answer_snapshot = ""
    if existing_rating and getattr(existing_rating, "answer_text", None):
        answer_snapshot = existing_rating.answer_text or ""
    if not answer_snapshot:
        session_snapshot = getattr(user_session, "last_answer_snapshot", None)
        if session_snapshot:
            answer_snapshot = session_snapshot
    username_display = format_user_display(callback.from_user)

    success = await add_rating_fn(
        request_id,
        user_id,
        rating_value,
        None,
        username=username_display,
        answer_text=answer_snapshot,
    )
    if not success:
        await callback.answer("Не удалось сохранить оценку")
        return

    if action == "like":
        await callback.answer("Спасибо за оценку! Рады, что ответ оказался полезным.")
        await callback.message.edit_text(
            "💬 <b>Спасибо за оценку!</b> ✅ Отмечено как полезное",
            parse_mode=ParseMode.HTML,
        )
        return

    await callback.answer("Спасибо за обратную связь!")
    feedback_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📝 Написать комментарий",
                    callback_data=f"feedback_{request_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Пропустить",
                    callback_data=f"skip_feedback_{request_id}",
                )
            ],
        ]
    )
    await callback.message.edit_text(
        "💬 <b>Что можно улучшить?</b>\n\nВаша обратная связь поможет нам стать лучше:",
        reply_markup=feedback_keyboard,
        parse_mode=ParseMode.HTML,
    )


async def handle_feedback_callback(callback: CallbackQuery):
    """Обработчик запроса обратной связи."""
    if not callback.data or not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        user_id = ensure_valid_user_id(callback.from_user.id, context="handle_feedback_callback")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in feedback callback: %s", exc)
        await callback.answer("❌ Ошибка пользователя", show_alert=True)
        return

    try:
        data = callback.data

        if data.startswith("feedback_"):
            action = "feedback"
            request_id = int(data.removeprefix("feedback_"))
        elif data.startswith("skip_feedback_"):
            action = "skip"
            request_id = int(data.removeprefix("skip_feedback_"))
        else:
            await callback.answer("❌ Неверный формат данных")
            return

        if action == "skip":
            await callback.message.edit_text(
                "💬 <b>Спасибо за оценку!</b> 👎 Отмечено для улучшения", parse_mode=ParseMode.HTML
            )
            await callback.answer("✅ Спасибо за обратную связь!")
            return

        user_session = get_user_session(user_id)
        if not hasattr(user_session, "pending_feedback_request_id"):
            user_session.pending_feedback_request_id = None
        user_session.pending_feedback_request_id = request_id

        await callback.message.edit_text(
            "💬 <b>Напишите ваш комментарий:</b>\n\n"
            "<i>Что можно улучшить в ответе? Отправьте текстовое сообщение.</i>",
            parse_mode=ParseMode.HTML,
        )
        await callback.answer("✏️ Напишите комментарий следующим сообщением")

    except Exception as exc:
        logger.error("Error in handle_feedback_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


def register_feedback_handlers(dp: Dispatcher) -> None:
    dp.callback_query.register(handle_rating_callback, F.data.startswith("rate_"))
    dp.callback_query.register(
        handle_feedback_callback, F.data.startswith(("feedback_", "skip_feedback_"))
    )
    dp.message.register(cmd_ratings_stats, Command("ratings"))


async def cmd_ratings_stats(message: Message) -> None:
    """Show aggregate rating stats (admin only)."""
    if not message.from_user:
        await message.answer("❌ Команда доступна только в диалоге с ботом")
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_ratings_stats")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in cmd_ratings_stats: %s", exc)
        await message.answer("❌ Ошибка идентификатора пользователя")
        return

    if user_id not in simple_context.ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    stats_fn = get_safe_db_method("get_ratings_statistics", default_return={})
    low_rated_fn = get_safe_db_method("get_low_rated_requests", default_return=[])
    if not stats_fn or not low_rated_fn:
        await message.answer("❌ Статистика рейтингов недоступна")
        return

    try:
        stats_7d = await stats_fn(7)
        stats_30d = await stats_fn(30)
        low_rated = await low_rated_fn(5)

        stats_text = f"""📊 <b>Статистика рейтингов</b>

📅 <b>За 7 дней:</b>
• Всего оценок: {stats_7d.get('total_ratings', 0)}
• 👍 Лайков: {stats_7d.get('total_likes', 0)}
• 👎 Дизлайков: {stats_7d.get('total_dislikes', 0)}
• 📈 Рейтинг лайков: {stats_7d.get('like_rate', 0):.1f}%
• 💬 С комментариями: {stats_7d.get('feedback_count', 0)}

📅 <b>За 30 дней:</b>
• Всего оценок: {stats_30d.get('total_ratings', 0)}
• 👍 Лайков: {stats_30d.get('total_likes', 0)}
• 👎 Дизлайков: {stats_30d.get('total_dislikes', 0)}
• 📈 Рейтинг лайков: {stats_30d.get('like_rate', 0):.1f}%
• 💬 С комментариями: {stats_30d.get('feedback_count', 0)}"""

        if low_rated:
            stats_text += "\n\n⚠️ <b>Запросы для улучшения:</b>\n"
            for req in low_rated[:3]:
                stats_text += (
                    f"• ID {req['request_id']}: "
                    f"рейтинг {req['avg_rating']:.1f} ({req['rating_count']} оценок)\n"
                )

        await message.answer(stats_text, parse_mode=ParseMode.HTML)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in cmd_ratings_stats: %s", exc)
        await message.answer("❌ Ошибка получения статистики рейтингов")

