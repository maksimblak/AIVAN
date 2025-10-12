from __future__ import annotations

import logging
from datetime import datetime
from html import escape as html_escape
from typing import Any, Optional

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.bot.ui_components import Emoji, sanitize_telegram_html
from src.core.bot_app import context as ctx
from src.core.bot_app.common import ensure_valid_user_id, get_user_session
from src.core.bot_app.payments import get_plan_pricing
from src.core.bot_app.stats import generate_user_stats_response, normalize_stats_period
from src.core.exceptions import ErrorContext, ValidationException

logger = logging.getLogger("ai-ivan.simple.menus")

__all__ = [
    "register_menu_handlers",
    "cmd_start",
    "cmd_status",
    "cmd_mystats",
]

SECTION_DIVIDER = "<code>────────────────────</code>"


def _main_menu_text() -> str:
    return (
        "🏠 <b>Главное меню</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "⚖️ <b>ИИ-ИВАН</b> — ваш виртуальный\n"
        "   юридический ассистент\n\n"
        "🎯 <b>Доступные возможности:</b>\n"
        "   • Поиск судебной практики\n"
        "   • Работа с документами\n"
        "   • Юридические консультации\n\n"
        "Выберите действие:"
    )


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔍 Поиск судебной практики", callback_data="search_practice")],
            [InlineKeyboardButton(text="🗂️ Работа с документами", callback_data="document_processing")],
            [
                InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
            ],
        ]
    )


async def _try_send_welcome_media(
    message: Message,
    caption_html: str,
    keyboard: Optional[InlineKeyboardMarkup],
) -> bool:
    welcome_media = ctx.WELCOME_MEDIA
    if not welcome_media:
        return False

    media_type = (welcome_media.media_type or "video").lower()
    media_source = None
    supports_streaming = False

    if welcome_media.file_id:
        media_source = welcome_media.file_id
        supports_streaming = media_type == "video"
    elif welcome_media.path and welcome_media.path.exists():
        media_source = FSInputFile(welcome_media.path)
        supports_streaming = media_type == "video"
    else:
        return False

    try:
        if media_type == "animation":
            await message.answer_animation(
                animation=media_source,
                caption=caption_html,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        elif media_type == "photo":
            await message.answer_photo(
                photo=media_source,
                caption=caption_html,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        else:
            await message.answer_video(
                video=media_source,
                caption=caption_html,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
                supports_streaming=supports_streaming,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to send welcome media: %s", exc)
        return False


async def cmd_start(message: Message) -> None:
    if not message.from_user:
        return

    error_handler = ctx.error_handler
    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_start")
    except ValidationException as exc:
        context = ErrorContext(function_name="cmd_start", chat_id=message.chat.id if message.chat else None)
        if error_handler:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in cmd_start: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Не удалось инициализировать сессию.</b>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    db = ctx.db
    if db is not None and hasattr(db, "ensure_user"):
        await db.ensure_user(
            user_id,
            default_trial=ctx.TRIAL_REQUESTS,
            is_admin=user_id in ctx.ADMIN_IDS,
        )

    get_user_session(user_id)

    user_name = message.from_user.first_name or "Пользователь"
    welcome_raw = f"""<b>Добро пожаловать, {user_name}!</b>

Меня зовут <b>ИИ-ИВАН</b>, я ваш виртуальный юридический ассистент.

<b>ЧТО Я УМЕЮ:</b>

<b>Юридические вопросы</b>
— составляю выигрышные стратегии, даю быстрые консультации, проверяю аргументы на ошибки
и «человеческий фактор».

<b>Поиск и анализ судебной практики</b>
— анализирую миллионы дел и подбираю релевантные решения: какова вероятность успеха и как суд
трактует норму.

<b>Работа с документами</b>
— подготавливаю (в том числе голосом) процессуальные документы, проверяю договоры на риски,
делаю саммари.

<b>ПРИМЕРЫ ОБРАЩЕНИЙ:</b>
💬 "Администрация отказала в согласовании — подбери стратегию обжалования со ссылками на
судебную практику".
💬 "Проанализируй различия между статьями 228 и 228.1 УК РФ".
💬 "Найди судебную практику по взысканию неустойки с застройщика".
💬 "Могут ли наследники оспорить завещание после 6 месяцев?".

<b> ПОПРОБУЙТЕ ПРЯМО СЕЙЧАС </b>👇👇👇"""
    welcome_html = sanitize_telegram_html(welcome_raw)
    main_menu_keyboard = _main_menu_keyboard()

    media_sent = await _try_send_welcome_media(
        message=message,
        caption_html=welcome_html,
        keyboard=None,
    )

    if not media_sent:
        await message.answer(welcome_html, parse_mode=ParseMode.HTML)

    await message.answer(
        _main_menu_text(),
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu_keyboard,
    )
    logger.info("User %s started bot", message.from_user.id)


def _profile_menu_text(
    user: Any | None,
    *,
    status_text: str | None = None,
    tariff_text: str | None = None,
    hint_text: str | None = None,
) -> str:
    username = sanitize_telegram_html(
        getattr(user, "full_name", None) or getattr(user, "first_name", "") or ""
    )
    status_text = status_text or "⭕ <i>нет активной подписки</i>"
    tariff_text = tariff_text or "<b>Триал</b>"
    hint_text = hint_text or ""

    return (
        f"👤 <b>Профиль</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n\n"
        f"🙂 {username}\n"
        f"🔔 Статус: {status_text}\n"
        f"🏷️ Тариф: {tariff_text}\n"
        f"{hint_text}"
    )


def _profile_menu_keyboard(
    subscribe_label: str | None = None,
    *,
    has_subscription: bool = False,
) -> InlineKeyboardMarkup:
    if has_subscription:
        change_button = InlineKeyboardButton(text="🔄 Сменить тариф", callback_data="buy_catalog")
        cancel_label = subscribe_label or "❌ Отменить подписку"
        cancel_button = InlineKeyboardButton(text=cancel_label, callback_data="cancel_subscription")
        back_button = InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main")
        return InlineKeyboardMarkup(inline_keyboard=[[change_button], [cancel_button], [back_button]])

    first_label = subscribe_label or "💳 Оформить подписку"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=first_label, callback_data="get_subscription")],
            [
                InlineKeyboardButton(text="📊 Моя статистика", callback_data="my_stats"),
                InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral_program"),
            ],
            [InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main")],
        ]
    )


async def handle_my_profile_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    db = ctx.db
    try:
        await callback.answer()

        status_text = None
        tariff_text = None
        hint_text = None
        subscribe_label = "💳 Оформить подписку"
        has_subscription = False

        if db is not None:
            try:
                user_id = callback.from_user.id
                user_record = await db.ensure_user(
                    user_id,
                    default_trial=ctx.TRIAL_REQUESTS,
                    is_admin=user_id in ctx.ADMIN_IDS,
                )
                has_subscription = await db.has_active_subscription(user_id)
                cancel_flag = bool(getattr(user_record, "subscription_cancelled", 0))

                plan_id = getattr(user_record, "subscription_plan", None)
                plan_info = get_plan_pricing(plan_id) if plan_id else None
                if plan_info:
                    tariff_text = plan_info.plan.name
                elif plan_id and plan_id not in (None, "—"):
                    tariff_text = str(plan_id)
                else:
                    tariff_text = "триал"

                if has_subscription and getattr(user_record, "subscription_until", 0):
                    until_dt = datetime.fromtimestamp(int(user_record.subscription_until))
                    purchase_ts = int(getattr(user_record, "subscription_last_purchase_at", 0) or 0)
                    if purchase_ts:
                        purchase_dt = datetime.fromtimestamp(purchase_ts)
                        status_text = (
                            f"подписка оформлена {purchase_dt:%d.%m.%y} (доступ до {until_dt:%d.%m.%y})"
                        )
                    else:
                        status_text = f"подписка активна до {until_dt:%d.%m.%y}"

                    if cancel_flag:
                        hint_text = "Отмена оформлена — доступ сохранится до даты окончания."
                        subscribe_label = "✅ Отмена оформлена"
                    else:
                        hint_text = "Пополнить пакет — команда /buy"
                        subscribe_label = "❌ Отменить подписку"
                else:
                    trial_remaining = int(getattr(user_record, "trial_remaining", 0) or 0)
                    status_text = "⭕ <i>нет активной подписки</i>"
                    tariff_text = f" <b>Триал</b> • <i>{trial_remaining} запросов</i>"
                    hint_text = ""
            except Exception as profile_error:  # pragma: no cover
                logger.debug("Failed to build profile header: %s", profile_error, exc_info=True)

        await callback.message.edit_text(
            _profile_menu_text(
                callback.from_user,
                status_text=status_text,
                tariff_text=tariff_text,
                hint_text=hint_text,
            ),
            parse_mode=ParseMode.HTML,
            reply_markup=_profile_menu_keyboard(subscribe_label, has_subscription=has_subscription),
        )

    except Exception as exc:
        logger.error("Error in handle_my_profile_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_my_stats_callback(callback: CallbackQuery) -> None:
    if not callback.from_user or callback.message is None:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    db = ctx.db
    try:
        await callback.answer()

        if db is None:
            await callback.message.edit_text(
                "Статистика временно недоступна",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        user_id = callback.from_user.id
        user = await db.ensure_user(
            user_id, default_trial=ctx.TRIAL_REQUESTS, is_admin=user_id in ctx.ADMIN_IDS
        )
        stats = await db.get_user_statistics(user_id, days=30)

        try:
            status_text, keyboard = await generate_user_stats_response(
                user_id,
                days=30,
                stats=stats,
                user=user,
                divider=SECTION_DIVIDER,
            )
        except RuntimeError as stats_error:
            logger.error("Failed to build user stats: %s", stats_error)
            await callback.message.edit_text(
                "Статистика временно недоступна",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        await callback.message.edit_text(
            status_text,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )

    except Exception as exc:
        logger.error("Error in handle_my_stats_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


def _resolve_bot_username() -> str:
    username = (ctx.BOT_USERNAME or "").strip()
    if username.startswith("@"):
        username = username[1:]
    if username:
        return username
    try:
        env_username = ctx.settings().get_str("TELEGRAM_BOT_USERNAME")
    except Exception:
        env_username = None
    if env_username:
        env_username = env_username.strip()
        if env_username.startswith("https://t.me/"):
            env_username = env_username[len("https://t.me/") :]
        elif env_username.startswith("t.me/"):
            env_username = env_username[len("t.me/") :]
        if env_username.startswith("@"):
            env_username = env_username[1:]
        if env_username:
            return env_username
    return ""


def _build_referral_link(referral_code: str | None) -> tuple[str | None, str | None]:
    if not referral_code or referral_code == "SYSTEM_ERROR":
        return None, None
    safe_code = html_escape(referral_code)
    username = _resolve_bot_username()
    if username:
        return f"https://t.me/{username}?start=ref_{safe_code}", referral_code
    try:
        fallback_base = ctx.settings().get_str("TELEGRAM_REFERRAL_BASE_URL")
    except Exception:
        fallback_base = None
    if fallback_base:
        base = fallback_base.strip().rstrip("/")
        if base:
            if not base.startswith("http"):
                base = f"https://{base.lstrip('/')}"
            return f"{base}?start=ref_{safe_code}", referral_code
    return None, referral_code


async def handle_referral_program_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    db = ctx.db
    try:
        await callback.answer()

        if db is None:
            await callback.message.edit_text(
                "Сервис временно недоступен",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        user_id = callback.from_user.id
        user = await db.get_user(user_id)

        if not user:
            await callback.message.edit_text(
                "Ошибка получения данных пользователя",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        referral_code: str | None = None
        stored_code = (getattr(user, "referral_code", None) or "").strip()

        if stored_code and stored_code != "SYSTEM_ERROR":
            referral_code = stored_code
        else:
            try:
                generated_code = (await db.generate_referral_code(user_id) or "").strip()
            except Exception as exc:
                logger.error("Error with referral code: %s", exc)
                generated_code = ""
            if generated_code and generated_code != "SYSTEM_ERROR":
                referral_code = generated_code
                try:
                    setattr(user, "referral_code", referral_code)
                except Exception:
                    pass

        referral_link, share_code = _build_referral_link(referral_code)

        try:
            referrals = await db.get_user_referrals(user_id)
        except Exception as exc:
            logger.error("Error getting referrals: %s", exc)
            referrals = []

        total_referrals = len(referrals)
        active_referrals = sum(1 for ref in referrals if ref.get("has_active_subscription", False))

        referral_bonus_days = getattr(user, "referral_bonus_days", 0)
        referrals_count = getattr(user, "referrals_count", 0)

        referral_lines: list[str] = [
            "👥 <b>Реферальная программа</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "🎁 <b>Ваши бонусы</b>",
            "",
            f"  🎉 Бонусных дней: <b>{referral_bonus_days}</b>",
            f"  👫 Приглашено друзей: <b>{referrals_count}</b>",
            f"  ✅ С активной подпиской: <b>{active_referrals}</b>",
            "",
        ]

        if referral_link:
            referral_lines.extend(
                [
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    "",
                    "🔗 <b>Ваша реферальная ссылка</b>",
                    "",
                    f"<code>{referral_link}</code>",
                    "",
                ]
            )
        elif share_code:
            referral_lines.extend(
                [
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    "",
                    "🔗 <b>Ваш реферальный код</b>",
                    "",
                    f"<code>ref_{html_escape(share_code)}</code>",
                    "",
                    "<i>Отправьте его друзьям, чтобы они\nуказали код при запуске бота</i>",
                    "",
                ]
            )
        else:
            referral_lines.extend(
                [
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    "",
                    "⚠️ <b>Ссылка временно недоступна</b>",
                    "",
                    "<i>Попробуйте позже или обратитесь\nв поддержку</i>",
                    "",
                ]
            )

        referral_lines.extend(
            [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "💡 <b>Как это работает</b>",
                "",
                "  1️⃣ Поделитесь ссылкой с друзьями",
                "  2️⃣ За каждого друга получите 3 дня",
                "  3️⃣ Друг получит скидку 20%",
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "📈 <b>Ваши рефералы</b>",
                "",
            ]
        )

        if referrals:
            referral_lines.append(f"  📊 Всего: <b>{total_referrals}</b>")
            referral_lines.append(f"  💎 С подпиской: <b>{active_referrals}</b>")
            for ref in referrals[:5]:
                join_date = datetime.fromtimestamp(ref["joined_at"]).strftime("%d.%m.%Y")
                status = "💎" if ref.get("has_active_subscription") else "👤"
                referral_lines.append(f"{status} Пользователь #{ref['user_id']} - {join_date}")
        else:
            referral_lines.append("• Пока никого нет")

        referral_text = "\n".join(referral_lines)

        keyboard_buttons: list[list[InlineKeyboardButton]] = []
        if share_code:
            copy_text = "📋 Скопировать ссылку" if referral_link else "📋 Скопировать код"
            keyboard_buttons.append(
                [
                    InlineKeyboardButton(
                        text=copy_text,
                        callback_data=f"copy_referral_{share_code}",
                    )
                ]
            )

        keyboard_buttons.append(
            [InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")]
        )

        referral_keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)

        await callback.message.edit_text(
            referral_text,
            parse_mode=ParseMode.HTML,
            reply_markup=referral_keyboard,
        )

    except Exception as exc:
        logger.error("Error in handle_referral_program_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_copy_referral_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        callback_data = callback.data or ""
        if callback_data.startswith("copy_referral_"):
            referral_code = callback_data.replace("copy_referral_", "")
            referral_link, share_code = _build_referral_link(referral_code)

            if referral_link:
                await callback.answer(f"📋 Ссылка скопирована!\n{referral_link}", show_alert=True)
                return
            if share_code:
                await callback.answer(f"📋 Код скопирован!\nref_{share_code}", show_alert=True)
                return

            await callback.answer("❌ Реферальная ссылка временно недоступна", show_alert=True)
            return

        await callback.answer("❌ Ошибка получения кода")

    except Exception as exc:
        logger.error("Error in handle_copy_referral_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_back_to_main_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()
        await callback.message.edit_text(
            _main_menu_text(),
            parse_mode=ParseMode.HTML,
            reply_markup=_main_menu_keyboard(),
        )
    except Exception as exc:
        logger.error("Error in handle_back_to_main_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_search_practice_callback(callback: CallbackQuery) -> None:
    """Handle 'search_practice' menu button."""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        instruction_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_to_main")],
                [InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile")],
            ]
        )

        await callback.message.edit_text(
            "🔍 <b>Поиск судебной практики</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚖️ <i>Найду релевантную судебную практику\n"
            "   для вашего юридического вопроса</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Что вы получите:</b>\n\n"
            "💡 <b>Краткая консультация</b>\n"
            "   └ 2 ссылки на практику и краткий анализ\n\n"
            "📊 <b>Углубленный анализ</b>\n"
            "   └ 6+ примеров с рекомендациями\n\n"
            "📄 <b>Подготовка документов</b>\n"
            "   └ На основе найденной практики\n"
            "   └ С учетом актуальных решений\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✍️ <i>Напишите ваш юридический вопрос\n"
            "   следующим сообщением...</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=instruction_keyboard,
        )

        user_session = get_user_session(callback.from_user.id)
        if not hasattr(user_session, "practice_search_mode"):
            user_session.practice_search_mode = False
        user_session.practice_search_mode = True

    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_search_practice_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_prepare_documents_callback(callback: CallbackQuery) -> None:
    """Handle 'prepare_documents' menu button."""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        await callback.message.answer(
            "📄 <b>Подготовка документов</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📑 <i>Помогу составить процессуальные\n"
            "   и юридические документы</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Типы документов:</b>\n\n"
            "⚖️ Исковые заявления\n"
            "📝 Ходатайства и запросы\n"
            "📧 Жалобы и возражения\n"
            "📜 Договоры и соглашения\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✍️ <i>Опишите какой документ нужен\n"
            "   и приложите детали дела...</i>",
            parse_mode=ParseMode.HTML,
        )

        user_session = get_user_session(callback.from_user.id)
        if not hasattr(user_session, "document_preparation_mode"):
            user_session.document_preparation_mode = False
        user_session.document_preparation_mode = True

    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_prepare_documents_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def handle_help_info_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        support_text = (
            "🔧 <b>Техническая поддержка</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📞 <b>Контакты поддержки</b>\n"
            "   ├ Telegram: @support_username\n"
            "   └ Email: support@example.com\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "❓ <b>Часто задаваемые вопросы</b>\n\n"
            "🤖 <b>Бот не отвечает</b>\n"
            "   ├ Попробуйте команду /start\n"
            "   └ Проверьте интернет-соединение\n\n"
            "📄 <b>Ошибка при обработке документа</b>\n"
            "   ├ Форматы: PDF, DOCX, DOC, TXT\n"
            "   ├ Максимальный размер: 20 МБ\n"
            "   └ Проверьте целостность файла\n\n"
            "⏳ <b>Долгое ожидание ответа</b>\n"
            "   ├ Сложные запросы: 2-3 минуты\n"
            "   └ Большие документы: до 5 минут\n\n"
            "💬 <b>Как задать вопрос боту?</b>\n"
            "   ├ Напишите свой вопрос\n"
            "   ├ Можете прикрепить документ\n"
            "   └ Бот учитывает контекст беседы\n\n"
            "🔄 <b>Как начать новую беседу?</b>\n"
            "   ├ Используйте команду /start\n"
            "   └ Или кнопку \"Новый диалог\"\n\n"
            "💰 <b>Как проверить баланс?</b>\n"
            "   └ Откройте раздел \"Профиль\"\n\n"
            "🎯 <b>Какие запросы понимает бот?</b>\n"
            "   ├ Вопросы на любые темы\n"
            "   ├ Анализ документов и текстов\n"
            "   ├ Генерация контента\n"
            "   └ Помощь с задачами\n\n"
            "🔒 <b>Безопасны ли мои данные?</b>\n"
            "   ├ Все данные зашифрованы\n"
            "   └ Не передаем данные третьим лицам"
        )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main")]]
        )

        await callback.message.answer(support_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        logger.info("Support info requested by user %s", callback.from_user.id)

    except Exception as exc:
        logger.error("Error in handle_help_info_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка")


async def cmd_status(message: Message) -> None:
    db = ctx.db
    if db is None:
        await message.answer("Статус временно недоступен")
        return

    if not message.from_user:
        await message.answer("Статус доступен только для авторизованных пользователей")
        return

    error_handler = ctx.error_handler
    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_status")
    except ValidationException as exc:
        context = ErrorContext(function_name="cmd_status", chat_id=message.chat.id if message.chat else None)
        if error_handler:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in cmd_status: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Не удалось получить статус.</b>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    user = await db.ensure_user(
        user_id,
        default_trial=ctx.TRIAL_REQUESTS,
        is_admin=user_id in ctx.ADMIN_IDS,
    )
    until_ts = int(getattr(user, "subscription_until", 0) or 0)
    now_ts = int(datetime.now().timestamp())
    has_active = until_ts > now_ts
    plan_id = getattr(user, "subscription_plan", None)
    plan_info = get_plan_pricing(plan_id) if plan_id else None
    if plan_info:
        plan_label = plan_info.plan.name
    elif plan_id:
        plan_label = plan_id
    elif has_active:
        plan_label = "Безлимит"
    else:
        plan_label = "нет"

    if until_ts > 0:
        until_dt = datetime.fromtimestamp(until_ts)
        if has_active:
            left_days = max(0, (until_dt - datetime.now()).days)
            until_text = f"{until_dt:%Y-%m-%d} (≈{left_days} дн.)"
        else:
            until_text = f"Истекла {until_dt:%Y-%m-%d}"
    else:
        until_text = "Не активна"

    quota_balance_raw = getattr(user, "subscription_requests_balance", None)
    quota_balance = int(quota_balance_raw) if quota_balance_raw is not None else None

    lines = [
        f"{Emoji.STATS} <b>Статус</b>",
        "",
        f"ID: <code>{user_id}</code>",
        f"Роль: {'админ' if getattr(user, 'is_admin', False) else 'пользователь'}",
        f"Триал: {getattr(user, 'trial_remaining', 0)} запрос(ов)",
        "Подписка:",
    ]
    if plan_info or plan_id or until_ts:
        lines.append(f"• План: {plan_label}")
        lines.append(f"• Доступ до: {until_text}")
        if plan_info and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif plan_id and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif has_active and not plan_id:
            lines.append("• Лимит: без ограничений")
    else:
        lines.append("• Не активна")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_mystats(message: Message) -> None:
    db = ctx.db
    if db is None:
        await message.answer("Статистика временно недоступна")
        return

    if not message.from_user:
        await message.answer("Статистика доступна только авторизованным пользователям")
        return

    days = 30
    if message.text:
        parts = message.text.strip().split()
        if len(parts) >= 2:
            try:
                days = int(parts[1])
            except ValueError:
                days = 30

    days = normalize_stats_period(days)

    try:
        stats_text, keyboard = await generate_user_stats_response(
            message.from_user.id,
            days,
            divider=SECTION_DIVIDER,
        )
        await message.answer(stats_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in cmd_mystats: %s", exc)
        await message.answer("❌ Ошибка получения статистики. Попробуйте позже.")


def register_menu_handlers(dp: Dispatcher) -> None:
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))

    dp.callback_query.register(handle_my_profile_callback, F.data == "my_profile")
    dp.callback_query.register(handle_my_stats_callback, F.data == "my_stats")
    dp.callback_query.register(handle_back_to_main_callback, F.data == "back_to_main")
    dp.callback_query.register(handle_search_practice_callback, F.data == "search_practice")
    dp.callback_query.register(handle_prepare_documents_callback, F.data == "prepare_documents")
    dp.callback_query.register(handle_referral_program_callback, F.data == "referral_program")
    dp.callback_query.register(handle_copy_referral_callback, F.data.startswith("copy_referral_"))
    dp.callback_query.register(handle_help_info_callback, F.data == "help_info")

