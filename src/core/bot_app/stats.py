from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from src.bot.ui_components import Emoji
from src.core.bot_app import context as ctx
from src.core.bot_app.formatting import (
    _format_number,
    _format_response_time,
    _format_stat_row,
    _format_trend_value,
)
from src.core.bot_app.formatting import _format_currency, _format_datetime
from src.core.bot_app.payments import get_plan_pricing, plan_stars_amount
from src.core.subscription_payments import (
    SubscriptionPayloadError,
    parse_subscription_payload,
)

PERIOD_OPTIONS: Sequence[int] = (7, 30, 90)
PROGRESS_BAR_LENGTH = 10
HEADER_DIVIDER = "━━━━━━━━━━━━"
DOCUMENT_OPERATION_LABELS: Mapping[str, str] = {
    "document_summarize": "📑 Краткая выжимка документа",
    "document_analyze_risks": "⚠️ Анализ рисков",
    "document_lawsuit_analysis": "⚖️ Анализ искового заявления",
    "document_anonymize": "🕶️ Анонимизация",
    "document_translate": "🌍 Перевод документов",
    "document_ocr": "🔍 Распознавание текста",
    "document_chat": "💬 Чат с документом",
    "document_draft": "✨ Создание юридического документа",
}
FEATURE_LABELS: Mapping[str, str] = {
    "legal_question": "🤖 Обращения к ИИ ассистенту",
    "document_processing": "📄 Работа с документами",
    "judicial_practice": "⚖️ Судебная практика",
    "document_draft": "📝 Черновики документов",
    "voice_message": "🎤 Голосовые ответы",
    "ocr_processing": "🔍 Распознавание текста",
    "document_chat": "💬 Чаты по документам",
    "document_work": "📄 Работа с документами",
    **DOCUMENT_OPERATION_LABELS,
}

DOCUMENT_REQUEST_PREFIXES: tuple[str, ...] = ("document_", "doc_")
DOCUMENT_REQUEST_ALIASES: frozenset[str] = frozenset(
    {
        "ocr_processing",
        "ocr",
        "lawsuit_analysis",
        "summarize",
        "analyze_risks",
        "anonymize",
        "translate",
    }
)


def _is_document_request_type(request_type: str | None) -> bool:
    """Определяет, относится ли тип запроса к операциям с документами."""
    if not request_type:
        return False
    normalized = request_type.lower()
    return normalized.startswith(DOCUMENT_REQUEST_PREFIXES) or normalized in DOCUMENT_REQUEST_ALIASES
DAY_NAMES: Mapping[str, str] = {
    "0": "Пн",
    "1": "Вт",
    "2": "Ср",
    "3": "Чт",
    "4": "Пт",
    "5": "Сб",
    "6": "Вс",
}


def normalize_stats_period(days: int) -> int:
    if days <= 0:
        return PERIOD_OPTIONS[0]
    for option in PERIOD_OPTIONS:
        if days <= option:
            return option
    return PERIOD_OPTIONS[-1]


def build_progress_bar(used: int, total: int) -> str:
    if total is None or total <= 0:
        return "<code>[██████████]</code> ∞ / <b>Безлимит</b>"

    total = max(total, 0)
    used = max(0, min(used, total))

    ratio = used / total if total else 0.0
    filled = min(PROGRESS_BAR_LENGTH, max(0, int(round(ratio * PROGRESS_BAR_LENGTH))))
    bar = f"[{'█' * filled}{'░' * (PROGRESS_BAR_LENGTH - filled)}]"
    bar_markup = f"<code>{bar}</code>"

    remaining = max(0, total - used)
    remaining_pct = max(0, min(100, int(round((remaining / total) * 100)))) if total else 0

    return f"{bar_markup} {used}/{total} · осталось <b>{remaining}</b> ({remaining_pct}%)"


def build_mini_bar(percentage: float, length: int = 8) -> str:
    """Создаёт компактный визуальный индикатор для процентов."""
    filled = min(length, max(0, int(round(percentage / 100 * length))))
    return f"<code>{'█' * filled}{'░' * (length - filled)}</code>"


def progress_line(label: str, used: int, total: int) -> str:
    return f"<b>{label}</b> {build_progress_bar(used, total)}"


def translate_payment_status(status: str) -> str:
    status_map = {
        "pending": "⏳ Ожидание",
        "processing": "🔄 Обработка",
        "succeeded": "✅ Успешно",
        "success": "✅ Успешно",
        "completed": "✅ Завершён",
        "failed": "❌ Ошибка",
        "cancelled": "🚫 Отменён",
        "canceled": "🚫 Отменён",
        "refunded": "↩️ Возврат",
        "unknown": "❓ Неизвестно",
    }
    return status_map.get(status.lower(), status)


def translate_plan_name(plan_id: str) -> str:
    plan_map = {
        "basic": "Базовый",
        "standard": "Стандарт",
        "premium": "Премиум",
        "pro": "Про",
        "trial": "Триал",
    }
    period_map = {
        "1m": "1 месяц",
        "3m": "3 месяца",
        "6m": "6 месяцев",
        "12m": "1 год",
        "1y": "1 год",
    }

    parts = plan_id.split("_")
    if len(parts) >= 2:
        plan_name = plan_map.get(parts[0].lower(), parts[0].capitalize())
        period = period_map.get(parts[1].lower(), parts[1])
        return f"{plan_name} • {period}"

    return plan_map.get(plan_id.lower(), plan_id)


def describe_primary_summary(summary: str, unit: str) -> str:
    if not summary or summary == "—":
        return "нет данных"
    if "(" in summary and summary.endswith(")"):
        label, count = summary.rsplit("(", 1)
        label = label.strip()
        count = count[:-1].strip()
        if count.isdigit():
            return f"{label} — {count} {unit}"
        return f"{label} — {count}"
    return summary


def describe_secondary_summary(summary: str, unit: str) -> str:
    if not summary:
        return "нет данных"
    parts: list[str] = []
    for raw in summary.split(","):
        item = raw.strip()
        if not item:
            continue
        tokens = item.split()
        if len(tokens) >= 2 and tokens[-1].isdigit():
            count = tokens[-1]
            label = " ".join(tokens[:-1])
            parts.append(f"{label} — {count}")
        else:
            parts.append(item)
    if not parts:
        return "нет данных"
    return "; ".join(parts)


def peak_summary(
    counts: Mapping[str, int],
    *,
    mapping: Mapping[str, str] | None = None,
    limit: int = 3,
    formatter: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    if not counts:
        return "—", "нет данных"

    best_keys = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    primary_parts: list[str] = []
    secondary_parts: list[str] = []
    for idx, (raw_key, value) in enumerate(best_keys, start=1):
        base_label = mapping.get(raw_key, raw_key) if mapping else raw_key
        label = str(base_label)
        if formatter:
            try:
                label = formatter(label)
            except Exception:
                label = str(base_label)
        entry = f"{label} ({value})"
        if idx == 1:
            primary_parts.append(entry)
        secondary_parts.append(entry)

    primary = ", ".join(primary_parts) if primary_parts else "—"
    secondary = ", ".join(secondary_parts)
    return primary, secondary


def top_labels(
    counts: Mapping[str, int],
    *,
    mapping: Mapping[str, str] | None = None,
    limit: int = 3,
    formatter: Callable[[str], str] | None = None,
) -> str:
    if not counts:
        return "—"
    top_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    labels: list[str] = []
    for key, value in top_items:
        label = mapping.get(key, key) if mapping else key
        if formatter:
            label = formatter(label)
        labels.append(f"{label}×{value}")
    return ", ".join(labels)


def build_recommendations(
    *,
    trial_remaining: int,
    has_subscription: bool,
    subscription_days_left: int,
    period_requests: int,
    previous_requests: int,
) -> list[str]:
    tips: list[str] = []
    if not has_subscription:
        if trial_remaining > 0:
            tips.append(
                f"Используйте оставшиеся {trial_remaining} запросов триала и оформите подписку через /buy."
            )
        else:
            tips.append("Подключите подписку для безлимитного доступа — /buy.")
    else:
        if subscription_days_left <= 5:
            tips.append("Продлите подписку заранее, чтобы не потерять доступ — кнопка ниже.")

    if period_requests == 0:
        tips.append("Задайте боту первый вопрос — начните с /start или загрузите документ.")
    elif period_requests < previous_requests:
        tips.append("Активность снизилась — попробуйте использовать подборки или вопросы к документам.")

    if not tips:
        tips.append("Продолжайте спрашивать — бот быстрее реагирует на частые запросы.")
    return tips[:3]


__all__ = [
    "PERIOD_OPTIONS",
    "PROGRESS_BAR_LENGTH",
    "FEATURE_LABELS",
    "DAY_NAMES",
    "normalize_stats_period",
    "build_progress_bar",
    "build_mini_bar",
    "progress_line",
    "translate_payment_status",
    "translate_plan_name",
    "describe_primary_summary",
    "describe_secondary_summary",
    "peak_summary",
    "top_labels",
    "build_recommendations",
    "build_stats_keyboard",
    "generate_user_stats_response",
]


def build_stats_keyboard(has_subscription: bool) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    if not has_subscription:
        buttons.append([InlineKeyboardButton(text="💳 Оформить подписку", callback_data="get_subscription")])
    buttons.append([InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


async def generate_user_stats_response(
    user_id: int,
    days: int,
    *,
    stats: dict[str, Any] | None = None,
    user: Any | None = None,
    divider: str = "──────────",
) -> tuple[str, InlineKeyboardMarkup]:
    db = ctx.db
    if db is None:
        raise RuntimeError("Database is not available")

    normalized_days = normalize_stats_period(days)
    if user is None:
        user = await db.ensure_user(
            user_id,
            default_trial=ctx.TRIAL_REQUESTS,
            is_admin=user_id in ctx.ADMIN_IDS,
        )

    if stats is None:
        stats = await db.get_user_statistics(user_id, days=normalized_days)
    if stats.get("error"):
        raise RuntimeError(stats.get("error"))

    plan_id = stats.get("subscription_plan") or getattr(user, "subscription_plan", None)
    plan_info = get_plan_pricing(plan_id) if plan_id else None

    subscription_until_ts = int(stats.get("subscription_until", 0) or 0)
    now_ts = int(time.time())
    has_subscription = subscription_until_ts > now_ts
    subscription_days_left = (
        max(0, math.ceil((subscription_until_ts - now_ts) / 86400)) if has_subscription else 0
    )

    subscription_status_text = "❌ Не активна"
    if has_subscription:
        until_text = _format_datetime(subscription_until_ts, default="—")
        subscription_status_text = f"✅ Активна до {until_text} (≈{subscription_days_left} дн.)"
    elif subscription_until_ts:
        until_text = _format_datetime(subscription_until_ts, default="—")
        subscription_status_text = f"⏰ Истекла {until_text}"

    trial_remaining = int(stats.get("trial_remaining", getattr(user, "trial_remaining", 0)) or 0)

    if plan_info:
        plan_label = plan_info.plan.name
    elif plan_id:
        plan_label = plan_id
    elif trial_remaining > 0:
        plan_label = "Триал"
    else:
        plan_label = "—"

    period_requests = int(stats.get("period_requests", 0) or 0)
    previous_requests = int(stats.get("previous_period_requests", 0) or 0)
    period_successful = int(stats.get("period_successful", 0) or 0)
    previous_successful = int(stats.get("previous_period_successful", 0) or 0)
    period_tokens = int(stats.get("period_tokens", 0) or 0)
    avg_response_time_ms = int(stats.get("avg_response_time_ms", 0) or 0)

    success_rate = (period_successful / period_requests * 100) if period_requests else 0.0

    day_counts = stats.get("day_of_week_counts") or {}
    hour_counts = stats.get("hour_of_day_counts") or {}
    raw_type_stats = stats.get("request_types") or {}
    document_requests_total = 0
    document_breakdown: dict[str, int] = {}
    type_stats: dict[str, int] = {}
    for raw_type, raw_value in raw_type_stats.items():
        if raw_type is None:
            continue
        try:
            count = int(raw_value)
        except (TypeError, ValueError):
            continue
        key = str(raw_type)
        if _is_document_request_type(key):
            document_requests_total += count
            document_breakdown[key] = document_breakdown.get(key, 0) + count
            continue
        type_stats[key] = type_stats.get(key, 0) + count
    if document_requests_total:
        type_stats["document_work"] = type_stats.get("document_work", 0) + document_requests_total

    day_primary, day_secondary = peak_summary(day_counts, mapping=DAY_NAMES)
    hour_primary, hour_secondary = peak_summary(
        hour_counts, formatter=lambda raw: f"{int(raw):02d}:00" if raw.isdigit() else raw
    )

    last_transaction = stats.get("last_transaction")

    created_at_ts = stats.get("created_at") or getattr(user, "created_at", 0)
    updated_at_ts = stats.get("updated_at") or getattr(user, "updated_at", 0)
    last_request_ts = stats.get("last_request_at", 0)

    subscription_balance_raw = stats.get("subscription_requests_balance")
    if subscription_balance_raw is None:
        subscription_balance_raw = getattr(user, "subscription_requests_balance", None)
    subscription_balance = int(subscription_balance_raw or 0)

    lines = [
        f"{Emoji.STATS} <b>Моя статистика</b>",
        HEADER_DIVIDER,
        "",
        f"📅 <i>Период: последние {normalized_days} дней</i>",
        "",
        divider,
        "",
        "👤 <b>Профиль</b>",
        "",
        _format_stat_row("  📆 Регистрация", _format_datetime(created_at_ts)),
        _format_stat_row("  🕐 Последний запрос", _format_datetime(last_request_ts)),
        _format_stat_row("  💳 Подписка", subscription_status_text),
        _format_stat_row("  🏷️ План", plan_label),
    ]

    lines.extend(["", divider, "", "🔋 <b>Лимиты</b>", ""])
    trial_requests = ctx.TRIAL_REQUESTS
    if trial_requests > 0:
        trial_used = max(0, trial_requests - trial_remaining)
        lines.append(progress_line("Триал", trial_used, trial_requests))
    else:
        lines.append(_format_stat_row("Триал", "недоступен"))

    if plan_info and plan_info.plan.request_quota > 0:
        used = max(0, plan_info.plan.request_quota - subscription_balance)
        lines.append(progress_line("  📊 Подписка", used, plan_info.plan.request_quota))
    elif has_subscription:
        lines.append(_format_stat_row("  📊 Подписка", "безлимит ♾️"))

    lines.extend(
        [
            "",
            divider,
            "",
            "📈 <b>Активность</b>",
            "",
            _format_stat_row("  📝 Запросов", _format_trend_value(period_requests, previous_requests)),
            _format_stat_row(
                "  ⏱️ Среднее время ответа",
                _format_response_time(avg_response_time_ms),
            ),
            "",
        ]
    )

    if day_primary != "—":
        lines.append(_format_stat_row("  📅 Активный день", describe_primary_summary(day_primary, "обращений")))
        if day_secondary.strip():
            lines.append(_format_stat_row("  📆 Другие дни", describe_secondary_summary(day_secondary, "обращений")))
    else:
        lines.append(_format_stat_row("  📅 Активный день", "нет данных"))

    if hour_primary != "—":
        lines.append(_format_stat_row("  🕐 Активный час", describe_primary_summary(hour_primary, "обращений")))
        if hour_secondary.strip():
            lines.append(_format_stat_row("  🕑 Другие часы", describe_secondary_summary(hour_secondary, "обращений")))
    else:
        lines.append(_format_stat_row("  🕐 Активный час", "нет данных"))

    lines.extend(["", divider, "", "📋 <b>Типы запросов</b>", ""])
    sorted_types = sorted(type_stats.items(), key=lambda item: item[1], reverse=True)

    if not sorted_types:
        lines.append("  <i>Нет данных за выбранный период</i>")

    for req_type, count in sorted_types:
        share_pct = (count / period_requests * 100) if period_requests else 0.0
        label = FEATURE_LABELS.get(req_type, req_type)
        lines.append(f"  {label} — <b>{count}</b> ({share_pct:.0f}%)")

    if document_breakdown:
        sorted_document_types = sorted(
            document_breakdown.items(), key=lambda item: item[1], reverse=True
        )
        for req_type, count in sorted_document_types:
            share_pct = (count / period_requests * 100) if period_requests else 0.0
            label = DOCUMENT_OPERATION_LABELS.get(req_type, FEATURE_LABELS.get(req_type, req_type))
            lines.append(f"      ↳ {label} — <b>{count}</b> ({share_pct:.0f}%)")

    if last_transaction:
        lines.extend(["", divider, "", "💳 <b>Последний платёж</b>", ""])
        currency = last_transaction.get("currency", "RUB") or "RUB"
        amount_minor = last_transaction.get("amount_minor_units") or last_transaction.get("amount")
        lines.append(_format_stat_row("  💰 Сумма", _format_currency(amount_minor, currency)))
        translated_status = translate_payment_status(last_transaction.get("status", "unknown"))
        lines.append(_format_stat_row("  📊 Статус", translated_status))
        lines.append(_format_stat_row("  📅 Дата", _format_datetime(last_transaction.get("created_at"))))
        payload_raw = last_transaction.get("payload")
        if payload_raw:
            try:
                payload = parse_subscription_payload(payload_raw)
                if payload.plan_id:
                    translated_plan = translate_plan_name(payload.plan_id)
                    lines.append(_format_stat_row("  🏷️ Тариф", translated_plan))
            except SubscriptionPayloadError:
                pass

    keyboard = build_stats_keyboard(has_subscription)
    return "\n".join(lines), keyboard

