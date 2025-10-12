from __future__ import annotations

from typing import Callable, Mapping, Sequence

PERIOD_OPTIONS: Sequence[int] = (7, 30, 90)
PROGRESS_BAR_LENGTH = 10
FEATURE_LABELS: Mapping[str, str] = {
    "legal_question": "Юридические вопросы",
    "document_processing": "Обработка документов",
    "judicial_practice": "Судебная практика",
    "document_draft": "Черновики документов",
    "voice_message": "Голосовые ответы",
    "ocr_processing": "Распознавание текста",
    "document_chat": "Чаты по документам",
}
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
) -> tuple[str, str]:
    if not counts:
        return "—", "нет данных"

    best_keys = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    primary_parts: list[str] = []
    secondary_parts: list[str] = []
    for idx, (raw_key, value) in enumerate(best_keys, start=1):
        label = mapping.get(raw_key, raw_key) if mapping else raw_key
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
    "progress_line",
    "translate_payment_status",
    "translate_plan_name",
    "describe_primary_summary",
    "describe_secondary_summary",
    "peak_summary",
    "top_labels",
    "build_recommendations",
]
