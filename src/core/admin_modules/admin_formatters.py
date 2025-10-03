"""
Shared formatting functions для admin dashboards
"""


def format_trend(trend: str) -> str:
    """
    Форматирование тренда с emoji

    Args:
        trend: "improving", "stable", "declining", "insufficient_data", "no_data"

    Returns:
        Formatted string with emoji
    """
    emoji_map = {
        "improving": "📈 Улучшается",
        "stable": "➡️ Стабильно",
        "declining": "📉 Ухудшается",
        "insufficient_data": "❓ Недостаточно данных",
        "no_data": "❌ Нет данных"
    }
    return emoji_map.get(trend, trend)


def nps_emoji(nps: float) -> str:
    """
    Emoji для NPS score

    Args:
        nps: NPS score (-100 to +100)

    Returns:
        Status string with emoji
    """
    if nps >= 50:
        return "🌟 Отлично"
    elif nps >= 30:
        return "✅ Хорошо"
    elif nps >= 0:
        return "⚠️ Средне"
    else:
        return "🔴 Плохо"


def growth_emoji(rate: float) -> str:
    """
    Emoji для growth rate

    Args:
        rate: Growth rate in %

    Returns:
        Emoji representing growth
    """
    if rate > 10:
        return "🚀"
    elif rate > 0:
        return "✅"
    elif rate > -10:
        return "⚠️"
    else:
        return "🔴"


def stickiness_emoji(ratio: float) -> str:
    """
    Emoji для stickiness ratio (DAU/MAU)

    Args:
        ratio: DAU/MAU ratio in %

    Returns:
        Emoji
    """
    if ratio >= 20:
        return "✅"
    elif ratio >= 10:
        return "⚠️"
    else:
        return "🔴"


def quick_ratio_status(ratio: float) -> str:
    """
    Статус Quick Ratio

    Args:
        ratio: Quick Ratio value

    Returns:
        Status string with emoji
    """
    if ratio > 4:
        return "🌟 Отлично"
    elif ratio > 2:
        return "✅ Хорошо"
    elif ratio > 1:
        return "⚠️ Нормально"
    else:
        return "🔴 Плохо"


def ltv_cac_status(ratio: float) -> str:
    """
    Статус LTV/CAC ratio

    Args:
        ratio: LTV/CAC ratio

    Returns:
        Emoji status
    """
    if ratio > 3:
        return "✅"
    elif ratio > 1:
        return "⚠️"
    else:
        return "🔴"


def pmf_status(achieved: bool) -> str:
    """
    Статус PMF (Product-Market Fit)

    Args:
        achieved: True if PMF achieved (>40% very disappointed)

    Returns:
        Status string
    """
    if achieved:
        return "✅ <b>PMF достигнут!</b>"
    else:
        return "⚠️ PMF пока не достигнут"


def pmf_rating_emoji(rating: str) -> str:
    """
    Emoji для PMF rating

    Args:
        rating: "strong", "moderate", "weak", "kill"

    Returns:
        Emoji
    """
    emoji_map = {
        "strong": "🌟",
        "moderate": "✅",
        "weak": "⚠️",
        "kill": "🗑️"
    }
    return emoji_map.get(rating, "")
