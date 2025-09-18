"""Форматирование ответов для Telegram."""
from __future__ import annotations

from typing import Iterable

from telegram.helpers import escape_markdown


def format_laws(laws: Iterable[str]) -> str:
    """Форматирует список законов в моноширинном стиле."""

    escaped_laws = [f"`{escape_markdown(law, version=2)}`" for law in laws if law]
    return "\n".join(escaped_laws) if escaped_laws else "`Нормы права не найдены.`"


def format_legal_response(summary: str, details: str, laws: Iterable[str]) -> str:
    """Возвращает красиво отформатированный ответ."""

    summary = escape_markdown(summary, version=2)
    details = escape_markdown(details, version=2)
    laws_block = format_laws(laws)

    return (
        "⚖️ *ЮРИДИЧЕСКАЯ КОНСУЛЬТАЦИЯ*\n\n"
        "📋 *Краткий ответ:*\n"
        f"{summary}\n\n"
        "📄 *Подробное разъяснение:*\n"
        f"{details}\n\n"
        "📚 *Применимые нормы права:*\n"
        f"{laws_block}\n\n"
        "⚠️ *Важно:*\n"
        "Данная консультация носит информационный характер и не заменяет профессиональную юридическую помощь."
    )
