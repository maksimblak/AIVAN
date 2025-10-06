import re

from src.bot.ui_components import sanitize_telegram_html
from src.core.safe_telegram import format_safe_html, split_html_for_telegram


def _is_balanced(html: str) -> bool:
    stack: list[str] = []
    for match in re.finditer(r"<(/?)([a-zA-Z0-9-]+)(?:[^>]*)>", html):
        name = match.group(2).lower()
        if name == "br":
            continue
        is_closing = bool(match.group(1))
        if is_closing:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
        else:
            stack.append(name)
    return not stack


def test_invalid_anchor_becomes_plain_text() -> None:
    raw = '<a href="javascript:alert(1)">Click me</a>'
    sanitized = sanitize_telegram_html(raw)
    assert "&lt;a" in sanitized
    assert "&lt;/a>" in sanitized
    assert "<a " not in sanitized


def test_split_html_balances_tags_for_long_anchor() -> None:
    long_text = '<a href="https://example.com/">' + ('hello ' * 200) + '</a>'
    formatted = format_safe_html(long_text)
    chunks = split_html_for_telegram(formatted, hard_limit=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _is_balanced(chunk)



def test_unbalanced_markup_is_normalised() -> None:
    raw = "<b><i>text</b></i>"
    sanitized = sanitize_telegram_html(raw)
    assert sanitized == "<b><i>text</i></b>"
