import pytest

from src.bot.ui_components import render_legal_html, sanitize_telegram_html


def test_render_legal_html_inserts_paragraph_and_italics():
    raw = (
        "Короткий ответ Да — договор можно расторгнуть за просрочку (_неоднократная_). "
        "(consultant.ru)Подробный разбор нормативной базы"
    )

    rendered = render_legal_html(raw)

    assert "<i>неоднократная</i>" in rendered
    assert "Подробный разбор" in rendered

    idx = rendered.index("(consultant.ru)")
    assert "<br><br>" in rendered[idx: idx + 40]



def test_sanitize_preserves_safe_html():
    raw = "<b>Привет</b><script>alert(1)</script><i>мир</i>"
    sanitized = sanitize_telegram_html(raw)
    assert "<script>" not in sanitized
    assert "<b>Привет</b>" in sanitized
    assert "<i>мир</i>" in sanitized
