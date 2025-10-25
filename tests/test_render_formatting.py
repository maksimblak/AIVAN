from core.bot_app.ui_components import sanitize_telegram_html


def test_sanitize_preserves_safe_html():
    raw = "<b>Привет</b><script>alert(1)</script><i>мир</i>"
    sanitized = sanitize_telegram_html(raw)
    assert "<script>" not in sanitized
    assert "<b>Привет</b>" in sanitized
    assert "<i>мир</i>" in sanitized
