import re

from core.bot_app.openai_gateway import format_legal_response_text


_ALLOWED_TAG_REGEX = re.compile(
    r"""
    (?:
        <br>|
        </?(?:b|i|u|s|code|pre|tg-spoiler|blockquote)>|
        </a>|
        <a\s+href=\"(?:https://|http://|tg://user\?id=)[^\"]+\">
    )
    """,
    re.VERBOSE,
)


def _strip_allowed_tags(html: str) -> str:
    """Remove allowed Telegram tags, ensuring nothing else remains."""
    previous = None
    current = html
    while previous != current:
        previous = current
        current = _ALLOWED_TAG_REGEX.sub("", current)
    return current


def test_formatter_produces_telegram_safe_html():
    raw = (
        "<details><summary>Plan</summary><p>First paragraph</p>"
        "<ul><li>Alpha</li><li>Beta</li></ul></details><div>After</div>"
        "<a href='ftp://bad'>bad</a><a href='https://valid'>ok</a>"
        "<span class='tg-spoiler'>secret</span>"
    )

    formatted = format_legal_response_text(raw)

    expected = (
        "<b>Plan</b><br><br>First paragraph<br>"
        "\u2022 Alpha<br>\u2022 Beta<br><br>"
        "Afterbad<a href=\"https://valid\">ok</a><tg-spoiler>secret"
    )

    assert formatted == expected
    assert "ftp://bad" not in formatted
    assert "<details" not in formatted

    stripped = _strip_allowed_tags(formatted)
    assert "<" not in stripped and ">" not in stripped


def test_formatter_escapes_unsupported_tags():
    raw = "<script>alert('x')</script><b>ok</b>"
    formatted = format_legal_response_text(raw)

    assert formatted == "alert(&#x27;x&#x27;)<b>ok</b>"
    assert "<script" not in formatted


def test_formatter_numbers_ordered_lists():
    raw = "<ol><li>One</li><li>Two</li></ol>"
    formatted = format_legal_response_text(raw)

    assert formatted == "<b>1.</b> One<br><b>2.</b> Two"
    stripped = _strip_allowed_tags(formatted)
    assert "<" not in stripped and ">" not in stripped
