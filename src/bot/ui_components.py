"""
UI компоненты для Telegram бота ИИ-Иван
Содержит клавиатуры, эмодзи, шаблоны сообщений и форматирование
"""

from __future__ import annotations
import re
from html import escape as html_escape
# ============ ЭМОДЗИ КОНСТАНТЫ ============


class Emoji:
    """Коллекция эмодзи для интерфейса"""

    ROBOT = "🤖"
    LAW = "⚖️"
    DOCUMENT = "📄"
    SEARCH = "🔍"
    IDEA = "💡"
    WARNING = "⚠️"
    SUCCESS = "✅"
    ERROR = "❌"
    LOADING = "⏳"
    STAR = "⭐"
    MAGIC = "✨"
    MICROPHONE = "🎙️"

    CIVIL = "📘"
    CRIMINAL = "🧑‍⚖️"
    CORPORATE = "🏢"
    CONTRACT = "📝"
    LABOR = "⚙️"
    TAX = "💰"
    REAL_ESTATE = "🏠"
    IP = "🧠"
    ADMIN = "🏛️"
    FAMILY = "👪"

    BACK = "⬅️"
    HELP = "❓"
    STATS = "📊"
    INFO = "ℹ️"

    DIAMOND = "🧾"
    DOWNLOAD = "📥"
    CLOCK = "🕒"
    CALENDAR = "📅"

# ============ ШАБЛОНЫ СООБЩЕНИЙ (MarkdownV2) ============

LEGAL_CATEGORIES = {
    "civil": {
        "name": "Гражданское право",
        "emoji": Emoji.CIVIL,
        "description": "Имущественные и личные неимущественные отношения",
        "examples": ["Договоры", "Собственность", "Обязательства", "Деликты"],
    },
    "corporate": {
        "name": "Корпоративное право",
        "emoji": Emoji.CORPORATE,
        "description": "Создание и деятельность юридических лиц",
        "examples": ["Учреждение ООО", "Корпоративные споры", "Реорганизация", "M&A"],
    },
    "contract": {
        "name": "Договорное право",
        "emoji": Emoji.CONTRACT,
        "description": "Заключение, исполнение и расторжение договоров",
        "examples": ["Поставка", "Подряд", "Аренда", "Займ"],
    },
    "labor": {
        "name": "Трудовое право",
        "emoji": Emoji.LABOR,
        "description": "Трудовые отношения и социальная защита",
        "examples": ["Увольнение", "Зарплата", "Отпуска", "Дисциплина"],
    },
    "tax": {
        "name": "Налоговое право",
        "emoji": Emoji.TAX,
        "description": "Налогообложение и взаимодействие с ФНС",
        "examples": ["НДС", "Налог на прибыль", "НДФЛ", "Проверки"],
    },
    "real_estate": {
        "name": "Право недвижимости",
        "emoji": Emoji.REAL_ESTATE,
        "description": "Сделки с недвижимостью и земельными участками",
        "examples": ["Купля-продажа", "Аренда", "Ипотека", "Кадастр"],
    },
    "ip": {
        "name": "Интеллектуальная собственность",
        "emoji": Emoji.IP,
        "description": "Авторские права, товарные знаки, патенты",
        "examples": ["Регистрация ТЗ", "Авторские права", "Патенты", "Лицензии"],
    },
    "admin": {
        "name": "Административное право",
        "emoji": Emoji.ADMIN,
        "description": "Взаимодействие с госорганами и административная ответственность",
        "examples": ["Лицензирование", "Штрафы", "Госуслуги", "Контроль"],
    },
    "criminal": {
        "name": "Уголовное право",
        "emoji": Emoji.CRIMINAL,
        "description": "Преступления и уголовная ответственность",
        "examples": ["Экономические преступления", "Должностные", "Налоговые", "Защита"],
    },
    "family": {
        "name": "Семейное право",
        "emoji": Emoji.FAMILY,
        "description": "Брак, развод, алименты, опека",
        "examples": ["Развод", "Алименты", "Раздел имущества", "Опека"],
    },
}


def get_category_info(category_id: str) -> dict:
    """Получить информацию о категории права"""
    return LEGAL_CATEGORIES.get(
        category_id,
        {
            "name": "Неизвестная категория",
            "emoji": Emoji.LAW,
            "description": "Общие правовые вопросы",
            "examples": [],
        },
    )


# ============ ФОРМАТИРОВАНИЕ (MarkdownV2) ============


def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для MarkdownV2"""
    special_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


# ============ HTML ФОРМАТИРОВАНИЕ ДЛЯ STREAMING ============

# --- Telegram HTML sanitizer (allowlist) ---
ALLOWED_TAGS = {"b","strong","i","em","u","ins","s","strike","del","code","pre","a","br","tg-spoiler","blockquote"}




def sanitize_telegram_html(html: str) -> str:
    """Sanitize Telegram HTML while keeping allowed markup balanced."""
    if not html:
        return ""

    tag_re = re.compile(r"<(/?)([a-zA-Z0-9-]+)([^>]*)>", re.IGNORECASE)
    href_re = re.compile(
        r"href\s*=\s*(\"([^\"]*)\"|'([^']*)'|([^\s\"'=`<>]+))",
        re.IGNORECASE,
    )

    simple_tags = {
        "b",
        "strong",
        "i",
        "em",
        "u",
        "ins",
        "s",
        "strike",
        "del",
        "code",
        "pre",
        "tg-spoiler",
        "blockquote",
    }

    parts: list[str] = []
    open_stack: list[str] = []
    ignored_open_counts: dict[str, int] = {}
    cursor = 0

    def _append_text(segment: str) -> None:
        if segment:
            parts.append(html_escape(segment))

    def _escape_tag(token: str) -> str:
        return token.replace('&', '&amp;').replace('<', '&lt;')


    for match in tag_re.finditer(html):
        start_pos, end_pos = match.span()
        _append_text(html[cursor:start_pos])

        slash, name_raw, attrs = match.groups()
        name = (name_raw or "").lower()
        is_closing = bool(slash)
        raw = match.group(0)

        if name not in ALLOWED_TAGS:
            parts.append(_escape_tag(raw))
            ignored_open_counts[name] = ignored_open_counts.get(name, 0) + 1
            cursor = end_pos
            continue

        if not is_closing:
            if name == "br":
                parts.append("<br>")
            elif name == "a":
                href_value = ""
                if attrs:
                    href_match = href_re.search(attrs)
                    if href_match:
                        href_candidate = next(
                            (group for group in href_match.groups()[1:] if group),
                            "",
                        )
                        if href_candidate.lower().startswith(("http://", "https://")):
                            href_value = html_escape(href_candidate, quote=True)
                if href_value:
                    parts.append(f'<a href="{href_value}">')
                    open_stack.append("a")
                else:
                    parts.append(_escape_tag(raw))
                    ignored_open_counts[name] = ignored_open_counts.get(name, 0) + 1
            else:
                parts.append(f"<{name}>")
                open_stack.append(name)
        else:
            if name not in open_stack:
                count = ignored_open_counts.get(name, 0)
                if count > 0:
                    parts.append(_escape_tag(raw))
                    if count == 1:
                        ignored_open_counts.pop(name, None)
                    else:
                        ignored_open_counts[name] = count - 1
                cursor = end_pos
                continue

            while open_stack:
                top = open_stack.pop()
                parts.append(f"</{top}>")
                if top == name:
                    break

        cursor = end_pos

    _append_text(html[cursor:])

    while open_stack:
        parts.append(f"</{open_stack.pop()}>")

    return "".join(parts)

