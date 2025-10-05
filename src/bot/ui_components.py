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

    DIAMOND = "💎"
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
    """
    Sanitize Telegram HTML: allow a limited subset of tags and normalise broken markup.
    """
    if not html:
        return ""

    allowed_pattern = "|".join(ALLOWED_TAGS)
    html = re.sub(f"<(?!/?(?:{allowed_pattern})\b)", "&lt;", html)

    tag_re = re.compile(r"</?([a-zA-Z0-9-]+)(\s[^>]*)?>", re.IGNORECASE)
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
    open_tags: list[str] = []

    def _pop_open(tag: str) -> bool:
        for idx in range(len(open_tags) - 1, -1, -1):
            if open_tags[idx] == tag:
                open_tags.pop(idx)
                return True
        return False

    def _clean_tag(match: re.Match[str]) -> str:
        full = match.group(0)
        name = (match.group(1) or "").lower()
        attrs = match.group(2) or ""
        is_closing = full.startswith("</")

        if name not in ALLOWED_TAGS:
            return html_escape(full)

        if name == "br":
            return "" if is_closing else "<br>"

        if is_closing:
            if _pop_open(name):
                return f"</{name}>"
            return html_escape(full)

        if name in simple_tags:
            open_tags.append(name)
            return f"<{name}>"

        if name == "a":
            href = ""
            if attrs:
                m = re.search(r'href\s*=\s*"(.*?)"', attrs, re.IGNORECASE)
                if not m:
                    m = re.search(r"href\s*=\s*'([^']*)'", attrs, re.IGNORECASE)
                if m:
                    cand = (m.group(1) or "").strip()
                    if cand.lower().startswith(("http://", "https://")):
                        href = html_escape(cand, quote=True)
            if href:
                open_tags.append("a")
                return f'<a href="{href}">'
            return html_escape(full)

        return html_escape(full)

    return tag_re.sub(_clean_tag, html)
