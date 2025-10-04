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
    Пропускает только безопасные теги: b,i,u,s,code,pre,a(href=http/https),br.
    У всех остальных тегов — экранирует угловые скобки.
    У <a> оставляет только допустимый href, прочие атрибуты выкидывает.
    """
    if not html:
        return ""

    # Экранируем угловые скобки, которые не похожи на валидный тег
    # Используем ALLOWED_TAGS для консистентности
    allowed_pattern = "|".join(ALLOWED_TAGS)
    html = re.sub(f"<(?!/?(?:{allowed_pattern})\\b)", "&lt;", html)

    tag_re = re.compile(r"</?([a-zA-Z0-9]+)(\s[^>]*)?>", re.IGNORECASE)

    def _clean_tag(match: re.Match) -> str:
        full = match.group(0)
        name = (match.group(1) or "").lower()
        attrs = match.group(2) or ""
        is_closing = full.startswith("</")

        # Неизвестные теги — экранируем полностью
        if name not in ALLOWED_TAGS:
            return html_escape(full)

        # <br> допускается без атрибутов; закрывающего нет
        if name == "br":
            return "" if is_closing else "<br>"

        # Закрывающие теги
        if is_closing:
            return f"</{name}>"

        # Открывающие простые теги без атрибутов (кроме <a>)
        if name in {"b", "strong", "i", "em", "u", "s", "del", "code", "pre"}:
            return f"<{name}>"

        # Специальная обработка <a ...>
        if name == "a":
            href = ""
            if attrs:
                # href="..." или href='...'
                m = re.search(r'href\s*=\s*"(.*?)"', attrs, re.IGNORECASE)
                if not m:
                    m = re.search(r"href\s*=\s*'([^']*)'", attrs, re.IGNORECASE)
                if m:
                    cand = (m.group(1) or "").strip()
                    if cand.lower().startswith(("http://", "https://")):
                        href = html_escape(cand, quote=True)
            # если href валидный — оставляем ссылку; иначе экранируем оригинал тега
            return f'<a href="{href}">' if href else html_escape(full)

        # На всякий случай
        return html_escape(full)

    return tag_re.sub(_clean_tag, html)
