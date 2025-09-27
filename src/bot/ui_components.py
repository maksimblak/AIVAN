"""
UI компоненты для Telegram бота ИИ-Иван
Содержит клавиатуры, эмодзи, шаблоны сообщений и форматирование
"""

from __future__ import annotations

# ============ ЭМОДЗИ КОНСТАНТЫ ============


class Emoji:
    """Коллекция эмодзи для интерфейса"""

    # Основные системные
    ROBOT = "🤖"
    LAW = "⚖️"
    DOCUMENT = "📋"
    SEARCH = "🔍"
    IDEA = "💡"
    WARNING = "⚠️"
    SUCCESS = "✅"
    ERROR = "❌"
    LOADING = "⏳"
    FIRE = "🔥"
    STAR = "⭐"
    MAGIC = "✨"

    # Категории права
    CIVIL = "🏠"
    CRIMINAL = "🚨"
    CORPORATE = "🏢"
    CONTRACT = "📝"
    LABOR = "👨‍💼"
    TAX = "💰"
    REAL_ESTATE = "🏘️"
    IP = "💼"
    ADMIN = "🏛️"
    FAMILY = "👪"

    # Навигация
    BACK = "◀️"
    HOME = "🏠"
    HELP = "❓"
    SETTINGS = "⚙️"
    STATS = "📊"
    UP = "🔺"
    DOWN = "🔻"

    # Действия
    SAVE = "💾"
    SHARE = "📤"
    COPY = "📄"
    PRINT = "🖨️"
    DOWNLOAD = "📥"

    # Статусы
    ONLINE = "🟢"
    OFFLINE = "🔴"
    PENDING = "🟡"
    CLOCK = "🕐"
    CALENDAR = "📅"


# ============ ЦВЕТОВЫЕ СХЕМЫ ============


class Colors:
    """Цвета для форматирования (не используются напрямую в Telegram, для документации)"""

    PRIMARY = "#2196F3"  # Синий
    SUCCESS = "#4CAF50"  # Зеленый
    WARNING = "#FF9800"  # Оранжевый
    ERROR = "#F44336"  # Красный
    INFO = "#00BCD4"  # Голубой


# ============ ШАБЛОНЫ СООБЩЕНИЙ (MarkdownV2) ============


class MessageTemplates:
    """Шаблоны сообщений с красивым форматированием (MarkdownV2)"""

    WELCOME = f"""{Emoji.LAW} **ИИ\\-Иван** — ваш юридический ассистент

{Emoji.ROBOT} Специализируюсь на российском праве и судебной практике
{Emoji.SEARCH} Анализирую дела, нахожу релевантную практику  
{Emoji.DOCUMENT} Готовлю черновики процессуальных документов

{Emoji.WARNING} *Важно*: все ответы требуют проверки юристом

Выберите действие:"""

    HELP = f"""{Emoji.HELP} **Справка по использованию**

{Emoji.MAGIC} **Для получения лучших результатов:**

{Emoji.IDEA} Указывайте конкретную юрисдикцию
{Emoji.CALENDAR} Упоминайте даты важных событий
{Emoji.DOCUMENT} Описывайте фактические обстоятельства
{Emoji.STAR} Формулируйте четкий правовой вопрос

{Emoji.LAW} **Что я умею:**
• Анализ судебной практики
• Поиск релевантных дел
• Подготовка процессуальных документов
• Оценка правовых рисков
• Разработка правовой стратегии

{Emoji.WARNING} **Ограничения:**
Не разглашайте персональные данные третьих лиц"""

    CATEGORIES = f"""{Emoji.LAW} **Выберите область права**

Выбор специализации поможет получить более точный и релевантный ответ:"""

    PROCESSING_STAGES = [
        f"{Emoji.SEARCH} Анализирую ваш вопрос...",
        f"{Emoji.LOADING} Ищу релевантную судебную практику...",
        f"{Emoji.DOCUMENT} Формирую структурированный ответ...",
        f"{Emoji.MAGIC} Финализирую рекомендации...",
    ]

    ERROR_GENERIC = f"""{Emoji.ERROR} **Произошла ошибка**

К сожалению, не удалось обработать ваш запрос\\.

{Emoji.HELP} *Рекомендации:*
• Проверьте формулировку вопроса
• Попробуйте через несколько минут
• Обратитесь к администратору если проблема повторяется"""

    NO_QUESTION = f"""{Emoji.WARNING} **Пустой запрос**

Пожалуйста, отправьте текст юридического вопроса\\."""


# ============ КАТЕГОРИИ ПРАВА ============

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


def format_legal_response(text: str, category: str | None = None) -> str:
    """Форматирует ответ с красивой разметкой MarkdownV2"""
    if category:
        category_info = get_category_info(category)
        header = f"{category_info['emoji']} **{escape_markdown_v2(category_info['name'])}**\n\n"
        text = header + text
    return text


def create_progress_message(stage: int, total: int = 4) -> str:
    """Создает сообщение с прогрессом (MarkdownV2)"""
    if stage >= len(MessageTemplates.PROCESSING_STAGES):
        stage = len(MessageTemplates.PROCESSING_STAGES) - 1
    progress_bar = "▓" * stage + "░" * (total - stage)
    percentage = int((stage / total) * 100)
    return f"{MessageTemplates.PROCESSING_STAGES[stage]}\n\n`{progress_bar}` {percentage}%"


def create_progress_message_html(stage: int, total: int = 4) -> str:
    """Альтернатива для HTML-режима (если отправляете parse_mode=HTML)"""
    if stage >= len(MessageTemplates.PROCESSING_STAGES):
        stage = len(MessageTemplates.PROCESSING_STAGES) - 1
    progress_bar = "▓" * stage + "░" * (total - stage)
    percentage = int((stage / total) * 100)
    return f"{MessageTemplates.PROCESSING_STAGES[stage]}<br><br><code>{progress_bar}</code> {percentage}%"


# ============ HTML ФОРМАТИРОВАНИЕ ДЛЯ STREAMING ============

import re
from html import escape as html_escape

# --- Telegram HTML sanitizer (allowlist) ---
ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "br"}


def sanitize_telegram_html(html: str) -> str:
    """
    Пропускает только безопасные теги: b,i,u,s,code,pre,a(href=http/https),br.
    У всех остальных тегов — экранирует угловые скобки.
    У <a> оставляет только допустимый href, прочие атрибуты выкидывает.
    """
    if not html:
        return ""

    # Экранируем угловые скобки, которые не похожи на тег, чтобы не ломать текст типа "a < b"
    html = re.sub(r"<(?!/?[a-zA-Z])", "&lt;", html)

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
        if name in {"b", "i", "u", "s", "code", "pre"}:
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


def _md_links_to_anchors(line: str) -> str:
    """Convert markdown links [text](url) into safe HTML anchors.

    Both link text and URL are escaped; only http/https URLs are allowed.
    """
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    result_parts: list[str] = []
    last = 0
    for m in pattern.finditer(line):
        # escape non-link part
        result_parts.append(html_escape(line[last : m.start()]))
        text = html_escape(m.group(1))
        url = html_escape(m.group(2), quote=True)
        result_parts.append(f'<a href="{url}">{text}</a>')
        last = m.end()
    # tail
    result_parts.append(html_escape(line[last:]))
    return "".join(result_parts)


def render_legal_html(raw: str) -> str:
    """Форматирует «сырой» ответ модели в аккуратный и безопасный HTML для Telegram.

    Делает:
    - Вставляет пустые строки после коротких заголовков (с учётом скобок, напр. 'Краткий ответ(Итого)')
    - Ставит пробел после нумерации (2)Текст -> 2) Текст)
    - Разрывает абзацы перед нумерацией и служебными заголовками
    - Преобразует [текст](url) в <a href="...">текст</a>
    - Делает жирными заголовки и короткие нумерованные тайтлы
    - Нормализует маркеры списков (•/—/- → «— »)
    - Санитизирует HTML под Telegram (b,i,u,s,code,pre,a,br)
    """
    if not raw:
        return ""

    # Если уже есть допустимые HTML-теги — просто санитизируем
    if "<" in raw and re.search(r"<\s*(?:b|i|u|s|code|pre|a|br)\b", raw, re.IGNORECASE):
        return sanitize_telegram_html(raw)

    # Нормализуем переносы строк
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 0) Принудительно добавим пробел после нумерации, если его нет: "2)Текст" -> "2) Текст"
    text = re.sub(r"(^|\n)(\d+[\.)])(?=\S)", r"\1\2 ", text)

    # 1) Вставим пустую строку ПОСЛЕ коротких заголовков (включая необязательные скобки сразу после фразы)
    #    Примеры: "1) Краткий ответ(Итого)", "2) Подробный разбор ситуации", "3) Примеры из судебной практики"
    short_header_re = re.compile(
        r"""
        (^|\n)                                   # старт строки/абзаца
        (?:\d+\s*[\.)]\s*)?                      # необязательная нумерация "1) " или "2. "
        (?P<title>                               # сама фраза заголовка
            Краткий\ ответ
            |Подробный\ разбор(?:\s*ситуации)?  # "Подробный разбор" (+ необяз. "ситуации")
            |Примеры\ из\ судебной\ практики
        )
        (?P<paren>\s*\([^)\n]{0,40}\))?          # необязательная скобочная приписка сразу после заголовка, напр. "(Итого)"
        \s*(?=\S)                                # и дальше сразу идёт текст без переноса — значит, надо вставить разрыв
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    text = short_header_re.sub(lambda m: f"{m.group(1)}{m.group(0).strip()}\n\n", text)

    # 2) Паузы перед нумерацией (1) 2) 1. 2.)
    text = re.sub(r"(?<!\n)(?=\b\d+[\.)]\s+)", "\n\n", text)

    # 3) Паузы перед служебными заголовками (TL;DR, Коротко, Итоги, Резюме, План, Что делать…)
    text = re.sub(
        r"(?<!\n)(?=\b(?:TL;DR|ТЛ;ДР|Коротко|Итоги|Резюме|План|Что делать|Подробный разбор|Примеры из судебной практики)\b)",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    lines = text.split("\n")
    out: list[str] = []
    prev_empty = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                out.append("")
                prev_empty = True
            continue
        prev_empty = False

        # Ссылки [text](url) -> <a>
        html_line = _md_links_to_anchors(stripped)

        # Маркеры списка •/—/- → «— » (если строка начинается с маркера)
        if re.match(r"^\s*(?:•|—|-)\s+", stripped):
            html_line = re.sub(r"^\s*(?:•|—|-)\s*", "— ", html_line)
            out.append(html_line)
            continue

        # Заголовки
        is_heading = (
            stripped.endswith(":")
            or re.match(r"^(?:tl;dr|тл;др|коротко|итоги|резюме|план|что делать)\b", stripped, re.IGNORECASE)
        )
        is_numbered = re.match(r"^\d+[\.)]\s+", stripped) is not None
        is_short_numbered_title = bool(
            is_numbered
            and len(stripped) <= 80
            and re.search(r"(кратк|разбор|пример|итог|резюме|план)", stripped, re.IGNORECASE)
        )

        if is_heading or is_short_numbered_title:
            out.append(f"<b>{html_line}</b>")
            out.append("")  # пустая строка после заголовка
            continue

        # Обычная строка
        out.append(html_line)

    # Переводы строк -> <br>
    html = "\n".join(out)
    html = html.replace("\n\n", "<br><br>").replace("\n", "<br>")

    # Итоговая санация
    return sanitize_telegram_html(html)

