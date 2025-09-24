"""
UI компоненты для Telegram бота ИИ-Иван
Содержит клавиатуры, эмодзи, шаблоны сообщений
"""

from __future__ import annotations

# Callback классы больше не нужны без inline клавиатур

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
    """Цвета для форматирования (не используются напрямую в Telegram, но для документации)"""
    PRIMARY = "#2196F3"    # Синий
    SUCCESS = "#4CAF50"    # Зеленый
    WARNING = "#FF9800"    # Оранжевый
    ERROR = "#F44336"      # Красный
    INFO = "#00BCD4"       # Голубой

# ============ ШАБЛОНЫ СООБЩЕНИЙ ============

class MessageTemplates:
    """Шаблоны сообщений с красивым форматированием"""

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
        f"{Emoji.MAGIC} Финализирую рекомендации..."
    ]

    ERROR_GENERIC = f"""{Emoji.ERROR} **Произошла ошибка**

К сожалению, не удалось обработать ваш запрос\\.

{Emoji.HELP} *Рекомендации:*
• Проверьте формулировку вопроса
• Попробуйте через несколько минут
• Обратитесь к администратору если проблема повторяется"""

    NO_QUESTION = f"""{Emoji.WARNING} **Пустой запрос**

Пожалуйста, отправьте текст юридического вопроса\\."""

# ============ КЛАВИАТУРЫ УБРАНЫ ============
# Все inline клавиатуры удалены по запросу пользователя

# ============ КАТЕГОРИИ ПРАВА ============

LEGAL_CATEGORIES = {
    "civil": {
        "name": "Гражданское право",
        "emoji": Emoji.CIVIL,
        "description": "Имущественные и личные неимущественные отношения",
        "examples": ["Договоры", "Собственность", "Обязательства", "Деликты"]
    },
    "corporate": {
        "name": "Корпоративное право",
        "emoji": Emoji.CORPORATE,
        "description": "Создание и деятельность юридических лиц",
        "examples": ["Учреждение ООО", "Корпоративные споры", "Реорганизация", "M&A"]
    },
    "contract": {
        "name": "Договорное право",
        "emoji": Emoji.CONTRACT,
        "description": "Заключение, исполнение и расторжение договоров",
        "examples": ["Поставка", "Подряд", "Аренда", "Займ"]
    },
    "labor": {
        "name": "Трудовое право",
        "emoji": Emoji.LABOR,
        "description": "Трудовые отношения и социальная защита",
        "examples": ["Увольнение", "Зарплата", "Отпуска", "Дисциплина"]
    },
    "tax": {
        "name": "Налоговое право",
        "emoji": Emoji.TAX,
        "description": "Налогообложение и взаимодействие с ФНС",
        "examples": ["НДС", "Налог на прибыль", "НДФЛ", "Проверки"]
    },
    "real_estate": {
        "name": "Право недвижимости",
        "emoji": Emoji.REAL_ESTATE,
        "description": "Сделки с недвижимостью и земельными участками",
        "examples": ["Купля-продажа", "Аренда", "Ипотека", "Кадастр"]
    },
    "ip": {
        "name": "Интеллектуальная собственность",
        "emoji": Emoji.IP,
        "description": "Авторские права, товарные знаки, патенты",
        "examples": ["Регистрация ТЗ", "Авторские права", "Патенты", "Лицензии"]
    },
    "admin": {
        "name": "Административное право",
        "emoji": Emoji.ADMIN,
        "description": "Взаимодействие с госорганами и административная ответственность",
        "examples": ["Лицензирование", "Штрафы", "Госуслуги", "Контроль"]
    },
    "criminal": {
        "name": "Уголовное право",
        "emoji": Emoji.CRIMINAL,
        "description": "Преступления и уголовная ответственность",
        "examples": ["Экономические преступления", "Должностные", "Налоговые", "Защита"]
    },
    "family": {
        "name": "Семейное право",
        "emoji": Emoji.FAMILY,
        "description": "Брак, развод, алименты, опека",
        "examples": ["Развод", "Алименты", "Раздел имущества", "Опека"]
    }
}

def get_category_info(category_id: str) -> dict:
    """Получить информацию о категории права"""
    return LEGAL_CATEGORIES.get(category_id, {
        "name": "Неизвестная категория",
        "emoji": Emoji.LAW,
        "description": "Общие правовые вопросы",
        "examples": []
    })

# ============ ФОРМАТИРОВАНИЕ ============

def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для MarkdownV2"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def format_legal_response(text: str, category: str | None = None) -> str:
    """Форматирует ответ с красивой разметкой MarkdownV2"""

    # Заголовок с категорией
    if category:
        category_info = get_category_info(category)
        header = f"{category_info['emoji']} **{escape_markdown_v2(category_info['name'])}**\n\n"
        text = header + text


    return text

def create_progress_message(stage: int, total: int = 4) -> str:
    """Создает сообщение с прогрессом"""
    if stage >= len(MessageTemplates.PROCESSING_STAGES):
        stage = len(MessageTemplates.PROCESSING_STAGES) - 1

    progress_bar = "▓" * stage + "░" * (total - stage)
    percentage = int((stage / total) * 100)

    return f"{MessageTemplates.PROCESSING_STAGES[stage]}\n\n`{progress_bar}` {percentage}%"

# ============ HTML ФОРМАТИРОВАНИЕ ДЛЯ STREAMING ============

import re
from html import escape as html_escape


def _md_links_to_anchors(line: str) -> str:
    """Convert markdown links [text](url) into safe HTML anchors.

    Both link text and URL are escaped; only http/https URLs are allowed.
    """
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    result_parts: list[str] = []
    last = 0
    for m in pattern.finditer(line):
        # escape non-link part
        result_parts.append(html_escape(line[last:m.start()]))
        text = html_escape(m.group(1))
        url = html_escape(m.group(2), quote=True)
        result_parts.append(f'<a href="{url}">{text}</a>')
        last = m.end()
    # tail
    result_parts.append(html_escape(line[last:]))
    return ''.join(result_parts)

def sanitize_telegram_html(raw: str) -> str:
    """Allow only Telegram-supported HTML tags; escape the rest.

    Allowed: b, i, u, s, code, pre, a[href=http/https], br
    """
    if not raw:
        return ""
    # Start from fully escaped text
    esc = html_escape(raw, quote=True)
    # Restore <br>, <br/>, <br />
    esc = re.sub(r"&lt;br\s*/?&gt;", "<br>", esc, flags=re.IGNORECASE)
    # Restore simple tags exactly
    for tag in ("b", "i", "u", "s", "code", "pre"):
        esc = re.sub(fr"&lt;{tag}&gt;", fr"<{tag}>", esc, flags=re.IGNORECASE)
        esc = re.sub(fr"&lt;/{tag}&gt;", fr"</{tag}>", esc, flags=re.IGNORECASE)
    # Restore anchors with http(s) only; keep entities like &amp; inside href
    esc = re.sub(r"&lt;a href=&quot;(https?://[^&quot;]+)&quot;&gt;", r'<a href="\1">', esc, flags=re.IGNORECASE)
    esc = re.sub(r"&lt;/a&gt;", "</a>", esc, flags=re.IGNORECASE)
    return esc

def render_legal_html(raw: str) -> str:
    """Beautify plain model text into simple, safe HTML.

    - Escapes HTML by default
    - Converts [text](url) markdown links to <a>
    - Bolds headings (lines ending with ':' or starting with 'N) ' or 'TL;DR')
    - Normalizes bullets (leading '-', '—', '•') to an em dash '— '
    - Replaces newlines with <br>
    """
    if not raw:
        return ""

    # If looks like HTML from the model, sanitize and keep structure
    if '<' in raw and re.search(r"<\s*(b|i|u|s|code|pre|a|br)\b", raw, re.IGNORECASE):
        return sanitize_telegram_html(raw)

    def _auto_paragraph_breaks(text: str) -> str:
        # Normalize spaces but preserve intentional structure
        t = re.sub(r"[ \t]+", " ", text)  # Only normalize spaces/tabs, keep newlines

        # Insert breaks before numbered items like "1) ", "2) ", "1.", "2."
        t = re.sub(r"(?<!\n)(?=\b\d+[\.)]\s)", "\n\n", t)

        # Insert breaks before section markers
        t = re.sub(r"(?<!\n)(?=\b(?:Коротко|Далее|Вариант|Итак|Резюме|Заключение)\b)", "\n\n", t)

        # Break after sentence end before em dash bullets or numbers
        t = re.sub(r"(?<=[\.!?])\s+(?=(?:—|•|-|\d+[\.)]\s))", "\n", t)

        # NEW: Break before em dashes that start new thoughts (после точки, скобки или в начале предложения)
        t = re.sub(r"(?<=[\.!?\)])\s+(?=—\s+[А-ЯA-Z])", "\n", t)

        # NEW: Break before em dashes in middle of text that indicate new bullet points
        t = re.sub(r"(?<=\.)\s+(?=—\s+[А-ЯA-Zа-я])", "\n", t)

        # Insert breaks before article references like "ст. 304", "Статья 222"
        t = re.sub(r"(?<=[\.!?])\s+(?=(?:—\s*)?(?:ст\.|Статья)\s*\d+)", "\n", t)

        # Break long sentences with semicolons into separate lines
        t = re.sub(r";\s+(?=и\s+\d+\))", ";\n— ", t)

        return t

    text = raw.replace('\r\n', '\n').replace('\r', '\n')

    # Always apply auto paragraph breaks for better structure
    text = _auto_paragraph_breaks(text)

    lines = text.split('\n')
    out: list[str] = []

    prev_was_empty = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle empty lines - create paragraph breaks
        if stripped == "":
            if not prev_was_empty:  # Avoid multiple consecutive breaks
                out.append("<br><br>")
                prev_was_empty = True
            continue

        prev_was_empty = False

        # Enhanced bullet detection
        if re.match(r"^\s*[-•—]\s+", line):
            line = re.sub(r"^\s*[-•—]\s+", "— ", line)

        # Transform md links and escape other parts FIRST
        html_line = _md_links_to_anchors(line)

        # Check if this is a numbered list item
        is_numbered_item = re.match(r"^\s*\d+[\.)]\s+", stripped)
        if is_numbered_item:
            html_line = re.sub(r"(\d+[\.)]\s+)", r"<b>\1</b>", html_line)

        # Enhanced heading detection (исключаем нумерованные элементы)
        is_heading = (
            stripped.endswith(":") and not is_numbered_item or
            stripped.upper().startswith(("КОРОТКО", "TL;DR", "РЕЗЮМЕ", "ЗАКЛЮЧЕНИЕ"))
        )

        # Special formatting for article references AFTER escaping
        if re.search(r"\b(?:ст\.|Статья)\s*\d+", stripped):
            html_line = re.sub(r"(\b(?:ст\.|Статья)\s*\d+[^\s]*)", r"<b>\1</b>", html_line)

        # Check if this line should start a new paragraph
        is_paragraph_start = (
            is_heading or
            is_numbered_item or
            re.match(r"^\s*[-•—]\s+", stripped) or     # Bullet point
            (i > 0 and lines[i-1].strip() == "")       # After empty line
        )

        if is_heading:
            html_line = f"<b>{html_line}</b>"
            out.append(html_line + "<br><br>")
        elif is_paragraph_start and out and not out[-1].endswith("<br><br>"):
            # Add paragraph break before this line if needed
            out.append("<br>" + html_line + "<br>")
        else:
            out.append(html_line + "<br>")

    # Clean up multiple breaks and ensure proper paragraph separation
    html_result = ''.join(out)

    # Remove excessive breaks (more than 2 consecutive) but keep paragraph structure
    html_result = re.sub(r"(?:<br>\s*){3,}", "<br><br>", html_result)

    # Clean up trailing breaks
    html_result = re.sub(r"(?:<br>\s*)+$", "", html_result)

    return html_result
