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
    """
    Универсальный форматтер «сырого» текста модели в аккуратный безопасный HTML для Telegram.

    Делает:
    - Экранирует HTML по умолчанию
    - Преобразует markdown-ссылки [текст](url) в <a>
    - Ставит переносы строк:
        • после концов предложений (. ! ? …), если дальше идёт заглавная/цифра/буллет/скобка
        • перед элементами нумерации: 1) / 2. / a) / IV. (в т.ч. слепленными без пробела)
        • перед буллетами — • - — (если они «прилипли»)
        • после коротких «заголовков» (строки до 80 симв., оканчивающиеся на :)
    - Выделяет жирным короткие нумерованные заголовки (не содержат точки в конце)
    - Нормализует маркеры списка к «— »
    - Заменяет переводы строк на <br>, чистит лишние <br>
    """
    if not raw:
        return ""

    # Если уже пришёл HTML с допустимыми тегами — лишь санитизируем
    if "<" in raw and re.search(r"<\s*(?:b|i|u|s|code|pre|a|br)\b", raw, re.IGNORECASE):
        return sanitize_telegram_html(raw)

    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 0) Чуть подчистим пробелы
    text = re.sub(r"[ \t]+", " ", text)

    # 1) Починим «слепленные» нумераторы: 1)Коротко → 1) Коротко (и 2.Пример → 2. Пример)
    text = re.sub(r"(?<=\b\d{1,3}[\.)])(?=\S)", " ", text)                    # 1)/2./10)
    text = re.sub(r"(?<=\b[IVXLCMivxlcm]{1,5}[\.)])(?=\S)", " ", text)        # I)/IV.
    text = re.sub(r"(?<=\b[A-Za-zА-Яа-яЁё][\.)])(?=\S)", " ", text)           # a)/б)

    # 2) Перенос после конца предложения, если дальше начинается «новая мысль»
    #    (буллет/заглавная/цифра/скобка). Не трогаем внутри чисел/сокращений.
    text = re.sub(
        r"(?<=[\.\!\?…])\s+(?=[—•\-A-ZА-ЯЁ0-9\(\[])",
        "\n",
        text,
    )

    # 3) Новая строка перед буллетами, если «прилипли»
    text = re.sub(r"(?<!\n)\s*(?=—\s+)", "\n", text)            # перед «— »
    text = re.sub(r"(?<!\n)\s*(?=•\s+)", "\n", text)            # перед «• »
    text = re.sub(r"(?<!\n)\s*(?=-\s+)", "\n", text)            # перед «- »

    # 4) Новая строка перед нумераторами (цифры/римские/буквенные), если «прилипли»
    enum_lookahead = r"(?=(?:\(?\s*(?:\d{1,3}|[IVXLCMivxlcm]{1,5}|[A-Za-zА-Яа-яЁё])[\.)])\s)"
    text = re.sub(rf"(?<!\n)\s*{enum_lookahead}", "\n\n", text)

    # 5) Избыточные пустые строки → максимум две
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6) Разобьём на строки для пост-обработки
    lines = text.split("\n")
    out: list[str] = []
    prev_empty = False

    for line in lines:
        raw_line = line
        stripped = raw_line.strip()

        # пустые
        if not stripped:
            if not prev_empty:
                out.append("<br>")
            prev_empty = True
            continue
        prev_empty = False

        # Преобразуем markdown-ссылки и экранируем остальной HTML
        html_line = _md_links_to_anchors(stripped)

        # Нормализуем маркеры списка к «— »
        if re.match(r"^(?:[-•—])\s+", stripped):
            html_line = re.sub(r"^(?:[-•—])\s*", "— ", html_line)

        # Короткие заголовки: <= 80 символов и оканчиваются двоеточием
        is_colon_heading = stripped.endswith(":") and len(stripped) <= 80

        # Нумерованные заголовки: "1) Текст" / "2. Текст" / "IV. Текст" / "а) Текст"
        is_numbered = bool(re.match(
            r"^(?:\(?\s*(?:\d{1,3}|[IVXLCMivxlcm]{1,5}|[A-Za-zА-Яа-яЁё])[\.)])\s+\S", stripped
        ))
        short_enumerated_title = is_numbered and (len(stripped) <= 80) and not stripped.endswith(".")

        # Если это «заголовок» — делаем жирным и добавляем пустую строку после
        if is_colon_heading or short_enumerated_title:
            # слегка подсветим сам префикс нумерации
            if is_numbered:
                html_line = re.sub(
                    r"^(\(?\s*(?:\d{1,3}|[IVXLCMivxlcm]{1,5}|[A-Za-zА-Яа-яЁё])[\.)])\s+",
                    r"<b>\1</b> ",
                    html_line,
                )
            out.append(f"<b>{html_line}</b><br><br>")
            continue

        # Обычная строка
        out.append(html_line + "<br>")

    # Соберём и подчистим лишние <br>
    html = "".join(out)
    html = re.sub(r"(?:<br>\s*){3,}", "<br><br>", html)
    html = re.sub(r"(?:<br>\s*)+$", "", html)

    # Финальная санация под Telegram
    return sanitize_telegram_html(html)


