"""
UI компоненты для Telegram бота ИИ-Иван
Содержит клавиатуры, эмодзи, шаблоны сообщений и форматирование
"""

from __future__ import annotations
import re
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

# --- простая линковка URL ---
_URL_RE = re.compile(r"(https?://[^\s<>{}]+)", re.IGNORECASE)

# --- стоп-слова (RU/EN) для авто-ключей ---
_STOPWORDS = {
    # ru
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было",
    "вот","от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
    "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
    "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где",
    "есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто",
    "чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому",
    "этого","какой","совсем","ним","здесь","этом","один","почти","мой","тем","чтобы",
    "нее","кажется","сейчас","были","куда","зачем","всех","никогда","можно","при",
    "наконец","два","об","другой","хоть","после","над","больше","тот","через",
    # en
    "the","and","or","to","of","a","an","in","on","for","with","by","at","as","is",
    "are","was","were","be","been","it","this","that","from","not","no","yes","but",
}
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
ALLOWED_TAGS = {"b","strong","i","em","u","s","del","code","pre","a","br"}
ALLOWED_ATTRS = {"a": {"href"}}


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


def _normalize(text: str) -> str:
    return (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")

def _is_heading(raw_line: str) -> bool:
    s = raw_line.strip()
    if not s:
        return False
    # "1) Заголовок" / "1. Заголовок"
    if re.match(r"^\d+[\.\)]\s+\S", s):
        return True
    # Короткая строка, заканчивающаяся двоеточием
    if len(s) <= 80 and s.endswith(":"):
        return True
    # Короткая фраза без точки в конце — вероятный заголовок
    if len(s) <= 70 and not s.endswith((".", "!", "?")) and len(s.split()) <= 10:
        return True
    # Почти все буквы — верхний регистр
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        up = sum(ch.isupper() for ch in letters)
        if up / len(letters) >= 0.6 and len(s) <= 80:
            return True
    return False

def _auto_keywords(text: str, limit: int = 8) -> list[str]:
    """
    Очень лёгкая эвристика: частотные слова/биграммы длиной >=4,
    без стоп-слов. Работает для любого языка с кириллицей/латиницей.
    """
    # упрощённая токенизация
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]{2,}", text, flags=re.UNICODE)
    norm = [w.lower() for w in words]
    # счёт частот
    freq = {}
    for w in norm:
        if len(w) < 4 or w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    # биграммы
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i+1]
        if len(a) < 4 or len(b) < 4 or a in _STOPWORDS or b in _STOPWORDS:
            continue
        big = f"{a} {b}"
        freq[big] = freq.get(big, 0) + 1

    if not freq:
        return []
    # топ по частоте/длине
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    keys = []
    seen_roots = set()
    for k, _ in ranked:
        root = k.split()[0][:6]  # примерно убираем дубль по корню
        if root in seen_roots:
            continue
        keys.append(k)
        seen_roots.add(root)
        if len(keys) >= limit:
            break
    return keys

def _linkify(text_escaped: str) -> str:
    return _URL_RE.sub(lambda m: f'<a href="{html_escape(m.group(1), quote=True)}">{html_escape(m.group(1))}</a>', text_escaped)

def render_legal_html(
    raw_text: str,
    *,
    underline_keywords: list[str] | None = None,
    auto_keywords: bool = True,
    max_keywords: int = 8,
) -> str:
    """
    Универсальный рендер для ЛЮБОГО текста (без словарей):
    - авто-заголовки (по форме строки),
    - разрыв встроенных списков: -, •, *, 1), 1. (даже если стоят после слова),
    - разрыв склеек типа "...словоСледующее" (буква + Заглавная + строчная),
    - жирнение "Термин: пояснение",
    - markdown-ссылки и автоссылки,
    - аккуратные абзацы <br><br>.
    """
    t = _normalize(raw_text)
    if not t:
        return "—"

    # ---------- СТРУКТУРНЫЕ РАЗРЫВЫ ----------
    # 1) После знаков препинания начинается маркер списка -> перенос
    t = re.sub(r'([:;.,)\]])\s*(?=[\-–—*•]\s+)', r'\1\n', t)
    # 2) После тех же знаков начинается нумерация "1) " / "1. " -> перенос
    t = re.sub(r'([:;.,)\]])\s*(?=\d+[\.\)]\s+)', r'\1\n', t)
    # 3) Несколько пробелов и далее маркер -> перенос
    t = re.sub(r'(?<!^)\s{2,}(?=[\-–—*•]\s+)', '\n', t)
    # 4) Случай "…слово- Пункт": перенос перед "- " если дальше Заглавная (чтобы "Краткий ответ- Пункт" не слипался)
    t = re.sub(r'(?<=\S)\s*(?=-\s+[A-ZА-ЯЁ])', '\n', t)
    # 5) Встроенные "• " без препинания перед ними -> перенос
    t = re.sub(r'(?<!^)(?<!\n)\s*(?=•\s+)', '\n', t)
    # 6) Разрыв склеек вида "…ответДа," (буква + Заглавная + строчная)
    t = re.sub(r'(?<=[A-Za-zА-Яа-яЁё])(?=[A-ZА-ЯЁ][a-zа-яё])', '\n', t)
    # -----------------------------------------

    # автоключи (по желанию)
    ukeys = underline_keywords
    if ukeys is None and auto_keywords:
        ukeys = _auto_keywords(t, limit=max_keywords)
    if ukeys is None:
        ukeys = []

    lines = t.split("\n")
    out: list[str] = []
    prev_blank = True

    for line in lines:
        raw = line.rstrip()

        # пустая строка -> абзац
        if not raw:
            if not prev_blank:
                out.append("<br>")
            prev_blank = True
            continue

        s = raw

        # Заголовки по форме (короткая строка / "1) ..."/"1. ..."/ строка оканчивается ":")
        if _is_heading(s):
            s2 = re.sub(r"^\d+[\.\)]\s+", "", s).strip()
            out.append(f"<b>{html_escape(s2)}</b>")
            out.append("<br>")  # небольшой «воздух» после заголовка
            prev_blank = False
            continue

        # Списки / нумерация
        bullet = ""
        m_num = re.match(r"^(\d+[\.\)])\s+(.*)$", s)
        if m_num:
            bullet = f"{m_num.group(1)} "
            s = m_num.group(2)
        elif re.match(r"^([\-–—*•])\s+(.*)$", s):
            s = re.sub(r"^([\-–—*•])\s+", "", s, count=1)
            bullet = "• "

        # Экранируем HTML
        e = html_escape(s)

        # "Термин: пояснение" -> жирный термин
        m_term = re.match(r"^([^:\n]{3,80}:\s*)(.*)$", e)
        if m_term:
            e = f"<b>{m_term.group(1)}</b>{m_term.group(2)}"

        # (опционально) подчёркивание автоключей — это частотные формы, не словарь
        for kw in ukeys:
            if not kw:
                continue
            safe = html_escape(kw)
            e = re.sub(
                rf"(?iu)(?<![A-Za-zА-Яа-яЁё0-9\-])({re.escape(safe)})(?![A-Za-zА-Яа-яЁё0-9\-])",
                r"<u>\1</u>",
                e,
            )

        # Markdown-ссылки -> <a>
        e = re.sub(
            r"\[([^\]]{1,200})\]\((https?://[^\s)]+)\)",
            lambda m: f'<a href="{html_escape(m.group(2), quote=True)}">{html_escape(m.group(1))}</a>',
            e,
        )
        # Автоссылки
        e = _linkify(e)

        out.append((bullet + e).strip())
        prev_blank = False

    # Склейка + нормализация абзацев
    html_parts: list[str] = []
    for piece in out:
        if piece == "<br>":
            if html_parts and html_parts[-1] != "<br><br>":
                html_parts.append("<br><br>")
            continue
        if html_parts:
            html_parts.append("<br>")
        html_parts.append(piece)

    result = "".join(html_parts).strip()
    result = re.sub(r"(?:<br>){3,}", "<br><br>", result)
    return result or "—"
