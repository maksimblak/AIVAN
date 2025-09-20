"""
UI компоненты для Telegram бота ИИ-Иван
Содержит клавиатуры, эмодзи, шаблоны сообщений
"""

from __future__ import annotations
from typing import Optional
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

def format_legal_response(text: str, category: Optional[str] = None) -> str:
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
