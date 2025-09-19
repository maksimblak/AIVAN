# telegram_legal_bot/utils/markdown_v2.py
"""
Правильное форматирование для Telegram MarkdownV2.
"""

import re


def format_md2_message(text: str) -> str:
    """
    Форматирует сообщение с правильным MarkdownV2 синтаксисом.
    Поддерживает:
    - **жирный текст**  
    - *курсивный текст*
    - `моноширинный текст`
    - [ссылки](URL)
    
    Автоматически экранирует остальные спецсимволы.
    """
    if not text:
        return ""
    
    # Сначала заменяем форматирование на временные маркеры
    text = re.sub(r'\*\*(.*?)\*\*', '__BOLD_START__\\1__BOLD_END__', text)  # **bold**
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)', '__ITALIC_START__\\1__ITALIC_END__', text)  # *italic*
    text = re.sub(r'`([^`]+)`', '__CODE_START__\\1__CODE_END__', text)  # `code`
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', '__LINK_START__\\1__LINK_MID__\\2__LINK_END__', text)  # [text](url)
    
    # Экранируем все спецсимволы MarkdownV2
    special_chars = r'_*[]()~`>#+-=|{}.!\\'
    for char in special_chars:
        if char not in ['*', '_', '`', '[', ']', '(', ')']:  # Не экранируем символы форматирования
            text = text.replace(char, '\\' + char)
    
    # Возвращаем правильное форматирование
    text = text.replace('__BOLD_START__', '*')
    text = text.replace('__BOLD_END__', '*')
    text = text.replace('__ITALIC_START__', '_') 
    text = text.replace('__ITALIC_END__', '_')
    text = text.replace('__CODE_START__', '`')
    text = text.replace('__CODE_END__', '`')
    text = text.replace('__LINK_START__', '[')
    text = text.replace('__LINK_MID__', '](')
    text = text.replace('__LINK_END__', ')')
    
    return text


def bold(text: str) -> str:
    """Жирный текст для MarkdownV2."""
    return f"*{escape_md2_text(text)}*"


def italic(text: str) -> str:
    """Курсивный текст для MarkdownV2.""" 
    return f"_{escape_md2_text(text)}_"


def code(text: str) -> str:
    """Моноширинный текст для MarkdownV2."""
    return f"`{text}`"  # Внутри ` экранирование не нужно


def link(text: str, url: str) -> str:
    """Ссылка для MarkdownV2."""
    escaped_url = url.replace('(', r'\(').replace(')', r'\)')
    return f"[{escape_md2_text(text)}]({escaped_url})"


def escape_md2_text(text: str) -> str:
    """Экранирует текст для использования внутри форматирования MarkdownV2."""
    # Список символов которые нужно экранировать в MarkdownV2
    special_chars = r'_*[]()~`>#+-=|{}.!\\'
    
    result = text
    for char in special_chars:
        result = result.replace(char, '\\' + char)
    
    return result


def create_message_template() -> str:
    """Создает шаблон сообщения с правильным форматированием."""
    return """
👋 {bold_welcome}

Я — {bold_name}, ваш персональный юридический ассистент ⚖️

🚀 {bold_abilities}
• Отвечаю на правовые вопросы по РФ
• Привожу ссылки на актуальные нормы
• Даю практические рекомендации  
• Помогаю с документооборотом

⭐ {bold_advantages}
• Быстрые и точные ответы
• Актуальная правовая база
• Понятное объяснение сложного
• Доступен 24/7

⚠️ {bold_important}
Я не заменяю консультацию с юристом, но помогу сориентироваться в правовых вопросах{escaped_exclamation}

Выберите действие в меню ниже 👍
    """.strip()


def format_welcome_message(user_name: str = None) -> str:
    """Создает правильно отформатированное приветственное сообщение."""
    name_part = f", {user_name}" if user_name else ""
    
    template = create_message_template()
    
    return template.format(
        bold_welcome=bold(f"Добро пожаловать{name_part}!"),
        bold_name=bold("AIVAN"),
        bold_abilities=bold("Что я умею:"),
        bold_advantages=bold("Мои преимущества:"),
        bold_important=bold("Важно помнить:"),
        escaped_exclamation="\\!"
    )


# Функция для быстрого форматирования часто используемых элементов
def format_section(title: str, items: list, emoji: str = "•") -> str:
    """Форматирует секцию с заголовком и списком."""
    result = f"\n{bold(title)}\n"
    for item in items:
        result += f"{emoji} {escape_md2_text(item)}\n"
    return result
