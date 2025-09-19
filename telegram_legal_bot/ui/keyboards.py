# telegram_legal_bot/ui/keyboards.py
"""
Красивые клавиатуры и интерактивные элементы для телеграм-бота.
"""

from aiogram import types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from typing import List, Optional


class BotKeyboards:
    """Класс для создания различных клавиатур бота."""
    
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        """Главное меню бота."""
        keyboard = [
            [
                InlineKeyboardButton(text="❓ Задать вопрос", callback_data="ask_question"),
                InlineKeyboardButton(text="📚 Категории права", callback_data="law_categories")
            ],
            [
                InlineKeyboardButton(text="👤 Профиль", callback_data="profile"),
                InlineKeyboardButton(text="📊 Статистика", callback_data="stats")
            ],
            [
                InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help"),
                InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def law_categories() -> InlineKeyboardMarkup:
        """Меню категорий права."""
        keyboard = [
            [
                InlineKeyboardButton(text="🏠 Гражданское право", callback_data="category_civil"),
                InlineKeyboardButton(text="⚖️ Уголовное право", callback_data="category_criminal")
            ],
            [
                InlineKeyboardButton(text="💼 Трудовое право", callback_data="category_labor"),
                InlineKeyboardButton(text="🏢 Налоговое право", callback_data="category_tax")
            ],
            [
                InlineKeyboardButton(text="🚗 Административное", callback_data="category_admin"),
                InlineKeyboardButton(text="👥 Семейное право", callback_data="category_family")
            ],
            [
                InlineKeyboardButton(text="🔙 Назад", callback_data="main_menu")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def help_menu() -> InlineKeyboardMarkup:
        """Меню помощи."""
        keyboard = [
            [
                InlineKeyboardButton(text="🚀 Быстрый старт", callback_data="quick_start"),
                InlineKeyboardButton(text="💡 Как задать вопрос", callback_data="how_to_ask")
            ],
            [
                InlineKeyboardButton(text="📋 Примеры вопросов", callback_data="examples"),
                InlineKeyboardButton(text="⚠️ Ограничения", callback_data="limitations")
            ],
            [
                InlineKeyboardButton(text="📞 Связаться с поддержкой", callback_data="contact"),
                InlineKeyboardButton(text="🔙 Назад", callback_data="main_menu")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def question_actions(question_id: Optional[str] = None) -> InlineKeyboardMarkup:
        """Действия после получения ответа на вопрос."""
        data_suffix = f"_{question_id}" if question_id else ""
        keyboard = [
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=f"rate_good{data_suffix}"),
                InlineKeyboardButton(text="👎 Не помогло", callback_data=f"rate_bad{data_suffix}")
            ],
            [
                InlineKeyboardButton(text="🔍 Уточнить вопрос", callback_data=f"clarify{data_suffix}"),
                InlineKeyboardButton(text="📋 Похожие вопросы", callback_data=f"similar{data_suffix}")
            ],
            [
                InlineKeyboardButton(text="🏠 Главное меню", callback_data="main_menu")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def confirm_action(action: str, data: str) -> InlineKeyboardMarkup:
        """Подтверждение действий."""
        keyboard = [
            [
                InlineKeyboardButton(text="✅ Да", callback_data=f"confirm_{action}_{data}"),
                InlineKeyboardButton(text="❌ Нет", callback_data=f"cancel_{action}")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def settings_menu() -> InlineKeyboardMarkup:
        """Меню настроек."""
        keyboard = [
            [
                InlineKeyboardButton(text="🔔 Уведомления", callback_data="settings_notifications"),
                InlineKeyboardButton(text="🎨 Тема", callback_data="settings_theme")
            ],
            [
                InlineKeyboardButton(text="📝 Формат ответов", callback_data="settings_format"),
                InlineKeyboardButton(text="🌍 Язык", callback_data="settings_language")
            ],
            [
                InlineKeyboardButton(text="🗑 Очистить историю", callback_data="clear_history"),
                InlineKeyboardButton(text="🔙 Назад", callback_data="main_menu")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def rate_limit_exceeded() -> InlineKeyboardMarkup:
        """Клавиатура при превышении лимита."""
        keyboard = [
            [
                InlineKeyboardButton(text="⏰ Когда можно снова?", callback_data="check_rate_limit"),
                InlineKeyboardButton(text="💎 Premium", callback_data="upgrade_premium")
            ],
            [
                InlineKeyboardButton(text="🏠 Главное меню", callback_data="main_menu")
            ]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def back_button(callback_data: str = "main_menu") -> InlineKeyboardMarkup:
        """Простая кнопка назад."""
        keyboard = [
            [InlineKeyboardButton(text="🔙 Назад", callback_data=callback_data)]
        ]
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    @staticmethod
    def typing_indicators() -> List[str]:
        """Анимированные индикаторы печати."""
        return [
            "🤔 Анализирую ваш вопрос...",
            "📚 Изучаю правовую базу...", 
            "⚖️ Формирую юридический ответ...",
            "✍️ Оформляю решение..."
        ]
    
    @staticmethod
    def loading_animation() -> List[str]:
        """Анимация загрузки."""
        return [
            "⏳ Загрузка",
            "⏳ Загрузка.",
            "⏳ Загрузка..",
            "⏳ Загрузка..."
        ]
