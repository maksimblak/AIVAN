# telegram_legal_bot/handlers/ui_demo.py
"""
Демонстрационные команды для нового UI.
"""

from __future__ import annotations

from aiogram import Router, types, F
from aiogram.filters import Command

from telegram_legal_bot.utils.message_formatter import md2
from telegram_legal_bot.ui.messages import BotMessages
from telegram_legal_bot.ui.animations import BotAnimations
from telegram_legal_bot.ui.user_profile import user_profile_manager

router = Router(name="ui_demo")


@router.message(Command("demo"))
async def cmd_demo(msg: types.Message) -> None:
    """Демонстрация нового UI."""
    try:
        demo_text = """
🎨 **ДЕМОНСТРАЦИЯ НОВОГО UI**

Добро пожаловать в обновленный AIVAN! 

✨ **Новые возможности:**
• Красивые интерактивные меню
• Анимации и эффекты
• Система профилей и достижений  
• Улучшенные сообщения об ошибках
• Прогресс-индикаторы
• Обратная связь и оценки

🚀 **Попробуйте функции через кнопки ниже!**
        """.strip()
        
        await msg.answer(
            md2(demo_text),
            parse_mode="MarkdownV2"
        )
    except Exception:
        fallback_text = "🎨 Демонстрация нового UI доступна!"
        await msg.answer(fallback_text, parse_mode=None)


@router.message(Command("animate"))
async def cmd_animate(msg: types.Message) -> None:
    """Демонстрация анимаций."""
    try:
        # Показываем различные анимации
        await BotAnimations.typing_animation(msg, 2.0)
        
        # Анимация загрузки
        await BotAnimations.loading_dots(msg, "⏳ **Загрузка анимации**", 3.0)
        
        # Анимация прогресса
        steps = [
            "🔄 Инициализация...",
            "📊 Загрузка данных...", 
            "🎨 Применение стилей...",
            "✅ **Анимация готова!**"
        ]
        
        await BotAnimations.progress_message(msg, steps, delay=1.5)
        
        # Финальный эффект
        await BotAnimations.celebration_effect(msg, "Анимации работают отлично!")
        
    except Exception:
        await msg.answer("Демонстрация анимаций завершена!")


@router.message(Command("profile_demo"))
async def cmd_profile_demo(msg: types.Message) -> None:
    """Демонстрация системы профилей."""
    try:
        user_id = msg.from_user.id if msg.from_user else 0
        
        # Добавляем тестовые данные
        user_profile_manager.add_question(user_id, "Тестовый вопрос для демо", "civil")
        user_profile_manager.add_feedback(user_id, True)
        
        # Получаем статистику  
        stats = user_profile_manager.get_stats(user_id)
        
        profile_text = BotMessages.profile(stats)
        await msg.answer(
            md2(profile_text),
            parse_mode="MarkdownV2"
        )
        
    except Exception:
        await msg.answer("Демонстрация профиля недоступна")


@router.message(Command("test_format"))
async def cmd_test_format(msg: types.Message) -> None:
    """Тестирование различных форматирований."""
    try:
        # HTML форматирование
        html_text = """
🧪 <b>ТЕСТ HTML ФОРМАТИРОВАНИЯ</b>

✅ <b>Жирный текст</b> - работает
✅ <i>Курсивный текст</i> - работает  
✅ <u>Подчеркнутый текст</u> - работает
✅ <code>Моноширинный текст</code> - работает
✅ <pre>Блок кода</pre> - работает

🎨 <b>Эмодзи и символы:</b>
• Список работает отлично
• Специальные символы: & < > "
• Восклицательные знаки!
• Двоеточия: работают

<b>Вывод:</b> HTML форматирование надежнее MarkdownV2!
        """.strip()
        
        await msg.answer(html_text, parse_mode="HTML")
        
        # MarkdownV2 для сравнения
        md2_text = """
🧪 *ТЕСТ MARKDOWNV2 ФОРМАТИРОВАНИЯ*

✅ *Курсивный текст* \\- работает
✅ `Моноширинный текст` \\- работает
✅ ||Скрытый текст|| \\- работает

⚠️ *Проблемы с символами:*
• Нужно экранировать: \\! \\. \\- \\( \\)
• Сложности с форматированием
• Легко ошибиться в синтаксисе

*Вывод:* MarkdownV2 сложнее в использовании\\!
        """.strip()
        
        await msg.answer(md2_text, parse_mode="MarkdownV2")
        
    except Exception as e:
        await msg.answer(f"Ошибка форматирования: {str(e)}")


@router.message(Command("ui_info"))
async def cmd_ui_info(msg: types.Message) -> None:
    """Информация о новом UI."""
    try:
        info_text = """
ℹ️ **ИНФОРМАЦИЯ О НОВОМ UI**

🎨 **Обновления интерфейса:**

**Клавиатуры:**
• Интерактивные меню
• Быстрая навигация
• Контекстные действия

**Сообщения:**
• Красивое форматирование
• Эмодзи и иконки
• Структурированная подача

**Анимации:**
• Индикаторы прогресса
• Эффекты загрузки
• Плавные переходы

**Профили:**
• Статистика пользователя
• Система достижений
• Персональные настройки

**Обратная связь:**
• Оценка ответов
• Улучшение качества
• Пользовательский опыт

🚀 **Все для удобства использования!**
        """.strip()
        
        await msg.answer(md2(info_text), parse_mode="MarkdownV2")
        
    except Exception:
        fallback_text = """
ℹ️ ИНФОРМАЦИЯ О НОВОМ UI

Обновленный интерфейс включает:
• Интерактивные меню
• Красивые сообщения  
• Анимации и эффекты
• Систему профилей
• Улучшенную обратную связь

Все для вашего удобства!
        """.strip()
        
        await msg.answer(fallback_text, parse_mode=None)


## Удалены все inline-кнопки и callback-обработчики по требованию: нет кнопок, только ответы.
