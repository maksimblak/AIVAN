# telegram_legal_bot/ui/messages.py
"""
Красивые сообщения и текстовые шаблоны для бота.
"""

from typing import Dict, Any, Optional
import datetime
from telegram_legal_bot.utils.message_formatter import md2


class BotMessages:
    """Класс для создания красивых сообщений."""
    
    # Эмодзи для разных типов сообщений
    EMOJI = {
        'welcome': '👋',
        'help': 'ℹ️',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'loading': '⏳',
        'thinking': '🤔',
        'book': '📚',
        'law': '⚖️',
        'question': '❓',
        'answer': '💡',
        'profile': '👤',
        'stats': '📊',
        'settings': '⚙️',
        'star': '⭐',
        'fire': '🔥',
        'rocket': '🚀',
        'heart': '❤️',
        'thumbs_up': '👍',
        'thumbs_down': '👎'
    }
    
    @staticmethod
    def welcome(user_name: Optional[str] = None) -> str:
        """Приветственное сообщение с правильным MarkdownV2."""
        name_part = f", {user_name}" if user_name else ""
        
        return f"""
{BotMessages.EMOJI['welcome']} *Добро пожаловать{name_part}\\!*

Я — *AIVAN*, ваш персональный юридический ассистент {BotMessages.EMOJI['law']}

{BotMessages.EMOJI['rocket']} *Что я умею:*
• Отвечаю на правовые вопросы по РФ
• Привожу ссылки на актуальные нормы  
• Даю практические рекомендации
• Помогаю с документооборотом

{BotMessages.EMOJI['star']} *Мои преимущества:*
• Быстрые и точные ответы
• Актуальная правовая база
• Понятное объяснение сложного
• Доступен 24/7

{BotMessages.EMOJI['warning']} *Важно помнить:*
Я не заменяю консультацию с юристом, но помогу сориентироваться в правовых вопросах\\!

Выберите действие в меню ниже {BotMessages.EMOJI['thumbs_up']}
        """.strip()
    
    @staticmethod
    def welcome_with_bold(user_name: Optional[str] = None) -> str:
        """Приветственное сообщение с жирным шрифтом (альтернативная версия)."""
        name_part = f", {user_name}" if user_name else ""
        
        # Используем HTML разметку для более надежного форматирования  
        return f"""
{BotMessages.EMOJI['welcome']} <b>Добро пожаловать{name_part}!</b>

Я — <b>AIVAN</b>, ваш персональный юридический ассистент {BotMessages.EMOJI['law']}

{BotMessages.EMOJI['rocket']} <b>Что я умею:</b>
• Отвечаю на правовые вопросы по РФ
• Привожу ссылки на актуальные нормы  
• Даю практические рекомендации
• Помогаю с документооборотом

{BotMessages.EMOJI['star']} <b>Мои преимущества:</b>
• Быстрые и точные ответы
• Актуальная правовая база
• Понятное объяснение сложного
• Доступен 24/7

{BotMessages.EMOJI['warning']} <b>Важно помнить:</b>
Я не заменяю консультацию с юристом, но помогу сориентироваться в правовых вопросах!

Выберите действие в меню ниже {BotMessages.EMOJI['thumbs_up']}
        """.strip()
    
    @staticmethod
    def help_main() -> str:
        """Основное сообщение помощи."""
        return f"""
{BotMessages.EMOJI['help']} <b>СПРАВКА ПО ИСПОЛЬЗОВАНИЮ</b>

{BotMessages.EMOJI['question']} <b>Как задать вопрос:</b>
1. Опишите ситуацию подробно
2. Укажите все важные детали
3. Упомяните документы и сроки
4. Получите развернутый ответ

{BotMessages.EMOJI['fire']} <b>Советы для лучшего результата:</b>
• Пишите конкретно: кто, что, где, когда
• Указывайте суммы, даты, документы
• Избегайте общих формулировок
• Задавайте уточняющие вопросы

{BotMessages.EMOJI['book']} <b>Категории права:</b>
• Гражданское • Уголовное
• Трудовое • Налоговое  
• Семейное • Административное

{BotMessages.EMOJI['star']} <b>Дополнительные функции:</b>
• История ваших вопросов
• Статистика консультаций
• Настройка уведомлений
• Экспорт консультаций
        """.strip()
    
    @staticmethod
    def quick_start() -> str:
        """Быстрый старт."""
        return f"""
{BotMessages.EMOJI['rocket']} <b>БЫСТРЫЙ СТАРТ</b>

<b>Шаг 1:</b> {BotMessages.EMOJI['question']} Опишите вашу ситуацию
<i>Пример:</i> "Работодатель задерживает зарплату на 2 недели. Что делать?"

<b>Шаг 2:</b> {BotMessages.EMOJI['thinking']} Дождитесь анализа
Я изучу вопрос и найду подходящие нормы права

<b>Шаг 3:</b> {BotMessages.EMOJI['answer']} Получите ответ
Вы получите подробный ответ со ссылками на законы

<b>Шаг 4:</b> {BotMessages.EMOJI['thumbs_up']} Оцените полезность
Поставьте оценку, чтобы я стал еще лучше!

{BotMessages.EMOJI['fire']} <b>Готовы начать? Просто напишите свой вопрос!</b>
        """.strip()
    
    @staticmethod
    def examples() -> str:
        """Примеры вопросов."""
        return f"""
{BotMessages.EMOJI['book']} **ПРИМЕРЫ ХОРОШИХ ВОПРОСОВ**

{BotMessages.EMOJI['law']} **Трудовое право:**
"Меня хотят уволить по сокращению штатов. Какие у меня права и компенсации?"

{BotMessages.EMOJI['law']} **Гражданское право:**
"Сосед затопил квартиру. Как получить компенсацию и кто должен платить за ремонт?"

{BotMessages.EMOJI['law']} **Семейное право:**
"Развожусь с женой. Как делится квартира, купленная в ипотеку в браке?"

{BotMessages.EMOJI['law']} **ДТП и страхование:**
"Попал в ДТП, ОСАГО не покрывает весь ущерб. Как взыскать недостающую сумму?"

{BotMessages.EMOJI['law']} **Налоги:**
"Продаю квартиру через 2 года после покупки. Какой налог буду платить?"

{BotMessages.EMOJI['star']} **Чем подробнее опишете — тем точнее ответ!**
        """.strip()
    
    @staticmethod
    def profile(stats: Dict[str, Any]) -> str:
        """Профиль пользователя."""
        return f"""
{BotMessages.EMOJI['profile']} <b>ВАШ ПРОФИЛЬ</b>

{BotMessages.EMOJI['stats']} <b>Статистика:</b>
• Вопросов задано: <b>{stats.get('questions', 0)}</b>
• Консультаций получено: <b>{stats.get('consultations', 0)}</b>
• Полезных ответов: <b>{stats.get('helpful_answers', 0)}</b>
• Дней с нами: <b>{stats.get('days_with_us', 0)}</b>

{BotMessages.EMOJI['fire']} <b>Активность:</b>
• Последний вопрос: {stats.get('last_question_date', 'Не задавали')}
• Любимая категория: <b>{stats.get('favorite_category', 'Не определена')}</b>

{BotMessages.EMOJI['star']} <b>Достижения:</b>
{BotMessages._format_achievements(stats.get('achievements', []))}

{BotMessages.EMOJI['heart']} Спасибо, что пользуетесь AIVAN!
        """.strip()
    
    @staticmethod
    def rate_limit_message(remaining_time: int) -> str:
        """Сообщение о превышении лимита."""
        hours = remaining_time // 3600
        minutes = (remaining_time % 3600) // 60
        
        time_str = ""
        if hours > 0:
            time_str += f"{hours} ч. "
        if minutes > 0:
            time_str += f"{minutes} мин."
        
        return f"""
{BotMessages.EMOJI['warning']} **ЛИМИТ ВОПРОСОВ ИСЧЕРПАН**

Вы достигли максимального количества вопросов в час.

{BotMessages.EMOJI['loading']} **Осталось ждать:** {time_str}

{BotMessages.EMOJI['star']} **Пока можете:**
• Изучить предыдущие ответы
• Ознакомиться с примерами вопросов  
• Настроить уведомления
• Рассмотреть Premium-подписку

{BotMessages.EMOJI['rocket']} **Premium дает:**
• Безлимитные вопросы
• Приоритетные ответы
• Расширенная аналитика
• Экспорт консультаций
        """.strip()
    
    @staticmethod
    def thinking_messages() -> list[str]:
        """Сообщения во время обработки."""
        return [
            f"{BotMessages.EMOJI['thinking']} Анализирую ваш вопрос...",
            f"{BotMessages.EMOJI['book']} Изучаю правовую базу...",
            f"{BotMessages.EMOJI['law']} Формирую юридическое заключение...",
            f"✍️ Оформляю ответ с ссылками на законы..."
        ]
    
    @staticmethod
    def error_message(error_type: str = "general") -> str:
        """Сообщения об ошибках."""
        messages = {
            "general": f"{BotMessages.EMOJI['error']} **Произошла ошибка**\n\nПопробуйте еще раз или обратитесь в поддержку.",
            "network": f"{BotMessages.EMOJI['warning']} **Проблемы с соединением**\n\nПроверьте интернет и повторите попытку.",
            "overload": f"{BotMessages.EMOJI['loading']} **Высокая нагрузка**\n\nПожалуйста, подождите немного и попробуйте снова.",
            "invalid_question": f"{BotMessages.EMOJI['question']} **Вопрос слишком короткий**\n\nОпишите ситуацию подробнее для лучшего ответа.",
        }
        return messages.get(error_type, messages["general"])
    
    @staticmethod
    def success_message(message_type: str) -> str:
        """Сообщения об успехе."""
        messages = {
            "feedback": f"{BotMessages.EMOJI['heart']} **Спасибо за обратную связь!**\n\nВаша оценка поможет мне стать лучше.",
            "settings_saved": f"{BotMessages.EMOJI['success']} **Настройки сохранены**\n\nИзменения вступили в силу.",
            "history_cleared": f"{BotMessages.EMOJI['success']} **История очищена**\n\nВсе ваши предыдущие вопросы удалены.",
        }
        return messages.get(message_type, f"{BotMessages.EMOJI['success']} Операция выполнена успешно!")
    
    @staticmethod
    def _format_achievements(achievements: list) -> str:
        """Форматирует достижения."""
        if not achievements:
            return "Пока нет достижений"
        
        achievement_emojis = {
            'first_question': '🎯 Первый вопрос',
            'active_user': '🔥 Активный пользователь',
            'expert_seeker': '🎓 Ищущий знания',
            'helpful_feedback': '👍 Благодарный пользователь'
        }
        
        formatted = []
        for achievement in achievements:
            if achievement in achievement_emojis:
                formatted.append(achievement_emojis[achievement])
        
        return '\n'.join(f"• {ach}" for ach in formatted) if formatted else "Пока нет достижений"
    
    @staticmethod  
    def format_legal_answer(answer: str, sources: list = None, confidence: float = None) -> str:
        """Форматирует юридический ответ."""
        result = f"{BotMessages.EMOJI['answer']} **ЮРИДИЧЕСКАЯ КОНСУЛЬТАЦИЯ**\n\n"
        result += f"{answer}\n\n"
        
        if sources:
            result += f"{BotMessages.EMOJI['book']} **ПРАВОВЫЕ ОСНОВАНИЯ:**\n"
            for i, source in enumerate(sources, 1):
                result += f"{i}. {source}\n"
            result += "\n"
        
        if confidence:
            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟠"
            result += f"{confidence_emoji} **Уверенность в ответе:** {confidence*100:.0f}%\n\n"
        
        result += f"{BotMessages.EMOJI['warning']} *Данная информация носит справочный характер и не заменяет консультацию юриста.*"
        
        return result
