# telegram_legal_bot/ui/user_profile.py
"""
Система управления профилями пользователей и статистикой.
"""

import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class UserProfile:
    """Класс для работы с профилями пользователей."""
    
    def __init__(self, data_dir: str = "user_data"):
        """Инициализация с директорией для данных."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def _get_user_file(self, user_id: int) -> Path:
        """Получить путь к файлу пользователя."""
        return self.data_dir / f"user_{user_id}.json"
    
    def get_profile(self, user_id: int) -> Dict[str, Any]:
        """Получить профиль пользователя."""
        user_file = self._get_user_file(user_id)
        
        if not user_file.exists():
            # Создаем новый профиль
            return self._create_default_profile(user_id)
        
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return self._create_default_profile(user_id)
    
    def _create_default_profile(self, user_id: int) -> Dict[str, Any]:
        """Создать профиль по умолчанию."""
        now = datetime.datetime.now().isoformat()
        profile = {
            'user_id': user_id,
            'created_at': now,
            'last_active': now,
            'questions_count': 0,
            'consultations_count': 0,
            'helpful_answers': 0,
            'total_feedback': 0,
            'favorite_categories': {},
            'achievements': [],
            'settings': {
                'notifications': True,
                'theme': 'default',
                'language': 'ru',
                'format': 'detailed'
            },
            'history': []
        }
        self.save_profile(user_id, profile)
        return profile
    
    def save_profile(self, user_id: int, profile: Dict[str, Any]) -> None:
        """Сохранить профиль пользователя."""
        user_file = self._get_user_file(user_id)
        profile['last_active'] = datetime.datetime.now().isoformat()
        
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving profile for user {user_id}: {e}")
    
    def add_question(self, user_id: int, question: str, category: str = "general") -> None:
        """Добавить вопрос в статистику."""
        profile = self.get_profile(user_id)
        
        # Обновляем счетчики
        profile['questions_count'] += 1
        profile['consultations_count'] += 1
        
        # Обновляем любимые категории
        if category not in profile['favorite_categories']:
            profile['favorite_categories'][category] = 0
        profile['favorite_categories'][category] += 1
        
        # Добавляем в историю
        history_entry = {
            'type': 'question',
            'content': question[:200],  # Ограничиваем длину для экономии места
            'category': category,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        profile['history'].append(history_entry)
        
        # Ограничиваем историю 100 записями
        if len(profile['history']) > 100:
            profile['history'] = profile['history'][-100:]
        
        # Проверяем достижения
        self._check_achievements(profile)
        
        self.save_profile(user_id, profile)
    
    def add_feedback(self, user_id: int, is_helpful: bool) -> None:
        """Добавить обратную связь."""
        profile = self.get_profile(user_id)
        
        profile['total_feedback'] += 1
        if is_helpful:
            profile['helpful_answers'] += 1
        
        # Добавляем в историю
        history_entry = {
            'type': 'feedback',
            'helpful': is_helpful,
            'timestamp': datetime.datetime.now().isoformat()
        }
        profile['history'].append(history_entry)
        
        self._check_achievements(profile)
        self.save_profile(user_id, profile)
    
    def _check_achievements(self, profile: Dict[str, Any]) -> None:
        """Проверить и добавить достижения."""
        achievements = profile.get('achievements', [])
        questions_count = profile.get('questions_count', 0)
        helpful_answers = profile.get('helpful_answers', 0)
        total_feedback = profile.get('total_feedback', 0)
        
        # Первый вопрос
        if questions_count >= 1 and 'first_question' not in achievements:
            achievements.append('first_question')
        
        # Активный пользователь (10+ вопросов)
        if questions_count >= 10 and 'active_user' not in achievements:
            achievements.append('active_user')
        
        # Ищущий знания (50+ вопросов)
        if questions_count >= 50 and 'expert_seeker' not in achievements:
            achievements.append('expert_seeker')
        
        # Благодарный пользователь (5+ положительных оценок)
        if helpful_answers >= 5 and 'helpful_feedback' not in achievements:
            achievements.append('helpful_feedback')
        
        profile['achievements'] = achievements
    
    def get_stats(self, user_id: int) -> Dict[str, Any]:
        """Получить статистику пользователя."""
        profile = self.get_profile(user_id)
        
        # Вычисляем дни с нами
        created_at = datetime.datetime.fromisoformat(profile.get('created_at', datetime.datetime.now().isoformat()))
        days_with_us = (datetime.datetime.now() - created_at).days
        
        # Определяем любимую категорию
        favorite_categories = profile.get('favorite_categories', {})
        favorite_category = "Не определена"
        if favorite_categories:
            favorite_category = max(favorite_categories, key=favorite_categories.get)
            category_map = {
                'civil': 'Гражданское право',
                'criminal': 'Уголовное право', 
                'labor': 'Трудовое право',
                'tax': 'Налоговое право',
                'family': 'Семейное право',
                'administrative': 'Административное право'
            }
            favorite_category = category_map.get(favorite_category, favorite_category.title())
        
        # Последний вопрос
        last_question_date = "Не задавали"
        history = profile.get('history', [])
        for entry in reversed(history):
            if entry.get('type') == 'question':
                date = datetime.datetime.fromisoformat(entry['timestamp'])
                delta = datetime.datetime.now() - date
                
                if delta.days == 0:
                    last_question_date = "Сегодня"
                elif delta.days == 1:
                    last_question_date = "Вчера"
                elif delta.days < 7:
                    last_question_date = f"{delta.days} дня назад"
                else:
                    last_question_date = date.strftime("%d.%m.%Y")
                break
        
        return {
            'questions': profile.get('questions_count', 0),
            'consultations': profile.get('consultations_count', 0),
            'helpful_answers': profile.get('helpful_answers', 0),
            'days_with_us': days_with_us,
            'last_question_date': last_question_date,
            'favorite_category': favorite_category,
            'achievements': profile.get('achievements', [])
        }
    
    def update_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        """Обновить настройки пользователя."""
        profile = self.get_profile(user_id)
        profile['settings'].update(settings)
        self.save_profile(user_id, profile)
    
    def clear_history(self, user_id: int) -> None:
        """Очистить историю пользователя."""
        profile = self.get_profile(user_id)
        profile['history'] = []
        profile['questions_count'] = 0
        profile['consultations_count'] = 0
        profile['helpful_answers'] = 0
        profile['total_feedback'] = 0
        profile['favorite_categories'] = {}
        profile['achievements'] = []
        self.save_profile(user_id, profile)


# Глобальный экземпляр для использования в боте
user_profile_manager = UserProfile()
