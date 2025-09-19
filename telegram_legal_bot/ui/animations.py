# telegram_legal_bot/ui/animations.py
"""
Анимации и эффекты для улучшения UX бота.
"""

import asyncio
from typing import List, Optional
from aiogram import types
from aiogram.exceptions import TelegramBadRequest


class BotAnimations:
    """Класс для создания анимаций в боте."""
    
    @staticmethod
    async def typing_animation(message: types.Message, duration: float = 3.0) -> None:
        """Показывает индикатор печати."""
        try:
            chat_id = message.chat.id
            # В реальном боте здесь был бы ChatActionSender
            # await message.bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(duration)
        except Exception:
            pass
    
    @staticmethod
    async def progress_message(
        message: types.Message, 
        steps: List[str], 
        delay: float = 1.5
    ) -> Optional[types.Message]:
        """Показывает прогресс выполнения через редактирование сообщения."""
        try:
            # Отправляем первое сообщение
            progress_msg = await message.answer(steps[0], parse_mode="MarkdownV2")
            
            # Обновляем сообщение для каждого шага
            for step in steps[1:]:
                await asyncio.sleep(delay)
                try:
                    await progress_msg.edit_text(step, parse_mode="MarkdownV2")
                except TelegramBadRequest:
                    # Если не удалось отредактировать, отправляем новое
                    progress_msg = await message.answer(step, parse_mode="MarkdownV2")
                except Exception:
                    # Fallback без форматирования
                    try:
                        await progress_msg.edit_text(step, parse_mode=None)
                    except Exception:
                        progress_msg = await message.answer(step, parse_mode=None)
            
            return progress_msg
            
        except Exception:
            return None
    
    @staticmethod
    async def loading_dots(
        message: types.Message, 
        base_text: str, 
        duration: float = 4.0
    ) -> Optional[types.Message]:
        """Анимация загрузки с точками."""
        try:
            loading_msg = await message.answer(base_text, parse_mode="MarkdownV2")
            
            dots = ["", ".", "..", "..."]
            steps = int(duration / 0.5)
            
            for i in range(steps):
                dot_index = i % len(dots)
                text = f"{base_text}{dots[dot_index]}"
                
                await asyncio.sleep(0.5)
                try:
                    await loading_msg.edit_text(text, parse_mode="MarkdownV2")
                except TelegramBadRequest:
                    try:
                        await loading_msg.edit_text(text, parse_mode=None)
                    except Exception:
                        break
                except Exception:
                    break
            
            return loading_msg
            
        except Exception:
            return None
    
    @staticmethod
    async def countdown_timer(
        message: types.Message, 
        seconds: int, 
        prefix: str = "Осталось: "
    ) -> None:
        """Обратный отсчет."""
        try:
            timer_msg = await message.answer(f"{prefix}{seconds} сек.")
            
            for remaining in range(seconds - 1, 0, -1):
                await asyncio.sleep(1)
                try:
                    await timer_msg.edit_text(f"{prefix}{remaining} сек.")
                except Exception:
                    break
            
            await asyncio.sleep(1)
            try:
                await timer_msg.edit_text("⏰ Время истекло!")
            except Exception:
                pass
                
        except Exception:
            pass
    
    @staticmethod
    async def celebration_effect(message: types.Message, text: str) -> None:
        """Эффект празднования."""
        celebration_frames = [
            f"🎉 {text} 🎉",
            f"✨ {text} ✨", 
            f"🎊 {text} 🎊",
            f"🌟 {text} 🌟",
            f"🎉 {text} 🎉"
        ]
        
        try:
            celebration_msg = await message.answer(celebration_frames[0])
            
            for frame in celebration_frames[1:]:
                await asyncio.sleep(0.8)
                try:
                    await celebration_msg.edit_text(frame)
                except Exception:
                    break
                    
        except Exception:
            pass
    
    @staticmethod
    def get_thinking_frames() -> List[str]:
        """Кадры анимации размышления."""
        return [
            "🤔 Анализирую...",
            "🧠 Размышляю...", 
            "💭 Обдумываю...",
            "🔍 Изучаю...",
            "📚 Ищу в базе знаний...",
            "⚖️ Формирую ответ...",
            "✍️ Оформляю..."
        ]
    
    @staticmethod
    def get_loading_frames() -> List[str]:
        """Кадры анимации загрузки."""
        return [
            "⏳ Загрузка",
            "⏳ Загрузка ▪️",
            "⏳ Загрузка ▪️▪️", 
            "⏳ Загрузка ▪️▪️▪️",
            "⏳ Загрузка ▪️▪️▪️▪️",
            "⏳ Загрузка ▪️▪️▪️▪️▪️"
        ]
    
    @staticmethod
    async def smooth_message_transition(
        old_message: types.Message,
        new_text: str, 
        parse_mode: Optional[str] = "MarkdownV2"
    ) -> Optional[types.Message]:
        """Плавный переход между сообщениями."""
        try:
            # Пытаемся отредактировать существующее сообщение
            await old_message.edit_text(new_text, parse_mode=parse_mode)
            return old_message
        except TelegramBadRequest:
            # Если редактирование не удалось, отправляем новое
            try:
                return await old_message.answer(new_text, parse_mode=parse_mode)
            except Exception:
                # Fallback без форматирования
                return await old_message.answer(new_text, parse_mode=None)
        except Exception:
            return None
    
    @staticmethod
    async def typewriter_effect(
        message: types.Message, 
        text: str, 
        delay: float = 0.05
    ) -> Optional[types.Message]:
        """Эффект печатной машинки."""
        try:
            # Для длинных текстов эффект может быть слишком медленным
            if len(text) > 100:
                delay = 0.02
            
            current_text = ""
            msg = await message.answer("_")
            
            for char in text:
                current_text += char
                await asyncio.sleep(delay)
                try:
                    await msg.edit_text(current_text)
                except TelegramBadRequest:
                    # Если слишком часто редактируем, пропускаем некоторые обновления
                    if len(current_text) % 10 == 0:
                        try:
                            await msg.edit_text(current_text)
                        except Exception:
                            break
                except Exception:
                    break
            
            return msg
            
        except Exception:
            # Если эффект не работает, просто отправляем обычное сообщение
            return await message.answer(text)
    
    @staticmethod
    async def pulsing_message(
        message: types.Message,
        text: str,
        pulses: int = 3
    ) -> None:
        """Пульсирующее сообщение."""
        try:
            pulse_msg = await message.answer(text)
            
            for _ in range(pulses):
                # "Затухание" - заменяем текст на точки
                await asyncio.sleep(0.5)
                try:
                    await pulse_msg.edit_text("• • •")
                except Exception:
                    break
                
                # "Появление" - возвращаем исходный текст
                await asyncio.sleep(0.5) 
                try:
                    await pulse_msg.edit_text(text)
                except Exception:
                    break
                    
        except Exception:
            pass
