from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from aiogram import Router, types, F
from aiogram.enums import ChatAction, ParseMode

from telegram_legal_bot.config import Settings, load_settings
from telegram_legal_bot.services.openai_service import OpenAIService, LegalAdvice
from telegram_legal_bot.utils.message_formatter import build_legal_reply, chunk_markdown_v2

logger = logging.getLogger(__name__)
router = Router()

# --------- простейший per-user rate limit (in-memory, 1h окно) ----------
@dataclass
class RateWindow:
    timestamps: Deque[float] = field(default_factory=deque)

    def hit(self, now: float, max_per_hour: int) -> bool:
        hour_ago = now - 3600.0
        while self.timestamps and self.timestamps[0] < hour_ago:
            self.timestamps.popleft()
        if len(self.timestamps) >= max_per_hour:
            return False
        self.timestamps.append(now)
        return True

_rate_map: Dict[int, RateWindow] = defaultdict(RateWindow)
_history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=5))

# DI контейнер-подобно: положим сервисы в "context" роутера
def setup_context(router: Router, settings: Settings, ai: OpenAIService) -> None:
    router.data["settings"] = settings
    router.data["ai"] = ai


def _looks_like_spam(text: str) -> bool:
    t = text.replace(" ", "")
    if len(t) > 20:
        from collections import Counter
        c = Counter(t)
        if c.most_common(1)[0][1] / len(t) > 0.6:
            return True
    words = [w for w in text.lower().split() if w]
    rep = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            rep += 1
            if rep >= 6:
                return True
        else:
            rep = 1
    return False


@router.message(F.text & ~F.text.startswith("/"))
async def handle_legal_query(message: types.Message, event_chat: types.Chat) -> None:
    """
    Обрабатывает любой текст без команд:
      - валидация
      - rate-limit
      - индикатор печати
      - запрос к GPT-5
      - форматирование и отправка чанками
    """
    settings: Settings = router.data["settings"]
    ai: OpenAIService = router.data["ai"]

    text = (message.text or "").strip()
    if len(text) < settings.min_question_length:
        await message.answer(
            f"🧐 Пожалуйста, опишите вопрос подробнее (минимум {settings.min_question_length} символов)."
        )
        return
    if _looks_like_spam(text):
        await message.answer("🤖 Похоже на спам/флуд. Если это ошибка — переформулируйте вопрос.")
        return

    user_id = message.from_user.id if message.from_user else 0
    now = time.time()
    if not _rate_map[user_id].hit(now, settings.max_requests_per_hour):
        await message.answer("⏳ Лимит запросов на час исчерпан. Попробуйте позже.")
        return

    # индикатор "печатает..."
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    try:
        advice: LegalAdvice = await ai.generate_legal_advice(
            user_question=text, short_history=list(_history[user_id])
        )
    except asyncio.TimeoutError:
        await message.answer("⏱️ Превышено время ожидания ответа. Попробуйте позже.")
        return
    except Exception as e:
        logger.exception("Ошибка OpenAI: %s", e)
        await message.answer("⚠️ Произошла ошибка при обработке. Попробуйте позже.")
        return

    reply = build_legal_reply(advice.summary, advice.details, advice.laws)
    for i, ch in enumerate(chunk_markdown_v2(reply)):
        prefix = "" if i == 0 else "…продолжение:\n\n"
        await message.answer(prefix + ch, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)

    if advice.summary:
        _history[user_id].append(advice.summary)
