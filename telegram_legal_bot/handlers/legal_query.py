from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.utils.chat_action import ChatActionSender  # ✅ aiogram v3 way


from telegram_legal_bot.config import Settings
from telegram_legal_bot.services import OpenAIService
from telegram_legal_bot.utils.rate_limiter import RateLimiter




router = Router(name="legal_query")

_settings: Optional[Settings] = None
_ai: Optional[OpenAIService] = None
_rl: Optional[RateLimiter] = None
_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=5))


def setup_context(settings: Settings, ai: OpenAIService) -> None:
    global _settings, _ai, _rl, _history
    _settings = settings
    _ai = ai
    _rl = RateLimiter(max_calls=settings.max_requests_per_hour, period_seconds=3600)
    _history = defaultdict(lambda: deque(maxlen=max(1, int(settings.history_size))))


@router.message(F.text & ~F.text.startswith("/"))
async def handle_legal_query(message: types.Message) -> None:
    assert _settings is not None and _ai is not None and _rl is not None

    user_id = message.from_user.id if message.from_user else 0
    text = (message.text or "").strip()

    if len(text) < _settings.min_question_length:
        await message.answer(
            "✋ Вопрос слишком короткий. Пожалуйста, опишите ситуацию подробней.",
            parse_mode=None,
        )
        return

    if not await _rl.check(user_id):
        remain = await _rl.remaining(user_id)
        msg = "⏳ Лимит вопросов на ближайший час исчерпан. Попробуйте позже."
        if remain:
            msg += f" Доступно ещё: {remain}."
        await message.answer(msg, parse_mode=None)
        return

    short_history: List[Dict[str, str]] = list(_history[user_id])

    # ✅ держим индикатор «печатает…» пока ждём LLM
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            result = await _ai.generate_legal_answer(text, short_history=short_history)

        answer: str = result.get("answer") or ""
        laws: List[str] = result.get("laws") or []

        _history[user_id].append({"role": "user", "content": text})
        _history[user_id].append({"role": "assistant", "content": answer})

        final = build_legal_reply(answer=answer, laws=laws)
        chunks = chunk_markdown_v2(final, limit=4096)

        for part in chunks:
            try:
                await message.answer(part, parse_mode=_settings.parse_mode)
            except TelegramBadRequest:
                await message.answer(part, parse_mode=None)

    except Exception:
        await message.answer(
            "😕 Произошла ошибка при обработке запроса. Попробуйте ещё раз через пару минут.",
            parse_mode=None,
        )
