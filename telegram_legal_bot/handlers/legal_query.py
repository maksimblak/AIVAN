from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

from config import Settings
from services.openai_service import LegalAdvice, OpenAIService
from utils.message_formatter import build_legal_reply, chunk_markdown_v2

logger = logging.getLogger(__name__)


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

# --------- короткая история кратких ответов per user (для контекста) ----
_history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=5))


def build_legal_message_handler(settings: Settings, ai: OpenAIService) -> MessageHandler:
    """
    Фабрика message-хэндлера: обрабатывает ВСЕ текстовые сообщения,
    кроме команд, и маршрутизирует их в OpenAI.
    """

    async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        user_id = update.effective_user.id if update.effective_user else 0
        text = (msg.text or "").strip()

        # 0) Валидация
        if not text:
            return
        if text.startswith("/"):
            return  # команды обрабатываются отдельно
        if len(text) < settings.min_question_length:
            short = (
                f"🧐 Пожалуйста, опишите вопрос подробнее (минимум {settings.min_question_length} символов)."
            )
            await msg.reply_text(short)
            return

        # 1) Антиспам: простой чек на сверхповторяющиеся символы
        if _looks_like_spam(text):
            await msg.reply_text(
                "🤖 Сообщение похоже на спам/флуд. Если это ошибка — переформулируйте вопрос."
            )
            return

        # 2) Rate-limit per user
        now = time.time()
        window = _rate_map[user_id]
        if not window.hit(now, settings.max_requests_per_hour):
            await msg.reply_text("⏳ Лимит запросов на час исчерпан. Попробуйте позже.")
            return

        # 3) Показываем "печатает"
        try:
            await context.bot.send_chat_action(msg.chat_id, action=ChatAction.TYPING)
        except Exception:  # noqa: BLE001
            pass

        # 4) Вызов OpenAI с ретраями и таймаутом
        try:
            advice: LegalAdvice = await ai.generate_legal_advice(
                user_question=text, short_history=list(_history[user_id])
            )
        except asyncio.TimeoutError:
            await msg.reply_text("⏱️ Превышено время ожидания ответа. Попробуйте позже.")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Ошибка при обращении к OpenAI: %s", e)
            await msg.reply_text("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")
            return

        # 5) Форматирование ответа и безопасная отправка чанками
        reply = build_legal_reply(advice.summary, advice.details, advice.laws)
        chunks = chunk_markdown_v2(reply)
        for i, ch in enumerate(chunks):
            prefix = "" if i == 0 else "…продолжение:\n\n"
            await msg.reply_text(
                prefix + ch,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
            )

        # 6) Сохраняем краткий ответ в историю пользователя
        if advice.summary:
            _history[user_id].append(advice.summary)

    return MessageHandler(filters.TEXT & ~filters.COMMAND, _handle)


def _looks_like_spam(text: str) -> bool:
    """
    Очень грубая эвристика — считается спамом, если:
      - доля самого частого символа > 0.6 и длина > 20
      - либо более 5 повторов одного и того же слова подряд
    """
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
