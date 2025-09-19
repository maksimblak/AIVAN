from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.utils.chat_action import ChatActionSender  # aiogram v3

from telegram_legal_bot.config import Settings
from telegram_legal_bot.services import OpenAIService
from telegram_legal_bot.utils.rate_limiter import RateLimiter
from telegram_legal_bot.utils.message_formatter import (
    md2,
    build_legal_reply,
    chunk_markdown_v2,
    strip_md2_escapes,
    _escape_md2_url as escape_md2_url,  # безопасные URL для MarkdownV2
)
from telegram_legal_bot.ui.messages import BotMessages
from telegram_legal_bot.ui.animations import BotAnimations

router = Router(name="legal_query")
log = logging.getLogger("legal_query")


def _looks_bad_markdown(s: str) -> bool:
    """
    Эвристика «плохого» ответа: короткий текст без ключевых секций.
    Триггерит фолбэк на альтернативный путь.
    """
    t = (s or "").strip()
    if len(t) >= 120:
        return False
    markers = ("*Кратко:*", "*Шаги:*", "*Аналитика:*", "*Источники:*", "*Черновики документов:*")
    return not any(m in t for m in markers)


class LegalQueryHandler:
    """
    Thread-safe обработчик юридических запросов.
    Инкапсулирует состояние: лимитер, история, служебные таймеры.
    """

    def __init__(self, settings: Settings, ai: OpenAIService):
        self.settings = settings
        self.ai = ai
        # корректная сигнатура RateLimiter: max_requests, period
        self.rate_limiter = RateLimiter(
            max_requests=settings.max_requests_per_hour,
            period=3600,
        )

        pairs = max(1, int(settings.history_size))
        self._history: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=pairs * 2))
        self._history_lock = asyncio.Lock()
        self._history_cleanup_time = time.time()
        self._last_access: Dict[int, float] = {}
        self.max_users_in_history = 10_000

    async def _cleanup_history_if_needed(self) -> None:
        now = time.time()
        if now - self._history_cleanup_time <= 6 * 60 * 60:
            return
        async with self._history_lock:
            if len(self._history) > self.max_users_in_history:
                overflow = len(self._history) - self.max_users_in_history
                sorted_users = sorted(self._history.keys(), key=lambda uid: self._last_access.get(uid, 0.0))
                to_remove = sorted_users[:overflow]
                for uid in to_remove:
                    self._history.pop(uid, None)
                    self._last_access.pop(uid, None)
                log.info("Cleaned up history for %d users", len(to_remove))
            self._history_cleanup_time = now

    async def get_user_history(self, user_id: int) -> List[Dict[str, str]]:
        await self._cleanup_history_if_needed()
        async with self._history_lock:
            self._last_access[user_id] = time.time()
            return list(self._history[user_id])

    async def add_to_history(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        async with self._history_lock:
            self._history[user_id].append({"role": "user", "content": user_msg})
            self._history[user_id].append({"role": "assistant", "content": assistant_msg[:1000]})
            self._last_access[user_id] = time.time()


_handler: Optional[LegalQueryHandler] = None


def setup_context(settings: Settings, ai: OpenAIService) -> None:
    global _handler
    _handler = LegalQueryHandler(settings, ai)


def _schema_to_markdown(d: Dict[str, Any]) -> str:
    """
    Рендер ответа по LEGAL_SCHEMA_V2 → MarkdownV2.
    """
    lines: List[str] = []

    def add(s: str) -> None:
        if s:
            lines.append(s)

    conclusion = (d.get("conclusion") or "").strip()
    if conclusion:
        add(f"*Кратко:* {md2(conclusion)}")

    legal_basis = d.get("legal_basis") or []
    if isinstance(legal_basis, list) and legal_basis:
        add("\n*Нормы права:*")
        for n in legal_basis:
            act = md2(str(n.get("act", "")).strip())
            article = md2(str(n.get("article", "")).strip())
            pin = md2(str(n.get("pinpoint", "")).strip()) if n.get("pinpoint") else ""
            quote = md2(str(n.get("quote", "")).strip()) if n.get("quote") else ""
            head = "• "
            head += f"{act}, " if act else ""
            head += f"{article}" if article else ""
            if pin:
                head += f" ({pin})"
            head_stripped = head.strip()
            add(head_stripped if head_stripped != "•" else "• Норма")
            if quote:
                add(f"  └ «{quote}»")

    cases = d.get("cases") or []
    if isinstance(cases, list) and cases:
        add("\n*Подборка дел:*")
        for c in cases:
            court = md2(str(c.get("court", "")).strip())
            date = md2(str(c.get("date", "")).strip())
            case_no = md2(str(c.get("case_no", "")).strip())
            url_raw = str(c.get("url") or "").strip()
            url = escape_md2_url(url_raw) if url_raw else ""
            holding = md2(str(c.get("holding", "")).strip())
            facts = md2(str(c.get("facts", "")).strip())
            norms = c.get("norms") or []
            sim = c.get("similarity")

            head = "• "
            head += f"{court}, " if court else ""
            head += f"{date}, " if date else ""
            head += f"№ {case_no}" if case_no else "дело"
            if url:
                head += f" — [ссылка]({url})"
            if isinstance(sim, (int, float)):
                head += f" (схожесть: {sim:.2f})"
            add(head)

            if holding:
                add(f"  └ Исход: {holding}")
            if facts:
                add(f"  └ Фабула: {facts}")
            if isinstance(norms, list) and norms:
                add("  └ Нормы: " + md2("; ".join(map(str, norms))))

    analysis = (d.get("analysis") or "").strip()
    if analysis:
        add("\n*Аналитика:*")
        add(md2(analysis))

    risks = d.get("risks") or []
    if isinstance(risks, list) and risks:
        add("\n*Риски:*")
        for r in risks:
            add("• " + md2(str(r)))

    steps = d.get("next_actions") or []
    if isinstance(steps, list) and steps:
        add("\n*Шаги:*")
        for s in steps:
            add("• " + md2(str(s)))

    sources = d.get("sources") or []
    if isinstance(sources, list) and sources:
        add("\n*Источники:*")
        for s in sources:
            title = md2(str(s.get("title") or "Источник"))
            u_raw = str(s.get("url") or "")
            u = escape_md2_url(u_raw) if u_raw else ""
            add(f"• [{title}]({u})" if u else f"• {title}")

    drafts = d.get("doc_drafts") or []
    if isinstance(drafts, list) and drafts:
        add("\n*Черновики документов:*")
        for doc in drafts:
            title = md2(str(doc.get("title") or "Документ"))
            dtype = md2(str(doc.get("doc_type") or ""))
            add(f"• *{title}*{f' ({dtype})' if dtype else ''} — доступен черновик")

    clar = d.get("clarifications") or []
    if isinstance(clar, list) and clar:
        add("\n*Что ещё нужно уточнить:*")
        for q in clar:
            add("• " + md2(str(q)))

    result = "\n".join(lines).strip()
    return result or md2(d.get("conclusion") or "")


@router.message(F.text & ~F.text.startswith("/"))
async def handle_legal_query(message: types.Message) -> None:
    """
    Основной хэндлер:
      • валидация, лимиты, индикатор «печатает…»;
      • строгий режим (V2) + quality-gate;
      • жёсткий фолбэк;
      • безопасная разбивка на 4096.
    """
    if _handler is None:
        log.error("Handler not initialized! Call setup_context() first.")
        await message.answer("⚠️ Сервис временно недоступен. Попробуйте позже.", parse_mode=None)
        return

    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    text = (message.text or "").strip()

    if not text:
        await message.answer("✋ Пожалуйста, отправьте текстовый вопрос.", parse_mode=None)
        return

    log.info("IN: user=%s chat=%s len=%s", user_id, chat_id, len(text))

    if len(text) < _handler.settings.min_question_length:
        error_text = BotMessages.error_message("invalid_question")
        try:
            await message.answer(md2(error_text), parse_mode="MarkdownV2")
        except Exception:
            await message.answer("✋ Вопрос слишком короткий. Пожалуйста, опишите ситуацию подробней.", parse_mode=None)
        return

    if not await _handler.rate_limiter.check(user_id):
        ttl_seconds: int = 3600
        retry_after_fn = getattr(_handler.rate_limiter, "retry_after", None)
        if callable(retry_after_fn):
            try:
                ttl_val = await retry_after_fn(user_id)  # type: ignore[misc]
                if isinstance(ttl_val, (int, float)):
                    ttl_seconds = max(0, int(ttl_val))
            except Exception:
                pass

        rate_limit_text = BotMessages.rate_limit_message(ttl_seconds)
        try:
            await message.answer(md2(rate_limit_text), parse_mode="MarkdownV2")
        except Exception:
            remain_text = ""
            remaining_fn = getattr(_handler.rate_limiter, "remaining", None)
            if callable(remaining_fn):
                try:
                    remain = await remaining_fn(user_id)  # type: ignore[misc]
                    if isinstance(remain, int):
                        remain_text = f" Доступно ещё: {remain}."
                except Exception:
                    pass
            await message.answer("⏳ Лимит вопросов на ближайший час исчерпан. Попробуйте позже." + remain_text, parse_mode=None)
        return

    short_history: List[Dict[str, str]] = await _handler.get_user_history(user_id)

    progress_msg: Optional[types.Message] = None
    try:
        thinking_steps = BotMessages.thinking_messages()
        progress_msg = await BotAnimations.progress_message(message, thinking_steps, delay=1.0)

        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            md_text: Optional[str] = None
            assistant_summary: str = ""

            # 1) Строгий режим со схемой + история + quality-gate
            try:
                rich = await _handler.ai.ask_ivan(text, short_history=short_history)
                data = rich.get("data") if rich else None
                citations = rich.get("citations") or []
                if isinstance(data, dict) and (data.get("conclusion") or data.get("sources") or data.get("cases")):
                    candidate = _schema_to_markdown(data)
                    if candidate and not _looks_bad_markdown(candidate):
                        md_text = candidate
                        if citations:
                            md_text += "\n\n*Ссылки из поиска:*\n" + "\n".join(
                                f"• [{md2(c.get('title') or 'Источник')}]({escape_md2_url(str(c.get('url') or ''))})"
                                for c in citations if c.get("url")
                            )
                        assistant_summary = (data.get("conclusion") or "")[:1000]
            except Exception as e_json:
                log.warning("ask_ivan failed, fallback to simple path: %r", e_json)

            # 2) Жёсткий фолбэк: JSON V2 → если тоже «так себе» → простой answer+laws
            if not md_text:
                result = await _handler.ai.generate_legal_answer(text, short_history=short_history)
                if isinstance(result, dict) and (result.get("conclusion") or result.get("sources") or result.get("cases")):
                    candidate = _schema_to_markdown(result) or ""
                    if candidate and not _looks_bad_markdown(candidate):
                        md_text = candidate
                        assistant_summary = (result.get("conclusion") or "")[:1000]

                if not md_text:
                    answer: str = (result.get("answer") or "") if result else ""
                    laws: List[str] = (result.get("laws") or []) if result else []
                    assistant_summary = answer[:1000] if answer else "Ответ не получен"
                    md_text = build_legal_reply(answer=answer, laws=laws)

        await _handler.add_to_history(user_id, text, assistant_summary)

        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass

        chunks = chunk_markdown_v2(md_text or "Не удалось получить ответ.", limit=4096)
        for part in chunks:
            try:
                await message.answer(part, parse_mode=_handler.settings.parse_mode)
            except TelegramBadRequest:
                await message.answer(strip_md2_escapes(part), parse_mode=None)

        log.info("OUT: user=%s chat=%s sent_chunks=%s", user_id, chat_id, len(chunks))

    except TelegramBadRequest as e:
        log.exception("TelegramBadRequest: user=%s chat=%s err=%r", user_id, chat_id, e)
        await message.answer("😕 Ошибка форматирования сообщения. Отправляю без разметки.", parse_mode=None)
    except Exception as e:
        log.exception("LLM handler error: user=%s chat=%s err=%r", user_id, chat_id, e)
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        error_text = BotMessages.error_message("general")
        try:
            await message.answer(md2(error_text), parse_mode="MarkdownV2")
        except Exception:
            await message.answer("😕 Произошла ошибка при обработке запроса. Попробуйте ещё раз через пару минут.", parse_mode=None)
