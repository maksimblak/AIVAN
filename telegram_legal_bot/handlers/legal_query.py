# telegram_legal_bot/handlers/legal_query.py
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.utils.chat_action import ChatActionSender  # aiogram v3

from telegram_legal_bot.config import Settings
from telegram_legal_bot.services import OpenAIService
from telegram_legal_bot.utils.rate_limiter import RateLimiter
from telegram_legal_bot.utils.message_formatter import (
    build_legal_reply,
    chunk_markdown_v2,
    strip_md2_escapes,
)

router = Router(name="legal_query")
log = logging.getLogger("legal_query")

# Контекст, инициализируется из main.py
_settings: Optional[Settings] = None
_ai: Optional[OpenAIService] = None
_rl: Optional[RateLimiter] = None
_history: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))


def setup_context(settings: Settings, ai: OpenAIService) -> None:
    """
    Инициализация зависимостей хэндлера (вызывается из main.py).
    """
    global _settings, _ai, _rl, _history
    _settings = settings
    _ai = ai
    _rl = RateLimiter(max_calls=settings.max_requests_per_hour, period_seconds=3600)

    # Храним ровно N обменов (user↔assistant), значит maxlen = N * 2
    pairs = max(1, int(settings.history_size))
    _history = defaultdict(lambda: deque(maxlen=pairs * 2))


def _fmt_md(text: str) -> str:
    """
    Мини-экранирование под MarkdownV2, чтобы гарантированно не ронять формат.
    """
    if not text:
        return ""
    # порядок важен: сначала обратный слеш
    text = text.replace("\\", "\\\\")
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, "\\" + ch)
    return text


def _schema_to_markdown(d: Dict[str, Any]) -> str:
    """
    Рендерит ответ по LEGAL_SCHEMA_V2 в читаемый MarkdownV2.
    Поля схемы опциональны — выводим то, что пришло.
    """
    lines: List[str] = []

    conclusion = (d.get("conclusion") or "").strip()
    if conclusion:
        lines.append(f"*Кратко:* {_fmt_md(conclusion)}")

    # Подборка дел
    cases = d.get("cases") or []
    if isinstance(cases, list) and cases:
        lines.append("\n*Подборка дел:*")
        for c in cases:
            court = _fmt_md(str(c.get("court", "")))
            date = _fmt_md(str(c.get("date", "")))
            case_no = _fmt_md(str(c.get("case_no", "")))
            url = str(c.get("url", "") or "")
            holding = _fmt_md(str(c.get("holding", "")))
            facts = _fmt_md(str(c.get("facts", "")))

            head = f"• {court}, {date}, № {case_no}"
            if url:
                # ссылка в круглых скобках — экранируем заранее
                head += f" — [ссылка]({url})"
            lines.append(head)
            if holding:
                lines.append(f"  └ Исход: {holding}")
            if facts:
                lines.append(f"  └ Фабула: {facts}")

    # Нормы права
    legal_basis = d.get("legal_basis") or []
    if isinstance(legal_basis, list) and legal_basis:
        lines.append("\n*Нормы права:*")
        for n in legal_basis:
            act = _fmt_md(str(n.get("act", "")))
            article = _fmt_md(str(n.get("article", "")))
            lines.append(f"• {act}, {article}")

    # Аналитика
    analysis = (d.get("analysis") or "").strip()
    if analysis:
        lines.append("\n*Аналитика:*")
        lines.append(_fmt_md(analysis))

    # Риски
    risks = d.get("risks") or []
    if isinstance(risks, list) and risks:
        lines.append("\n*Риски:*")
        for r in risks:
            lines.append(f"• {_fmt_md(str(r))}")

    # Рекомендуемые действия
    steps = d.get("next_actions") or []
    if isinstance(steps, list) and steps:
        lines.append("\n*Шаги:*")
        for s in steps:
            lines.append(f"• {_fmt_md(str(s))}")

    # Источники (минимум 2 домена ожидается схемой, но рендерим всё, что пришло)
    sources = d.get("sources") or []
    if isinstance(sources, list) and sources:
        lines.append("\n*Источники:*")
        for s in sources:
            title = _fmt_md(str(s.get("title", "") or "Источник"))
            url = str(s.get("url", "") or "")
            if url:
                lines.append(f"• [{title}]({url})")
            else:
                lines.append(f"• {title}")

    # Уточняющие вопросы (если модель их вернула)
    clar = d.get("clarifications") or []
    if isinstance(clar, list) and clar:
        lines.append("\n*Что ещё нужно уточнить:*")
        for q in clar:
            lines.append(f"• {_fmt_md(str(q))}")

    return "\n".join(lines).strip() or _fmt_md(d.get("conclusion") or "")


@router.message(F.text & ~F.text.startswith("/"))
async def handle_legal_query(message: types.Message) -> None:
    """
    Основной хэндлер юридических вопросов.
    — Валидация длины
    — Rate-limit на пользователя
    — Индикация «печатает…»
    — Ответ + разбиение сообщения по 4096
    """
    assert _settings is not None and _ai is not None and _rl is not None

    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    text = (message.text or "").strip()

    log.info("IN: user=%s chat=%s len=%s", user_id, chat_id, len(text))

    # Мини-длина
    if len(text) < _settings.min_question_length:
        await message.answer(
            "✋ Вопрос слишком короткий. Пожалуйста, опишите ситуацию подробней.",
            parse_mode=None,
        )
        return

    # Rate-limit
    if not await _rl.check(user_id):
        remain = await _rl.remaining(user_id)
        msg = "⏳ Лимит вопросов на ближайший час исчерпан. Попробуйте позже."
        if remain:
            msg += f" Доступно ещё: {remain}."
        await message.answer(msg, parse_mode=None)
        return

    short_history: List[Dict[str, str]] = list(_history[user_id])

    try:
        # Индикатор «печатает…»
        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            # 1) Пытаемся получить строгий ответ по LEGAL_SCHEMA_V2
            md_text: Optional[str] = None
            assistant_summary: str = ""

            try:
                rich = await _ai.ask_ivan(text)
                data = rich.get("data")
                if isinstance(data, dict) and (data.get("conclusion") or data.get("sources") or data.get("cases")):
                    md_text = _schema_to_markdown(data)
                    assistant_summary = (data.get("conclusion") or "")[:1000]
            except Exception as e_json:
                log.warning("ask_ivan failed, fallback to simple path: %r", e_json)

            # 2) Фолбэк: упрощённый путь (answer + laws)
            if not md_text:
                result = await _ai.generate_legal_answer(text, short_history=short_history)
                answer: str = result.get("answer") or ""
                laws: List[str] = result.get("laws") or []
                assistant_summary = answer[:1000]
                md_text = build_legal_reply(answer=answer, laws=laws)

        # Обновляем короткую историю (user→assistant)
        _history[user_id].append({"role": "user", "content": text})
        _history[user_id].append({"role": "assistant", "content": assistant_summary})

        # Отправляем по кускам
        chunks = chunk_markdown_v2(md_text, limit=4096)
        for part in chunks:
            try:
                await message.answer(part, parse_mode=_settings.parse_mode)
            except TelegramBadRequest:
                # фолбэк если MarkdownV2 сломался — убираем экранирование
                await message.answer(strip_md2_escapes(part), parse_mode=None)

        log.info("OUT: user=%s chat=%s sent_chunks=%s", user_id, chat_id, len(chunks))

    except TelegramBadRequest as e:
        log.exception("TelegramBadRequest: user=%s chat=%s err=%r", user_id, chat_id, e)
        await message.answer(
            "😕 Ошибка форматирования сообщения. Отправляю без разметки.",
            parse_mode=None,
        )
    except Exception as e:
        log.exception("LLM handler error: user=%s chat=%s err=%r", user_id, chat_id, e)
        await message.answer(
            "😕 Произошла ошибка при обработке запроса. Попробуйте ещё раз через пару минут.",
            parse_mode=None,
        )
