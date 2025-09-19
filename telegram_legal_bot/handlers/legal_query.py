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
    md2,
    build_legal_reply,
    chunk_markdown_v2,
    strip_md2_escapes,
    _escape_md2_url as escape_md2_url,  # безопасные URL для MarkdownV2
)

router = Router(name="legal_query")
log = logging.getLogger("legal_query")

# ── Глобальный контекст хэндлера (инициализируется из main.py) ────────────────
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


def _schema_to_markdown(d: Dict[str, Any]) -> str:
    """
    Рендер ответа по LEGAL_SCHEMA_V2 → MarkdownV2.
    Покрыты поля: conclusion, legal_basis(act, article, pinpoint, quote),
    cases(court, date, case_no, url, holding, facts, norms, similarity),
    analysis, risks, next_actions, sources(title, url), doc_drafts(title, doc_type),
    clarifications.
    """
    lines: List[str] = []

    def add(s: str) -> None:
        if s:
            lines.append(s)

    # Краткий вывод
    conclusion = (d.get("conclusion") or "").strip()
    if conclusion:
        add(f"*Кратко:* {md2(conclusion)}")

    # Нормы права
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
            add(head.strip() if head.strip() != "•" else "• Норма")
            if quote:
                add(f"  └ «{quote}»")

    # Подборка дел
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

    # Аналитика
    analysis = (d.get("analysis") or "").strip()
    if analysis:
        add("\n*Аналитика:*")
        add(md2(analysis))

    # Риски
    risks = d.get("risks") or []
    if isinstance(risks, list) and risks:
        add("\n*Риски:*")
        for r in risks:
            add("• " + md2(str(r)))

    # Шаги
    steps = d.get("next_actions") or []
    if isinstance(steps, list) and steps:
        add("\n*Шаги:*")
        for s in steps:
            add("• " + md2(str(s)))

    # Источники
    sources = d.get("sources") or []
    if isinstance(sources, list) and sources:
        add("\n*Источники:*")
        for s in sources:
            title = md2(str(s.get("title") or "Источник"))
            u_raw = str(s.get("url") or "")
            u = escape_md2_url(u_raw) if u_raw else ""
            add(f"• [{title}]({u})" if u else f"• {title}")

    # Черновики документов
    drafts = d.get("doc_drafts") or []
    if isinstance(drafts, list) and drafts:
        add("\n*Черновики документов:*")
        for doc in drafts:
            title = md2(str(doc.get("title") or "Документ"))
            dtype = md2(str(doc.get("doc_type") or ""))
            add(f"• *{title}*{f' ({dtype})' if dtype else ''} — доступен черновик")

    # Уточнения
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
    Основной хэндлер юридических вопросов:
      • валидация длины,
      • per-user rate-limit,
      • индикатор «печатает…»,
      • попытка строгого режима (LEGAL_SCHEMA_V2) и фолбэк,
      • безопасная разбивка сообщения по 4096 символов.
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
                rich = await _ai.ask_ivan(text)  # точный режим (без истории)
                data = rich.get("data")
                if isinstance(data, dict) and (data.get("conclusion") or data.get("sources") or data.get("cases")):
                    md_text = _schema_to_markdown(data)
                    assistant_summary = (data.get("conclusion") or "")[:1000]
            except Exception as e_json:
                log.warning("ask_ivan failed, fallback to simple path: %r", e_json)

            # 2) Фолбэк: упрощённый путь (answer + laws) — с краткой историей
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
                # Фолбэк: если MarkdownV2 сломался — убираем экранирование
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
