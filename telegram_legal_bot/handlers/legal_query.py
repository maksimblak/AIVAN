# telegram_legal_bot/handlers/legal_query.py
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


class LegalQueryHandler:
    """
    Thread-safe обработчик юридических запросов.
    Заменяет глобальные переменные на инкапсулированное состояние.
    """
    
    def __init__(self, settings: Settings, ai: OpenAIService):
        self.settings = settings
        self.ai = ai
        self.rate_limiter = RateLimiter(
            max_calls=settings.max_requests_per_hour, 
            period_seconds=3600
        )
        
        # Thread-safe история с автоматической очисткой старых записей
        pairs = max(1, int(settings.history_size))
        self._history: Dict[int, Deque[Dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=pairs * 2)
        )
        self._history_lock = asyncio.Lock()
        self._history_cleanup_time = time.time()
        # Последнее обращение пользователя к истории (для корректной очистки наименее активных)
        self._last_access: Dict[int, float] = {}
        
        # Максимальное количество пользователей в истории (защита от утечки памяти)
        self.max_users_in_history = 10000
        
    async def _cleanup_history_if_needed(self) -> None:
        """Периодическая очистка истории для предотвращения утечек памяти.
        Запускается редко и сама берёт lock, чтобы не создавать дедлоков.
        """
        now = time.time()
        # Очищаем историю не чаще, чем раз в 6 часов
        if now - self._history_cleanup_time <= 21600:  # 6 * 60 * 60
            return
        async with self._history_lock:
            if len(self._history) > self.max_users_in_history:
                overflow = len(self._history) - self.max_users_in_history
                # Сортируем по последнему доступу (наименее активные — вперёд)
                sorted_users = sorted(
                    self._history.keys(),
                    key=lambda uid: self._last_access.get(uid, 0.0)
                )
                to_remove = sorted_users[:overflow]
                for uid in to_remove:
                    self._history.pop(uid, None)
                    self._last_access.pop(uid, None)
                log.info("Cleaned up history for %d users", len(to_remove))
            self._history_cleanup_time = now
    
    async def get_user_history(self, user_id: int) -> List[Dict[str, str]]:
        """Безопасное получение истории пользователя без реэнтрантного захвата lock."""
        # Сначала, вне lock, решаем вопрос с периодической очисткой (сама возьмёт lock внутри)
        await self._cleanup_history_if_needed()
        async with self._history_lock:
            self._last_access[user_id] = time.time()
            return list(self._history[user_id])
    
    async def add_to_history(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        """Безопасное добавление в историю."""
        async with self._history_lock:
            self._history[user_id].append({"role": "user", "content": user_msg})
            self._history[user_id].append({"role": "assistant", "content": assistant_msg[:1000]})
            self._last_access[user_id] = time.time()


# Глобальный экземпляр обработчика (инициализируется в setup_context)
_handler: Optional[LegalQueryHandler] = None


def setup_context(settings: Settings, ai: OpenAIService) -> None:
    """
    Инициализация thread-safe обработчика (вызывается из main.py).
    """
    global _handler
    _handler = LegalQueryHandler(settings, ai)


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
            head_stripped = head.strip()
            add(head_stripped if head_stripped != "•" else "• Норма")
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
    if _handler is None:
        log.error("Handler not initialized! Call setup_context() first.")
        await message.answer(
            "⚠️ Сервис временно недоступен. Попробуйте позже.",
            parse_mode=None,
        )
        return

    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    text = (message.text or "").strip()

    # Валидация входных данных
    if not text:
        await message.answer(
            "✋ Пожалуйста, отправьте текстовый вопрос.",
            parse_mode=None,
        )
        return

    log.info("IN: user=%s chat=%s len=%s", user_id, chat_id, len(text))

    # Мини-длина
    if len(text) < _handler.settings.min_question_length:
        error_text = BotMessages.error_message("invalid_question")
        
        try:
            await message.answer(
                md2(error_text),
                parse_mode="MarkdownV2"
            )
        except Exception:
            await message.answer(
                "✋ Вопрос слишком короткий. Пожалуйста, опишите ситуацию подробней.",
                parse_mode=None
            )
        return

    # Rate-limit
    if not await _handler.rate_limiter.check(user_id):
        # Вычисляем оставшееся время до сброса лимита
        remaining_time = 3600  # Примерное время в секундах
        
        rate_limit_text = BotMessages.rate_limit_message(remaining_time)
        
        try:
            await message.answer(
                md2(rate_limit_text),
                parse_mode="MarkdownV2"
            )
        except Exception:
            remain = await _handler.rate_limiter.remaining(user_id)
            fallback_text = "⏳ Лимит вопросов на ближайший час исчерпан. Попробуйте позже."
            if remain:
                fallback_text += f" Доступно ещё: {remain}."
            await message.answer(
                fallback_text,
                parse_mode=None
            )
        return

    # Получаем историю thread-safe способом
    short_history: List[Dict[str, str]] = await _handler.get_user_history(user_id)

    progress_msg: Optional[types.Message] = None
    try:
        # Показываем красивую анимацию обработки
        thinking_steps = BotMessages.thinking_messages()
        progress_msg = await BotAnimations.progress_message(message, thinking_steps, delay=1.0)
        
        # Индикатор «печатает…»
        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            # 1) Пытаемся получить строгий ответ по LEGAL_SCHEMA_V2
            md_text: Optional[str] = None
            assistant_summary: str = ""

            try:
                rich = await _handler.ai.ask_ivan(text)  # точный режим (без истории)
                data = rich.get("data") if rich else None
                citations = rich.get("citations") or []
                if isinstance(data, dict) and (data.get("conclusion") or data.get("sources") or data.get("cases")):
                    md_text = _schema_to_markdown(data)
                    # Добавим блок «Ссылки из поиска» из citations для повышения проверяемости
                    if citations:
                        md_text += "\n\n*Ссылки из поиска:*\n" + "\n".join(
                            f"• [{md2(c.get('title') or 'Источник')}]({escape_md2_url(str(c.get('url') or ''))})" for c in citations if c.get('url')
                        )
                    assistant_summary = (data.get("conclusion") or "")[:1000]
            except Exception as e_json:
                log.warning("ask_ivan failed, fallback to simple path: %r", e_json)

            # 2) Фолбэк: пробуем отрисовать V2 JSON напрямую; если формат другой — используем старый answer/laws
            if not md_text:
                result = await _handler.ai.generate_legal_answer(text, short_history=short_history)
                if isinstance(result, dict) and (result.get("conclusion") or result.get("sources") or result.get("cases")):
                    md_text = _schema_to_markdown(result)
                    assistant_summary = (result.get("conclusion") or "")[:1000]
                else:
                    answer: str = result.get("answer") or "" if result else ""
                    laws: List[str] = result.get("laws") or [] if result else []
                    assistant_summary = answer[:1000] if answer else "Ответ не получен"
                    md_text = build_legal_reply(answer=answer, laws=laws)

        # Обновляем короткую историю thread-safe способом
        await _handler.add_to_history(user_id, text, assistant_summary)

        # Удаляем сообщение о прогрессе если удалось создать
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        
        # Отправляем по кускам  
        chunks = chunk_markdown_v2(md_text or "Не удалось получить ответ.", limit=4096)
        
        for i, part in enumerate(chunks):
            try:
                await message.answer(
                    part, 
                    parse_mode=_handler.settings.parse_mode
                )
            except TelegramBadRequest:
                # Фолбэк: если MarkdownV2 сломался — убираем экранирование
                await message.answer(
                    strip_md2_escapes(part), 
                    parse_mode=None
                )

        log.info("OUT: user=%s chat=%s sent_chunks=%s", user_id, chat_id, len(chunks))

    except TelegramBadRequest as e:
        log.exception("TelegramBadRequest: user=%s chat=%s err=%r", user_id, chat_id, e)
        await message.answer(
            "😕 Ошибка форматирования сообщения. Отправляю без разметки.",
            parse_mode=None,
        )
    except Exception as e:
        log.exception("LLM handler error: user=%s chat=%s err=%r", user_id, chat_id, e)
        
        # Удаляем прогресс-сообщение в случае ошибки
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        
        error_text = BotMessages.error_message("general")
        
        try:
            await message.answer(
                md2(error_text),
                parse_mode="MarkdownV2"
            )
        except Exception:
            await message.answer(
                "😕 Произошла ошибка при обработке запроса. Попробуйте ещё раз через пару минут.",
                parse_mode=None
            )

