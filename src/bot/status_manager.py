# status_manager.py
# Полноценный прогресс-бар для Telegram (aiogram v3):
# - единое сообщение с полосой, таймером и чек-листом из 7 шагов;
# - авто-цикл процентов, безопасное редактирование с троттлингом;
# - тумблер «Отображать контекстные вопросы» (inline-кнопка).

from __future__ import annotations
import asyncio
import time
from typing import Optional, Callable

from aiogram import Router, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

progress_router = Router()  # Роутер для обработчика тумблера

class ProgressStatus:
    PROGRESS_FLOW = [
        {"label": "Анализирую ваш запрос"},
        {"label": "Ищу законы и нормы права"},
        {"label": "Изучаю позицию высших судов"},
        {"label": "Формирую правовую позицию"},
        {"label": "Разрабатываю аргументацию"},
        {"label": "Проверяю точность и полноту"},
        {"label": "Формирую ответ"},
    ]

    def __init__(
        self,
        bot,
        chat_id: int,
        *,
        show_checklist: bool = True,
        show_context_toggle: bool = True,
        context_enabled_default: bool = False,
        min_edit_interval: float = 0.9,
        total_stages: int = 100,
        on_context_toggle: Optional[Callable[[bool], None]] = None,
    ):
        self.bot = bot
        self.chat_id = chat_id
        self.show_checklist = show_checklist
        self.show_context_toggle = show_context_toggle
        self.context_enabled = context_enabled_default
        self.min_edit_interval = float(min_edit_interval)
        self.total_stages = total_stages
        self.on_context_toggle = on_context_toggle

        self.message_id: Optional[int] = None
        self.current_percent: int = 0
        self.current_stage: int = 0  # 0..7
        self._last_edit_ts: float = 0.0
        self._running: bool = False
        self.start_time: Optional[float] = None
        self._lock = asyncio.Lock()

    # -------------------- ПУБЛИЧНЫЕ МЕТОДЫ --------------------

    async def start(self, auto_cycle: bool = True, interval: float = 2.0) -> None:
        """Создаёт сообщение прогресса и, при необходимости, запускает тикер."""
        self.start_time = time.monotonic()
        self._running = True
        content = self._render()
        msg = await self.bot.send_message(
            self.chat_id,
            content,
            parse_mode="HTML",
            reply_markup=self._kb() if self.show_context_toggle else None,
            disable_web_page_preview=True,
        )
        self.message_id = msg.message_id

        # регистрируемся, чтобы callback мог найти объект
        reg = getattr(self.bot, "_ps_registry", None)
        if reg is None:
            reg = self.bot._ps_registry = {}
        reg[(self.chat_id, self.message_id)] = self

        if auto_cycle:
            asyncio.create_task(self._ticker(float(interval)))

    async def complete(self, note: Optional[str] = None) -> None:
        """Успешное завершение."""
        self._running = False
        self.current_percent = 100
        if self.current_stage < len(self.PROGRESS_FLOW):
            self.current_stage = len(self.PROGRESS_FLOW)
        await self._safe_edit(self._render(completed=True, note=note))
        try:
            getattr(self.bot, "_ps_registry", {}).pop((self.chat_id, self.message_id), None)
        except Exception:
            pass

    async def fail(self, note: Optional[str] = None) -> None:
        """Завершение с ошибкой."""
        self._running = False
        await self._safe_edit(self._render(failed=True, note=note))
        try:
            getattr(self.bot, "_ps_registry", {}).pop((self.chat_id, self.message_id), None)
        except Exception:
            pass

    async def update_stage(self, percent: int, label: Optional[str] = None) -> None:
        """Обновить проценты и (опционально) продвинуть чек-лист до шага по label."""
        self.current_percent = max(self.current_percent, min(int(percent), 100))
        if label:
            for idx, item in enumerate(self.PROGRESS_FLOW, start=1):
                if item["label"] == label:
                    self.current_stage = max(self.current_stage, idx)
                    break
        await self._safe_edit(self._render())

    def duration_text(self) -> str:
        if not self.start_time:
            return "00:00"
        sec = int(time.monotonic() - self.start_time)
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # -------------------- ВНУТРЕННЕЕ --------------------

    async def _ticker(self, interval: float) -> None:
        """Небольшой автопрогресс, пока ждём ответ модели."""
        while self._running:
            await asyncio.sleep(interval)
            if self.current_percent < 90:
                self.current_percent += 1
            await self._safe_edit(self._render())

    async def _safe_edit(self, html: str) -> None:
        """Безопасно редактирует сообщение с троттлингом и блокировкой."""
        if not self.message_id:
            return
        now = time.monotonic()
        if now - self._last_edit_ts < self.min_edit_interval:
            return
        async with self._lock:
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                    text=html,
                    parse_mode="HTML",
                    reply_markup=self._kb() if self.show_context_toggle else None,
                    disable_web_page_preview=True,
                )
                self._last_edit_ts = now
            except Exception:
                # Игнорируем редкие гонки/ограничения Telegram
                pass

    def _kb(self) -> Optional[InlineKeyboardMarkup]:
        if not self.show_context_toggle:
            return None
        title = "Отображать контекстные вопросы: " + ("Вкл" if self.context_enabled else "Выкл")
        btn = InlineKeyboardButton(
            text=title,
            callback_data=f"ctx:{self.chat_id}:{self.message_id}",
        )
        return InlineKeyboardMarkup(inline_keyboard=[[btn]])

    def _render(self, *, completed: bool = False, failed: bool = False, note: Optional[str] = None) -> str:
        status = (
            "Действие выполнено" if completed else
            "Ошибка" if failed else
            "Действие выполняется"
        )
        bar = self._progress_bar(self.current_percent)
        lines = [
            f"<b>{status}</b> за <code>{self.duration_text()}</code>",
            bar,
            "",
        ]
        if self.show_checklist:
            for i, step in enumerate(self.PROGRESS_FLOW, start=1):
                mark = "●" if i <= self.current_stage else "○"
                lines.append(f"{mark} {step['label']}")
        if note:
            lines += ["", note]
        # небольшой запас до лимита 4096 с учётом тегов
        return "\n".join(lines)[:3990]

    @staticmethod
    def _progress_bar(pct: int) -> str:
        pct = max(0, min(100, int(pct)))
        blocks = 20
        filled = int(blocks * pct / 100)
        return "▰" * filled + "▱" * (blocks - filled) + f"  {pct}%"

# -------------------- CALLBACK: тумблер «контекстные вопросы» --------------------

@progress_router.callback_query(F.data.startswith("ctx:"))
async def _toggle_context(cb: CallbackQuery):
    try:
        _, chat_id, message_id = cb.data.split(":")
        key = (int(chat_id), int(message_id))
        reg = getattr(cb.bot, "_ps_registry", {})
        ps: Optional[ProgressStatus] = reg.get(key)
        if not ps or not cb.message or cb.message.chat.id != int(chat_id):
            await cb.answer()
            return
        ps.context_enabled = not ps.context_enabled
        await ps._safe_edit(ps._render())
        if callable(ps.on_context_toggle):
            try:
                await asyncio.shield(ps.on_context_toggle(ps.context_enabled))  # если on_context_toggle async
            except TypeError:
                ps.on_context_toggle(ps.context_enabled)  # если sync
        await cb.answer("Переключено")
    except Exception:
        await cb.answer()
