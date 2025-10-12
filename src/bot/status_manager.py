# src/bot/status_manager.py
from __future__ import annotations
import asyncio
import time
import math
from typing import Optional, Callable, List, Dict
from html import escape as esc

from aiogram import Dispatcher, Router, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

__all__ = ["ProgressStatus", "progress_router", "register_progressbar"]

# Роутер для callback-кнопки в самом сообщении прогресса
progress_router = Router()


class ProgressStatus:
    """
    Двухстрочный прогресс:
      <b>● Текущий шаг</b>
      ○ Следующий шаг

    Предыдущие завершённые шаги не отображаются.
    После complete() шаги скрываются (только заголовок/полоса/таймер).
    """

    BAR_WIDTH = 30  # ширина текстовой полосы прогресса

    def __init__(
        self,
        bot,
        chat_id: int,
        *,
        steps: Optional[List[Dict[str, str]] | List[str]] = None,
        show_checklist: bool = True,
        show_context_toggle: bool = True,
        context_enabled_default: bool = False,
        min_edit_interval: float = 0.9,
        total_stages: int = 100,
        on_context_toggle: Optional[Callable[[bool], None]] = None,
        auto_advance_stages: bool = True,
        percent_thresholds: Optional[List[int]] = None,
        two_step_only: bool = True,
        bold_current: bool = True,
        hide_steps_on_complete: bool = True,
        display_total_seconds: int | None = None,
    ):
        self.bot = bot
        self.chat_id = chat_id
        self.show_checklist = show_checklist
        self.show_context_toggle = show_context_toggle
        self.context_enabled = context_enabled_default
        self.min_edit_interval = float(min_edit_interval)
        self.total_stages = total_stages
        self.on_context_toggle = on_context_toggle

        default_steps = [
            {"icon": "🔎", "label": "Анализирую ваш запрос"},
            {"icon": "📚", "label": "Ищу законы и нормы права"},
            {"icon": "⚖️", "label": "Изучаю позицию высших судов"},
            {"icon": "🧠", "label": "Формирую правовую позицию"},
            {"icon": "🧩", "label": "Разрабатываю аргументацию"},
            {"icon": "✅", "label": "Проверяю точность и полноту"},
            {"icon": "📝", "label": "Формирую ответ"},
        ]
        self.steps: List[Dict[str, str]] = self._normalize_steps(steps) if steps else default_steps

        self.auto_advance = auto_advance_stages
        self.percent_thresholds = percent_thresholds

        self.two_step_only = bool(two_step_only)
        self.bold_current = bool(bold_current)
        self.hide_steps_on_complete = bool(hide_steps_on_complete)
        self.display_total_seconds = (
            int(display_total_seconds) if display_total_seconds and display_total_seconds > 0 else None
        )

        self.message_id: Optional[int] = None
        self.current_percent: int = 0
        self.current_stage: int = 1  # 1..len(self.steps)
        self._last_edit_ts: float = 0.0
        self._running: bool = False
        self.start_time: Optional[float] = None
        self._lock = asyncio.Lock()

    # ---------- ПУБЛИЧНЫЕ МЕТОДЫ ----------

    async def start(self, auto_cycle: bool = True, interval: float = 2.0) -> None:
        self.start_time = time.monotonic()
        self._running = True
        msg = await self.bot.send_message(
            self.chat_id,
            self._render(),
            parse_mode="HTML",
            reply_markup=self._kb() if self.show_context_toggle else None,
            disable_web_page_preview=True,
        )
        self.message_id = msg.message_id
        reg = getattr(self.bot, "_ps_registry", None) or {}
        reg[(self.chat_id, self.message_id)] = self
        self.bot._ps_registry = reg
        if auto_cycle:
            asyncio.create_task(self._ticker(float(interval)))

    async def complete(self, note: Optional[str] = None) -> None:
        self._running = False
        self.current_percent = 100
        self.current_stage = len(self.steps)
        await self._safe_edit(self._render(completed=True, note=note))
        getattr(self.bot, "_ps_registry", {}).pop((self.chat_id, self.message_id), None)

    async def fail(self, note: Optional[str] = None) -> None:
        self._running = False
        await self._safe_edit(self._render(failed=True, note=note))
        getattr(self.bot, "_ps_registry", {}).pop((self.chat_id, self.message_id), None)

    async def update_stage(
        self,
        percent: Optional[int] = None,
        label: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if percent is not None:
            self.current_percent = max(self.current_percent, min(int(percent), 100))

        target_stage = None
        if isinstance(step, int) and 1 <= step <= len(self.steps):
            target_stage = step
        elif label:
            norm = self._norm(label)
            for idx, s in enumerate(self.steps, start=1):
                if self._norm(s["label"]) == norm:
                    target_stage = idx
                    break
        if target_stage is not None:
            self.current_stage = max(self.current_stage, target_stage)

        await self._safe_edit(self._render())

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _elapsed_seconds(self) -> int:
        if not self.start_time:
            return 0
        return int(time.monotonic() - self.start_time)

    def duration_text(self, percent: int) -> str:
        if self.display_total_seconds is not None:
            projected = int(round(self.display_total_seconds * max(0, min(percent, 100)) / 100))
            return self._format_duration(projected)
        return self._format_duration(self._elapsed_seconds())

    # ---------- ВНУТРЕННЕЕ ----------

    async def _ticker(self, interval: float) -> None:
        while self._running:
            await asyncio.sleep(interval)
            percent_changed = False
            if self.display_total_seconds:
                duration = max(1, self.display_total_seconds)
                elapsed = self._elapsed_seconds()
                target = int((elapsed / duration) * 90)
                target = max(0, min(target, 90))
                if target > self.current_percent:
                    self.current_percent = target
                    percent_changed = True
            else:
                if self.current_percent < 90:
                    self.current_percent += 1
                    percent_changed = True
            stage_changed = False
            if self.auto_advance:
                new_stage = self._stage_by_percent(self.current_percent)
                if new_stage is not None and new_stage > self.current_stage:
                    self.current_stage = new_stage
                    stage_changed = True
            if percent_changed or stage_changed:
                await self._safe_edit(self._render())

    def _stage_by_percent(self, pct: int) -> Optional[int]:
        if not self.steps:
            return None
        pct = max(0, min(100, int(pct)))
        if self.percent_thresholds:
            stage = 1
            for i, thr in enumerate(self.percent_thresholds, start=1):
                if pct >= int(thr):
                    stage = i
            return max(1, min(stage, len(self.steps)))
        bins = len(self.steps)
        if bins <= 1:
            return 1
        return max(1, min(len(self.steps), math.ceil(pct / (100 / bins))))

    async def _safe_edit(self, html: str) -> None:
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
                pass

    def _kb(self) -> Optional[InlineKeyboardMarkup]:
        if not self.show_context_toggle:
            return None
        title = "Отображать контекстные вопросы: " + ("Вкл" if self.context_enabled else "Выкл")
        return InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text=title, callback_data=f"ctx:{self.chat_id}:{self.message_id}")
        ]])

    def _render(self, *, completed: bool = False, failed: bool = False, note: str | None = None) -> str:
        header = "✅ ГОТОВО" if completed else "❌ ОШИБКА" if failed else "📱 ВЫПОЛНЕНИЕ..."
        pct = max(0, min(100, int(self.current_percent)))
        duration_display = self.duration_text(pct)
        lines: list[str] = [
            f"<b>{header}</b>",
            f"<code>{'━' * self.BAR_WIDTH}</code>",
            f"<code>{self._progress_bar(pct)}</code>",
            f"{pct}% • {esc(duration_display)}",
            "",
        ]

        # Только текущий и следующий; предыдущие шаги не показываем
        if self.show_checklist and self.steps:
            if not (completed and self.hide_steps_on_complete):
                idx = max(1, min(self.current_stage, len(self.steps)))
                cur_label = esc(self.steps[idx - 1].get("label", ""))
                cur_line = f"● {cur_label}"
                if self.bold_current:
                    cur_line = f"<b>{cur_line}</b>"
                lines.append(cur_line)

                if idx < len(self.steps):  # следующий (если есть)
                    next_label = esc(self.steps[idx].get("label", ""))
                    lines.append(f"○ {next_label}")

        if note:
            lines += ["", esc(note)]
        return "\n".join(lines)[:3990]

    def _progress_bar(self, pct: int) -> str:
        pct = max(0, min(100, int(pct)))
        filled = int(round(self.BAR_WIDTH * pct / 100))
        return "█" * filled + "░" * (self.BAR_WIDTH - filled)

    @staticmethod
    def _normalize_steps(steps: List[Dict[str, str]] | List[str]) -> List[Dict[str, str]]:
        norm: List[Dict[str, str]] = []
        for s in steps:
            if isinstance(s, str):
                norm.append({"icon": "•", "label": s})
            elif isinstance(s, dict):
                norm.append({"label": s.get("label", ""), "icon": s.get("icon", "•")})
        return norm

    @staticmethod
    def _norm(s: str) -> str:
        import re
        s = re.sub(r"[^\w\sА-Яа-яЁё-]+", "", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s


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
                await asyncio.shield(ps.on_context_toggle(ps.context_enabled))
            except TypeError:
                ps.on_context_toggle(ps.context_enabled)
        await cb.answer("Переключено")
    except Exception:
        await cb.answer()


def register_progressbar(dp: Dispatcher) -> None:
    """Attach the progress-router callbacks to the dispatcher."""
    dp.include_router(progress_router)
