import asyncio
import html
from typing import List, Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest

DEFAULT_STEPS_RU = [
    "Анализирую ваш запрос",
    "Ищу законы и нормы права",
    "Изучаю позицию высших судов",
    "Формирую правовую позицию",
    "Разрабатываю аргументацию",
    "Проверяю точность и полноту",
    "Формирую ответ",
]


class AwaitChecklist:
    """
    Прогресс-виджет в Telegram: прогресс-бар + таймер + чек-лист шагов.
    - Авто-тикает раз в секунду (с троттлингом правок).
    - Можно двигать прогресс вручную (по ходу стрима).
    - Безопасный HTML, короткий текст (не бьёт лимиты Telegram).
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        steps: Optional[List[str]] = None,
        *,
        throttle_sec: float = 1.2,         # не редактировать чаще
        expected_duration_sec: int = 45,   # для авто-прогресса по времени
    ):
        self.bot = bot
        self.chat_id = chat_id
        self.steps = steps or DEFAULT_STEPS_RU
        self.n = len(self.steps)

        # состояние
        loop = asyncio.get_event_loop()
        self._t0 = loop.time()
        self._progress = 0.0               # 0..1
        self._current_step = 0
        self._throttle = throttle_sec
        self._expected = max(10, expected_duration_sec)

        # tg
        self.message_id: Optional[int] = None
        self._tick_task: Optional[asyncio.Task] = None
        self._last_edit_ts = 0.0
        self._stopped = False

    # ---------- публичный API ----------

    async def start(self) -> int:
        text = self._render()
        msg = await self.bot.send_message(
            self.chat_id, text,
            parse_mode="HTML", disable_web_page_preview=True,
        )
        self.message_id = msg.message_id
        self._tick_task = asyncio.create_task(self._ticker())
        return msg.message_id

    async def finish(self, final_text: Optional[str] = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._tick_task:
            self._tick_task.cancel()
        if not self.message_id:
            return
        # финальный рендер — все шаги зелёные
        txt = final_text or self._render(final=True)
        try:
            await self.bot.edit_message_text(
                txt, chat_id=self.chat_id, message_id=self.message_id,
                parse_mode="HTML", disable_web_page_preview=True
            )
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e).lower():
                raise

    def set_progress(self, fraction: float) -> None:
        """0..1 — можно дергать из стрим-колбэка; без await."""
        fraction = max(0.0, min(1.0, fraction))
        self._progress = fraction
        # переводим прогресс в текущий шаг, но сохраняем последний «бОльший» шаг
        est_step = min(self.n - 1, int(self._progress * self.n))
        if est_step > self._current_step:
            self._current_step = est_step

    async def advance(self, step_index: Optional[int] = None) -> None:
        """Ручной прыжок на шаг (или +1)."""
        if step_index is None:
            step_index = self._current_step + 1
        self._current_step = max(0, min(self.n - 1, step_index))
        await self._maybe_edit()

    # ---------- внутренности ----------

    async def _ticker(self):
        try:
            while not self._stopped:
                await asyncio.sleep(1.0)
                # авто-прогресс по времени: плавно двигаем индикатор,
                # если стрим ещё не прислал реальный прогресс
                elapsed = asyncio.get_event_loop().time() - self._t0
                time_progress = max(self._progress, min(1.0, elapsed / self._expected))
                if time_progress > self._progress:
                    self._progress = time_progress
                    est_step = min(self.n - 1, int(self._progress * self.n))
                    self._current_step = max(self._current_step, est_step)
                await self._maybe_edit()
        except asyncio.CancelledError:
            pass

    async def _maybe_edit(self):
        if not self.message_id:
            return
        now = asyncio.get_event_loop().time()
        if (now - self._last_edit_ts) < self._throttle:
            return
        self._last_edit_ts = now
        txt = self._render()
        try:
            await self.bot.edit_message_text(
                txt, chat_id=self.chat_id, message_id=self.message_id,
                parse_mode="HTML", disable_web_page_preview=True
            )
        except TelegramBadRequest as e:
            # игнорируем «message is not modified»
            if "message is not modified" not in str(e).lower():
                raise

    # ---------- рендер ----------

    def _elapsed_hhmmss(self) -> str:
        s = int(asyncio.get_event_loop().time() - self._t0)
        return f"{s//60:02d}:{s%60:02d}"

    def _progress_bar(self, width: int = 24) -> str:
        p = max(0.0, min(1.0, self._progress))
        filled = int(round(width * p))
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"<code>[{bar}] {int(p*100):02d}%</code>"

    def _render(self, final: bool = False) -> str:
        # заголовок
        header = f"<b>Действие выполняется</b> за <code>{self._elapsed_hhmmss()}</code>"
        # список шагов
        lines = []
        for i, title in enumerate(self.steps):
            safe_title = html.escape(title)
            if final or i < self._current_step:
                bullet = "✅"
            elif i == self._current_step:
                bullet = "🟡"
            else:
                bullet = "⚪️"
            lines.append(f"{bullet} {safe_title}")
        checklist = "<br>".join(lines)
        return f"{self._progress_bar()}<br>{header}<br><br>{checklist}"
