# rich_reply.py
from __future__ import annotations

import re
from dataclasses import dataclass

from aiogram.enums import ParseMode
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

# ---- Настройки ----
MAX_MSG = 3900  # безопасный лимит под HTML
ALLOWED_TAGS = (
    "b",
    "i",
    "u",
    "a",
    "code",
    "pre",
    "blockquote",
    "span",
)  # spoiler поддерживается span class="tg-spoiler"


@dataclass
class Section:
    id: str
    title: str
    html: str


@dataclass
class Article:
    request_id: str
    tldr_html: str
    sections: list[Section]


def _build_nav_keyboard(sections: list[Section], req_id: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=f"• {s.title}", callback_data=f"sec:{req_id}:{s.id}")]
        for s in sections
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _split_html_by_paragraphs(html: str, hard_limit: int = MAX_MSG) -> list[str]:
    chunks, cur, length = [], [], 0
    for para in re.split(r"\n{2,}", html.strip()):
        p = para.strip()
        if not p:
            continue
        add = p + "\n\n"
        if length + len(add) > hard_limit and cur:
            chunks.append("".join(cur).strip())
            cur, length = [add], len(add)
        else:
            cur.append(add)
            length += len(add)
    if cur:
        chunks.append("".join(cur).strip())
    return chunks


async def _send_chunked(message: Message, html: str):
    for part in _split_html_by_paragraphs(html):
        await message.answer(part, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


# ---- Автосегментация монолита (простой и надёжный хелпер) ----
HEAD_RE = re.compile(
    r"^\s*(?:\d+\)\s+|[•\-]\s+|—\s+)?(?P<title>(tl;dr|тл;др|итоги|что сделать|шаги|куда обращаться|шаблон|правовые основания).{0,40})$",
    re.I,
)


def article_from_monolith(request_id: str, text: str) -> Article:
    # TL;DR = первые 1–2 абзаца (до ~350 символов)
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    tldr = []
    total = 0
    for p in paras:
        tldr.append(p)
        total += len(p)
        if total >= 350 or len(tldr) >= 2:
            break
    tldr_html = "<b>Коротко</b> ✅\n" + "<br/>\n".join(tldr)

    # Поиск заголовков будущих секций
    sections: list[Section] = []
    current_title = None
    current_buf: list[str] = []

    def flush():
        nonlocal current_title, current_buf
        if current_buf:
            title = current_title or "Детали"
            body = "\n\n".join(current_buf).strip()
            sections.append(Section(id=f"s{len(sections)+1}", title=title, html=body))
        current_title, current_buf = None, []

    for p in paras[len(tldr) :]:
        m = HEAD_RE.match(p.lower())
        if m:
            flush()
            title = m.group("title").strip().capitalize()
            # вычистим маркер в оригинале
            clean_title = re.sub(r"^(?:\d+\)|[•\-]|—)\s*", "", title, flags=re.I)
            current_title = clean_title
        else:
            current_buf.append(p)

    flush()
    if not sections:
        sections = [Section(id="s1", title="Детали", html="\n\n".join(paras[len(tldr) :]) or "—")]

    # лёгкая HTML-разметка: абзацы -> <br/>, списки сохраним как есть
    def to_html(txt: str) -> str:
        # Превращаем URL в ссылки
        txt = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', txt)
        return (
            txt.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>\n")
        )

    return Article(
        request_id=request_id,
        tldr_html=to_html(tldr_html),
        sections=[Section(s.id, s.title, to_html(s.html)) for s in sections],
    )


# ---- Публичное API ----
class RichReply:
    """
    Помогает красиво показать ответ: TL;DR + кнопки секций + отправка длинных секций чанками.
    Хранит секции в session_store под ключом article:{request_id}
    """

    def __init__(self, bot, session_store):
        self.bot = bot
        self.session_store = session_store  # ожидание интерфейса get_or_create()/set()

    async def send_article(self, message: Message, article: Article):
        # сохраним секции в сессии (TTL живёт по настройкам вашего SessionStore)
        await self.session_store.set(
            f"article:{article.request_id}",
            {
                "sections": [
                    {"id": s.id, "title": s.title, "html": s.html} for s in article.sections
                ]
            },
        )
        kb = _build_nav_keyboard(article.sections, article.request_id)
        await message.answer(
            article.tldr_html,
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
            disable_web_page_preview=True,
        )

    async def handle_callback(self, call: CallbackQuery):
        # ожидаем callback_data вида sec:{req_id}:{sec_id}
        try:
            _, req_id, sec_id = call.data.split(":", 2)
        except Exception:
            await call.answer("Неизвестная секция", show_alert=False)
            return

        data: dict | None = await self.session_store.get(f"article:{req_id}")
        if not data:
            await call.answer("Данные устарели. Сформируйте ответ заново.", show_alert=True)
            return

        sec = next((s for s in data["sections"] if s["id"] == sec_id), None)
        if not sec:
            await call.answer("Секция не найдена", show_alert=False)
            return

        await call.answer()
        # длинные секции — порежем и отправим несколькими сообщениями
        html = f"<b>{sec['title']}</b>\n" + sec["html"]
        await _send_chunked(call.message, html)
