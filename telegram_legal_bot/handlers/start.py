from __future__ import annotations

from aiogram import Router, types
from aiogram.filters import Command

from telegram_legal_bot.utils.message_formatter import md2

router = Router(name="start")


WELCOME_TEXT = (
    "Привет! Я — юридический ассистент. Помогаю быстро разобраться "
    "в вопросах по праву РФ: от бытовых ситуаций до деловой переписки.\n\n"
    "Как пользоваться:\n"
    "• Опиши свою ситуацию или задай вопрос\n"
    "• Я отвечу кратко и по делу, а также приложу используемые нормы\n\n"
    "⚠️ Помни: я не заменяю юриста, а помогаю сориентироваться."
)

HELP_TEXT = (
    "Отправь текст вопроса — я постараюсь ответить и показать нормы права.\n\n"
    "Подсказки:\n"
    "• Пиши контекст: кто, где, когда и что именно произошло\n"
    "• Если есть документы/сроки — укажи их\n"
    "• Чем точнее вводные, тем полезнее ответ"
)


@router.message(Command("start"))
async def cmd_start(msg: types.Message) -> None:
    try:
        await msg.answer(md2(WELCOME_TEXT), parse_mode="MarkdownV2")
    except Exception:
        await msg.answer(WELCOME_TEXT, parse_mode=None)


@router.message(Command("help"))
async def cmd_help(msg: types.Message) -> None:
    try:
        await msg.answer(md2(HELP_TEXT), parse_mode="MarkdownV2")
    except Exception:
        await msg.answer(HELP_TEXT, parse_mode=None)
