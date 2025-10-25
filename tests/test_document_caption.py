# mypy: ignore-errors
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_aiogram_stub() -> None:
    if "aiogram" in sys.modules:
        return

    aiogram_module = types.ModuleType("aiogram")

    class _Dispatcher:
        def __init__(self, *args, **kwargs): ...

    class _Router:
        def __init__(self, *args, **kwargs): ...

        def callback_query(self, *args, **kwargs):
            def decorator(handler):
                return handler

            return decorator

        def message(self, *args, **kwargs):
            def decorator(handler):
                return handler

            return decorator

    class _FilterAttr:
        def startswith(self, *args, **kwargs):
            return self

    class _F:
        data = _FilterAttr()

    class _Bot:
        def __init__(self, *args, **kwargs): ...
        async def send_chat_action(self, *args, **kwargs):
            return None

    aiogram_module.Dispatcher = _Dispatcher
    aiogram_module.Router = _Router
    aiogram_module.F = _F
    aiogram_module.Bot = _Bot

    enums_module = types.ModuleType("aiogram.enums")

    class _ParseMode:
        HTML = "HTML"

    enums_module.ParseMode = _ParseMode

    exceptions_module = types.ModuleType("aiogram.exceptions")
    exceptions_module.TelegramBadRequest = type(
        "TelegramBadRequest",
        (Exception,),
        {},
    )

    filters_module = types.ModuleType("aiogram.filters")
    filters_module.Command = object

    fsm_module = types.ModuleType("aiogram.fsm")
    fsm_context_module = types.ModuleType("aiogram.fsm.context")

    class _FSMContext:
        def __init__(self, *args, **kwargs): ...

    fsm_context_module.FSMContext = _FSMContext

    fsm_state_module = types.ModuleType("aiogram.fsm.state")

    class _State:
        def __init__(self, *args, **kwargs): ...

    class _StatesGroup:
        def __init__(self, *args, **kwargs): ...

    fsm_state_module.State = _State
    fsm_state_module.StatesGroup = _StatesGroup

    types_module = types.ModuleType("aiogram.types")

    class _CallbackQuery:
        def __init__(self, *args, **kwargs): ...

    class _FSInputFile:
        def __init__(self, *args, **kwargs): ...

    class _InlineKeyboardButton:
        def __init__(self, *args, **kwargs): ...

    class _InlineKeyboardMarkup:
        def __init__(self, *args, **kwargs): ...

    class _Message:
        def __init__(self, *args, **kwargs): ...

    types_module.CallbackQuery = _CallbackQuery
    types_module.FSInputFile = _FSInputFile
    types_module.InlineKeyboardButton = _InlineKeyboardButton
    types_module.InlineKeyboardMarkup = _InlineKeyboardMarkup
    types_module.Message = _Message

    sys.modules["aiogram"] = aiogram_module
    sys.modules["aiogram.enums"] = enums_module
    sys.modules["aiogram.exceptions"] = exceptions_module
    sys.modules["aiogram.filters"] = filters_module
    sys.modules["aiogram.fsm"] = fsm_module
    sys.modules["aiogram.fsm.context"] = fsm_context_module
    sys.modules["aiogram.fsm.state"] = fsm_state_module
    sys.modules["aiogram.types"] = types_module

    aiogram_module.enums = enums_module
    aiogram_module.exceptions = exceptions_module
    aiogram_module.filters = filters_module
    aiogram_module.fsm = fsm_module
    aiogram_module.types = types_module
    enums_module.ParseMode = _ParseMode
    fsm_module.context = fsm_context_module
    fsm_module.state = fsm_state_module


_ensure_aiogram_stub()


def _ensure_bot_app_stubs() -> None:
    async def _async_noop(*args, **kwargs):
        return None

    modules_to_stub = {
        "src.core.bot_app.menus": {"cmd_start": _async_noop},
        "src.core.bot_app.voice": {"download_voice_to_temp": _async_noop},
        "src.core.bot_app.payments": {},
    }

    for module_name, attributes in modules_to_stub.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        for attr_name, attr_value in attributes.items():
            setattr(module, attr_name, attr_value)
        sys.modules[module_name] = module


_ensure_bot_app_stubs()

from src.core.bot_app import documents as documents_module


class DummyBot:
    def __init__(self) -> None:
        self.deleted_messages: list[tuple] = []

    async def delete_message(self, *args, **kwargs) -> None:
        self.deleted_messages.append((args, kwargs))


class DummyMessage:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(id=123)
        self.bot = DummyBot()
        self.documents: list[dict] = []
        self.answers: list[dict] = []

    async def answer_document(self, document, caption: str, parse_mode: str | None = None):
        self.documents.append(
            {
                "document": document,
                "caption": caption,
                "parse_mode": parse_mode,
            }
        )

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append({"text": text, **kwargs})


class DummyState:
    def __init__(self) -> None:
        self._cleared = False

    async def get_data(self) -> dict:
        return {
            "draft_request": "request",
            "draft_plan": {"title": "Исковое заявление"},
            "draft_answers": [],
        }

    async def clear(self) -> None:
        self._cleared = True


class DummyProgressStatus:
    message_id: int | None = None

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        """Collect constructor params but do nothing."""

    async def start(self, **kwargs) -> None:  # noqa: D401
        """No-op for tests."""

    async def update_stage(self, **kwargs) -> None:  # noqa: D401
        """No-op for tests."""

    async def complete(self, **kwargs) -> None:  # noqa: D401
        """No-op for tests."""

    async def fail(self, **kwargs) -> None:  # noqa: D401
        """No-op for tests."""


class DummyTypingCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyFSInputFile:
    def __init__(self, path: str, filename: str | None = None) -> None:
        self.path = path
        self.filename = filename


class DummyResult:
    status = "ok"
    markdown = "# Отчёт"
    follow_up_questions: list[str] | None = None
    data: dict = {}

    def __init__(self, title: str, validated: list[str], issues: list[str]) -> None:
        self.title = title
        self.validated = validated
        self.issues = issues


def _setup_monkeypatch(monkeypatch):
    module = documents_module

    monkeypatch.setattr(module, "_get_openai_service", lambda: object())
    monkeypatch.setattr(module, "ProgressStatus", DummyProgressStatus)
    monkeypatch.setattr(module, "typing_action", lambda *args, **kwargs: DummyTypingCtx())
    monkeypatch.setattr(module, "FSInputFile", DummyFSInputFile)
    original_sleep = asyncio.sleep
    monkeypatch.setattr(module.asyncio, "sleep", lambda *args, **kwargs: original_sleep(0))

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(module.asyncio, "to_thread", immediate_to_thread)

    def fake_build_docx(markdown: str, output_path: str) -> None:
        Path(output_path).write_text("dummy docx content", encoding="utf-8")

    monkeypatch.setattr(module, "build_docx_from_markdown", fake_build_docx)

    repeated_segment = "А" * 400
    validated_items = [f"Подтверждение {idx}: {repeated_segment}" for idx in range(6)]
    issues_items = [f"Риск {idx}: {repeated_segment}" for idx in range(6)]

    async def fake_generate(openai_service, request_text, title, answers):
        return DummyResult(
            title="Исковое заявление", validated=validated_items, issues=issues_items
        )

    monkeypatch.setattr(module, "generate_document", fake_generate)

    return module


def test_finalize_draft_truncates_caption_and_preserves_html(monkeypatch):
    module = _setup_monkeypatch(monkeypatch)
    state = DummyState()
    message = DummyMessage()

    async def run():
        await module._finalize_draft(message, state)

    asyncio.run(run())

    assert state._cleared is True
    assert message.documents, "Document should be sent"
    doc_call = message.documents[0]
    caption = doc_call["caption"]

    assert len(caption) <= module._CAPTION_MAX_LENGTH
    assert "А" * 250 not in caption  # ensure long tail truncated
    assert "…" in caption
    assert "Документ успешно создан" in caption
    assert "<b>Проверено</b>" in caption
    assert doc_call["parse_mode"] == module.ParseMode.HTML
