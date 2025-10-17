from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from contextlib import asynccontextmanager
from src.core.app_context import get_settings
from src.core.settings import AppSettings

import html
from html.parser import HTMLParser
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from urllib.parse import quote, urlparse

import httpx
from openai import AsyncOpenAI
from src.core.attachments import QuestionAttachment

logger = logging.getLogger(__name__)


def _build_user_message_content(user_text: str, attachments: Sequence[QuestionAttachment] | None) -> Any:
    if not attachments:
        return user_text

    content: list[dict[str, Any]] = []
    base_text = (user_text or "").strip()
    if base_text:
        content.append({"type": "text", "text": base_text})

    inline_limit = _get_env_non_negative_int("OPENAI_INLINE_ATTACHMENT_MAX_BYTES", 65536)

    for item in attachments:
        if not item or not getattr(item, "data", b""):
            continue
        mime = (item.mime_type or "application/octet-stream").lower()
        data_bytes = getattr(item, "data", b"") or b""
        size_bytes = getattr(item, "size", None)
        if size_bytes is None:
            size_bytes = len(data_bytes)

        if mime.startswith("image/"):
            encoded = base64.b64encode(data_bytes).decode("ascii") if data_bytes else ""
            data_url = f"data:{mime};base64,{encoded}"
            content.append({"type": "input_image", "image_url": {"url": data_url}})
        else:
            details = (
                f"Attached file: {item.filename or 'file'}\n"
                f"MIME type: {item.mime_type}\n"
                f"Size: {size_bytes} bytes\n"
            )
            if inline_limit and data_bytes and size_bytes <= inline_limit:
                encoded = base64.b64encode(data_bytes).decode("ascii")
                chunk = f"{details}Base64 contents:\n{encoded}"
            elif inline_limit and size_bytes > inline_limit:
                chunk = (
                    f"{details}Contents omitted: size exceeds inline limit of {inline_limit} bytes."
                )
            else:
                chunk = f"{details}Contents omitted: inline attachments are disabled."
            content.append({"type": "text", "text": chunk})

    if not content:
        return base_text or user_text
    return content



MODEL_CAPABILITIES: dict[str, dict[str, bool]] = {}
_VALIDATED_MODELS: set[str] = set()

_client_lock = asyncio.Lock()
_shared_async_client: AsyncOpenAI | None = None


def _infer_model_caps(model: str) -> dict[str, bool]:
    lower = model.lower()
    caps: dict[str, bool] = {}
    if lower.startswith("gpt-5") or (lower.startswith("o") and not lower.startswith("omni")):
        caps["supports_sampling"] = False
    return caps



def _normalise_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-convert Mapping instances into plain dict/list structures."""
    def _convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return {str(key): _convert(val) for key, val in data.items()}


def _format_structured_legal_response(data: Mapping[str, Any]) -> str:
    summary = str(data.get('summary') or '').strip()
    analysis = str(data.get('analysis') or '').strip()
    disclaimer = str(data.get('disclaimer') or '').strip()
    legal_basis = data.get('legal_basis') or []
    risks = data.get('risks') or []

    parts: list[str] = []
    if summary:
        parts.append(f"<b>РљСЂР°С‚РєРѕ:</b> {html.escape(summary)}")
    if analysis:
        parts.append(f"<b>РђРЅР°Р»РёР·:</b> {html.escape(analysis)}")

    basis_lines: list[str] = []
    for item in legal_basis:
        if not isinstance(item, Mapping):
            continue
        reference = str(item.get('reference') or '').strip()
        explanation = str(item.get('explanation') or '').strip()
        if reference and explanation:
            basis_lines.append(f"вЂў {html.escape(reference)} вЂ” {html.escape(explanation)}")
        elif reference:
            basis_lines.append(f"вЂў {html.escape(reference)}")
    if basis_lines:
        basis_block = '\n'.join(basis_lines)
        parts.append(f"<b>РџСЂР°РІРѕРІРѕРµ РѕСЃРЅРѕРІР°РЅРёРµ:</b>\n{basis_block}")

    risk_lines: list[str] = []
    for risk in risks:
        risk_text = str(risk or '').strip()
        if risk_text:
            risk_lines.append(f"вЂў {html.escape(risk_text)}")
    if risk_lines:
        risk_block = '\n'.join(risk_lines)
        parts.append(f"<b>Р РёСЃРєРё:</b>\n{risk_block}")

    if disclaimer:
        parts.append(f"<i>{html.escape(disclaimer)}</i>")

    combined = '\n\n'.join(part for part in parts if part).strip()
    return combined


def _extract_text_from_content(
    content: Any,
    structured_sink: list[Mapping[str, Any]] | None = None,
) -> str | None:
    def _store(candidate: Mapping[str, Any] | None) -> None:
        if candidate and structured_sink is not None and not structured_sink:
            structured_sink.append(_normalise_mapping(candidate))

    if hasattr(content, "model_dump"):
        try:
            dumped = content.model_dump()
            if isinstance(dumped, Mapping):
                _store(dumped)
            return json.dumps(dumped, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            pass
    if hasattr(content, "json") and callable(getattr(content, "json")):
        try:
            return content.json(exclude_none=True)
        except Exception:
            pass

    text_value = getattr(content, 'text', None)
    if text_value:
        return text_value
    if isinstance(content, dict):
        dict_text = content.get('text')
        if dict_text:
            return str(dict_text)

    for key in ('parsed', 'data', 'json'):
        candidate = getattr(content, key, None)
        if candidate is None and isinstance(content, dict):
            candidate = content.get(key)
        if candidate is not None:
            if isinstance(candidate, Mapping):
                _store(candidate)
                formatted = _format_structured_legal_response(candidate)
                if formatted:
                    return formatted
            try:
                return json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
            except Exception:
                return str(candidate)

    json_schema = getattr(content, 'json_schema', None)
    if json_schema is None and isinstance(content, dict):
        json_schema = content.get('json_schema')
    if isinstance(json_schema, Mapping):
        _store(json_schema)
        for key in ('parsed', 'data'):
            candidate = json_schema.get(key)
            if isinstance(candidate, Mapping):
                _store(candidate)
                formatted = _format_structured_legal_response(candidate)
                if formatted:
                    return formatted
        formatted = _format_structured_legal_response(json_schema)
        if formatted:
            return formatted

    return None

StreamCallback = Callable[[str, bool], Awaitable[None] | None]

def _settings() -> AppSettings:
    return get_settings()


__all__ = [
    "ask_legal",
    "ask_legal_stream",
    "format_legal_response_text",
    "get_async_openai_client",
    "shared_openai_client",
    "close_async_openai_client",
]


def _get_env_float(name: str, default: float) -> float:
    settings = _settings()
    raw = settings.get_str(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Invalid %s value '%s'; falling back to %s", name, raw, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    settings = _settings()
    raw = settings.get_str(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Invalid %s value '%s'; falling back to %s", name, raw, default)
        return default




def _get_env_non_negative_int(name: str, default: int) -> int:
    settings = _settings()
    raw = settings.get_str(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Invalid %s value '%s'; falling back to %s", name, raw, default)
        return default


def _resolve_proxy_url() -> str | None:
    settings = _settings()
    proxy = settings.get_str("OPENAI_HTTP_PROXY") or settings.get_str("OPENAI_PROXY")
    if not proxy:
        return None
    proxy = proxy.strip()
    if not proxy:
        return None
    if '://' not in proxy:
        proxy = f"http://{proxy}"

    user = settings.get_str("OPENAI_PROXY_USER")
    password = settings.get_str("OPENAI_PROXY_PASSWORD") or settings.get_str("OPENAI_PROXY_PASS")
    if user and password and '@' not in proxy:
        parsed = urlparse(proxy)
        netloc = parsed.netloc or parsed.path
        netloc = netloc.lstrip('@')
        credentials = f"{quote(user.strip(), safe='')}:{quote(password.strip(), safe='')}"
        proxy = parsed._replace(netloc=f"{credentials}@{netloc}").geturl()
    return proxy


def _build_http_client() -> httpx.AsyncClient:
    timeout_total = _get_env_float("OPENAI_HTTP_TIMEOUT", 600.0)
    connect_timeout = _get_env_float("OPENAI_HTTP_CONNECT_TIMEOUT", min(timeout_total, 15.0))
    read_timeout = _get_env_float("OPENAI_HTTP_READ_TIMEOUT", timeout_total)
    write_timeout = _get_env_float("OPENAI_HTTP_WRITE_TIMEOUT", timeout_total)
    pool_timeout = _get_env_float("OPENAI_HTTP_POOL_TIMEOUT", min(connect_timeout, 10.0))

    timeout = httpx.Timeout(
        timeout_total,
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )

    limits = httpx.Limits(
        max_connections=_get_env_int("OPENAI_HTTP_MAX_CONNECTIONS", 20),
        max_keepalive_connections=_get_env_int("OPENAI_HTTP_MAX_KEEPALIVE", 10),
    )

    proxy = _resolve_proxy_url()
    client_kwargs: dict[str, Any] = {
        'timeout': timeout,
        'limits': limits,
    }
    if proxy:
        client_kwargs['proxies'] = proxy

    verify = _settings().get_str("OPENAI_CA_BUNDLE")
    if verify:
        client_kwargs['verify'] = verify

    return httpx.AsyncClient(**client_kwargs)


async def _create_async_client() -> AsyncOpenAI:
    settings = _settings()
    api_key = (
        settings.openai_api_key
        or settings.get_str("OPENAI_KEY")
        or settings.get_str("AZURE_OPENAI_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    base_url = (
        settings.get_str("OPENAI_BASE_URL")
        or settings.get_str("OPENAI_API_BASE")
        or settings.get_str("AZURE_OPENAI_ENDPOINT")
    )
    organization = (
        settings.get_str("OPENAI_ORGANIZATION")
        or settings.get_str("OPENAI_ORG_ID")
    )
    project = settings.get_str("OPENAI_PROJECT")

    http_client = _build_http_client()
    max_retries = _get_env_non_negative_int("OPENAI_MAX_RETRIES", 1)
    client_kwargs: dict[str, Any] = {
        'api_key': api_key,
        'http_client': http_client,
        'max_retries': max_retries,
    }
    if base_url:
        client_kwargs['base_url'] = base_url.rstrip('/')
    if organization:
        client_kwargs['organization'] = organization
    if project:
        client_kwargs['project'] = project

    try:
        return AsyncOpenAI(**client_kwargs)
    except Exception:
        await http_client.aclose()
        raise


async def get_async_openai_client() -> AsyncOpenAI:
    """Return a shared AsyncOpenAI client instance."""
    global _shared_async_client

    client = _shared_async_client
    if client is not None:
        return client

    async with _client_lock:
        client = _shared_async_client
        if client is None:
            client = await _create_async_client()
            _shared_async_client = client
        return client


@asynccontextmanager
async def shared_openai_client() -> AsyncIterator[AsyncOpenAI]:
    """Async context manager yielding the shared AsyncOpenAI client."""
    client = await get_async_openai_client()
    try:
        yield client
    finally:
        # The shared client is retained for reuse; shutdown occurs separately.
        pass


async def close_async_openai_client() -> None:
    """Close the shared AsyncOpenAI client if it was created."""
    global _shared_async_client

    client_to_close: AsyncOpenAI | None = None
    async with _client_lock:
        client_to_close = _shared_async_client
        _shared_async_client = None

    if client_to_close is None:
        return

    try:
        await client_to_close.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to close OpenAI client: %s", exc)



async def ask_legal(
    system_prompt: str,
    user_text: str,
    *,
    attachments: Sequence[QuestionAttachment] | None = None,
) -> dict[str, Any]:
    """Public wrapper used by OpenAIService for non-streaming replies."""
    return await _ask_legal_internal(system_prompt, user_text, stream=False, attachments=attachments)


async def ask_legal_stream(
    system_prompt: str,
    user_text: str,
    callback: StreamCallback | None = None,
    *,
    attachments: Sequence[QuestionAttachment] | None = None,
) -> dict[str, Any]:
    """Public wrapper that enables streaming responses through a callback."""
    return await _ask_legal_internal(
        system_prompt, user_text, stream=True, callback=callback, attachments=attachments
    )


LEGAL_RESPONSE_SCHEMA = {
    "name": "legal_response",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Short summary for the user",
            },
            "legal_basis": {
                "type": "array",
                "description": "List of cited legal sources with references",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "string",
                            "description": "For example: Civil Code, Article 432",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Why the norm is applicable",
                        },
                    },
                    "required": ["reference"],
                    "additionalProperties": False,
                },
            },
            "analysis": {
                "type": "string",
                "description": "Key reasoning and conclusions",
            },
            "risks": {
                "type": "array",
                "description": "Potential risks or alternative interpretations",
                "items": {"type": "string"},
            },
            "disclaimer": {
                "type": "string",
                "description": "Disclaimer that the answer is informational",
            },
        },
        "required": ["summary", "legal_basis", "analysis", "disclaimer"],
        "additionalProperties": False,
    },
}

class _TelegramHTMLFormatter(HTMLParser):
    """Convert model output into Telegram-compatible HTML."""

    _TAG_SYNONYMS = {
        "strong": "b",
        "em": "i",
        "ins": "u",
        "del": "s",
        "strike": "s",
    }
    _BULLET_SYMBOL = "в–« "
    _INDENT_UNIT = "&nbsp;&nbsp;"

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.tag_stack: list[str | None] = []
        self.list_stack: list[dict[str, Any]] = []
        self.in_pre = 0
        self.in_details = 0
        self.in_summary = False
        self.summary_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        tag_norm = self._TAG_SYNONYMS.get(tag_lower, tag_lower)
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag_norm == "br":
            self.parts.append("<br>")
            return

        if tag_norm in {"p", "div"}:
            self._ensure_breaks(2)
            self.tag_stack.append(None)
            return

        if tag_norm in {"ul", "ol"}:
            self._ensure_breaks(1)
            self.list_stack.append({"type": tag_norm, "index": 1})
            self.tag_stack.append(tag_norm)
            return

        if tag_norm == "li":
            self._ensure_breaks(1)
            indent = self._INDENT_UNIT * max(len(self.list_stack) - 1, 0)
            bullet = indent + self._BULLET_SYMBOL
            if self.list_stack:
                current = self.list_stack[-1]
                if current["type"] == "ol":
                    bullet = f"{indent}<b>{current['index']}.</b> "
                    current["index"] += 1
            self.parts.append(bullet)
            self.tag_stack.append("li")
            return

        if tag_norm == "details":
            self.in_details += 1
            self.tag_stack.append("details")
            return

        if tag_norm == "summary":
            if self.in_details:
                self.in_summary = True
                self.summary_buffer = []
            self.tag_stack.append("summary")
            return

        if tag_norm == "span" and "class" in attrs_dict and "tg-spoiler" in attrs_dict["class"]:
            self.parts.append("<tg-spoiler>")
            self.tag_stack.append("tg-spoiler")
            return

        if tag_norm == "tg-spoiler":
            self.parts.append("<tg-spoiler>")
            self.tag_stack.append("tg-spoiler")
            return

        if tag_norm == "a":
            href = attrs_dict.get("href", "")
            if href and href.lower().startswith(("http://", "https://", "tg://user?id=")):
                safe_href = html.escape(href, quote=True)
                self.parts.append(f'<a href="{safe_href}">')
                self.tag_stack.append("a")
            else:
                self.tag_stack.append(None)
            return

        if tag_norm == "pre":
            self._ensure_breaks(1)
            self.parts.append("<pre>")
            self.tag_stack.append("pre")
            self.in_pre += 1
            return

        if tag_norm == "code":
            self.parts.append("<code>")
            self.tag_stack.append("code")
            return

        if tag_norm == "blockquote":
            self._ensure_breaks(1)
            self.parts.append("<blockquote>")
            self.tag_stack.append("blockquote")
            return

        if tag_norm in {"b", "i", "u", "s"}:
            self.parts.append(f"<{tag_norm}>")
            self.tag_stack.append(tag_norm)
            return

        self.tag_stack.append(None)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        tag_norm = self._TAG_SYNONYMS.get(tag_lower, tag_lower)

        if tag_norm == "li":
            if self._pop_expected("li"):
                self._ensure_breaks(1)
            return

        if tag_norm in {"ul", "ol"}:
            if self.list_stack and self.list_stack[-1]["type"] == tag_norm:
                self.list_stack.pop()
            self._pop_expected(tag_norm)
            self._ensure_breaks(1)
            return

        if tag_norm == "details":
            if self.in_details:
                self.in_details -= 1
            self._pop_expected("details")
            return

        if tag_norm == "summary":
            self._flush_summary()
            self._pop_expected("summary")
            return

        if tag_norm == "pre":
            if self._pop_expected("pre"):
                self.in_pre = max(0, self.in_pre - 1)
                self.parts.append("</pre>")
                self._ensure_breaks(1)
            return

        if tag_norm == "code":
            if self._pop_expected("code"):
                self.parts.append("</code>")
            return

        if tag_norm == "a":
            if self._pop_expected("a"):
                self.parts.append("</a>")
            return

        if tag_norm in {"tg-spoiler", "blockquote"}:
            if self._pop_expected(tag_norm):
                self.parts.append(f"</{tag_norm}>")
                if tag_norm == "blockquote":
                    self._ensure_breaks(1)
            return

        if tag_norm in {"b", "i", "u", "s"}:
            if self._pop_expected(tag_norm):
                self.parts.append(f"</{tag_norm}>")
            return

        self._pop_expected(tag_norm)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "br":
            self.parts.append("<br>")
            return
        if tag_lower == "tg-spoiler":
            self.parts.append("<tg-spoiler></tg-spoiler>")
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self.in_summary:
            self.summary_buffer.append(data)
            return
        escaped = html.escape(data)
        escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
        if self.in_pre:
            self.parts.append(escaped)
        else:
            self.parts.append(escaped.replace("\n", "<br>"))

    def close(self) -> None:
        self._flush_summary()
        super().close()

    def get_text(self) -> str:
        result = "".join(self.parts)
        result = re.sub(r"(?:<br>\s*){3,}", "<br><br>", result)
        result = re.sub(r"^(?:<br>\s*)+", "", result)
        result = re.sub(r"(?:<br>\s*)+$", "", result)
        return result.strip()

    def _ensure_breaks(self, amount: int) -> None:
        if amount <= 0:
            return
        if not self.parts:
            return
        existing = 0
        for part in reversed(self.parts):
            if part != "<br>":
                break
            existing += 1
        for _ in range(max(0, amount - existing)):
            self.parts.append("<br>")

    def _pop_expected(self, expected: str) -> bool:
        while self.tag_stack:
            current = self.tag_stack.pop()
            if current is None:
                continue
            if current == expected:
                return True
        return False

    def _flush_summary(self) -> None:
        if not self.in_summary:
            return
        summary = "".join(self.summary_buffer).strip()
        self.in_summary = False
        self.summary_buffer = []
        if not summary:
            return
        summary = html.escape(summary)
        summary = summary.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary).strip()
        if not summary:
            return
        self._ensure_breaks(1)
        self.parts.append(f"<b>{summary}</b>")
        self._ensure_breaks(1)


def format_legal_response_text(raw: str) -> str:
    """Convert raw response text to Telegram-compatible HTML."""
    raw_text = (raw or '').strip()
    if not raw_text:
        return ''
    formatter = _TelegramHTMLFormatter()
    formatter.feed(raw_text)
    formatter.close()
    return formatter.get_text()



async def _ask_legal_internal(
    system_prompt: str,
    user_text: str,
    stream: bool = False,
    callback=None,
    attachments: Sequence[QuestionAttachment] | None = None,
) -> dict[str, Any]:
    """Unified Responses API invocation with optional streaming."""
    settings = _settings()
    model = (settings.get_str("OPENAI_MODEL") or "gpt-5").strip()
    max_out = settings.get_int("MAX_OUTPUT_TOKENS", 4096)
    verb = (settings.get_str("OPENAI_VERBOSITY") or "medium").lower()
    effort = (settings.get_str("OPENAI_REASONING_EFFORT") or "medium").lower()
    temperature = settings.get_float("OPENAI_TEMPERATURE", 0.15)
    top_p = settings.get_float("OPENAI_TOP_P", 0.3)

    user_message = _build_user_message_content(user_text, attachments)

    base_core: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "text": {"verbosity": verb},
        "reasoning": {"effort": effort},
        "max_output_tokens": max_out,
    }
    sampling_payload = {"temperature": temperature, "top_p": top_p}

    model_caps = MODEL_CAPABILITIES.setdefault(model, {})
    inferred_caps = _infer_model_caps(model)
    for key, value in inferred_caps.items():
        model_caps[key] = value
    include_sampling = model_caps.get("supports_sampling", True)

    if not settings.get_bool("DISABLE_WEB", False):
        base_core |= {"tools": [{"type": "web_search"}], "tool_choice": "auto"}

    async with shared_openai_client() as oai:
        stored_schema = model_caps.get("supports_schema")
        if stored_schema is None:
            schema_supported = True
            model_caps["supports_schema"] = schema_supported
        else:
            schema_supported = stored_schema

        def build_attempts(
            include_schema: bool,
            include_sampling_params: bool,
            use_response_format: bool,
        ) -> list[dict[str, Any]]:
            payload_base: dict[str, Any] = {**base_core}
            text_config = dict(base_core.get("text") or {})
            extra_body = dict(payload_base.get("extra_body") or {})

            if include_schema:
                if use_response_format:
                    extra_body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": LEGAL_RESPONSE_SCHEMA,
                    }
                else:
                    text_config["format"] = {
                        "type": "json_schema",
                        "json_schema": LEGAL_RESPONSE_SCHEMA,
                    }

            if extra_body:
                payload_base["extra_body"] = extra_body
            if text_config:
                payload_base["text"] = text_config
            if include_sampling_params:
                payload_base |= sampling_payload

            with_tools = payload_base
            without_tools = {k: v for k, v in payload_base.items() if k != "tools"}
            boosted = without_tools | {"max_output_tokens": max_out * 2}
            return [with_tools, without_tools, boosted]

        supports_response_format = model_caps.get("supports_response_format", True)
        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)

        last_err = None
        if model not in _VALIDATED_MODELS:
            if not settings.get_bool("OPENAI_SKIP_MODEL_CHECK", False):
                for i in range(2):
                    try:
                        await oai.models.retrieve(model)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = str(e)
                        if i == 1:
                            return {"ok": False, "error": last_err}
                        await asyncio.sleep(0.6)
            _VALIDATED_MODELS.add(model)

        structured_payload: dict[str, Any] | None = None

        while True:
            retry_payload = False
            for payload in attempts:
                try:
                    if stream and callback:
                        accumulated_text = ""
                        usage_info = None

                        async with oai.responses.stream(**payload) as s:
                            async for event in s:
                                etype = getattr(event, "type", "") or ""

                                if etype.endswith(".delta"):
                                    delta = getattr(event, "delta", None)
                                    piece = ""
                                    if isinstance(delta, dict):
                                        piece = delta.get("text") or ""
                                    if not piece:
                                        piece = getattr(event, "output_text", "") or ""

                                    if piece:
                                        accumulated_text += piece
                                        try:
                                            cb_result = callback(piece, False)
                                            if inspect.isawaitable(cb_result):
                                                await cb_result
                                        except Exception as cb_err:
                                            logger.warning("Callback error during streaming: %s", cb_err)

                                if etype == "response.completed":
                                    continue

                            final = await s.get_final_response()
                            usage_info = getattr(final, "usage", None)
                            text = getattr(final, "output_text", None)
                            items = getattr(final, "output", []) or []
                            structured_collector: list[Mapping[str, Any]] = []
                            if not text:
                                chunks: list[str] = []
                                for it in items:
                                    contents = []
                                    if hasattr(it, "content") and getattr(it, "content"):
                                        contents = getattr(it, "content")
                                    elif isinstance(it, dict):
                                        contents = it.get("content") or []
                                    before = len(chunks)
                                    for c in contents:
                                        extracted = _extract_text_from_content(c, structured_collector)
                                        if extracted:
                                            chunks.append(extracted)
                                    if not contents or len(chunks) == before:
                                        extracted_item = _extract_text_from_content(it, structured_collector)
                                        if extracted_item:
                                            chunks.append(extracted_item)
                                text = "\n\n".join(chunks) if chunks else ""
                            else:
                                for it in items:
                                    contents = []
                                    if hasattr(it, "content") and getattr(it, "content"):
                                        contents = getattr(it, "content")
                                    elif isinstance(it, dict):
                                        contents = it.get("content") or []
                                    if contents:
                                        for c in contents:
                                            _extract_text_from_content(c, structured_collector)
                                    else:
                                        _extract_text_from_content(it, structured_collector)
                            if not structured_payload and structured_collector:
                                structured_payload = structured_collector[0]

                            final_raw = (text or accumulated_text or "").strip()
                            logger.debug("OpenAI raw response (stream): %s", final_raw)
                            formatted_final = final_raw
                            if callback and formatted_final:
                                try:
                                    cb_result = callback(formatted_final, True)
                                    if inspect.isawaitable(cb_result):
                                        await cb_result
                                except Exception as cb_err:
                                    logger.warning("Final callback error: %s", cb_err)

                            if formatted_final:
                                return {
                                    "ok": True,
                                    "text": formatted_final,
                                    "usage": usage_info,
                                    "structured": structured_payload,
                                }

                            logger.warning("OpenAI stream returned no text; trying next payload option")
                            last_err = "empty_response"
                            continue

                    resp = await oai.responses.create(**payload)
                    text = getattr(resp, "output_text", None)
                    items = getattr(resp, "output", []) or []
                    structured_collector: list[Mapping[str, Any]] = []
                    if items:
                        for it in items:
                            contents = []
                            if hasattr(it, "content") and getattr(it, "content"):
                                contents = getattr(it, "content")
                            elif isinstance(it, dict):
                                contents = it.get("content") or []
                            if contents:
                                for c in contents:
                                    _extract_text_from_content(c, structured_collector)
                            else:
                                _extract_text_from_content(it, structured_collector)
                    if not structured_payload and structured_collector:
                        structured_payload = structured_collector[0]
                    if text and text.strip():
                        raw = text.strip()
                        logger.debug("OpenAI raw response: %s", raw)
                        return {
                            "ok": True,
                            "text": raw,
                            "usage": getattr(resp, "usage", None),
                            "structured": structured_payload,
                        }

                    chunks: list[str] = []
                    for it in items:
                        contents = []
                        if hasattr(it, "content") and getattr(it, "content"):
                            contents = getattr(it, "content")
                        elif isinstance(it, dict):
                            contents = it.get("content") or []
                        before = len(chunks)
                        for c in contents:
                            extracted = _extract_text_from_content(c, structured_collector)
                            if extracted:
                                chunks.append(extracted)
                        if not contents or len(chunks) == before:
                            extracted_item = _extract_text_from_content(it, structured_collector)
                            if extracted_item:
                                chunks.append(extracted_item)
                    if chunks:
                        joined = "\n\n".join(chunks).strip()
                        logger.debug("OpenAI raw response (joined): %s", joined)
                        return {
                            "ok": True,
                            "text": joined,
                            "usage": getattr(resp, "usage", None),
                            "structured": structured_payload,
                        }

                    logger.warning("OpenAI response contained no text output; trying next payload option")
                    last_err = "empty_response"
                    continue

                except TypeError as type_err:
                    message = str(type_err)
                    schema_payload_used = (
                        schema_supported
                        and (
                            (
                                isinstance(payload.get("text"), Mapping)
                                and isinstance(payload["text"].get("format"), Mapping)
                            )
                            or (
                                isinstance(payload.get("extra_body"), Mapping)
                                and isinstance(payload["extra_body"].get("response_format"), Mapping)
                            )
                        )
                    )
                    if schema_payload_used and any(token in message for token in ("format", "json_schema")):
                        logger.warning(
                            "Responses API rejected structured output format; retrying without schema enforcement"
                        )
                        schema_supported = False
                        model_caps["supports_schema"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    if include_sampling and any(token in message for token in ("temperature", "top_p")):
                        logger.warning(
                            "Responses API rejected sampling parameters; retrying with defaults"
                        )
                        include_sampling = False
                        model_caps["supports_sampling"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    if supports_response_format and "response_format" in message:
                        logger.warning(
                            "Responses API client rejected response_format parameter; retrying with legacy schema embedding"
                        )
                        supports_response_format = False
                        model_caps["supports_response_format"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    last_err = message
                    break
                except Exception as e:
                    message = str(e)
                    schema_payload_used = (
                        schema_supported
                        and (
                            (
                                isinstance(payload.get("text"), Mapping)
                                and isinstance(payload["text"].get("format"), Mapping)
                            )
                            or (
                                isinstance(payload.get("extra_body"), Mapping)
                                and isinstance(payload["extra_body"].get("response_format"), Mapping)
                            )
                        )
                    )
                    if schema_payload_used and any(token in message for token in ("json_schema", "format", "structured")):
                        logger.warning(
                            "OpenAI API error while using structured output; retrying without schema enforcement"
                        )
                        schema_supported = False
                        model_caps["supports_schema"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    if include_sampling and any(token in message for token in ("temperature", "top_p")):
                        logger.warning(
                            "Responses API rejected sampling parameters; retrying with defaults"
                        )
                        include_sampling = False
                        model_caps["supports_sampling"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    if supports_response_format and "response_format" in message:
                        logger.warning(
                            "Responses API rejected response_format parameter; retrying with legacy schema embedding"
                        )
                        supports_response_format = False
                        model_caps["supports_response_format"] = False
                        attempts = build_attempts(schema_supported, include_sampling, supports_response_format)
                        retry_payload = True
                        break
                    last_err = message

            if retry_payload:
                continue
            break

        return {"ok": False, "error": last_err or "unknown_error", "structured": structured_payload}

