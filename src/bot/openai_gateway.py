from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from dotenv import load_dotenv
import inspect
from openai import AsyncOpenAI

load_dotenv()



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



def _format_legal_sections(payload: dict[str, Any]) -> str:
    sections = [
        ("summary", "Summary"),
        ("legal_basis", "Legal Basis"),
        ("analysis", "Analysis"),
        ("risks", "Risks"),
        ("disclaimer", "Disclaimer"),
    ]
    lines: list[str] = []

def format_legal_response_text(raw: str) -> str:
    if not raw:
        return ''
    candidate = raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return candidate
    if isinstance(data, dict):
        formatted = _format_legal_sections(data)
        if formatted:
            return formatted
    return candidate


    def _ensure_lines(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            collected: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    collected.append(item.strip())
                elif isinstance(item, dict):
                    ref = (item.get('reference') or item.get('citation') or '').strip()
                    expl = (item.get('explanation') or item.get('comment') or '').strip()
                    parts = [p for p in (ref, expl) if p]
                    if parts:
                        collected.append(': '.join(parts))
            return collected
        return []

    for key, title in sections:
        values = _ensure_lines(payload.get(key))
        if not values:
            continue
        lines.append(f"{title}:")
        if key in {"legal_basis", "risks"}:
            lines.extend(f"- {item}" for item in values)
        else:
            lines.extend(values)
        lines.append('')

    return "\n".join(item for item in lines if item).strip()


logger = logging.getLogger(__name__)
REDACT_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "openai-api-key"}


def _bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _ssl_verify():
    if _bool(os.getenv("SSL_NO_VERIFY"), False):
        return False
    ca = os.getenv("SSL_CA_BUNDLE", "").strip()
    return ca or True


def _redact_headers(headers: dict) -> dict:
    safe = {}
    for k, v in (headers or {}).items():
        lk = k.lower()
        if any(tok in lk for tok in ["key", "token", "secret", "cookie"]) or lk in REDACT_HEADERS:
            safe[k] = "***"
        else:
            safe[k] = v
    return safe


def build_proxy_url() -> str | None:
    url = (os.getenv("OPENAI_PROXY_URL") or os.getenv("TELEGRAM_PROXY_URL") or "").strip()
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    u = urlparse(url)
    user = (os.getenv("OPENAI_PROXY_USER") or os.getenv("TELEGRAM_PROXY_USER") or "").strip()
    pwd = (os.getenv("OPENAI_PROXY_PASS") or os.getenv("TELEGRAM_PROXY_PASS") or "").strip()
    if user and pwd and not u.username:
        host = u.hostname or ""
        port = f":{u.port}" if u.port else ""
        return f"{u.scheme}://{quote(user, safe='')}:{quote(pwd, safe='')}@{host}{port}"
    return url


async def _make_async_client() -> AsyncOpenAI:
    proxy = build_proxy_url()
    http2 = _bool(os.getenv("HTTP2", "1"), True)
    verify = _ssl_verify()

    async def on_req(request: httpx.Request):
        request.extensions["start"] = time.perf_counter()

    logger = logging.getLogger("http.client")

    async def on_resp(response: httpx.Response):
        start = response.request.extensions.get("start")
        dur = (time.perf_counter() - start) if start else None
        try:
            logger.info(
                json.dumps(
                    {
                        "event": "http",
                        "method": response.request.method,
                        "url": str(response.request.url).split("://", 1)[-1],
                        "status": response.status_code,
                        "duration_ms": int((dur or 0) * 1000),
                        "headers": _redact_headers(dict(response.request.headers)),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass

    transport = httpx.AsyncHTTPTransport(http2=http2, verify=verify)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=20.0, read=180.0, write=120.0, pool=60.0),
        proxies=proxy,
        transport=transport,
        event_hooks={"request": [on_req], "response": [on_resp]},
        trust_env=False,
    )
    return AsyncOpenAI(http_client=client)


async def ask_legal(system_prompt: str, user_text: str) -> dict[str, Any]:
    """Запрос на Responses API без стриминга.

    Возвращает dict: { ok: bool, text?: str, usage?: Any, error?: str }
    """
    return await _ask_legal_internal(system_prompt, user_text, stream=False)


async def ask_legal_stream(system_prompt: str, user_text: str, callback=None):
    """Streaming запрос к Responses API.

    Callback принимает (partial_text: str, is_final: bool)
    Возвращает dict: { ok: bool, text?: str, usage?: Any, error?: str }
    """
    return await _ask_legal_internal(system_prompt, user_text, stream=True, callback=callback)


async def _ask_legal_internal(
    system_prompt: str, user_text: str, stream: bool = False, callback=None
) -> dict[str, Any]:
    """Unified Responses API invocation with optional streaming."""
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    max_out = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
    verb = os.getenv("OPENAI_VERBOSITY", "medium").lower()
    effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").lower()
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.15"))
    except (TypeError, ValueError):
        temperature = 0.15
    try:
        top_p = float(os.getenv("OPENAI_TOP_P", "0.3"))
    except (TypeError, ValueError):
        top_p = 0.3

    base_common: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "text": {"verbosity": verb},
        "reasoning": {"effort": effort},
        "max_output_tokens": max_out,
        "temperature": temperature,
        "top_p": top_p,
    }

    if not (os.getenv("DISABLE_WEB", "0").strip() in ("1", "true", "yes", "on")):
        base_common |= {"tools": [{"type": "web_search"}], "tool_choice": "auto"}

    async with await _make_async_client() as oai:
        schema_supported = "response_format" in inspect.signature(oai.responses.create).parameters
        schema_payload: dict[str, Any] = {}
        if schema_supported:
            schema_payload = {"response_format": {"type": "json_schema", "json_schema": LEGAL_RESPONSE_SCHEMA}}
        else:
            logger.debug("Responses API does not accept response_format; falling back to raw JSON parsing")

        def build_attempts(include_schema: bool) -> list[dict[str, Any]]:
            payload_base = base_common | (schema_payload if include_schema else {})
            with_tools = payload_base
            without_tools = {k: v for k, v in payload_base.items() if k != "tools"}
            boosted = without_tools | {"max_output_tokens": max_out * 2}
            return [with_tools, without_tools, boosted]

        attempts = build_attempts(schema_supported)

        last_err = None
        for i in range(2):
            try:
                await oai.models.retrieve(model)
                break
            except Exception as e:
                last_err = str(e)
                if i == 1:
                    return {"ok": False, "error": last_err}
                await asyncio.sleep(0.6)

        while True:
            schema_retry = False
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
                                            if asyncio.iscoroutinefunction(callback):
                                                await callback(accumulated_text, False)
                                            else:
                                                callback(accumulated_text, False)
                                        except Exception as cb_err:
                                            logger.warning("Callback error during streaming: %s", cb_err)

                                if etype == "response.completed":
                                    continue

                            final = await s.get_final_response()
                            usage_info = getattr(final, "usage", None)
                            text = getattr(final, "output_text", None)
                            if not text:
                                items = getattr(final, "output", []) or []
                                chunks: list[str] = []
                                for it in items:
                                    for c in getattr(it, "content", []) or []:
                                        t = getattr(c, "text", None)
                                        if t:
                                            chunks.append(t)
                                text = "\n\n".join(chunks) if chunks else ""



                            final_raw = (text or accumulated_text or "").strip()
                            formatted_final = format_legal_response_text(final_raw) if final_raw else ""
                            if callback and formatted_final:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(formatted_final, True)
                                    else:
                                        callback(formatted_final, True)
                                except Exception as cb_err:
                                    logger.warning("Final callback error: %s", cb_err)

                            if formatted_final:
                                return {"ok": True, "text": formatted_final, "usage": usage_info}

                    resp = await oai.responses.create(**payload)
                    text = getattr(resp, "output_text", None)
                    if text and text.strip():
                        formatted_text = format_legal_response_text(text)
                        return {"ok": True, "text": formatted_text, "usage": getattr(resp, "usage", None)}

                    items = getattr(resp, "output", []) or []
                    chunks: list[str] = []
                    for it in items:
                        for c in getattr(it, "content", []) or []:
                            t = getattr(c, "text", None)
                            if t:
                                chunks.append(t)
                    if chunks:
                        joined = "\n\n".join(chunks).strip()
                        return {
                            "ok": True,
                            "text": format_legal_response_text(joined),
                            "usage": getattr(resp, "usage", None),
                        }

                except TypeError as type_err:
                    if schema_supported and "response_format" in payload and "response_format" in str(type_err):
                        logger.warning(
                            "Responses API rejected response_format; retrying without JSON schema enforcement"
                        )
                        schema_supported = False
                        attempts = build_attempts(False)
                        schema_retry = True
                        break
                    last_err = str(type_err)
                    break
                except Exception as e:
                    last_err = str(e)

            if schema_retry:
                continue
            break

        return {"ok": False, "error": last_err or "unknown_error"}



def _extract_text_from_chunk(chunk) -> str:
    """Извлекает текстовую часть из stream chunk"""
    try:
        if hasattr(chunk, "output_text") and chunk.output_text:
            return chunk.output_text

        if hasattr(chunk, "delta") and chunk.delta:
            if hasattr(chunk.delta, "content") and chunk.delta.content:
                return chunk.delta.content
            if hasattr(chunk.delta, "text") and chunk.delta.text:
                return chunk.delta.text

        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                if hasattr(choice.delta, "content") and choice.delta.content:
                    return choice.delta.content
                if hasattr(choice.delta, "text") and choice.delta.text:
                    return choice.delta.text

        if hasattr(chunk, "output") and chunk.output:
            for item in chunk.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text") and content.text:
                            return content.text

        return ""
    except Exception:
        return ""
