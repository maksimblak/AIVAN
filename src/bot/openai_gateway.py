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
from openai import AsyncOpenAI

load_dotenv()

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
    """Единая точка запроса к Responses API.

    Возвращает dict: { ok: bool, text?: str, usage?: Any, error?: str }
    """
    return await _ask_legal_internal(system_prompt, user_text, stream=False)


async def ask_legal_stream(system_prompt: str, user_text: str, callback=None):
    """Streaming версия запроса к Responses API.

    Callback вызывается с частичными ответами: callback(partial_text: str, is_final: bool)
    Возвращает финальный dict: { ok: bool, text?: str, usage?: Any, error?: str }
    """
    return await _ask_legal_internal(system_prompt, user_text, stream=True, callback=callback)


async def _ask_legal_internal(
    system_prompt: str, user_text: str, stream: bool = False, callback=None
) -> dict[str, Any]:
    """
    Внутренняя реализация запроса к Responses API с поддержкой streaming.
    Возвращает dict: { ok: bool, text?: str, usage?: Any, error?: str }
    """
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    max_out = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
    verb = os.getenv("OPENAI_VERBOSITY", "medium").lower()
    effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").lower()

    base = dict(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        text={"verbosity": verb},
        reasoning={"effort": effort},
        max_output_tokens=max_out,
    )
    # web-инструмент по умолчанию можно отключить переменной окружения
    if not (os.getenv("DISABLE_WEB", "0").strip() in ("1", "true", "yes", "on")):
        base |= {"tools": [{"type": "web_search"}], "tool_choice": "auto"}

    async with await _make_async_client() as oai:
        # Проверка модели с небольшим ретраем
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

        attempts = [base]
        # Повтор без tools
        attempts.append({k: v for k, v in base.items() if k != "tools"})
        # Буст по количеству токенов
        boosted = attempts[-1] | {"max_output_tokens": max_out * 2}

        for payload in (attempts[0], attempts[1], boosted):
            try:
                if stream and callback:
                    # --- ПРАВИЛЬНЫЙ streaming через responses.stream ---
                    accumulated_text = ""
                    usage_info = None

                    try:
                        async with oai.responses.stream(**payload) as s:
                            async for event in s:
                                etype = getattr(event, "type", "") or ""

                                # Основные дельты текста
                                if etype.endswith(".delta"):
                                    # Пытаемся достать текст из разных мест (зависит от версии SDK)
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
                                            logger.warning(f"Callback error during streaming: {cb_err}")

                                # Можно ловить финал/usage
                                if etype == "response.completed":
                                    pass

                            # финальный объект ответа
                            final = await s.get_final_response()
                            usage_info = getattr(final, "usage", None)
                            # Сам текст
                            text = getattr(final, "output_text", None)
                            if not text:
                                # запасной разбор
                                items = getattr(final, "output", []) or []
                                chunks: list[str] = []
                                for it in items:
                                    for c in getattr(it, "content", []) or []:
                                        t = getattr(c, "text", None)
                                        if t:
                                            chunks.append(t)
                                text = "\n\n".join(chunks) if chunks else ""

                            # Финальный колбэк
                            if callback and (accumulated_text or text):
                                try:
                                    full = text or accumulated_text
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(full, True)
                                    else:
                                        callback(full, True)
                                except Exception as cb_err:
                                    logger.warning(f"Final callback error: {cb_err}")

                            full_text = (text or accumulated_text or "").strip()
                            if full_text:
                                return {"ok": True, "text": full_text, "usage": usage_info}

                    except Exception as stream_error:
                        logger.error(f"Streaming error: {stream_error}")
                        raise  # пойдём в обычный режим/следующую попытку

                # --- Обычный (нестриминговый) режим ---
                resp = await oai.responses.create(**payload)
                text = getattr(resp, "output_text", None)
                if text and text.strip():
                    return {"ok": True, "text": text.strip(), "usage": getattr(resp, "usage", None)}

                # запасной парсер
                items = getattr(resp, "output", []) or []
                chunks: list[str] = []
                for it in items:
                    for c in getattr(it, "content", []) or []:
                        t = getattr(c, "text", None)
                        if t:
                            chunks.append(t)
                if chunks:
                    return {
                        "ok": True,
                        "text": "\n\n".join(chunks).strip(),
                        "usage": getattr(resp, "usage", None),
                    }

            except Exception as e:  # переходим к следующей попытке
                last_err = str(e)

        return {"ok": False, "error": last_err or "unknown_error"}



def _extract_text_from_chunk(chunk) -> str:
    """Извлекает текст из stream chunk"""
    try:
        # Пробуем разные способы извлечения текста
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
