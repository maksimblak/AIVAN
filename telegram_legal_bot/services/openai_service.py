# telegram_legal_bot/services/openai_service.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

try:
    # OpenAI SDK >= 1.0
    from openai import AsyncOpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Не удалось импортировать openai. Установите пакет: pip install 'openai>=1.0'"
    ) from exc

from telegram_legal_bot.config import Settings

# ── корректные импорты системного промпта и схемы (V2 → фолбэки) ───────────────
SYSTEM_PROMPT: str
LEGAL_SCHEMA: Dict[str, Any]

try:
    # Приоритет: новые имена (PROMPT_V2/LEGAL_SCHEMA_V2)
    from telegram_legal_bot.promt import (  # type: ignore
        PROMPT_V2 as SYSTEM_PROMPT,
        LEGAL_SCHEMA_V2 as LEGAL_SCHEMA,
    )
except Exception:
    try:
        # Совместимость: старое имя PROMPT, без продвинутой схемы
        from telegram_legal_bot.promt import PROMPT as LONG_PROMPT  # type: ignore

        SYSTEM_PROMPT = LONG_PROMPT
        LEGAL_SCHEMA = {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "laws": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["answer", "laws"],
                    "additionalProperties": False,
                },
            },
        }
    except Exception:
        try:
            # Плоские пути
            from promt import PROMPT as LONG_PROMPT  # type: ignore

            SYSTEM_PROMPT = LONG_PROMPT
            LEGAL_SCHEMA = {
                "type": "json_schema",
                "json_schema": {
                    "name": "legal_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "laws": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["answer", "laws"],
                        "additionalProperties": False,
                    },
                },
            }
        except Exception:
            from prompt import PROMPT as LONG_PROMPT  # type: ignore

            SYSTEM_PROMPT = LONG_PROMPT
            LEGAL_SCHEMA = {
                "type": "json_schema",
                "json_schema": {
                    "name": "legal_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "laws": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["answer", "laws"],
                        "additionalProperties": False,
                    },
                },
            }

log = logging.getLogger("openai_service")


@dataclass
class LegalAdvice:
    """Контейнер ответа ИИ: текст и список применённых норм."""
    answer: str
    laws: List[str]

    # совместимость со старым кодом, который ожидает dict API
    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "laws": self.laws}

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class OpenAIService:
    """
    Обёртка над OpenAI для получения юридического ответа.
    Фишки:
      • поддержка HTTP-прокси (OPENAI_PROXY_* + фоллбек на TELEGRAM_PROXY_*)
      • мягкие ретраи с backoff
      • явное закрытие собственного httpx-клиента (aclose)
      • автосовместимость: Responses (JSON Schema) → Responses (plain) → Chat Completions
      • подробное логирование (latency, попытки, статусы, усечённые тела ошибок)
      • дауншифт неподдержанных параметров (temperature/response_format/…)
      • ask_ivan(...) с web_search/file_search/citations и LEGAL_SCHEMA_V2
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._own_httpx: Optional[httpx.AsyncClient] = None

        # ➊ Фоллбек: если OPENAI_* не заданы — берём TELEGRAM_*
        url = getattr(settings, "openai_proxy_url", None) or getattr(settings, "telegram_proxy_url", None)
        usr = getattr(settings, "openai_proxy_user", None) or getattr(settings, "telegram_proxy_user", None)
        pwd = getattr(settings, "openai_proxy_pass", None) or getattr(settings, "telegram_proxy_pass", None)
        proxy_url = self._build_proxy_url(url, usr, pwd)

        if proxy_url:
            self._own_httpx = self._make_async_httpx_client_with_proxy(proxy_url)

        self._client = AsyncOpenAI(
            http_client=self._own_httpx,        # None → SDK создаст свой клиент без прокси
            api_key=getattr(self._s, "openai_api_key", None),
        )
        log.debug("OpenAI client init: using_proxy=%s", bool(self._own_httpx))

    async def aclose(self) -> None:
        """Закрыть внутренний httpx-клиент и OpenAI-клиент."""
        try:
            if hasattr(self._client, "aclose"):
                await self._client.aclose()  # корректно закрыть SDK
        finally:
            if self._own_httpx is not None:
                await self._own_httpx.aclose()

    # ──────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ──────────────────────────────────────────────────────────────────────
    def _extract_url_citations(self, resp: Any) -> List[Dict[str, str]]:
        cites: List[Dict[str, str]] = []
        try:
            output = getattr(resp, "output", None) or []
            for item in output:
                content = getattr(item, "content", None) or (
                    item.get("content") if isinstance(item, dict) else None) or []
                for seg in content:
                    annotations = getattr(seg, "annotations", None) or (
                        seg.get("annotations") if isinstance(seg, dict) else None) or []
                    for ann in annotations:
                        atype = getattr(ann, "type", None) or (ann.get("type") if isinstance(ann, dict) else None)
                        if atype == "url_citation":
                            title = getattr(ann, "title", None) or (
                                ann.get("title") if isinstance(ann, dict) else "") or ""
                            url = getattr(ann, "url", None) or (ann.get("url") if isinstance(ann, dict) else "") or ""
                            if url:
                                cites.append({"title": str(title), "url": str(url)})
        except Exception:
            pass
        return cites

    async def ask_ivan(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Высокоточный режим: строго структурированный юридический ответ с инструментами поиска.
        Возвращает dict: {"data": <json|raw>, "citations": <list|None>, "raw_text": <str|None>}
        Теперь поддерживает короткую историю диалога (последние 6 сообщений user/assistant).
        """
        # Настройки с дефолтами на случай отсутствия полей в Settings
        recency_days = getattr(self._s, "web_search_recency_days", 3650)
        max_results = getattr(self._s, "web_search_max_results", 8)
        tool_choice = getattr(self._s, "tool_choice", "required")
        reasoning_effort = (
            getattr(self._s, "reasoning_effort", None)
            or getattr(self._s, "openai_reasoning_effort", "medium")
        )
        max_output_tokens = getattr(self._s, "max_output_tokens", None) or getattr(self._s, "openai_max_tokens", 1800)
        temperature = getattr(self._s, "temperature", None) or getattr(self._s, "openai_temperature", 0.3)
        top_p = getattr(self._s, "top_p", 1.0)
        seed = getattr(self._s, "seed", 7)
        search_domains = getattr(self._s, "search_domains", None)
        file_search_enabled = getattr(self._s, "file_search_enabled", True)

        # История (хвост до 6 сообщений, только роли user/assistant)
        history_msgs: List[Dict[str, str]] = []
        if short_history:
            for h in short_history[-6:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    history_msgs.append({"role": r, "content": str(c)})

        payload: Dict[str, Any] = {
            "model": getattr(self._s, "openai_model", "gpt-5"),
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_msgs,
                {"role": "user", "content": question},
            ],
            "response_format": LEGAL_SCHEMA,  # ← подаём всю обёртку V2
            "tools": [
                {"type": "web_search"},
            ],
            "tool_choice": tool_choice,
            "reasoning": {"effort": reasoning_effort},
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "parallel_tool_calls": True,
            "seed": seed,
        }

        if search_domains:
            payload["extra_body"]["domains"] = [
                d.strip() for d in str(search_domains).split(",") if d.strip()
            ]

        if file_search_enabled:
            pass
             # payload["tools"].append({"type": "file_search"})

        # Вызов с авто-дауншифтом неподдержанных ключей
        optional_keys = {
            "response_format",
            "tools",
            "tool_choice",
            "extra_body",
            "reasoning",
            "temperature",
            "top_p",
            "max_output_tokens",
            "parallel_tool_calls",
            "seed",
        }
        resp = await self._responses_call_with_optional(payload, optional_keys)

        # Достаём текст/цитации
        raw_text: Optional[str] = self._extract_text_from_responses(resp)
        # было:
        # citations = getattr(resp, "citations", None)
        # if citations is None:
        #     try:
        #         out = getattr(resp, "output", None) or []
        #         if out and hasattr(out[0], "citations"):
        #             citations = getattr(out[0], "citations", None)
        #     except Exception:
        #         citations = None

        # стало:
        citations = self._extract_url_citations(resp)

        # Парсим JSON, если есть
        if raw_text:
            try:
                data: Any = json.loads(raw_text)
            except Exception:
                data = {"raw": raw_text}
        else:
            data = {"raw": repr(resp)}

        return {"data": data, "citations": citations, "raw_text": raw_text}

    async def generate_legal_answer(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
        retries: int = 2,
    ) -> LegalAdvice:
        """
        Бэк-совместимый путь: запросить у модели краткий юридический ответ и список норм права.
        Возвращает LegalAdvice(answer, laws). Бросает RuntimeError при неуспехе после ретраев.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if short_history:
            for h in short_history[-5:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": question})

        # Берём «внутреннюю» часть вашей LEGAL_SCHEMA_V2, если она есть.
        try:
            inner_schema = LEGAL_SCHEMA["json_schema"]["schema"]  # type: ignore[index]
        except Exception:
            inner_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "laws": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer", "laws"],
                "additionalProperties": False,
            }

        log.debug(
            "LLM request: model=%s temp=%s max_out=%s effort=%s proxy=%s",
            getattr(self._s, "openai_model", None),
            getattr(self._s, "openai_temperature", None),
            getattr(self._s, "openai_max_tokens", None),
            getattr(self._s, "openai_reasoning_effort", None),
            bool(getattr(self._s, "openai_proxy_url", None) or getattr(self._s, "telegram_proxy_url", None)),
        )

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            t0 = perf_counter()
            try:
                # 1) Попытка №1: Responses API с response_format (JSON Schema)
                content = await self._responses_create_compat(
                    messages=messages,
                    json_schema=inner_schema,
                    use_schema=True,
                )
                took_ms = int((perf_counter() - t0) * 1000)

                data = self._safe_json_extract(content)
                answer = str(data.get("answer") or "").strip()
                laws = data.get("laws") or []
                if not isinstance(laws, list):
                    laws = []

                log.info(
                    "LLM ok: attempt=%s api=responses(schema) took_ms=%s answer_len=%s laws=%s",
                    attempt, took_ms, len(answer), len(laws)
                )
                return LegalAdvice(answer=answer, laws=list(map(str, laws)))

            except TypeError as e:
                last_err = e
                log.warning("Responses(schema) not supported by SDK: %r. Falling back.", e)

                try:
                    # 2) Попытка №2: Responses API без schema (просим вернуть JSON текстом)
                    forced_messages = messages.copy()
                    forced_messages[0] = {
                        "role": "system",
                        "content": SYSTEM_PROMPT + " Всегда возвращай ТОЛЬКО JSON вида {\"answer\": \"...\", \"laws\": [\"...\"]}.",
                    }
                    content = await self._responses_create_compat(
                        messages=forced_messages,       # <-- используем действительно forced_messages
                        json_schema=inner_schema,
                        use_schema=False,
                    )
                    took_ms = int((perf_counter() - t0) * 1000)

                    data = self._safe_json_extract(content)
                    answer = str(data.get("answer") or "").strip()
                    laws = data.get("laws") or []
                    if not isinstance(laws, list):
                        laws = []

                    log.info(
                        "LLM ok: attempt=%s api=responses(plain) took_ms=%s answer_len=%s laws=%s",
                        attempt, took_ms, len(answer), len(laws)
                    )
                    return LegalAdvice(answer=answer, laws=list(map(str, laws)))

                except Exception as e2:
                    last_err = e2
                    log.warning("Responses(plain) fail: %r. Falling back to chat.completions.", e2)

                    try:
                        # 3) Попытка №3: Chat Completions
                        content = await self._chat_completions_compat(
                            messages=messages,
                            force_json=True,
                        )
                        took_ms = int((perf_counter() - t0) * 1000)

                        data = self._safe_json_extract(content)
                        answer = str(data.get("answer") or "").strip()
                        laws = data.get("laws") or []
                        if not isinstance(laws, list):
                            laws = []

                        log.info(
                            "LLM ok: attempt=%s api=chat.completions took_ms=%s answer_len=%s laws=%s",
                            attempt, took_ms, len(answer), len(laws)
                        )
                        return LegalAdvice(answer=answer, laws=list(map(str, laws)))
                    except Exception as e3:
                        last_err = e3
                        log.warning("chat.completions fail: %r", e3)

            except Exception as e:
                took_ms = int((perf_counter() - t0) * 1000)
                last_err = e
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                body = None
                if getattr(e, "response", None) is not None:
                    try:
                        body = getattr(e.response, "text", None)
                        if body and len(body) > 1000:
                            body = body[:1000] + "…"
                    except Exception:
                        body = "<unreadable>"
                log.warning(
                    "LLM fail: attempt=%s took_ms=%s status=%s err=%r body=%s",
                    attempt, took_ms, status, e, body
                )

            # backoff между попытками
            await asyncio.sleep(0.75 * (attempt + 1))

        log.exception("LLM fatal after retries: %r", last_err)
        raise RuntimeError(f"OpenAI error: {last_err}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal: универсальный вызов Responses с авто-удалением ключей
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_call_with_optional(self, payload: Dict[str, Any], optional_keys: set[str]) -> Any:
        """
        Вызывает client.responses.create(**payload), удаляя неподдержанные параметры.
        Обрабатывает как ошибки SDK (TypeError), так и серверные 400 с указанием параметра.
        """
        # Рабочая копия
        pl = dict(payload)

        while True:
            try:
                return await self._client.responses.create(**pl)
            except TypeError as e:
                # пример: unexpected keyword argument 'response_format'
                m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
                bad = m.group(1) if m else None
                if bad and (bad in pl or bad in optional_keys):
                    log.debug("SDK doesn't support '%s' — removing and retrying...", bad)
                    pl.pop(bad, None)
                    continue
                raise
            except Exception as e:
                # Разбор серверного ответа
                resp = getattr(e, "response", None)
                err_param = None
                err_message = ""
                try:
                    if resp is not None and hasattr(resp, "json"):
                        j = resp.json()
                        err = (j or {}).get("error") or {}
                        err_param = err.get("param")
                        err_message = (err.get("message") or "").lower()
                except Exception:
                    pass

                removed = False
                for cand in list(optional_keys):
                    if (err_param == cand) or (cand in pl and f"'{cand}'" in err_message):
                        log.debug("Server says param '%s' unsupported — removing and retrying...", cand)
                        pl.pop(cand, None)
                        removed = True
                if removed:
                    continue
                raise

    # ──────────────────────────────────────────────────────────────────────
    # Internal: Responses API с авто-удалением неподдержанных параметров
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_create_compat(
        self,
        *,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        use_schema: bool,
    ) -> str:
        """
        Пытается вызвать client.responses.create(**payload) с мягким удалением неподдержанных ключей.
        Возвращает объединённый текст ответа.
        """
        payload: Dict[str, Any] = {
            "model": getattr(self._s, "openai_model", "gpt-5"),
            "input": messages,
        }

        optional_keys = set()

        # temperature — у reasoning-моделей часто не поддерживается
        if getattr(self._s, "openai_temperature", None) is not None:
            payload["temperature"] = self._s.openai_temperature
            optional_keys.add("temperature")

        # max_output_tokens (в Responses)
        if getattr(self._s, "openai_max_tokens", None):
            payload["max_output_tokens"] = self._s.openai_max_tokens
            optional_keys.add("max_output_tokens")

        # reasoning/text — «новые» допы
        if getattr(self._s, "openai_reasoning_effort", None):
            payload["reasoning"] = {"effort": self._s.openai_reasoning_effort}
            optional_keys.add("reasoning")
        if getattr(self._s, "openai_verbosity", None):
            payload["text"] = {"verbosity": self._s.openai_verbosity}
            optional_keys.add("text")

        # JSON Schema (может не поддерживаться)
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "legal_answer", "schema": json_schema},
            }
            optional_keys.add("response_format")

        resp = await self._responses_call_with_optional(payload, optional_keys)
        return self._extract_text_from_responses(resp) or ""

    # ──────────────────────────────────────────────────────────────────────
    # Internal: Chat Completions fallback
    # ──────────────────────────────────────────────────────────────────────
    async def _chat_completions_compat(
        self,
        *,
        messages: List[Dict[str, str]],
        force_json: bool,
    ) -> str:
        """
        Пытается вызвать client.chat.completions.create(...).
        Если SDK/сервер не знает response_format={"type":"json_object"} или temperature — убираем и повторяем.
        """
        chat_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        # Базовые аргументы
        kwargs: Dict[str, Any] = dict(
            model=getattr(self._s, "openai_model", "gpt-5"),
            messages=chat_msgs,
        )
        removable = set()

        if getattr(self._s, "openai_temperature", None) is not None:
            kwargs["temperature"] = self._s.openai_temperature
            removable.add("temperature")

        if getattr(self._s, "openai_max_tokens", None):
            kwargs["max_tokens"] = self._s.openai_max_tokens

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
            removable.add("response_format")

        # Попробуем один-два прохода с удалением неподдержанных параметров
        for _ in range(2):
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                return self._extract_text_from_chat(resp)
            except TypeError as e:
                if "response_format" in str(e) and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    # усилим system-подсказку в сообщениях
                    patched = chat_msgs.copy()
                    if patched and patched[0]["role"] == "system":
                        patched[0] = {
                            "role": "system",
                            "content": patched[0]["content"] + " Всегда возвращай ТОЛЬКО JSON {\"answer\":\"...\",\"laws\":[\"...\"]}.",
                        }
                    kwargs["messages"] = patched
                    continue
                raise
            except Exception as e:
                resp = getattr(e, "response", None)
                err_param = None
                err_message = ""
                try:
                    if resp is not None and hasattr(resp, "json"):
                        j = resp.json()
                        err = (j or {}).get("error") or {}
                        err_param = err.get("param")
                        err_message = (err.get("message") or "").lower()
                except Exception:
                    pass

                removed = False
                for cand in list(removable):
                    if (err_param == cand) or (f"'{cand}'" in err_message and cand in kwargs):
                        log.debug("Server says chat param '%s' unsupported — removing and retrying...", cand)
                        kwargs.pop(cand, None)
                        removed = True
                if removed:
                    continue
                raise

    # ──────────────────────────────────────────────────────────────────────
    # Helpers: прокси и извлечение текста
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_proxy_url(url: Optional[str], user: Optional[str], pwd: Optional[str]) -> Optional[str]:
        """
        Возвращает корректный proxy-URL вида http(s)://user:pass@host:port.
        Логин/пароль экранируются (quote) на случай спецсимволов.
        Добавляем схему http:// если не указана.
        """
        if not url:
            return None
        if "://" not in url:
            url = "http://" + url
        if user and pwd and "@" not in url:
            scheme, rest = url.split("://", 1)
            u = quote(user, safe="")
            p = quote(pwd, safe="")
            return f"{scheme}://{u}:{p}@{rest}"
        return url

    @staticmethod
    def _make_async_httpx_client_with_proxy(proxy_url: str) -> httpx.AsyncClient:
        """
        Создаёт httpx.AsyncClient с прокси.
        Кросс-версионно пробует сначала proxy= (httpx>=0.28), затем proxies= (старые версии).
        """
        try:
            return httpx.AsyncClient(
                proxy=proxy_url,                 # httpx >= 0.28
                timeout=httpx.Timeout(45.0),
                verify=True,
                trust_env=False,
            )
        except TypeError:
            return httpx.AsyncClient(
                proxies=proxy_url,               # старые версии httpx
                timeout=httpx.Timeout(45.0),
                verify=True,
                trust_env=False,
            )

    @staticmethod
    def _extract_text_from_responses(resp: Any) -> str:
        """
        Унифицированно достаём текст из Responses API объекта.
        """
        # 1) новое поле/метод output_text
        out = getattr(resp, "output_text", None)
        if callable(out):
            try:
                return out()
            except Exception:
                pass
        if isinstance(out, str):
            return out

        # 2) универсальный обход output[]
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                content = None
                if isinstance(item, dict):
                    content = item.get("content")
                else:
                    content = getattr(item, "content", None)

                if isinstance(content, list):
                    for seg in content:
                        if isinstance(seg, dict):
                            if seg.get("type") in {"output_text", "text"}:
                                txt = seg.get("text")
                                if isinstance(txt, str):
                                    parts.append(txt)
                        else:
                            seg_type = getattr(seg, "type", None)
                            if seg_type in {"output_text", "text"}:
                                txt = getattr(seg, "text", None)
                                if isinstance(txt, str):
                                    parts.append(txt)
            if parts:
                return "".join(parts)

        # 3) фоллбек — как есть
        return str(resp)

    @staticmethod
    def _extract_text_from_chat(resp: Any) -> str:
        """
        Достаёт контент из Chat Completions ответа.
        """
        try:
            choices = getattr(resp, "choices", None) or []
            if choices and hasattr(choices[0], "message"):
                msg = choices[0].message
                return getattr(msg, "content", "") or ""
        except Exception:
            pass
        return str(resp)

    @staticmethod
    def _safe_json_extract(text: str) -> Dict[str, Any]:
        """
        Пытается распарсить JSON даже если модель вернула «обёртку».
        Поддерживает удаление ```json-блоков. Возвращает dict, минимум: {"answer": <string>, "laws": []}
        """
        text = (text or "").strip()

        # срежем ```json ... ``` если есть
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {"answer": text, "laws": []}
