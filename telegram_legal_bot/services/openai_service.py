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
        # Безопасное логирование без раскрытия чувствительной информации
        log.info("OpenAI client initialized: proxy_enabled=%s", bool(self._own_httpx))

    def _get_model(self) -> str:
        """
        Возвращает имя модели из настроек.
        Фоллбек отключён — если модель не задана, кидаем явную ошибку.
        """
        m = (getattr(self._s, "openai_model", "") or "").strip()
        if not m:
            raise RuntimeError(
                "OPENAI_MODEL не задан. Укажите доступную модель в переменной окружения OPENAI_MODEL."
            )
        return m

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


    async def ask_ivan(self,question: str,short_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Высокоточный режим: строго структурированный юридический ответ с инструментами поиска.
        Возвращает dict: {"data": <json|raw>, "citations": <list|None>, "raw_text": <str|None>}
        Теперь поддерживает короткую историю диалога (последние 6 сообщений user/assistant).
        """
        # Настройки
        recency_days = getattr(self._s, "web_search_recency_days", 3650)
        max_results = getattr(self._s, "web_search_max_results", 8)
        tool_choice = getattr(self._s, "tool_choice", "required")
        reasoning_effort = getattr(self._s, "reasoning_effort", None) or getattr(self._s, "openai_reasoning_effort",
                                                                                 "medium")
        max_output_tokens = getattr(self._s, "max_output_tokens", None) or getattr(self._s, "openai_max_tokens", 1800)
        temperature = getattr(self._s, "temperature", None) or getattr(self._s, "openai_temperature", 0.3)
        top_p = getattr(self._s, "top_p", 1.0)
        seed = getattr(self._s, "seed", 7)
        search_domains = getattr(self._s, "search_domains", None)
        file_search_enabled = getattr(self._s, "file_search_enabled", True)

        # История (хвост до 6 сообщений user/assistant)
        history_msgs: List[Dict[str, str]] = []
        if short_history:
            for h in short_history[-6:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    history_msgs.append({"role": r, "content": str(c)})

        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_msgs,
                {"role": "user", "content": question},
            ],
            "response_format": LEGAL_SCHEMA,  # работаем строго по LEGAL_SCHEMA_V2 (если поддерживается)
            # инструменты будут добавлены ниже только если включены
            "tool_choice": tool_choice,
            "reasoning": {"effort": reasoning_effort},
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "parallel_tool_calls": True,
            "seed": seed,
        }

        # --- инструменты и extra_body ---
        extra_body: Dict[str, Any] = {}

        if search_domains:
            extra_body["search_domains"] = search_domains

        if getattr(self._s, "web_search_enabled", False):
            extra_body["web_search"] = {
                "recency_days": int(recency_days),
                "max_results": int(max_results),
            }

        if extra_body:
            payload["extra_body"] = extra_body
        else:
            payload.pop("extra_body", None)

        tools: List[Dict[str, Any]] = []
        if getattr(self._s, "web_search_enabled", False):
            tools.append({"type": "web_search"})
        if file_search_enabled:
            tools.append({"type": "file_search"})
        if tools:
            payload["tools"] = tools
        else:
            payload.pop("tools", None)

        payload["tool_choice"] = getattr(self._s, "tool_choice", "auto")

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
        try:
            resp = await self._responses_call_with_optional(payload, optional_keys)
            # Достаём текст и url-citations
            raw_text: Optional[str] = self._extract_text_from_responses(resp)
            citations = self._extract_url_citations(resp)
            # Парсим JSON, если это он
            if raw_text:
                try:
                    data: Any = json.loads(raw_text)
                except Exception:
                    data = {"raw": raw_text}
            else:
                data = {"raw": repr(resp)}
            return {"data": data, "citations": citations, "raw_text": raw_text}
        except Exception:
            # Фолбэк: Chat Completions с просьбой вернуть JSON по схеме V2 (мягко)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_msgs,
                {"role": "user", "content": question},
            ]
            content = await self._chat_completions_compat(messages=messages, force_json=True)
            data = self._safe_json_extract(content) or {}
            return {"data": data, "citations": [], "raw_text": content}

    async def generate_legal_answer(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Получить развернутый юридический ответ строго по LEGAL_SCHEMA_V2.
        Возвращает dict с полями схемы. Бросает RuntimeError при неуспехе.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if short_history:
            for h in short_history[-5:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": question})

        # Внутренняя JSON-схема V2
        try:
            inner_schema: Dict[str, Any] = LEGAL_SCHEMA["json_schema"]["schema"]  # type: ignore[index]
        except Exception as e:
            raise RuntimeError("LEGAL_SCHEMA_V2 недоступна, проверь импорт PROMPT_V2/LEGAL_SCHEMA_V2") from e

        # Безопасное логирование запроса без чувствительных данных
        log.debug(
            "LLM request: model=%s temp=%s max_tokens=%s effort=%s proxy_enabled=%s",
            getattr(self._s, "openai_model", "unknown"),
            getattr(self._s, "openai_temperature", 0.3),
            getattr(self._s, "openai_max_tokens", 1500),
            getattr(self._s, "openai_reasoning_effort", "medium"),
            bool(getattr(self._s, "openai_proxy_url", None) or getattr(self._s, "telegram_proxy_url", None)),
        )

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            t0 = perf_counter()
            try:
                # 1) Responses API с response_format (JSON Schema V2)
                content = await self._responses_create_compat(
                    messages=messages,
                    json_schema=inner_schema,
                    use_schema=True,
                )
                took_ms = int((perf_counter() - t0) * 1000)
                data = self._safe_json_extract(content) or {}
                if not (isinstance(data, dict) and data.get("conclusion") and data.get("sources")):
                    raise ValueError("V2 schema parse: missing key fields")
                log.info("LLM ok: attempt=%s api=responses(schema) took_ms=%s", attempt, took_ms)
                return data

            except TypeError as e:
                # SDK не знает response_format → убираем схему и просим JSON текстом
                last_err = e
                log.warning("Responses(schema) not supported by SDK: %r. Falling back.", e)

                try:
                    # 2) Responses API без schema (но формат — V2 JSON)
                    content = await self._responses_create_compat(
                        messages=messages,
                        json_schema=inner_schema,
                        use_schema=False,
                    )
                    took_ms = int((perf_counter() - t0) * 1000)
                    data = self._safe_json_extract(content) or {}
                    if not (isinstance(data, dict) and data.get("conclusion") and data.get("sources")):
                        raise ValueError("V2 parse (responses no-schema) failed")
                    log.info("LLM ok: attempt=%s api=responses took_ms=%s", attempt, took_ms)
                    return data

                except Exception as e2:
                    last_err = e2
                    log.warning("Responses call failed: %r", e2)

                try:
                    # 3) Chat Completions (просим вернуть JSON V2)
                    content = await self._chat_completions_compat(messages=messages, force_json=True)
                    took_ms = int((perf_counter() - t0) * 1000)
                    data = self._safe_json_extract(content) or {}
                    if not (isinstance(data, dict) and data.get("conclusion") and data.get("sources")):
                        raise ValueError("V2 parse (chat.completions) failed")
                    log.info("LLM ok: attempt=%s api=chat.completions took_ms=%s", attempt, took_ms)
                    return data

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
                        body = self._safe_truncate_error_message(body)
                    except Exception:
                        body = "<unreadable>"
                log.warning("LLM fail: attempt=%s took_ms=%s status=%s err=%r body=%s", attempt, took_ms, status, e, body)

            await asyncio.sleep(0.75 * (attempt + 1))

        log.exception("LLM fatal after retries: %r", last_err)
        raise RuntimeError(f"OpenAI error: {last_err}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal: универсальный вызов Responses с авто-удалением ключей
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_call_with_optional(self, payload: Dict[str, Any]) -> Any:
        """
        Вызывает Responses API, поэтапно снимая неподдержанные поля.
        Важно: модель не изменяем. Если модель недоступна — кидаем явную ошибку.
        """
        max_attempts = 4
        pl: Dict[str, Any] = dict(payload)

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._client.responses.create(**pl)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                removed_any = False

                # 1) SDK-уровень: неожиданные аргументы
                if "response_format" in low:
                    pl.pop("response_format", None)
                    removed_any = True
                if "seed" in low:
                    pl.pop("seed", None)
                    removed_any = True
                if "temperature" in low:
                    pl.pop("temperature", None)
                    removed_any = True
                if "top_p" in low:
                    pl.pop("top_p", None)
                    removed_any = True
                if "max_output_tokens" in low:
                    pl.pop("max_output_tokens", None)
                    removed_any = True
                if "parallel_tool_calls" in low:
                    pl.pop("parallel_tool_calls", None)
                    removed_any = True

                # 2) Сервер: инструменты и веб-поиск недоступны
                if "web_search" in low or ("unknown" in low and "tools" in low):
                    pl.pop("tools", None)
                    eb = pl.get("extra_body") or {}
                    if isinstance(eb, dict):
                        eb.pop("web_search", None)
                        if eb:
                            pl["extra_body"] = eb
                        else:
                            pl.pop("extra_body", None)
                    removed_any = True

                # 3) Модель недоступна/неизвестна — НЕ фолбэчим, а объясняем
                if ("model" in low) and (
                    "not found" in low or "does not exist" in low or "unknown" in low
                ):
                    raise RuntimeError(
                        f"Модель '{pl.get('model')}' недоступна на аккаунте/в регионе. Укажите корректную модель через OPENAI_MODEL."
                    ) from e

                if removed_any and attempt < max_attempts:
                    continue
                raise
    
    def _safe_parse_error_response(self, resp: Any) -> tuple[Optional[str], str]:
        """
        Безопасно парсит ошибку из HTTP ответа.
        Возвращает (err_param, err_message).
        """
        try:
            # Проверяем наличие метода json и Content-Type
            if not hasattr(resp, "json"):
                return None, ""
            
            # Дополнительная проверка Content-Type, если доступно
            content_type = ""
            if hasattr(resp, "headers"):
                content_type = resp.headers.get("content-type", "").lower()
            elif hasattr(resp, "content_type"):
                content_type = str(resp.content_type).lower()
            
            # Проверяем, что это действительно JSON
            if content_type and "application/json" not in content_type:
                log.warning("Response Content-Type is not JSON: %s", content_type)
                return None, ""
            
            j = resp.json()
            if not isinstance(j, dict):
                return None, ""
                
            err = j.get("error") or {}
            if not isinstance(err, dict):
                return None, ""
                
            err_param = err.get("param")
            err_message = (err.get("message") or "").lower()
            
            return err_param, err_message
            
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
            log.debug("Failed to parse error response: %r", e)
            return None, ""
        except Exception as e:
            log.warning("Unexpected error parsing response: %r", e)
            return None, ""

    def _validate_domains(self, domains_str: str) -> List[str]:
        """
        Валидирует и очищает список доменных имен.
        Возвращает список валидных доменов.
        """
        if not domains_str or not isinstance(domains_str, str):
            return []
        
        import re
        # Простая regex для проверки доменных имен
        domain_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        )
        
        valid_domains = []
        for domain in domains_str.split(","):
            domain = domain.strip().lower()
            if not domain:
                continue
            
            # Удаляем протокол если есть
            if domain.startswith(("http://", "https://")):
                domain = domain.split("//", 1)[1]
            
            # Удаляем путь если есть
            if "/" in domain:
                domain = domain.split("/", 1)[0]
            
            # Валидация доменного имени
            if domain_pattern.match(domain) and len(domain) <= 253:
                valid_domains.append(domain)
            else:
                log.warning("Invalid domain name ignored: %s", domain)
        
        log.debug("Validated domains: %s -> %s", domains_str, valid_domains)
        return valid_domains

    @staticmethod
    def _safe_truncate_error_message(message: Optional[str], max_length: int = 1000) -> str:
        """
        Безопасно усекает сообщения об ошибках для предотвращения переполнения логов.
        Удаляет потенциально чувствительные данные.
        """
        if not message:
            return "<empty>"
        
        if not isinstance(message, str):
            message = str(message)
        
        # Маскируем потенциально чувствительные данные
        import re
        # Маскируем API ключи
        message = re.sub(r'sk-[a-zA-Z0-9-_]{20,}', 'sk-***MASKED***', message)
        # Маскируем токены
        message = re.sub(r'[Bb]earer\s+[a-zA-Z0-9-_]{20,}', 'Bearer ***MASKED***', message)
        # Маскируем пароли в URL
        message = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', message)
        
        # Усекаем сообщение если необходимо
        if len(message) > max_length:
            return message[:max_length] + "…[TRUNCATED]"
        
        return message

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
            "model": self._get_model(),
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

        resp = await self._responses_call_with_optional(payload)
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
            model=self._get_model(),
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
                
                if resp is not None:
                    err_param, err_message = self._safe_parse_error_response(resp)

                removed = False
                # Модель недоступна/не найдена — не фолбэчим, объясняем явно
                if (err_param == "model") or ("model" in err_message and ("not found" in err_message or "does not exist" in err_message or "unknown" in err_message)):
                    raise RuntimeError(
                        f"Модель '{kwargs.get('model')}' недоступна. Задайте корректную OPENAI_MODEL."
                    ) from e
                
                # Параметр max_tokens не поддержан — пробуем max_completion_tokens
                if (err_param == "max_tokens") or ("max_tokens" in err_message and "max_completion_tokens" in err_message):
                    if "max_tokens" in kwargs:
                        val = kwargs.pop("max_tokens")
                        kwargs["max_completion_tokens"] = val
                        log.debug("Chat: switching 'max_tokens' -> 'max_completion_tokens' and retrying...")
                        removed = True
                
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
        Безопасно достаёт контент из Chat Completions ответа.
        """
        try:
            choices = getattr(resp, "choices", None)
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                return ""
            
            first_choice = choices[0]
            if not hasattr(first_choice, "message"):
                return ""
            
            msg = first_choice.message
            content = getattr(msg, "content", None)
            if content is None:
                return ""
                
            return str(content)
        except (IndexError, AttributeError, TypeError) as e:
            log.debug("Failed to extract text from chat response: %r", e)
            return ""
        except Exception as e:
            log.warning("Unexpected error extracting chat text: %r", e)
            return ""

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
