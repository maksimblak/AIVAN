# telegram_legal_bot/services/openai_service.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
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
      • только Responses API (поддержка reasoning-моделей, напр. gpt-5)
      • поддержка HTTP-прокси (OPENAI_PROXY_* + фоллбек на TELEGRAM_PROXY_*)
      • мягкие ретраи с backoff
      • явное закрытие собственного httpx-клиента (aclose)
      • подробное логирование (latency, попытки, статусы, усечённые тела ошибок)
      • авто-совместимость: переключение input↔messages, снятие response_format при старом SDK
      • ask_ivan(...) возвращает citations, реальные ссылки из ответа Responses
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

        client_kwargs: Dict[str, Any] = {
            "http_client": self._own_httpx,        # None → SDK создаст свой клиент без прокси
            "api_key": getattr(self._s, "openai_api_key", None),
        }
        # если используется частный шлюз (совместимый API)
        base_url = getattr(self._s, "openai_base_url", None)
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

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
        """
        Пытается найти URL-цитаты в structure Responses:
        - в content[].annotations[] (type='url_citation')
        - в top-level attachments/citations, если такие поля есть
        Возвращает список словарей: {"title": str, "url": str}
        """
        cites: List[Dict[str, str]] = []

        # a) content[].annotations[]
        try:
            output = getattr(resp, "output", None) or []
            if isinstance(output, list):
                for item in output:
                    content = getattr(item, "content", None) or (
                        item.get("content") if isinstance(item, dict) else None) or []
                    if isinstance(content, list):
                        for seg in content:
                            annotations = getattr(seg, "annotations", None) or (
                                seg.get("annotations") if isinstance(seg, dict) else None) or []
                            if isinstance(annotations, list):
                                for ann in annotations:
                                    atype = getattr(ann, "type", None) or (ann.get("type") if isinstance(ann, dict) else None)
                                    if atype == "url_citation":
                                        title = getattr(ann, "title", None) or (ann.get("title") if isinstance(ann, dict) else "") or ""
                                        url = getattr(ann, "url", None) or (ann.get("url") if isinstance(ann, dict) else "") or ""
                                        if url:
                                            cites.append({"title": str(title), "url": str(url)})
        except Exception:
            pass

        # b) top-level attachments/citations
        try:
            for field in ("attachments", "citations"):
                raw = getattr(resp, field, None)
                if isinstance(raw, list):
                    for it in raw:
                        url = None
                        title = ""
                        if isinstance(it, dict):
                            url = it.get("url") or it.get("source_url") or it.get("href")
                            title = it.get("title") or ""
                        else:
                            url = getattr(it, "url", None) or getattr(it, "source_url", None) or getattr(it, "href", None)
                            title = getattr(it, "title", "") or ""
                        if isinstance(url, str) and url:
                            cites.append({"title": str(title), "url": url})
        except Exception:
            pass

        # уникализируем по URL, сохраняя порядок
        seen = set()
        uniq: List[Dict[str, str]] = []
        for c in cites:
            u = c.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            uniq.append({"title": c.get("title", "") or "", "url": u})
        return uniq

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
        # Настройки
        recency_days = getattr(self._s, "web_search_recency_days", 3650)
        max_results = getattr(self._s, "web_search_max_results", 8)
        reasoning_effort = getattr(self._s, "reasoning_effort", None) or getattr(self._s, "openai_reasoning_effort", "medium")
        max_output_tokens = getattr(self._s, "max_output_tokens", None) or getattr(self._s, "openai_max_tokens", 1800)
        temperature = getattr(self._s, "openai_temperature", 0.3)
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
            "modalities": ["text"],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "response_format": LEGAL_SCHEMA,
            "tool_choice": getattr(self._s, "tool_choice", "auto"),
            # Дополнительные параметры генерации (оставляем как «мягкие»)
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }

        # --- инструменты и extra_body ---
        extra_body: Dict[str, Any] = {}
        if search_domains:
            valid = self._validate_domains(search_domains)
            if valid:
                extra_body["search_domains"] = valid

        if getattr(self._s, "web_search_enabled", False):
            extra_body["web_search"] = {
                "recency_days": int(recency_days),
                "max_results": int(max_results),
            }

        if extra_body:
            payload["extra_body"] = extra_body

        tools: List[Dict[str, Any]] = []
        if getattr(self._s, "web_search_enabled", False):
            tools.append({"type": "web_search"})
        if file_search_enabled:
            tools.append({"type": "file_search"})
        if tools:
            payload["tools"] = tools

        try:
            raw_text, citations = await self._responses_call_with_optional(payload)
            if raw_text:
                try:
                    data: Any = json.loads(raw_text)
                except Exception:
                    data = {"raw": raw_text}
            else:
                data = {"raw": ""}

            return {"data": data, "citations": citations, "raw_text": raw_text}
        except Exception as e:
            log.exception("ask_ivan failed: %r", e)
            raise

    async def generate_legal_answer(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Получить развернутый юридический ответ строго по LEGAL_SCHEMA_V2 (если доступна).
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

        # Безопасное логирование для Responses
        log.debug(
            "LLM request (Responses): model=%s effort=%s max_output_tokens=%s proxy_enabled=%s",
            getattr(self._s, "openai_model", "unknown"),
            getattr(self._s, "openai_reasoning_effort", "medium"),
            getattr(self._s, "openai_max_tokens", 1500),
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
                # Желательно наличие ключевых полей; если их нет — вернём как есть
                if isinstance(data, dict) and data:
                    log.info("LLM ok: attempt=%s api=responses(schema) took_ms=%s", attempt, took_ms)
                    return data
                raise ValueError("V2 schema parse: empty data")

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
                    if isinstance(data, dict) and data:
                        log.info("LLM ok: attempt=%s api=responses took_ms=%s", attempt, took_ms)
                        return data
                    raise ValueError("V2 parse (responses no-schema) failed")

                except Exception as e2:
                    last_err = e2
                    log.warning("Responses call failed: %r", e2)

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
    # Internal: универсальный вызов Responses с авто-совместимостью
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_call_with_optional(self, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Универсальный вызов Responses API:
        1) пробуем payload как есть;
        2) если ругается на input/messages — автоматически переключаем;
        3) если ругается на response_format — повторяем без схемы.
        Возвращает (text, citations).
        """
        allowed = {
            "model", "input", "messages", "modalities", "audio", "metadata",
            "max_output_tokens", "reasoning", "text", "response_format",
            "temperature", "top_p", "seed", "stop", "extra_body",
            "tools", "tool_choice", "presence_penalty", "frequency_penalty",
            "parallel_tool_calls",
        }
        pl = {k: v for k, v in payload.items() if k in allowed and v is not None}

        # У reasoning-моделей некоторые параметры не поддерживаются — убираем на всякий случай
        if self._is_reasoning_model():
            for k in ("presence_penalty", "frequency_penalty", "stop", "max_tokens"):
                pl.pop(k, None)

        log_local = logging.getLogger("openai_service")
        m = pl.get("model", self._get_model())
        try:
            effort = (pl.get("reasoning") or {}).get("effort", "")
        except Exception:
            effort = ""
        log_local.debug(
            "LLM request (Responses): model=%s effort=%s max_output_tokens=%s proxy_enabled=%s",
            m,
            effort or "<unset>",
            pl.get("max_output_tokens"),
            bool(getattr(self._s, "openai_proxy_url", None) or getattr(self._s, "telegram_proxy_url", None)),
        )

        # 1) первая попытка
        try:
            resp = await self._client.responses.create(**pl)
            return self._extract_text_and_citations(resp)
        except Exception as e:
            resp_obj = getattr(e, "response", None)
            err_param, err_message = (None, "")
            if resp_obj is not None:
                err_param, err_message = self._safe_parse_error_response(resp_obj)
            log_local.debug("Responses 1st attempt failed: param=%r detail=%r", err_param, err_message)

            msg = (err_message or "").lower()

            # 2) автопереключение input ↔ messages
            if ("unrecognized request argument: input" in msg) or (err_param == "input"):
                if "messages" not in pl and "input" in pl:
                    pl2 = dict(pl)
                    pl2["messages"] = pl2.pop("input")
                    resp2 = await self._client.responses.create(**pl2)
                    return self._extract_text_and_citations(resp2)

            if ("unrecognized request argument: messages" in msg) or (err_param == "messages"):
                if "input" not in pl and "messages" in pl:
                    pl2 = dict(pl)
                    pl2["input"] = pl2.pop("messages")
                    resp2 = await self._client.responses.create(**pl2)
                    return self._extract_text_and_citations(resp2)

            # 3) убрать response_format, если не поддерживается вашей версией SDK
            if (err_param == "response_format") or ("response_format" in msg):
                pl2 = dict(pl)
                pl2.pop("response_format", None)
                resp2 = await self._client.responses.create(**pl2)
                return self._extract_text_and_citations(resp2)

            # иначе — пробрасываем дальше
            raise

    def _safe_parse_error_response(self, resp: Any) -> tuple[Optional[str], str]:
        """
        Безопасно парсит ошибку из HTTP ответа.
        Возвращает (err_param, err_message).
        """
        try:
            if not hasattr(resp, "json"):
                return None, ""

            # Попробуем извлечь Content-Type (если доступен)
            content_type = ""
            if hasattr(resp, "headers"):
                content_type = resp.headers.get("content-type", "").lower()
            elif hasattr(resp, "content_type"):
                content_type = str(resp.content_type).lower()

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

        domain_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        )

        valid_domains: List[str] = []
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
    # Internal: Responses API «совместимый» вызов (со схемой/без)
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_create_compat(
        self,
        *,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        use_schema: bool,
    ) -> str:
        """
        Вызывает client.responses.create(**payload) с «мягким» снятием неподдержанных ключей.
        Возвращает итоговый текст ответа (string).
        """
        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "input": messages,
        }

        # temperature — у reasoning-моделей может игнорироваться; передаём как «мягкий» параметр
        if getattr(self._s, "openai_temperature", None) is not None:
            payload["temperature"] = self._s.openai_temperature
        # max_output_tokens (в Responses)
        if getattr(self._s, "openai_max_tokens", None):
            payload["max_output_tokens"] = self._s.openai_max_tokens
        # reasoning/text — дополнительные параметры
        if getattr(self._s, "openai_reasoning_effort", None):
            payload["reasoning"] = {"effort": self._s.openai_reasoning_effort}
        if getattr(self._s, "openai_verbosity", None):
            payload["text"] = {"verbosity": self._s.openai_verbosity}

        # JSON Schema (может не поддерживаться старым SDK)
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "legal_answer", "schema": json_schema},
            }

        text, _cites = await self._responses_call_with_optional(payload)
        return text or ""

    # ──────────────────────────────────────────────────────────────────────
    # Helpers: модель/прокси/извлечение текста и цитат
    # ──────────────────────────────────────────────────────────────────────
    def _is_reasoning_model(self) -> bool:
        """
        True для reasoning-моделей, которые надо вызывать через Responses API
        (gpt-5, o4, o3, o1 и т.п.).
        """
        try:
            m = self._get_model()
        except Exception:
            return False
        return m.startswith(("gpt-5", "o4", "o3", "o1"))

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
                proxies=proxy_url,               # ← ключевой фикс для старых версий httpx
                timeout=httpx.Timeout(45.0),
                verify=True,
                trust_env=False,
            )

    def _extract_text_and_citations(self, resp: Any) -> Tuple[str, List[Dict[str, str]]]:
        """
        Унифицированный проход по объекту Responses:
        - собираем финальный text (включая output_text, если есть),
        - достаём URL-цитаты.
        """
        text = self._extract_text_from_responses(resp)
        cites = self._extract_url_citations(resp)
        return text, cites

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
                obj = json.loads(text[start: end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {"answer": text, "laws": []}
