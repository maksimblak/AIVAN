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
      • поддержка HTTP-прокси (OPENAI_PROXY_URL/USER/PASS)
      • мягкие ретраи с backoff
      • явное закрытие собственного httpx-клиента (aclose)
      • автосовместимость: Responses API (c JSON Schema) → упрощённый Responses → Chat Completions
      • подробное логирование (latency, попытки, статусы, усечённые тела ошибок)
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._own_httpx: Optional[httpx.AsyncClient] = None

        http_client: Optional[httpx.AsyncClient] = None
        if getattr(settings, "openai_proxy_url", None):
            proxy_url = self._build_proxy_url(
                settings.openai_proxy_url,
                settings.openai_proxy_user,
                settings.openai_proxy_pass,
            )
            self._own_httpx = httpx.AsyncClient(
                proxies=proxy_url,
                timeout=httpx.Timeout(45.0),
            )
            http_client = self._own_httpx

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            http_client=http_client,
        )

    async def aclose(self) -> None:
        """Закрыть внутренний httpx-клиент, если он создавался здесь."""
        if self._own_httpx is not None:
            await self._own_httpx.aclose()

    @staticmethod
    def _build_proxy_url(url: Optional[str], user: Optional[str], pwd: Optional[str]) -> Optional[str]:
        """
        Возвращает корректный proxy-URL вида http(s)://user:pass@host:port
        Логин/пароль экранируются (quote) на случай спецсимволов.
        """
        if not url:
            return None
        if user and pwd and "@" not in url and "://" in url:
            scheme, rest = url.split("://", 1)
            u = quote(user, safe="")
            p = quote(pwd, safe="")
            return f"{scheme}://{u}:{p}@{rest}"
        return url

    # ──────────────────────────────────────────────────────────────────────
    # Публичный метод
    # ──────────────────────────────────────────────────────────────────────
    async def generate_legal_answer(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
        retries: int = 2,
    ) -> LegalAdvice:
        """
        Запрашивает у модели краткий юридический ответ и список норм права.
        Возвращает LegalAdvice(answer, laws). Бросает RuntimeError при неуспехе после ретраев.
        """
        sys_prompt = (
            "Ты юридический ассистент для РФ. Отвечай кратко, ясно и корректно. "
            "Если данных недостаточно, попроси уточнить. Выделяй списки с новой строки. "
            "Строго соблюдай фактологию, не выдумывай нормы права."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        if short_history:
            for h in short_history[-5:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": question})

        # Первичная схема JSON (если SDK умеет response_format)
        json_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "laws": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "laws"],
            "additionalProperties": False,
        }

        # Базовая мета-логика (без промптов)
        log.debug(
            "LLM request: model=%s temp=%.2f max_out=%s effort=%s verbosity=%s proxy=%s",
            self._s.openai_model,
            self._s.openai_temperature,
            self._s.openai_max_tokens,
            getattr(self._s, "openai_reasoning_effort", None),
            getattr(self._s, "openai_verbosity", None),
            bool(getattr(self._s, "openai_proxy_url", None)),
        )

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            t0 = perf_counter()
            try:
                # 1) Попытка №1: Responses API с response_format (JSON Schema)
                content = await self._responses_create_compat(
                    messages=messages,
                    json_schema=json_schema,
                    use_schema=True,
                )
                took_ms = int((perf_counter() - t0) * 1000)

                data = self._safe_json_extract(content)
                answer = str(data.get("answer") or "").strip()
                laws = data.get("laws") or []
                if not isinstance(laws, list):
                    laws = []

                log.info("LLM ok: attempt=%s api=responses(schema) took_ms=%s answer_len=%s laws=%s",
                         attempt, took_ms, len(answer), len(laws))
                return LegalAdvice(answer=answer, laws=list(map(str, laws)))

            except TypeError as e:
                # Обычно сюда попадаем при старом SDK (нет response_format / text / reasoning / max_output_tokens).
                last_err = e
                log.warning("Responses(schema) not supported by SDK: %r. Falling back.", e)

                try:
                    # 2) Попытка №2: Responses API без response_format (жёсткая подсказка вернуть JSON)
                    # Добавим инструкцию в system: вернуть строго JSON-объект с ключами answer, laws
                    forced_messages = messages.copy()
                    forced_messages[0] = {
                        "role": "system",
                        "content": sys_prompt + " Всегда возвращай ТОЛЬКО JSON вида {\"answer\": \"...\", \"laws\": [\"...\"]}.",
                    }
                    content = await self._responses_create_compat(
                        messages=forced_messages,
                        json_schema=json_schema,
                        use_schema=False,
                    )
                    took_ms = int((perf_counter() - t0) * 1000)

                    data = self._safe_json_extract(content)
                    answer = str(data.get("answer") or "").strip()
                    laws = data.get("laws") or []
                    if not isinstance(laws, list):
                        laws = []

                    log.info("LLM ok: attempt=%s api=responses(plain) took_ms=%s answer_len=%s laws=%s",
                             attempt, took_ms, len(answer), len(laws))
                    return LegalAdvice(answer=answer, laws=list(map(str, laws)))

                except Exception as e2:
                    last_err = e2
                    log.warning("Responses(plain) fail: %r. Falling back to chat.completions.", e2)

                    try:
                        # 3) Попытка №3: Chat Completions (при наличии)
                        content = await self._chat_completions_compat(
                            messages=messages,
                            force_json=True,  # попробуем json_object, если SDK умеет
                        )
                        took_ms = int((perf_counter() - t0) * 1000)

                        data = self._safe_json_extract(content)
                        answer = str(data.get("answer") or "").strip()
                        laws = data.get("laws") or []
                        if not isinstance(laws, list):
                            laws = []

                        log.info("LLM ok: attempt=%s api=chat.completions took_ms=%s answer_len=%s laws=%s",
                                 attempt, took_ms, len(answer), len(laws))
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
                log.warning("LLM fail: attempt=%s took_ms=%s status=%s err=%r body=%s",
                            attempt, took_ms, status, e, body)

            # backoff между попытками
            await asyncio.sleep(0.75 * (attempt + 1))

        log.exception("LLM fatal after retries: %r", last_err)
        raise RuntimeError(f"OpenAI error: {last_err}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal: Responses API с авто-удалением неподдерживаемых параметров
    # ──────────────────────────────────────────────────────────────────────
    async def _responses_create_compat(
        self,
        *,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        use_schema: bool,
    ) -> str:
        """
        Пытается вызвать client.responses.create(**payload).
        Если SDK старый и не знает аргументы (response_format/text/reasoning/max_output_tokens),
        удаляем их по сообщению TypeError("unexpected keyword argument '...'") и повторяем.
        Возвращает объединённый текст ответа.
        """
        payload: Dict[str, Any] = {
            "model": self._s.openai_model,
            "input": messages,
            "temperature": self._s.openai_temperature,
        }

        # эти ключи не все версии понимают — будем удалять по мере необходимости
        optional_keys = []

        # max_output_tokens (в Responses), но в старых реализациях могло не быть
        if getattr(self._s, "openai_max_tokens", None):
            payload["max_output_tokens"] = self._s.openai_max_tokens
            optional_keys.append("max_output_tokens")

        # reasoning/text — «новые» допы
        if getattr(self._s, "openai_reasoning_effort", None):
            payload["reasoning"] = {"effort": self._s.openai_reasoning_effort}
            optional_keys.append("reasoning")
        if getattr(self._s, "openai_verbosity", None):
            payload["text"] = {"verbosity": self._s.openai_verbosity}
            optional_keys.append("text")

        # JSON Schema (может не поддерживаться)
        if use_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "legal_answer", "schema": json_schema}}
            optional_keys.append("response_format")

        # будем пробовать удалять по сообщению об ошибке
        while True:
            try:
                resp = await self._client.responses.create(**payload)
                text = self._extract_text_from_responses(resp)
                if not text:
                    # На всякий — пусть не упадёт, вернём пустую строку
                    return ""
                return text
            except TypeError as e:
                # ищем "unexpected keyword argument 'xxx'"
                m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
                if not m:
                    raise
                bad = m.group(1)
                if bad in payload:
                    log.debug("SDK doesn't support '%s' — removing and retrying...", bad)
                    payload.pop(bad, None)
                    continue
                # если ключ вложенный, например reasoning/text — удалим их, если добавляли
                if bad in {"reasoning", "text", "response_format", "max_output_tokens"} and bad in optional_keys:
                    payload.pop(bad, None)
                    continue
                # что-то иное
                raise

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
        Если SDK не знает response_format={"type":"json_object"}, убираем и просим JSON подсказкой.
        """
        # сконвертим сообщения 1:1
        chat_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        kwargs: Dict[str, Any] = dict(
            model=self._s.openai_model,
            messages=chat_msgs,
            temperature=self._s.openai_temperature,
        )
        # В chat.completions max_tokens — не max_output_tokens
        if getattr(self._s, "openai_max_tokens", None):
            kwargs["max_tokens"] = self._s.openai_max_tokens

        # Попробуем json_object
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.chat.completions.create(**kwargs)
            return self._extract_text_from_chat(resp)
        except TypeError as e:
            # старые SDK не знают response_format — убираем и просим JSON подсказкой
            if "response_format" in str(e) and force_json:
                # усилим system-подсказку
                patched = chat_msgs.copy()
                if patched and patched[0]["role"] == "system":
                    patched[0] = {
                        "role": "system",
                        "content": patched[0]["content"] + " Всегда возвращай ТОЛЬКО JSON {\"answer\":\"...\",\"laws\":[\"...\"]}.",
                    }
                kwargs.pop("response_format", None)
                kwargs["messages"] = patched
                resp = await self._client.chat.completions.create(**kwargs)
                return self._extract_text_from_chat(resp)
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Helpers: вытаскивание текста
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_text_from_responses(resp: Any) -> str:
        """
        Унифицированно достаём текст из Responses API объекта.
        """
        # Новый SDK даёт resp.output_text (str) или метод
        out = getattr(resp, "output_text", None)
        if callable(out):
            try:
                return out()  # в некоторых версиях это метод
            except Exception:
                pass
        if isinstance(out, str):
            return out

        # В некоторых билдах — список "output" c "content" → "text"
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for seg in content:
                        if isinstance(seg, dict) and seg.get("type") == "output_text":
                            txt = seg.get("text")
                            if isinstance(txt, str):
                                parts.append(txt)
            return "".join(parts)
        # Фоллбек — как есть
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
        Возвращает dict, минимум: {"answer": <string>, "laws": []}
        """
        text = (text or "").strip()
        # прямая попытка
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # попытка вырезать первый {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # фоллбек
        return {"answer": text, "laws": []}
