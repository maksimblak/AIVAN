from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
try:
    from openai import AsyncOpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Не удалось импортировать openai. Установите пакет: pip install openai>=1.0") from exc

try:
    # пакетный импорт
    from telegram_legal_bot.config import Settings
except ImportError:  # fallback для «плоской» структуры
    from config import Settings


@dataclass
class LegalAdvice:
    """Единый контейнер ответа ИИ."""
    answer: str
    laws: List[str]

    # совместимость со старым кодом, который ожидает dict
    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "laws": self.laws}

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class OpenAIService:
    """
    Обёртка над OpenAI Responses API.
    - Поддержка прокси (OPENAI_PROXY_URL/USER/PASS)
    - Мягкие ретраи
    - Аккуратное закрытие http-клиента (aclose)
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._own_httpx: Optional[httpx.AsyncClient] = None
        http_client: Optional[httpx.AsyncClient] = None

        if getattr(settings, "openai_proxy_url", None):
            proxy_url = self._inject_basic_auth(
                settings.openai_proxy_url or "",
                settings.openai_proxy_user,
                settings.openai_proxy_pass,
            )
            self._own_httpx = httpx.AsyncClient(
                proxies=proxy_url,
                timeout=httpx.Timeout(45.0),
            )
            http_client = self._own_httpx

        self._client = AsyncOpenAI(api_key=settings.openai_api_key, http_client=http_client)

    async def aclose(self) -> None:
        if self._own_httpx is not None:
            await self._own_httpx.aclose()

    def _inject_basic_auth(self, url: str, user: Optional[str], pwd: Optional[str]) -> str:
        if not url:
            return url
        if user and pwd and "@" not in url and "://" in url:
            scheme, rest = url.split("://", 1)
            return f"{scheme}://{user}:{pwd}@{rest}"
        return url

    async def generate_legal_answer(
        self,
        question: str,
        short_history: Optional[List[Dict[str, str]]] = None,
        retries: int = 2,
    ) -> LegalAdvice:
        """
        Возвращает LegalAdvice(answer, laws).
        Совместимо со старым кодом через .get() / __getitem__.
        """
        sys_prompt = (
            "Ты юридический ассистент для РФ. Отвечай кратко, ясно и корректно. "
            "Если данных недостаточно, проси уточнить. Выделяй списки с новой строки. "
            "Строго соблюдай фактологию, не выдумывай нормы права."
        )

        json_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "laws": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "laws"],
            "additionalProperties": False,
        }

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        if short_history:
            for h in short_history[-5:]:
                r, c = h.get("role"), h.get("content")
                if r in {"user", "assistant"} and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": question})

        payload = dict(
            model=self._s.openai_model,
            input=messages,
            max_output_tokens=self._s.openai_max_tokens,
            temperature=self._s.openai_temperature,
            reasoning={"effort": self._s.openai_reasoning_effort},
            text={"verbosity": self._s.openai_verbosity},
            response_format={"type": "json_schema", "json_schema": {"name": "legal_answer", "schema": json_schema}},
        )

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.responses.create(**payload)
                content = (resp.output_text or "").strip()
                data = self._safe_json_extract(content)
                answer = str(data.get("answer") or "").strip()
                laws = data.get("laws") or []
                if not isinstance(laws, list):
                    laws = []
                return LegalAdvice(answer=answer, laws=list(map(str, laws)))
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.75 * (attempt + 1))

        raise RuntimeError(f"OpenAI error: {last_err}")

    @staticmethod
    def _safe_json_extract(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                obj = json.loads(text[s : e + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {"answer": text, "laws": []}
