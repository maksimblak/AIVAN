from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Sequence

try:
    # базовый запрос (обязателен)
    from src.bot.openai_gateway import ask_legal as oai_ask_legal
except Exception as e:
    raise ImportError("Не найден src.bot.openai_gateway.ask_legal") from e

# стриминговый запрос – может отсутствовать в gateway (обрабатываем это)
try:
    from src.bot.openai_gateway import ask_legal_stream as oai_ask_legal_stream  # type: ignore
except Exception:  # noqa: BLE001
    oai_ask_legal_stream = None  # type: ignore

from src.bot.openai_gateway import (
    close_async_openai_client,
    format_legal_response_text,
)
from src.core.attachments import QuestionAttachment

from .cache import ResponseCache

logger = logging.getLogger(__name__)

# Колбэк принимает либо (delta: str), либо (delta: str, is_final: bool)
StreamCallback = Callable[..., Awaitable[None]]


async def _safe_fire_callback(
    cb: StreamCallback | None, delta: str, is_final: bool = False
) -> None:
    """Вызов колбэка с поддержкой 1 или 2 аргументов, sync/async."""
    if not cb:
        return
    try:
        # пробуем (delta, is_final)
        res = cb(delta, is_final)
    except TypeError:
        # колбэк ожидает только один аргумент
        res = cb(delta)
    if inspect.isawaitable(res):
        await res


class OpenAIService:
    """
    Application-facing сервис над OpenAI gateway.

    Возможности:
    - ask_legal: обычный запрос с кэшированием
    - ask_legal_stream: стрим с кэш-фолбэком (и «псевдостримом», если gateway-стрим недоступен)
    - простая статистика и методы обслуживания кэша
    """

    def __init__(self, cache: ResponseCache | None = None, enable_cache: bool = True):
        self.cache = cache
        self.enable_cache = enable_cache

        # Статистика
        self.total_requests = 0
        self.cached_requests = 0
        self.failed_requests = 0

        # последний «полный» текст из стрима (удобно читать после показа дельт)
        self.last_full_text: str = ""

    # ------------------------ обычный запрос ------------------------

    async def ask_legal(
        self,
        system_prompt: str,
        user_text: str,
        *,
        attachments: Sequence[QuestionAttachment] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Запрос к OpenAI с кэшированием и обработкой ошибок."""
        self.total_requests += 1

        use_cache = bool(self.cache and self.enable_cache and not force_refresh and not attachments)
        cache_params = {"schema": "legal_json_v2"}

        if use_cache:
            try:
                cached = await self.cache.get_cached_response(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    model_params=cache_params,
                )
                if cached:
                    self.cached_requests += 1
                    logger.info("Cache HIT for ask_legal (len=%s)", len(user_text))
                    return cached
            except Exception as e:  # noqa: BLE001
                logger.warning("Cache get failed: %s", e)

        # сеть
        try:
            response = await oai_ask_legal(system_prompt, user_text, attachments=attachments)

            # кэшируем только успешный и непустой ответ
            if use_cache and response.get("ok") and response.get("text"):
                try:
                    await self.cache.cache_response(
                        system_prompt=system_prompt,
                        user_text=user_text,
                        response=response,
                        model_params=cache_params,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Cache set failed: %s", e)

            return response

        except Exception as e:  # noqa: BLE001
            self.failed_requests += 1
            logger.error("OpenAI request failed: %s", e)
            raise

    # ------------------------ стриминговый запрос ------------------------

    async def ask_legal_stream(
        self,
        system_prompt: str,
        user_text: str,
        callback: StreamCallback | None = None,
        *,
        force_refresh: bool = False,
        pseudo_chunk: int = 600,
        attachments: Sequence[QuestionAttachment] | None = None,
    ) -> dict[str, Any]:
        """
        Стрим с кэш-фолбэком. Возвращает финальный dict, а дельты отдает через `callback`.

        Поведение:
        1) Если в кэше есть готовый ответ — отдаем его и «симулируем стрим» через callback.
        2) Если в gateway есть `ask_legal_stream`:
           - поддерживаем оба варианта: (а) async-генератор дельт; (б) функция с колбэком.
        3) Если стрима нет — обычный запрос + «псевдострим» кусками.
        """
        self.total_requests += 1
        self.last_full_text = ""
        parts: list[str] = []

        # кэш (и псевдострим из кэша)
        use_cache = bool(self.cache and self.enable_cache and not force_refresh and not attachments)
        cache_params = {"schema": "legal_json_v2"}
        if use_cache:
            try:
                cached = await self.cache.get_cached_response(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    model_params=cache_params,
                )
                if cached and cached.get("ok") and cached.get("text"):
                    self.cached_requests += 1
                    text = str(cached.get("text", ""))
                    formatted_text = format_legal_response_text(text)
                    # «поток» из кэша
                    if callback:
                        if len(formatted_text) <= pseudo_chunk:
                            await _safe_fire_callback(callback, formatted_text, True)
                        else:
                            for i in range(0, len(formatted_text), pseudo_chunk):
                                await _safe_fire_callback(
                                    callback, formatted_text[i : i + pseudo_chunk], False
                                )
                            await _safe_fire_callback(callback, "", True)
                    self.last_full_text = formatted_text
                    cached = dict(cached)
                    cached["text"] = formatted_text
                    return cached
            except Exception as e:  # noqa: BLE001
                logger.warning("Cache get failed (stream): %s", e)

        # вспомогательный on_delta: накапливает и пробрасывает наружу
        async def on_delta(delta: str) -> None:
            if not delta:
                return
            parts.append(delta)
            await _safe_fire_callback(callback, delta, False)

        # если в gateway есть стрим — пробуем оба режима
        if oai_ask_legal_stream:
            try:
                candidate = oai_ask_legal_stream(
                    system_prompt,
                    user_text,
                    on_delta,
                    attachments=attachments,
                )  # type: ignore[misc]
            except TypeError:
                if attachments:
                    raise
                # сигнатура без колбэка — возможно, async-генератор
                candidate = oai_ask_legal_stream(  # type: ignore[misc]
                    system_prompt,
                    user_text,
                    attachments=attachments,
                )

            try:
                # режим async-генератора
                if hasattr(candidate, "__aiter__"):
                    async for delta in candidate:  # type: ignore[attr-defined]
                        await on_delta(str(delta))
                    text = "".join(parts).strip()
                    formatted_text = format_legal_response_text(text)
                    await _safe_fire_callback(callback, "", True)
                    self.last_full_text = formatted_text

                    resp = {"ok": True, "text": formatted_text}
                    # кэшируем
                    if self.cache and self.enable_cache and not attachments and text:
                        try:
                            await self.cache.cache_response(
                                system_prompt,
                                user_text,
                                resp,
                                model_params=cache_params,
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Cache set failed (stream/gen): %s", e)
                    return resp

                # режим «функция с колбэком» -> нужно дождаться результата
                result = await candidate  # type: ignore[misc]
                text = (result or {}).get("text") if isinstance(result, dict) else None
                if not text:
                    text = "".join(parts).strip()
                    result = {"ok": True, "text": text}
                formatted_text = format_legal_response_text(text or "")
                await _safe_fire_callback(callback, "", True)
                self.last_full_text = formatted_text

                if isinstance(result, dict):
                    result["text"] = formatted_text
                else:
                    result = {"ok": True, "text": formatted_text}

                if self.cache and self.enable_cache and not attachments and formatted_text:
                    try:
                        await self.cache.cache_response(
                            system_prompt,
                            user_text,
                            result,  # type: ignore[arg-type]
                            model_params=cache_params,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Cache set failed (stream/callback): %s", e)

                return result  # type: ignore[return-value]

            except Exception as e:  # noqa: BLE001
                logger.warning("Gateway streaming failed, fallback to non-stream. Error: %s", e)

        # фолбэк: обычный запрос + псевдострим кусками
        try:
            result = await oai_ask_legal(system_prompt, user_text, attachments=attachments)
            original_text = str(result.get("text", "") or "")
            formatted_text = format_legal_response_text(original_text)

            if callback:
                if len(formatted_text) <= pseudo_chunk:
                    await _safe_fire_callback(callback, formatted_text, True)
                else:
                    for i in range(0, len(formatted_text), pseudo_chunk):
                        await _safe_fire_callback(callback, formatted_text[i : i + pseudo_chunk], False)
                    await _safe_fire_callback(callback, "", True)

            self.last_full_text = formatted_text

            if isinstance(result, dict):
                result["text"] = formatted_text

            # кэшируем обычным способом (если еще не закэшировано)
            if self.cache and self.enable_cache and not attachments and result.get("ok") and result.get("text"):
                try:
                    await self.cache.cache_response(
                        system_prompt,
                        user_text,
                        result,
                        model_params=cache_params,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Cache set failed (fallback): %s", e)

            return result

        except Exception as e:  # noqa: BLE001
            self.failed_requests += 1
            logger.error("OpenAI streaming request failed: %s", e)
            raise

    async def answer_question(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        attachments: Sequence[QuestionAttachment] | None = None,
        stream_callback: StreamCallback | None = None,
        force_refresh: bool = False,
        model: str | None = None,  # reserved for future routing
        user_id: int | None = None,  # reserved for future telemetry
    ) -> dict[str, Any]:
        """
        Высокоуровневый помощник для обработки пользовательского вопроса.

        Позволяет прокинуть системный промт и выбрать стриминговый или обычный сценарий.
        Параметры `model` и `user_id` зарезервированы для совместимости с существующими
        вызовами и могут использоваться позже для маршрутизации запросов.
        """
        _ = model  # placeholders for future use
        _ = user_id

        if not system_prompt:
            from src.bot.promt import LEGAL_SYSTEM_PROMPT

            system_prompt = LEGAL_SYSTEM_PROMPT

        if stream_callback:
            return await self.ask_legal_stream(
                system_prompt,
                user_text,
                callback=stream_callback,
                force_refresh=force_refresh,
                attachments=attachments,
            )

        return await self.ask_legal(
            system_prompt,
            user_text,
            attachments=attachments,
            force_refresh=force_refresh,
        )

    # ------------------------ служебные методы ------------------------

    async def get_stats(self) -> dict[str, Any]:
        stats = {
            "total_requests": self.total_requests,
            "cached_requests": self.cached_requests,
            "failed_requests": self.failed_requests,
            "cache_enabled": self.enable_cache,
            "cache_hit_rate": (
                (self.cached_requests / self.total_requests) if self.total_requests else 0.0
            ),
        }
        if self.cache:
            try:
                stats["cache_stats"] = await self.cache.get_cache_stats()
            except Exception as e:  # noqa: BLE001
                stats["cache_error"] = str(e)
        return stats

    async def clear_cache(self) -> None:
        if self.cache:
            await self.cache.clear_cache()
            logger.info("OpenAI response cache cleared")

    async def close(self) -> None:
        if self.cache:
            await self.cache.close()
        await close_async_openai_client()
