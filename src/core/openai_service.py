from __future__ import annotations

import logging
from typing import Any, Optional

from src.bot.openai_gateway import ask_legal as oai_ask_legal
from .cache import ResponseCache

logger = logging.getLogger(__name__)

class OpenAIService:
    """Application-facing service for legal Q&A over OpenAI Responses API.

    Encapsulates gateway calls with caching, retry logic and monitoring.
    """
    
    def __init__(self, cache: Optional[ResponseCache] = None, enable_cache: bool = True):
        self.cache = cache
        self.enable_cache = enable_cache
        
        # Статистика
        self.total_requests = 0
        self.cached_requests = 0
        self.failed_requests = 0

    async def ask_legal(
        self, 
        system_prompt: str, 
        user_text: str,
        force_refresh: bool = False
    ) -> dict[str, Any]:
        """Запрос к OpenAI с кешированием и обработкой ошибок"""
        self.total_requests += 1
        
        # Проверяем кеш если включен
        if self.cache and self.enable_cache and not force_refresh:
            try:
                cached_response = await self.cache.get_cached_response(
                    system_prompt=system_prompt,
                    user_text=user_text
                )
                
                if cached_response:
                    self.cached_requests += 1
                    logger.info(f"Returning cached response for user query (length: {len(user_text)})")
                    return cached_response
                    
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        # Выполняем запрос к OpenAI
        try:
            response = await oai_ask_legal(system_prompt, user_text)
            
            # Кешируем успешный ответ
            if (self.cache and 
                self.enable_cache and 
                response.get("ok") and 
                response.get("text")):
                try:
                    await self.cache.cache_response(
                        system_prompt=system_prompt,
                        user_text=user_text,
                        response=response
                    )
                    logger.debug("Response cached successfully")
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            return response
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"OpenAI request failed: {e}")
            raise
    
    async def get_stats(self) -> dict[str, Any]:
        """Статистика сервиса"""
        stats = {
            "total_requests": self.total_requests,
            "cached_requests": self.cached_requests,
            "failed_requests": self.failed_requests,
            "cache_enabled": self.enable_cache,
            "cache_hit_rate": (
                self.cached_requests / self.total_requests 
                if self.total_requests > 0 else 0
            )
        }
        
        # Добавляем статистику кеша если доступен
        if self.cache:
            try:
                cache_stats = await self.cache.get_cache_stats()
                stats["cache_stats"] = cache_stats
            except Exception as e:
                stats["cache_error"] = str(e)
        
        return stats
    
    async def clear_cache(self) -> None:
        """Очистка кеша"""
        if self.cache:
            await self.cache.clear_cache()
            logger.info("OpenAI response cache cleared")
    
    async def close(self) -> None:
        """Закрытие сервиса"""
        if self.cache:
            await self.cache.close()


