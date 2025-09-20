from __future__ import annotations

from typing import Any

from src.bot.openai_gateway import ask_legal as oai_ask_legal


class OpenAIService:
    """Application-facing service for legal Q&A over OpenAI Responses API.

    Encapsulates gateway calls for easier substitution/mocking and testing.
    """

    async def ask_legal(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        return await oai_ask_legal(system_prompt, user_text)


