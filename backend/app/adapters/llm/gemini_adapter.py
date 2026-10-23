from typing import AsyncIterator, List, Optional
from openai import AsyncOpenAI
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage


class GeminiAdapter(LLMProvider):
    """Utiliza a compatibilidade oficial OpenAI da Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.client = AsyncOpenAI(
            api_key=api_key or "dummy-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage],
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
