from typing import AsyncIterator, List, Optional
from openai import AsyncOpenAI
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage


class OpenAIAdapter(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key or "dummy-key")
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
