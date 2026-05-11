"""
DeepSeek LLM Implementation

DeepSeek uses OpenAI-compatible API, so we can use the OpenAI client.
"""

from typing import Any, Dict, Iterator, Mapping
from openai import OpenAI
from .base import BaseLLM, parse_json_object


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM implementation.

    Supports DeepSeek models via OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        **kwargs
    ):
        """
        Initialize DeepSeek LLM.

        Args:
            api_key: DeepSeek API key
            model: Model name (default: deepseek-v4-flash)
            base_url: API base URL (default: https://api.deepseek.com)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.last_usage = None
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using DeepSeek API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        self.last_usage = response.usage.model_dump() if response.usage else None
        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        schema: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a JSON object using DeepSeek's OpenAI-compatible JSON mode."""
        params = {**self.config, **kwargs}
        params.setdefault("response_format", {"type": "json_object"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        self.last_usage = response.usage.model_dump() if response.usage else None
        return parse_json_object(response.choices[0].message.content or "{}")

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream generate text using DeepSeek API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **params
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
