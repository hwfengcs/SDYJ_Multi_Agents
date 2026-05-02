"""
Gemini LLM Implementation
"""

from typing import Iterator
from google import genai
from google.genai import types
from .base import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM implementation.

    Supports Gemini Pro, Gemini Ultra, and other Gemini models.
    """

    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        """
        Initialize Gemini LLM.

        Args:
            api_key: Google API key
            model: Model name (default: gemini-pro)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.client = genai.Client(api_key=api_key)

    def _build_config(self, params: dict) -> types.GenerateContentConfig | None:
        """Translate common generation parameters to Gemini config."""
        config = {}
        if "temperature" in params:
            config["temperature"] = params.pop("temperature")
        if "max_tokens" in params:
            config["max_output_tokens"] = params.pop("max_tokens")
        if "max_output_tokens" in params:
            config["max_output_tokens"] = params.pop("max_output_tokens")

        return types.GenerateContentConfig(**config) if config else None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_output_tokens, etc.)

        Returns:
            Generated text
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        config = self._build_config(params)
        request = {
            "model": self.model,
            "contents": prompt,
        }
        if config:
            request["config"] = config

        response = self.client.models.generate_content(**request)
        return response.text

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream generate text using Gemini API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        config = self._build_config(params)
        request = {
            "model": self.model,
            "contents": prompt,
        }
        if config:
            request["config"] = config

        response = self.client.models.generate_content_stream(**request)

        for chunk in response:
            if chunk.text:
                yield chunk.text
