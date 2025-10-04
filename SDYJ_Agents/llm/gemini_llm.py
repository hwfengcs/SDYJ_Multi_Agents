"""
Gemini LLM Implementation
"""

from typing import Iterator
import google.generativeai as genai
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
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

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

        response = self.client.generate_content(prompt, **params)
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

        response = self.client.generate_content(prompt, stream=True, **params)

        for chunk in response:
            if chunk.text:
                yield chunk.text
