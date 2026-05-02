"""
Claude LLM Implementation
"""

from typing import Iterator
from anthropic import Anthropic
from .base import BaseLLM


class ClaudeLLM(BaseLLM):
    """
    Anthropic Claude LLM implementation.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        """
        Initialize Claude LLM.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-5-sonnet-20241022)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.last_usage = None
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Claude API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        # Set default max_tokens if not provided
        if 'max_tokens' not in params:
            params['max_tokens'] = 4096

        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        # Anthropic returns input_tokens / output_tokens; SDYJ_Agents.utils.cost
        # normalizes both naming conventions, so we expose the raw dict.
        if response.usage:
            self.last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            self.last_usage = None
        return response.content[0].text

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream generate text using Claude API.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        # Merge default config with kwargs
        params = {**self.config, **kwargs}

        # Set default max_tokens if not provided
        if 'max_tokens' not in params:
            params['max_tokens'] = 4096

        with self.client.messages.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **params
        ) as stream:
            for text in stream.text_stream:
                yield text
