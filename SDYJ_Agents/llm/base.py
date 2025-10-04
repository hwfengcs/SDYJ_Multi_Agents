"""
LLM Base Class

This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    All LLM implementations (OpenAI, Claude, Gemini) should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the LLM.

        Args:
            api_key: API key for the LLM provider
            model: Model name/identifier
            **kwargs: Additional configuration parameters
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        pass

    def __repr__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(model={self.model})"
