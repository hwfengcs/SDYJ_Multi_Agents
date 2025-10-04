"""
LLM Factory

This module provides a factory pattern for creating LLM instances.
"""

from typing import Optional
from .base import BaseLLM


class LLMFactory:
    """
    Factory class for creating LLM instances.

    This factory allows easy instantiation of different LLM providers
    without needing to import them directly.
    """

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        Register a new LLM provider.

        Args:
            name: Provider name (e.g., 'openai', 'claude', 'gemini')
            provider_class: The LLM class to register
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create_llm(
        cls,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance.

        Args:
            provider: Provider name ('openai', 'claude', 'gemini', 'deepseek')
            api_key: API key for the provider
            model: Optional model name. If not provided, uses provider's default
            **kwargs: Additional configuration parameters

        Returns:
            An instance of the requested LLM provider

        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()

        if provider not in cls._providers:
            # Try to lazy load the provider
            cls._lazy_load_provider(provider)

        if provider not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Available providers: {available}"
            )

        llm_class = cls._providers[provider]

        # Create instance with or without model parameter
        if model:
            return llm_class(api_key=api_key, model=model, **kwargs)
        else:
            return llm_class(api_key=api_key, **kwargs)

    @classmethod
    def _lazy_load_provider(cls, provider: str):
        """
        Lazy load a provider when needed.

        Args:
            provider: Provider name to load
        """
        try:
            if provider == 'openai':
                from .openai_llm import OpenAILLM
                cls.register_provider('openai', OpenAILLM)
            elif provider == 'claude':
                from .claude_llm import ClaudeLLM
                cls.register_provider('claude', ClaudeLLM)
            elif provider == 'gemini':
                from .gemini_llm import GeminiLLM
                cls.register_provider('gemini', GeminiLLM)
            elif provider == 'deepseek':
                from .deepseek_llm import DeepSeekLLM
                cls.register_provider('deepseek', DeepSeekLLM)
        except ImportError as e:
            # Provider module not available
            pass

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
