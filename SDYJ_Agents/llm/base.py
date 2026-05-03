"""
LLM Base Class

This module defines the abstract base class for all LLM providers.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Mapping


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse one JSON object from an LLM response.

    Native JSON mode should return a bare object, but replay traces and older
    models may still include Markdown fences or short prose. This parser keeps
    the fallback path centralized so agents do not each hand-roll ``find('{')``.
    """
    if not text:
        raise ValueError("empty JSON response")

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.lower().startswith("json\n"):
            candidate = candidate[5:].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = candidate.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(candidate[start:])
                break
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
        else:
            raise

    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object")
    return parsed


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

    def generate_json(
        self,
        prompt: str,
        schema: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON object.

        Providers can override this to use native JSON/object mode. The base
        fallback appends a short schema reminder and parses the response with
        ``parse_json_object`` so tests, replay, and custom local LLM adapters
        keep working without implementing a new method immediately.
        """
        schema_hint = ""
        if schema:
            schema_hint = (
                "\n\nReturn a single valid JSON object matching this schema. "
                "Do not include Markdown fences or prose.\n"
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
            )
        response = self.generate(prompt + schema_hint, **kwargs)
        return parse_json_object(response)

    def __repr__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(model={self.model})"
