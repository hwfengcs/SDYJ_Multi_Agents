"""
Prompt Loader

This module provides functionality to load and render prompt templates.
"""

import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template


class PromptLoader:
    """
    Centralized prompt management system.

    Loads prompt templates from markdown files and renders them with variables.
    """

    def __init__(self, prompts_dir: str = None):
        """
        Initialize the PromptLoader.

        Args:
            prompts_dir: Directory containing prompt templates.
                        If None, uses the default prompts directory.
        """
        if prompts_dir is None:
            # Default to the prompts directory in the same folder as this file
            prompts_dir = Path(__file__).parent

        self.prompts_dir = Path(prompts_dir)

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )

    def load(self, prompt_name: str, **variables: Any) -> str:
        """
        Load and render a prompt template.

        Args:
            prompt_name: Name of the prompt file (without .md extension)
            **variables: Variables to render in the template

        Returns:
            Rendered prompt string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Add current time by default
        if 'CURRENT_TIME' not in variables:
            variables['CURRENT_TIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Load and render template
        try:
            template = self.env.get_template(f"{prompt_name}.md")
            rendered = template.render(**variables)
            return rendered
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load prompt '{prompt_name}' from {self.prompts_dir}: {e}"
            )

    def load_raw(self, prompt_name: str) -> str:
        """
        Load a prompt template without rendering.

        Args:
            prompt_name: Name of the prompt file (without .md extension)

        Returns:
            Raw prompt template string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / f"{prompt_name}.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}"
            )

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def render_string(self, template_str: str, **variables: Any) -> str:
        """
        Render a template string with variables.

        Args:
            template_str: Template string to render
            **variables: Variables to render in the template

        Returns:
            Rendered string
        """
        # Add current time by default
        if 'CURRENT_TIME' not in variables:
            variables['CURRENT_TIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        template = Template(template_str)
        return template.render(**variables)


# Global instance for convenience
_default_loader = None


def get_default_loader() -> PromptLoader:
    """
    Get the default global PromptLoader instance.

    Returns:
        Default PromptLoader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader
