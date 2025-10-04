"""
Logging Utility

This module provides logging functionality for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "SDYJ_Research",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setLevel(level)
    
    if not use_rich:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "SDYJ_Research") -> logging.Logger:
    return logging.getLogger(name)


class LoggerMixin:
    @property
    def logger(self) -> logging.Logger:
        name = f"SDYJ_Research.{self.__class__.__name__}"
        return logging.getLogger(name)


console = Console()


def print_success(message: str):
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str):
    console.print(f"[blue]ℹ[/blue] {message}")


def print_step(message: str):
    console.print(f"[cyan]▶[/cyan] {message}")
