"""Shared UI utilities — console and logging configuration."""

import logging

from rich.console import Console

# Single Console instance used across all modules (width 200 for consistency).
console = Console(width=200)

# Noise loggers to suppress from third-party libraries.
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpcore",
    "httpx",
    "datasets",
    "torchao",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "bitsandbytes",
    "urllib3",
    "gptqmodel",
)


def silence_noisy_loggers() -> None:
    """Suppress noisy DEBUG logs from HF/HTTP/torch libraries."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
