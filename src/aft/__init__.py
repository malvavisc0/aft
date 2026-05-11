"""Aria Finetune — standalone fine-tuning and GPTQ quantization CLI."""

__version__ = "0.0.1"

from aft.cli import app


def main() -> None:
    """Entry point for the `aft` CLI."""
    app()
