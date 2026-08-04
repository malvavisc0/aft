"""Shared user-facing error type for the aft pipeline."""

from __future__ import annotations


class AftError(RuntimeError):
    """User-facing error (caught by CLI for clean output)."""
