"""Shared test fixtures for aft test suite."""

from __future__ import annotations

from typing import Any


class FakeTokenizer:
    """Minimal tokenizer stub that returns predictable token counts."""

    eos_token = "<eos>"

    def __init__(self, tokens_per_char: int = 1) -> None:
        self._tokens_per_char = tokens_per_char

    def __call__(
        self, text: str, return_length: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        length = max(1, len(text) * self._tokens_per_char)
        result: dict[str, Any] = {}
        if return_length:
            result["length"] = [length]
        return result

    def save_pretrained(self, path: str) -> None:
        pass


class FakeDataset:
    """Minimal in-memory dataset that mimics HuggingFace Dataset API."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self._rows]

    def select(self, indices: range) -> "FakeDataset":
        return FakeDataset([self._rows[i] for i in indices])

    def map(self, fn: Any) -> "FakeDataset":
        return FakeDataset([fn(row) for row in self._rows])

    def filter(self, fn: Any) -> "FakeDataset":
        return FakeDataset([row for row in self._rows if fn(row)])


def make_fake_dataset(texts: list[str]) -> FakeDataset:
    """Create a FakeDataset from a list of text strings."""
    return FakeDataset([{"text": t} for t in texts])
