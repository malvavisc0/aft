"""Dataset cleaning utilities for fine-tuning."""

import hashlib
import re

from loguru import logger

from aft.ui import console


def clean_dataset(
    dataset,
    tokenizer,
    dedup: bool = False,
    min_tokens: int = 10,
    max_tokens: int = 2048,
    languages: list[str] | None = None,
    max_special_ratio: float = 0.3,
):
    """Apply cleaning steps to a HuggingFace dataset.

    Steps applied in order: whitespace cleanup → special char filter →
    length filter → language filter → deduplication.

    Args:
        dataset: A HF ``Dataset`` with a ``text`` column.
        tokenizer: Model tokenizer for token-count filtering.
        dedup: Remove exact duplicate texts.
        min_tokens: Minimum token count (shorter samples dropped).
        max_tokens: Maximum token count (longer samples dropped).
        languages: If set, keep only samples in these language codes.
        max_special_ratio: Drop samples where non-alphanumeric characters
            exceed this fraction of total length.

    Returns:
        Cleaned dataset.
    """
    n_start = len(dataset)

    # ── 1. Whitespace normalization ────────────────────────────────────
    def _clean_whitespace(example: dict) -> dict:
        text = example["text"]
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = text.strip()
        return {"text": text}

    dataset = dataset.map(_clean_whitespace)

    # ── 2. Special character ratio filter ──────────────────────────────
    def _special_char_ok(example: dict) -> bool:
        text = example["text"]
        if not text:
            return False
        alpha = sum(1 for c in text if c.isalnum() or c.isspace())
        ratio = 1.0 - (alpha / len(text))
        return ratio <= max_special_ratio

    n_before = len(dataset)
    dataset = dataset.filter(_special_char_ok)
    if len(dataset) != n_before:
        logger.debug(
            "Cleaning: dropped {} samples with high special-char ratio",
            n_before - len(dataset),
        )

    # ── 3. Token length filter ─────────────────────────────────────────
    def _token_length_ok(example: dict) -> bool:
        ids = tokenizer(example["text"], return_length=True)
        length = ids["length"][0] if isinstance(ids["length"], list) else ids["length"]
        return min_tokens <= length <= max_tokens

    n_before = len(dataset)
    dataset = dataset.filter(_token_length_ok)
    if len(dataset) != n_before:
        logger.debug(
            "Cleaning: dropped {} samples outside token range [{}, {}]",
            n_before - len(dataset),
            min_tokens,
            max_tokens,
        )

    # ── 4. Language filter (optional) ──────────────────────────────────
    if languages:
        try:
            from langdetect import detect as detect_lang

            def _lang_ok(example: dict) -> bool:
                try:
                    return detect_lang(example["text"]) in languages
                except Exception:
                    return False

            n_before = len(dataset)
            dataset = dataset.filter(_lang_ok)
            if len(dataset) != n_before:
                logger.debug(
                    "Cleaning: dropped {} non-{} samples",
                    n_before - len(dataset),
                    ",".join(languages),
                )
        except ImportError:
            console.print(
                "[yellow]⚠ langdetect not installed — skipping language filter[/yellow]"
            )

    # ── 5. Deduplication ──────────────────────────────────────────────
    if dedup:
        seen: set[str] = set()

        def _not_dup(example: dict) -> bool:
            h = hashlib.sha256(example["text"].encode()).hexdigest()
            if h in seen:
                return False
            seen.add(h)
            return True

        n_before = len(dataset)
        dataset = dataset.filter(_not_dup)
        if len(dataset) != n_before:
            logger.debug(
                "Cleaning: dropped {} duplicate samples",
                n_before - len(dataset),
            )

    n_end = len(dataset)
    if n_start != n_end:
        console.print(
            f"[cyan]Cleaning: {n_start} → {n_end} samples "
            f"({n_start - n_end} removed)[/cyan]"
        )
    else:
        console.print("[cyan]Cleaning: no samples removed[/cyan]")

    return dataset
