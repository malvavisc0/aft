"""Dataset cleaning utilities for fine-tuning."""

import hashlib
import re

from loguru import logger

from aft.errors import AftError
from aft.ui import console

# ── Dataset row normalization ──────────────────────────────────────────────

#: Flat text field names tried in order when no preferred field is given.
#: Different corpora expose their content under different keys (web text →
#: ``text``, code corpora → ``content``), and assuming a single key silently
#: yields empty data for half the datasets out there.
_FLAT_TEXT_FIELDS: tuple[str, ...] = ("text", "content", "code", "body")

#: Message-list field names: a list of single-turn dicts describing a
#: conversation.  ``flatten_row_to_text`` joins these into one string so aft
#: can train/calibrate on chat datasets (Nemotron-Agentic, ShareGPT,
#: OpenAI-format) without a per-dataset loader.
_MESSAGE_LIST_FIELDS: tuple[str, ...] = ("messages", "conversations")

#: ``(role_key, content_key)`` pairs for the message-dict shapes in the wild.
#: OpenAI/HF chat uses ``role``/``content``; ShareGPT uses ``from``/``value``.
_MESSAGE_ROLE_CONTENT_KEYS: tuple[tuple[str, str], ...] = (
    ("role", "content"),
    ("from", "value"),
    ("sender", "text"),
)


def _block_text(block) -> str:
    """Text from a single content block (dict or bare string)."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("text") or block.get("content") or ""
    return ""


def _message_text(content) -> str:
    """Extract text from a message ``content`` field.

    ``content`` may be a plain string or a list of content blocks (OpenAI
    multi-part, e.g. ``[{"type": "text", "text": "..."}]``).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b for b in (_block_text(x) for x in content) if b)
    return str(content) if content is not None else ""


def _flatten_messages(msgs) -> str | None:
    """Join a list of message dicts into a single ``"role: text"`` string."""
    if not isinstance(msgs, list) or not msgs:
        return None
    parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        for role_key, content_key in _MESSAGE_ROLE_CONTENT_KEYS:
            if role_key in m and content_key in m:
                text = _message_text(m[content_key])
                if text:
                    parts.append(f"{m[role_key]}: {text}")
                break
    return "\n".join(parts) if parts else None


def flatten_row_to_text(row: dict, preferred_field: str | None = None) -> str | None:
    """Normalize a dataset row into a single text string, or ``None``.

    Handles flat text columns and conversational message-list columns so the
    same loader works for web text, code, and chat formats.  A preferred
    field is honored first (used by the calibration registry); then flat
    text fields; then message-list fields.
    """
    if preferred_field:
        val = row.get(preferred_field)
        if isinstance(val, str):
            return val
        text = _flatten_messages(val)
        if text is not None:
            return text

    for field in _FLAT_TEXT_FIELDS:
        val = row.get(field)
        if isinstance(val, str):
            return val

    for field in _MESSAGE_LIST_FIELDS:
        text = _flatten_messages(row.get(field))
        if text is not None:
            return text

    return None


def supported_text_columns() -> str:
    """Human-readable list of columns flatten_row_to_text understands."""
    flat = ", ".join(_FLAT_TEXT_FIELDS)
    msgs = ", ".join(_MESSAGE_LIST_FIELDS)
    return f"flat text ({flat}) or message-list ({msgs})"


def parse_dataset_spec(dataset_id: str) -> tuple[str, str]:
    """Split a ``repo_id:split`` spec into ``(repo_id, split)``.

    HuggingFace repo ids never contain colons, so this is unambiguous.  Lets
    users pick a named subset, e.g. ``nvidia/Nemotron-Agentic-v1:interactive_agent``.
    Without a colon the split defaults to ``"train"`` (resolved later by
    :func:`resolve_dataset_split`).
    """
    if ":" in dataset_id:
        repo, split = dataset_id.rsplit(":", 1)
        if repo and split:
            return repo, split
    return dataset_id, "train"


def resolve_dataset_split(
    repo_id: str, preferred: str = "train", *, token: str | None = None
) -> str:
    """Pick a usable split, falling back when ``preferred`` is absent.

    Some datasets (e.g. ``nvidia/Nemotron-Agentic-v1``) expose named subsets
    (``interactive_agent``, ``tool_calling``) instead of a ``train`` split.
    Hard-coding ``split="train"`` silently breaks on those, so we query the
    available splits and fall back to the first one with a notice.
    """
    import datasets as hf_datasets

    try:
        splits = hf_datasets.get_dataset_split_names(repo_id, token=token)
    except Exception:
        # Can't enumerate (offline, gated, network) — let load_dataset raise
        # the real, actionable error rather than guessing here.
        return preferred
    if preferred in splits:
        return preferred
    if splits:
        logger.info(
            "Split '{}' not found in {}; using '{}'", preferred, repo_id, splits[0]
        )
        return splits[0]
    return preferred


# ── Cleaning pipeline ──────────────────────────────────────────────────────


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
        # ``ids["length"]`` may be a list, numpy array, or tensor depending
        # on the tokenizer backend. Coerce via int() to handle all cases.
        length = int(ids["length"][0])
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
            from langdetect.lang_detect_exception import (
                LangDetectException,
            )
        except ImportError:
            raise AftError(
                "Language filtering requested (--languages) but"
                " `langdetect` is not installed.\n"
                "  Install it with: pip install langdetect\n"
                "  Or remove --languages to skip language filtering."
            ) from None

        def _lang_ok(example: dict) -> bool:
            try:
                return detect_lang(example["text"]) in languages
            except LangDetectException:
                # "Can't detect language" — drop the sample.
                return False

        n_before = len(dataset)
        dataset = dataset.filter(_lang_ok)
        if len(dataset) != n_before:
            logger.debug(
                "Cleaning: dropped {} non-{} samples",
                n_before - len(dataset),
                ",".join(languages),
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
