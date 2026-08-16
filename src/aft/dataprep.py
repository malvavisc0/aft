"""Pre-tokenize structured chat rows into ``input_ids`` + ``labels``.

The local v22 chat template (``experiments/qwen3.5/chat_template.jinja``)
renders ``messages`` + ``tools=`` but, like most templates, carries no
``{% generation %}`` markers, so ``apply_chat_template`` cannot tell us which
tokens are model output. Labels are therefore built by **turn-diff**: the
conversation is rendered cumulatively up to and including each assistant
turn's ``<|im_end|>``, and the token delta of that turn becomes the label
span. Rendered prefixes are genuine char-prefixes under the template's
defaults (``preserve_thinking`` is never disabled here), so tokenizing each
grows the id list monotonically — no re-splitting is needed.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from aft.errors import AftError

#: Id used to mask out context (non-assistant) tokens in ``labels``.
IGNORE_INDEX: int = -100

#: Content value permitted for these roles.
_TEXT_ROLES: frozenset[str] = frozenset({"system", "user", "tool"})


def _coerce_content(value) -> str:
    """Flatten a message ``content`` field to a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if value is None:
        return ""
    return str(value)


def _coerce_arguments(arguments) -> dict[str, Any]:
    """Return ``tool_calls[*].function.arguments`` as a dict.

    The template iterates ``arguments.items()`` to render ``<parameter=>``
    tags (XML format), so a bare JSON string would render a headerless body.
    Accept dicts and parseable JSON strings; hard-fail on anything else.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise AftError(
                "tool_calls[*].function.arguments must be a dict or a valid"
                f" JSON string; got unparseable string: {arguments[:80]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AftError(
                "tool_calls[*].function.arguments must decode to a dict; got"
                f" JSON {type(parsed).__name__}."
            )
        return parsed
    raise AftError(
        "tool_calls[*].function.arguments must be a dict (or JSON string);"
        f" got {type(arguments).__name__}."
    )


def _normalize_tool_calls(tool_calls):
    """Canonicalize a message's ``tool_calls`` list."""
    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise AftError(f"tool_calls entry is not a dict: {call!r}")
        function = call.get("function")
        if not isinstance(function, dict):
            raise AftError(f"tool_call has no function dict: {call!r}")
        function = dict(function)
        if "arguments" in function:
            function["arguments"] = _coerce_arguments(function.pop("arguments"))
        normalized.append({**call, "function": function})
    return normalized


def _assistant_fields(message: dict[str, Any]) -> dict[str, Any]:
    """Role-specific fields for assistant messages (reasoning + tool calls)."""
    fields: dict[str, Any] = {}
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("thinking")
    if reasoning is not None:
        fields["reasoning_content"] = str(reasoning)
    if message.get("tool_calls"):
        fields["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    return fields


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw message into the canonical schema used for training.

    - content (string or list of blocks) → string;
    - ``tool_calls[*].function.arguments`` → dict (hard-fail on bad string);
    - ``thinking`` → ``reasoning_content``;
    - drops ``tool_call_id`` for non-tool roles (harmless already).
    """
    if not isinstance(message, dict):
        raise AftError(f"Message is not a dict: {message!r}")
    role = message.get("role")
    if role not in _TEXT_ROLES and role != "assistant":
        raise AftError(f"Unrecognized message role: {role!r}")
    normalized: dict[str, Any] = {"role": role}
    if "content" in message:
        normalized["content"] = _coerce_content(message.get("content"))
    if role == "assistant":
        normalized.update(_assistant_fields(message))
    if role == "tool" and message.get("tool_call_id") is not None:
        normalized["tool_call_id"] = str(message["tool_call_id"])
    return normalized


def _validate_tools(tools) -> list[dict[str, Any]]:
    """Validate a row's ``tools`` list (passed to the template via ``tools=``)."""
    if tools is None:
        return []
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise AftError(
            "tools must be a list of tool-definition dicts;"
            f" got {type(tools).__name__}."
        )
    return tools


def _loads_if_str(value: Any) -> Any:
    """Parse a JSON-encoded column value; pass through anything else."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def stack_messages(dataset) -> Any:
    """Return a dataset of normalized rows as JSON strings.

    Columns ``messages`` and ``tools`` are stored **JSON-encoded**: message
    dicts are heterogeneous (assistant rows may carry ``tool_calls`` /
    ``reasoning_content``, tool rows ``tool_call_id``), and heterogeneous
    structs do not fit an Arrow column. Tool definitions travel in ``tools``
    — never inside a system message — because the template only renders the
    <tools> block and its format instructions when they arrive via the
    ``tools=`` kwarg. Source columns may themselves be lists or JSON strings
    (both shapes occur in the wild).
    """
    from datasets import Dataset

    column = "messages" if "messages" in dataset.column_names else "conversations"
    has_tools = "tools" in dataset.column_names
    messages = [
        list(map(normalize_message, _loads_if_str(row))) for row in dataset[column]
    ]
    tools = (
        [_validate_tools(_loads_if_str(row)) for row in dataset["tools"]]
        if has_tools
        else [[] for _ in messages]
    )
    return Dataset.from_dict(
        {
            "messages": [json.dumps(m) for m in messages],
            "tools": [json.dumps(t) for t in tools],
        }
    )


def _assistant_spans(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools,
    chat_template: str,
    enable_thinking: bool,
    reasoning_effort: str,
    tool_call_format: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(full_ids, spans)`` via cumulative turn-diff.

    Renders each growing prefix with the chat template, records its token
    length, and derives each assistant turn's span as the delta between the
    render that includes it and the one that does not.
    """
    if not messages:
        return [], []
    rendered = ""
    boundaries: list[int] = [0]
    for i in range(1, len(messages) + 1):
        rendered = tokenizer.apply_chat_template(
            messages[:i],
            tools=tools,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            tool_call_format=tool_call_format,
            add_generation_prompt=False,
            tokenize=False,
        )
        ids = tokenizer(rendered)["input_ids"]
        boundaries.append(len(ids))
    full_ids = list(tokenizer(rendered)["input_ids"])
    spans: list[tuple[int, int]] = []
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    for idx in assistant_idx:
        spans.append((boundaries[idx], boundaries[idx + 1]))
    return full_ids, spans


def tokenize_with_template(
    tokenizer,
    messages: list[dict[str, Any]],
    tools=None,
    *,
    max_seq_len: int,
    mask_strategy: str,
    chat_template: str,
    enable_thinking: bool = True,
    reasoning_effort: str = "xhigh",
    tool_call_format: str = "xml",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize one conversation into ``(input_ids, labels, attention_mask)``.

    ``mask_strategy``:
    - ``"full"``: every assistant span (reasoning + content + tool_calls) is a
      label; system/user/tool context is masked.
    - ``"cumulative"``: only the final assistant span is a label.
    """
    if mask_strategy not in ("full", "cumulative"):
        raise AftError(f"Unknown mask_strategy: {mask_strategy!r}")
    normalized = [normalize_message(m) for m in messages]
    full_ids, spans = _assistant_spans(
        tokenizer,
        normalized,
        tools=tools,
        chat_template=chat_template,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        tool_call_format=tool_call_format,
    )
    seq = min(max_seq_len, len(full_ids))
    full_ids = full_ids[:seq]
    labels = torch.full((seq,), IGNORE_INDEX, dtype=torch.long)
    for start, end in spans:
        start = max(0, start)
        end = min(seq, end)
        if end <= start:
            continue
        if mask_strategy == "cumulative" and (start, end) != spans[-1]:
            continue
        labels[start:end] = torch.tensor(full_ids[start:end], dtype=torch.long)
    input_ids = torch.full((max_seq_len,), 0, dtype=torch.long)
    input_ids[:seq] = torch.tensor(full_ids, dtype=torch.long)
    attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
    attention_mask[:seq] = 1
    labels_pad = torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long)
    labels_pad[:seq] = labels
    return input_ids, labels_pad, attention_mask


def tokenize_dataset(
    dataset,
    tokenizer,
    *,
    max_seq_len: int,
    mask_strategy: str,
    chat_template: str,
    enable_thinking: bool = True,
    reasoning_effort: str = "xhigh",
    tool_call_format: str = "xml",
):
    """Pre-tokenize every row of a messages dataset.

    Expects ``messages`` and ``tools`` columns of JSON strings (as produced
    by :func:`stack_messages`); each row's ``tools`` list is passed to the
    template via the ``tools=`` kwarg. Returns a HF ``Dataset`` with
    ``input_ids``/``labels``/``attention_mask`` columns, ready for a trl
    ``SFTTrainer`` (no ``dataset_text_field``).
    """
    from datasets import Dataset

    rows = {"input_ids": [], "labels": [], "attention_mask": []}
    for raw_msgs, raw_tools in zip(dataset["messages"], dataset["tools"], strict=True):
        inp, lab, attn = tokenize_with_template(
            tokenizer,
            json.loads(raw_msgs),
            tools=json.loads(raw_tools),
            max_seq_len=max_seq_len,
            mask_strategy=mask_strategy,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            tool_call_format=tool_call_format,
        )
        rows["input_ids"].append(inp)
        rows["labels"].append(lab)
        rows["attention_mask"].append(attn)
    return Dataset.from_dict(rows)
