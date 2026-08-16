"""Unit tests for aft.dataprep — structured messages tokenization + masking."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from aft.dataprep import IGNORE_INDEX, normalize_message, tokenize_with_template
from aft.errors import AftError

TEMPLATE = open("experiments/qwen3.5/chat_template.jinja").read()


class _FakeTokenizer:
    """Tokenizes by character but renders via the real v22 template path.

    ``apply_chat_template``/``__call__`` mirror the production surface used
    by :mod:`aft.dataprep` (which itself is tokenizer-agnostic), so the
    turn-diff masking logic is exercised against the real template.
    """

    chat_template = TEMPLATE

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        chat_template=None,
        enable_thinking=True,
        reasoning_effort="xhigh",
        tool_call_format="xml",
        add_generation_prompt=False,
        tokenize=False,
    ) -> Any:
        from transformers.utils.chat_template_utils import render_jinja_template

        text = render_jinja_template(
            [messages],
            tools=tools,
            chat_template=chat_template or self.chat_template,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            tool_call_format=tool_call_format,
            add_generation_prompt=add_generation_prompt,
        )[0][0]
        if tokenize:
            return {"input_ids": [ord(c) for c in text], "attention_mask": None}
        return text


TOK = _FakeTokenizer()


MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Fix the test"},
    {
        "role": "assistant",
        "reasoning_content": "think first",
        "content": "using tool",
        "tool_calls": [
            {"id": "c1", "function": {"name": "read", "arguments": {"path": "a.py"}}}
        ],
    },
    {"role": "tool", "content": "contents here", "tool_call_id": "c1"},
    {"role": "assistant", "reasoning_content": "done", "content": "Fixed."},
]


class TestNormalizeMessage:
    def test_coerces_content_blocks_to_string(self) -> None:
        msg = normalize_message(
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )
        assert msg["content"] == "hi"

    def test_parses_json_string_arguments_to_dict(self) -> None:
        msg = normalize_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}
                ],
            }
        )
        assert msg["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_hard_fails_on_invalid_arguments_string(self) -> None:
        with pytest.raises(AftError, match="arguments"):
            normalize_message(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "f", "arguments": "nope"}}
                    ],
                }
            )

    def test_maps_thinking_field_to_reasoning_content(self) -> None:
        msg = normalize_message(
            {"role": "assistant", "thinking": "think!", "content": "answer"}
        )
        assert msg["reasoning_content"] == "think!"
        assert "thinking" not in msg


def _render_text(messages: list[dict[str, Any]]) -> str:
    text = TOK.apply_chat_template(messages, chat_template=TEMPLATE)
    assert isinstance(text, str)
    return text


def _label_string(labels: torch.Tensor, ids: torch.Tensor, size: int) -> str:
    parts: list[str] = []
    for lab, i in zip(labels[:size].tolist(), ids[:size].tolist()):
        parts.append(chr(i) if lab != IGNORE_INDEX else "_")
    return "".join(parts)


class TestTokenizeWithTemplate:
    def test_full_masks_all_assistant_spans(self) -> None:
        ids, labels, attn = tokenize_with_template(
            TOK,
            MESSAGES,
            max_seq_len=4096,
            mask_strategy="full",
            chat_template=TEMPLATE,
        )
        # labeled tokens are exactly the input ids at those positions
        assert (
            labels[labels != IGNORE_INDEX].tolist()
            == ids[labels != IGNORE_INDEX].tolist()
        )
        # context (system/user/tool) is masked out
        assert (labels == IGNORE_INDEX).any()

    def test_full_labels_reasoning_and_tool_calls(self) -> None:
        ids, labels, _ = tokenize_with_template(
            TOK,
            MESSAGES,
            max_seq_len=4096,
            mask_strategy="full",
            chat_template=TEMPLATE,
        )
        labeled = _label_string(labels, ids, len(labels))
        # both assistant turns' thinking + content + tool_call are labeled
        assert "think first" in labeled
        assert "<function=read>" in labeled
        assert "Fixed." in labeled

    def test_cumulative_labels_only_final_assistant(self) -> None:
        ids, labels, _ = tokenize_with_template(
            TOK,
            MESSAGES,
            max_seq_len=4096,
            mask_strategy="cumulative",
            chat_template=TEMPLATE,
        )
        labeled = _label_string(labels, ids, len(labels))
        assert labeled.count("_") > 0
        assert "Fixed." in labeled
        # earlier assistant turn (reasoning + tool call) is masked, not labeled
        assert "think first" not in labeled

    def test_truncates_to_max_seq_len(self) -> None:
        full_len = len(TOK(_render_text(MESSAGES))["input_ids"])
        small = max(1, full_len // 2)
        ids, labels, attn = tokenize_with_template(
            TOK,
            MESSAGES,
            max_seq_len=small,
            mask_strategy="full",
            chat_template=TEMPLATE,
        )
        assert len(ids) == small
        assert len(attn) == small
        assert attn.sum().item() == small

    def test_rejects_unknown_mask_strategy(self) -> None:
        with pytest.raises(AftError, match="mask_strategy"):
            tokenize_with_template(
                TOK,
                MESSAGES,
                max_seq_len=128,
                mask_strategy="nope",
                chat_template=TEMPLATE,
            )


class TestStacking:
    def test_stack_retains_normalized_messages_column(self) -> None:
        import json

        from datasets import Dataset

        from aft.dataprep import stack_messages

        ds = Dataset.from_dict({"messages": [json.dumps(MESSAGES)]})
        out = stack_messages(ds)
        assert "messages" in out.column_names
        assert json.loads(out[0]["messages"])[0]["content"] == "You are an agent."
        assert json.loads(out[0]["tools"]) == []

    def test_stack_retains_and_validates_tools_column(self) -> None:
        import json

        from datasets import Dataset

        from aft.dataprep import stack_messages

        tool_def = {"type": "function", "function": {"name": "read"}}
        ds = Dataset.from_dict(
            {
                "messages": [json.dumps(MESSAGES)],
                "tools": [json.dumps([tool_def])],
            }
        )
        out = stack_messages(ds)
        assert json.loads(out[0]["tools"]) == [tool_def]

        bad = Dataset.from_dict(
            {"messages": [json.dumps(MESSAGES)], "tools": [json.dumps(["oops"])]}
        )
        with pytest.raises(AftError, match="tools"):
            stack_messages(bad)


class TestTokenizeDataset:
    def test_per_row_tools_reach_the_template(self) -> None:
        import json

        from datasets import Dataset

        from aft.dataprep import tokenize_dataset

        tool_def = {
            "type": "function",
            "function": {"name": "read", "description": "Read a file"},
        }
        ds = Dataset.from_dict(
            {
                "messages": [json.dumps(MESSAGES)],
                "tools": [json.dumps([tool_def])],
            }
        )
        out = tokenize_dataset(
            ds, TOK, max_seq_len=4096, mask_strategy="full", chat_template=TEMPLATE
        )
        text = "".join(chr(i) for i in out[0]["input_ids"] if i != 0)
        assert "# Tools" in text
        assert '"name": "read"' in text
        assert "<IMPORTANT>" in text

    def test_empty_tools_column_renders_no_tools_block(self) -> None:
        import json

        from datasets import Dataset

        from aft.dataprep import tokenize_dataset

        ds = Dataset.from_dict(
            {"messages": [json.dumps(MESSAGES)], "tools": [json.dumps([])]}
        )
        out = tokenize_dataset(
            ds, TOK, max_seq_len=4096, mask_strategy="full", chat_template=TEMPLATE
        )
        text = "".join(chr(i) for i in out[0]["input_ids"] if i != 0)
        assert "# Tools" not in text
