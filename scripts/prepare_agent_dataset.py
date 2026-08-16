#!/usr/bin/env python
"""Prepare an agentic tool-use SFT dataset from four public sources.

Converts raw datasets into a canonical chat schema and publishes train/eval
splits. Each output row has two JSON-string columns:

    messages  list of {"role": "system"|"user"|"assistant"|"tool", ...}
              where assistant messages may carry "reasoning_content" and
              "tool_calls" (arguments always as dicts), and tool messages
              carry "tool_call_id"
    tools     list of OpenAI function-schema dicts, [] when none

Tool definitions live in the ``tools`` column, never inside system content,
because chat templates only render tool blocks and their call-format
instructions when definitions arrive via the ``tools=`` kwarg.

Sources (see the per-source converters for format details):
    A  greghavens/fable-5-coding-and-debugging-traces   cumulative coding-agent
                                                       traces (default cap 300)
    B  PolarSeeker/OpenSeeker-v1-Data                   search-agent traces with
                                                       inline markup (cap 100)
    C  saidutta69/fable-5-premium (openai_chat)         cleaned agent traces
                                                       (cap 200)
    D  KNipun/ai-humanizer                              flat prompt/completion
                                                       rewrites (cap 100)

Pipeline:
    1. Convert each source to canonical rows.
    2. Drop rows containing CJK text — ideographs (Chinese/Mandarin, kanji),
       Japanese kana, Korean hangul (OpenSeeker has a large Mandarin subset;
       the filter runs on every source as a safety net).
    3. Validate every row (hard-fail on schema violations).
    4. Dedup, cross-source and intra-source (two SHA-256 keys, see dedup()).
    5. Seeded per-source cap draw.
    6. Drop rows over the token cap, measured by rendering with the chat
       template and tokenizing.
    7. Trajectory-disjoint train/eval split; push two private HF repos, or
       write local JSONL with --no-push.

Usage:
    HF_TOKEN=hf_... uv run python scripts/prepare_agent_dataset.py
    uv run python scripts/prepare_agent_dataset.py --no-push --output-dir data/smoke
    uv run python scripts/prepare_agent_dataset.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT, "experiments", "qwen3.5", "chat_template.jinja")

TOKENIZER_ID = "unsloth/Qwen3.5-9B"
MAX_TOKENS = 8192
SPLIT = 0.95
SEED = 20260813
NAMESPACE = os.environ.get("HF_NAMESPACE", "")  # required for --push
TRAIN_REPO = "qwen3.5-9b-agent-smoke-train"
EVAL_REPO = "qwen3.5-9b-agent-smoke-eval"

SOURCE_CAPS = {"A": 300, "B": 100, "C": 200, "D": 100}

# CJK scripts: ideographs (Chinese/Mandarin, Japanese kanji), Japanese kana,
# Korean hangul. Any row containing these is dropped.
_CJK_RE = re.compile(
    "["
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # Compatibility Ideographs
    "\U00020000-\U0002a6df"  # CJK Extension B
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "가-힯"  # Hangul syllables
    "ᄀ-ᄿ"  # Hangul Jamo
    "]+"
)

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class BadRow(ValueError):
    """A row violates the canonical schema (hard-fail)."""


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _to_str(content: Any) -> str:
    """Coerce a message ``content`` to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _row_has_cjk(messages: Sequence[dict[str, Any]]) -> bool:
    """True if any message content/reasoning/tool arg contains CJK text."""
    for msg in messages:
        fields = [_to_str(msg.get("content"))]
        for key in ("reasoning_content", "thinking"):
            if msg.get(key):
                fields.append(str(msg[key]))
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                fields.append(str(fn.get("name", "")))
                fields.append(_to_str(fn.get("arguments", "")))
        if any(_CJK_RE.search(f) for f in fields):
            return True
    return False


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #


def _parse_lenient_json(text: str) -> Any:
    """Parse JSON, repairing lone backslashes (invalid escapes) if needed.

    Teacher-generated argument strings sometimes contain unescaped
    backslashes (regex ``\\d``, Windows paths). Valid JSON escapes are
    preserved; anything else becomes a literal backslash.
    """
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?![\\/bfnrtu"])', r"\\\\", text)
        return json.loads(repaired, strict=False)


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Return ``tool_calls[*].function.arguments`` as a dict (hard-fail)."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = _parse_lenient_json(arguments)
        except json.JSONDecodeError as exc:
            raise BadRow(
                f"arguments is an unparseable JSON string: {arguments[:80]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise BadRow(
                "tool_calls[].function.arguments must decode to a dict; got"
                f" {type(parsed).__name__}."
            )
        return parsed
    raise BadRow(
        "tool_calls[].function.arguments must be a dict or JSON string; got"
        f" {type(arguments).__name__}."
    )


def _canonicalize_tool_call(tc: Any, msg_index: int) -> dict[str, Any]:
    """Coerce one tool call; ``arguments`` becomes a dict."""
    if not isinstance(tc, dict):
        raise BadRow(f"message {msg_index}: tool call is not a dict: {tc!r}")
    fn = tc.get("function")
    if not isinstance(fn, dict):
        fn = {"name": tc.get("name"), "arguments": tc.get("arguments")}
    return {
        "id": str(tc["id"]) if tc.get("id") is not None else f"call_{msg_index}",
        "type": str(tc.get("type") or "function"),
        "function": {
            "name": str(fn.get("name", "")),
            "arguments": _coerce_arguments(fn.get("arguments")),
        },
    }


def _canonicalize_message(msg: Any, index: int) -> dict[str, Any]:
    """Coerce a raw message into the canonical dict shape."""
    if not isinstance(msg, dict):
        raise BadRow(f"message {index} is not a dict: {msg!r}")
    canonical: dict[str, Any] = {
        "role": msg.get("role"),
        "content": _to_str(msg.get("content")),
    }
    if msg.get("role") == "assistant":
        reasoning = msg.get("reasoning_content")
        if reasoning is None:
            reasoning = msg.get("thinking")
        if reasoning is not None and str(reasoning).strip():
            canonical["reasoning_content"] = str(reasoning)
        if msg.get("tool_calls"):
            canonical["tool_calls"] = [
                _canonicalize_tool_call(tc, index) for tc in msg["tool_calls"]
            ]
    elif msg.get("role") == "tool" and msg.get("tool_call_id"):
        canonical["tool_call_id"] = str(msg["tool_call_id"])
    return canonical


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_role_sequence(messages: list[dict[str, Any]]) -> None:
    """Roles known; every tool message answered by an issued call; args dicts."""
    issued: set[str] = set()
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise BadRow(f"message {i}: unknown role {role!r}")
        if role == "assistant":
            if not isinstance(msg.get("content"), str):
                raise BadRow(f"message {i}: assistant content must be a string")
            for j, tc in enumerate(msg.get("tool_calls", [])):
                fn = tc.get("function")
                if not isinstance(fn, dict) or not isinstance(
                    fn.get("arguments"), dict
                ):
                    raise BadRow(f"message {i} call {j}: arguments must be a dict.")
                issued.add(tc["id"])
        if role == "tool":
            call_id = msg.get("tool_call_id")
            if not call_id or call_id not in issued:
                raise BadRow(
                    f"message {i}: tool message references unknown tool_call_id"
                    f" {call_id!r} (orphan or forwards-reference)."
                )


def _validate_final_message(messages: list[dict[str, Any]]) -> None:
    """The last message must be an assistant turn with content or a tool call."""
    if not messages:
        raise BadRow("empty message list")
    last = messages[-1]
    if last.get("role") != "assistant":
        raise BadRow(
            f"final message must be role 'assistant'; got {last.get('role')!r}."
        )
    if not last.get("content", "").strip() and not last.get("tool_calls"):
        raise BadRow("final assistant message must have content or tool_calls")


def validate_row(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate one canonical row; hard-fail on violation."""
    if not isinstance(messages, list):
        raise BadRow("messages must be a list")
    _validate_role_sequence(messages)
    _validate_final_message(messages)
    return messages


# --------------------------------------------------------------------------- #
# Tokenization / hashing
# --------------------------------------------------------------------------- #


def render_row(tokenizer: Any, template: str, messages: Any, tools: Any) -> str:
    """Render a row with the chat template using the training render config."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        chat_template=template,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
        tool_call_format="xml",
        add_generation_prompt=False,
    )


def count_tokens(tokenizer: Any, template: str, messages: Any, tools: Any) -> int:
    """Number of tokens a row occupies when rendered with the chat template."""
    rendered = render_row(tokenizer, template, messages, tools)
    return len(tokenizer(rendered)["input_ids"])


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_sha(messages: Sequence[dict[str, Any]]) -> str:
    """SHA-256 over the canonical messages list (exact-duplicate key)."""
    return _sha(json.dumps(list(messages), sort_keys=True, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Source A: greghavens/fable-5-coding-and-debugging-traces
# --------------------------------------------------------------------------- #


def _convert_a(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cumulative rows into full trajectories; args string→dict.

    Each trajectory appears several times with growing prefixes; keep the
    longest step per ``source_trajectory_id`` (it contains the whole prefix
    by construction). ``seed-authoring`` trajectories are dropped: they are
    meta data-authoring work, not coding-agent behavior.
    """
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_id[str(r["source_trajectory_id"])].append(r)

    out: list[dict[str, Any]] = []
    dropped = 0
    for tid, group in by_id.items():
        best = max(group, key=lambda r: len(r["messages"]))
        if best.get("category") == "seed-authoring":
            continue
        try:
            messages = [
                _canonicalize_message(m, i) for i, m in enumerate(best["messages"])
            ]
        except BadRow:
            dropped += 1
            continue
        out.append({"messages": messages, "tools": [], "trajectory_id": tid})
    if dropped:
        print(f"[A] dropped {dropped} trajectories that failed canonicalization")
    return out


# --------------------------------------------------------------------------- #
# Source B: PolarSeeker/OpenSeeker-v1-Data
# --------------------------------------------------------------------------- #

_B_THINK_RE = re.compile(r"<think>\n?(.*?)\n?</think>\n*", re.DOTALL)
_B_CALLS_BLOCK_RE = re.compile(
    r"<tool_calls_begin>\n?(.*?)\n?</tool_calls_end>", re.DOTALL
)
_B_TOOL_CALL_RE = re.compile(r"<tool_call>\n?(.*?)\n?</tool_call>", re.DOTALL)
_B_TOOLS_BLOCK_RE = re.compile(r"<tools>\n?(.*?)\n?</tools>", re.DOTALL)


def _parse_json_objects(block: str) -> list[dict[str, Any]]:
    """Parse consecutive JSON objects from a text block (newline-agnostic)."""
    decoder = json.JSONDecoder()
    tools: list[dict[str, Any]] = []
    idx = 0
    while idx < len(block):
        while idx < len(block) and block[idx] != "{":
            idx += 1
        if idx >= len(block):
            break
        try:
            obj, end = decoder.raw_decode(block, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            tools.append(obj)
        idx = end
    return tools


def _extract_b_tool_defs(system_text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool defs from a B system message's real ``<tools>`` block.

    The system prompt *documents* the format with an empty ``<tools></tools>``
    mention before the actual block; pick the first block that yields at
    least one JSON object. Returns ``(tools, text_without_block)``.
    """
    for match in _B_TOOLS_BLOCK_RE.finditer(system_text):
        tools = _parse_json_objects(match.group(1))
        if tools:
            stripped = system_text[: match.start()] + system_text[match.end() :]
            return tools, re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return [], system_text


def _split_b_assistant(text: str) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    """Split a B assistant turn into (reasoning, content, parsed tool calls).

    B format: ``<think>…</think>`` followed by an optional
    ``<tool_calls_begin><tool_call>{JSON}</tool_call></tool_calls_end>``
    block. Both wrappers are stripped from the content.
    """
    reasoning = ""
    think = _B_THINK_RE.search(text)
    if think:
        reasoning = think.group(1).strip()
        text = text[: think.start()] + text[think.end() :]
    calls: list[tuple[str, dict[str, Any]]] = []
    block = _B_CALLS_BLOCK_RE.search(text)
    if block:
        for payload in _B_TOOL_CALL_RE.findall(block.group(1)):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                args = _coerce_arguments(parsed.get("arguments", {}))
                calls.append((str(parsed["name"]), args))
        text = text[: block.start()] + text[block.end() :]
    return reasoning, text.strip(), calls


def _strip_tool_response(text: str) -> str:
    """Remove the literal <tool_response>…</tool_response> wrapper from B."""
    text = text.strip()
    if text.startswith("<tool_response>"):
        text = text[len("<tool_response>") :]
        if text.endswith("</tool_response>"):
            text = text[: -len("</tool_response>")]
    return text.strip()


def _convert_b(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse thinking/tools/tool-responses; drop rows with 0 tool calls."""
    out: list[dict[str, Any]] = []
    dropped = 0
    for idx, r in enumerate(rows):
        try:
            row = _convert_b_row(r, idx)
        except BadRow:
            dropped += 1
            continue
        if row is not None:
            out.append(row)
    if dropped:
        print(f"[B] dropped {dropped} rows that failed canonicalization")
    return out


def _convert_b_row(r: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Convert one B trajectory; ``None`` when it has 0 tool calls."""
    counter = itertools.count(1)
    pending: list[str] = []
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    n_calls = 0
    for m in r["trajectory"]:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "system":
            tools, stripped = _extract_b_tool_defs(content)
            messages.append({"role": "system", "content": stripped})
        elif role == "user":
            if "<tool_response>" in content:
                call_id = pending.pop(0) if pending else f"call_{next(counter)}"
                messages.append(
                    {
                        "role": "tool",
                        "content": _strip_tool_response(content),
                        "tool_call_id": call_id,
                    }
                )
            else:
                messages.append({"role": "user", "content": content})
        elif role == "assistant":
            reasoning, text, calls = _split_b_assistant(content)
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = []
                for name, args in calls:
                    call_id = f"call_{next(counter)}"
                    pending.append(call_id)
                    msg["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    )
                n_calls += len(calls)
            messages.append(msg)
    if n_calls == 0:
        return None
    return {
        "messages": messages,
        "tools": tools,
        "trajectory_id": f"openseeker-{idx}",
    }


# --------------------------------------------------------------------------- #
# Source C: saidutta69/fable-5-premium (openai_chat view)
# --------------------------------------------------------------------------- #


def _convert_c(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse the JSON-string messages (quality filter happens in the loader)."""
    out: list[dict[str, Any]] = []
    dropped = 0
    for i, r in enumerate(rows):
        try:
            messages = [
                _canonicalize_message(m, j)
                for j, m in enumerate(json.loads(r["messages"]))
            ]
        except BadRow, json.JSONDecodeError:
            dropped += 1
            continue
        out.append(
            {
                "messages": messages,
                "tools": [],
                "trajectory_id": r.get("source_row_hash") or f"fable5-{i}",
            }
        )
    if dropped:
        print(f"[C] dropped {dropped} rows that failed canonicalization")
    return out


# --------------------------------------------------------------------------- #
# Source D: KNipun/ai-humanizer
# --------------------------------------------------------------------------- #

_MIN_COMPLETION_CHARS = 30


def _convert_d(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map prompt/completion → single-turn messages; drop degenerate rows."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        prompt, completion = str(r["prompt"]).strip(), str(r["completion"]).strip()
        if not prompt or len(completion) < _MIN_COMPLETION_CHARS:
            continue
        out.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "tools": [],
                "trajectory_id": f"humanizer-{i}",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def _rows(ds: Any) -> list[dict[str, Any]]:
    """Materialize a HF dataset as plain dicts."""
    return [dict(r) for r in ds]


def _load_a() -> list[dict[str, Any]]:
    from datasets import load_dataset as hf_load

    rows = _rows(
        hf_load("greghavens/fable-5-coding-and-debugging-traces", split="train")
    )
    print(f"[A] loaded {len(rows)} rows from fable-5")
    return _convert_a(rows)


# B search results are ~2-4k tokens each; trajectories with more calls than
# this cannot fit the token cap, so they are excluded at load time.
_B_MAX_TOOL_CALLS = 8


def _load_b() -> list[dict[str, Any]]:
    from datasets import load_dataset as hf_load

    ds = hf_load("PolarSeeker/OpenSeeker-v1-Data", split="train")
    correct = [
        r
        for r in _rows(ds)
        if r["trajectory correctness"] == "Correct"
        and 0 < r["number of tool calls"] <= _B_MAX_TOOL_CALLS
    ]
    print(
        f"[B] loaded {len(correct)} Correct rows (<= {_B_MAX_TOOL_CALLS} tool calls)"
        " from OpenSeeker"
    )
    return _convert_b(correct)


def _load_c() -> list[dict[str, Any]]:
    from datasets import load_dataset as hf_load

    ds = hf_load(
        "saidutta69/fable-5-premium",
        data_files={"train": "openai_chat/train.parquet"},
        split="train",
    )
    rows = []
    for r in _rows(ds):
        try:
            q = json.loads(r["quality_scores"]).get("overall", 0)
        except json.JSONDecodeError, AttributeError:
            q = 0
        if q >= 0.85:
            rows.append(r)
    print(f"[C] loaded {len(rows)} rows with quality>=0.85 from fable-5-premium")
    return _convert_c(rows)


def _load_d() -> list[dict[str, Any]]:
    from datasets import load_dataset as hf_load

    rows = _rows(hf_load("KNipun/ai-humanizer", split="train"))
    print(f"[D] loaded {len(rows)} rows from ai-humanizer")
    return _convert_d(rows)


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #


def dedup(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop exact duplicate trajectories (full-messages SHA-256).

    Returns ``(kept, dropped)``. Task-level (fuzzy) dedup is deliberately
    not done: C's trajectories share long boilerplate preambles as their
    first user message, so any first-message key collapses thousands of
    distinct trajectories (measured: 3895/4638 false positives).
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for r in rows:
        digest = row_sha(r["messages"])
        if digest in seen:
            dropped.append(r)
            continue
        seen.add(digest)
        kept.append(r)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


def split_by_trajectory(
    rows: Sequence[dict[str, Any]], rng: random.Random, split: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trajectory-disjoint split over distinct trajectory ids."""
    by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_traj[r["trajectory_id"]].append(r)
    traj_ids = list(by_traj)
    rng.shuffle(traj_ids)
    boundary = int(math.floor(len(traj_ids) * split))
    train_ids = set(traj_ids[:boundary])
    train = [r for r in rows if r["trajectory_id"] in train_ids]
    eval_rows = [r for r in rows if r["trajectory_id"] not in train_ids]
    return train, eval_rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class SourceStats:
    """Per-source counters through the pipeline stages."""

    name: str
    loaded: int = 0
    validated: int = 0
    lang_filtered: int = 0
    dedup_dropped: int = 0
    length_dropped: int = 0
    kept: int = 0
    token_counts: list[int] = field(default_factory=list)
    tool_calls: int = 0


def _histogram(values: Sequence[int], buckets: Sequence[int]) -> str:
    counts = [0] * (len(buckets) + 1)
    for v in values:
        for i, b in enumerate(buckets):
            if v <= b:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    parts = [f"≤{b}:{c}" for b, c in zip(buckets, counts[:-1])]
    parts.append(f">{buckets[-1]}:{counts[-1]}")
    return " ".join(parts)


def _report(stats: Sequence[SourceStats]) -> None:
    header = (
        f"{'src':<4} {'loaded':>7} {'valid':>6} {'lang':>5} {'dedup':>6}"
        f" {'len>max':>8} {'kept':>6} {'avg_tok':>8} {'avg_tc':>7}"
    )
    print("\n" + "-" * len(header))
    print("Per-source stats")
    print("-" * len(header))
    print(header)
    for s in stats:
        avg_tok = sum(s.token_counts) / len(s.token_counts) if s.token_counts else 0
        avg_tc = s.tool_calls / s.kept if s.kept else 0
        print(
            f"{s.name:<4} {s.loaded:>7} {s.validated:>6} {s.lang_filtered:>5}"
            f" {s.dedup_dropped:>6} {s.length_dropped:>8} {s.kept:>6}"
            f" {avg_tok:>8.1f} {avg_tc:>7.2f}"
        )
    hist = [t for s in stats for t in s.token_counts]
    if hist:
        print(
            "\nToken histogram (kept rows): "
            + _histogram(hist, [1024, 2048, 4096, 6144, 8192])
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

_SYNTHETIC_B_ROW = {
    "trajectory correctness": "Correct",
    "trajectory": [
        {
            "role": "system",
            "content": (
                "You are a tool-augmented QA agent.\nYou are provided with"
                " function signatures within <tools></tools> XML tags:\n"
                '<tools>{"name": "search", "description": "Web search",'
                ' "parameters": {"type": "object", "properties": {}}}</tools>\n'
                "Call them well."
            ),
        },
        {"role": "user", "content": "Who wrote Die Weltbühne?"},
        {
            "role": "assistant",
            "content": (
                "<think>\nNeed to search for this.\n</think>\n\n"
                "<tool_calls_begin>\n"
                '<tool_call>{"name": "search", "arguments": {"query": ["Die'
                ' Weltbühne author"]}}</tool_call>\n'
                "</tool_calls_end>"
            ),
        },
        {
            "role": "user",
            "content": (
                "<tool_response>Carl von Ossietzky wrote for it.</tool_response>"
            ),
        },
        {
            "role": "assistant",
            "content": "<think>\nFound it.\n</think>\n\nCarl von Ossietzky.",
        },
    ],
}


def _self_test_b_conversion() -> None:
    """The B converter must produce a valid canonical row (regression)."""
    out = _convert_b([_SYNTHETIC_B_ROW])
    if len(out) != 1:
        raise AssertionError(f"expected 1 row, got {len(out)}")
    row = out[0]
    validate_row(row["messages"])
    assert row["tools"] and row["tools"][0]["name"] == "search", row["tools"]
    system = row["messages"][0]["content"]
    assert '"name": "search"' not in system, system
    assistant = row["messages"][2]
    assert assistant["reasoning_content"] == "Need to search for this."
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "query": ["Die Weltbühne author"]
    }
    tool = row["messages"][3]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1"
    assert "<tool_response>" not in tool["content"]


def _plant_bad_rows() -> list[tuple[str, Callable[[], None]]]:
    """Return (name, fn) pairs where fn() must raise BadRow."""

    def string_arguments() -> None:
        validate_row(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                },
            ]
        )

    def orphan_tool() -> None:
        validate_row(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "tool",
                    "content": "not a call",
                    "tool_call_id": "call_missing",
                },
                {"role": "assistant", "content": "x"},
            ]
        )

    def unknown_role() -> None:
        validate_row(
            [
                {"role": "giraffe", "content": "moo"},
                {"role": "user", "content": "hi"},
            ]
        )

    return [
        ("string-arguments", string_arguments),
        ("orphan-tool-message", orphan_tool),
        ("unknown-role", unknown_role),
    ]


def run_self_test() -> int:
    """Prove the validators reject bad rows and the B converter works."""
    failures = 0
    for name, fn in _plant_bad_rows():
        try:
            fn()
        except BadRow as exc:
            print(f"  [ok] {name}: rejected ({exc})")
        except Exception as exc:
            print(f"  [FAIL] {name}: raised {type(exc).__name__}: {exc}")
            failures += 1
        else:
            print(f"  [FAIL] {name}: validator accepted an invalid row")
            failures += 1
    try:
        _self_test_b_conversion()
        print("  [ok] b-conversion: synthetic row converts and validates")
    except Exception as exc:
        print(f"  [FAIL] b-conversion: {type(exc).__name__}: {exc}")
        failures += 1
    if failures:
        print(f"\nSelf-test FAILED ({failures} failure(s))")
        return 1
    print("\nSelf-test PASSED")
    return 0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _to_dataset(rows: Sequence[dict[str, Any]]) -> Any:
    """Build a HF dataset with ``messages``/``tools`` as JSON strings."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "messages": [json.dumps(r["messages"], ensure_ascii=False) for r in rows],
            "tools": [json.dumps(r["tools"], ensure_ascii=False) for r in rows],
        }
    )


def _write_local(rows: Sequence[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "messages": json.dumps(r["messages"], ensure_ascii=False),
                        "tools": json.dumps(r["tools"], ensure_ascii=False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an agentic tool-use SFT dataset from four public sources."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the validator/converter self-test and exit.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push to HF hub; write local JSONL instead.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/smoke",
        help="Local output dir with --no-push (default: %(default)s).",
    )
    parser.add_argument(
        "--namespace",
        default=NAMESPACE,
        help="HF namespace for the repos (default: HF_NAMESPACE env var).",
    )
    parser.add_argument(
        "--template",
        default=TEMPLATE_PATH,
        help="Chat template file for token counting (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Sampling seed (default: %(default)s)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Per-row token cap (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _process_source(
    label: str,
    loader: Callable[[], list[dict[str, Any]]],
    cap: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], SourceStats]:
    """Load → language filter → validate (hard-fail) → seeded cap."""
    stats = SourceStats(name=label)
    rows = loader()
    stats.loaded = len(rows)
    rows = [r for r in rows if not _row_has_cjk(r["messages"])]
    stats.lang_filtered = stats.loaded - len(rows)
    for r in rows:
        validate_row(r["messages"])
    stats.validated = len(rows)
    rng.shuffle(rows)
    kept = rows[:cap]
    for r in kept:
        r["source"] = label
    return kept, stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    from transformers import AutoTokenizer

    if not os.path.exists(args.template):
        print(f"template not found: {args.template}", file=sys.stderr)
        return 1
    with open(args.template) as f:
        template = f.read()
    print(f"Loading tokenizer {TOKENIZER_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)

    loaders = {"A": _load_a, "B": _load_b, "C": _load_c, "D": _load_d}
    rng = random.Random(args.seed)

    all_rows: list[dict[str, Any]] = []
    stats: dict[str, SourceStats] = {}
    for label in sorted(loaders):
        cap = SOURCE_CAPS[label]
        print(f"\n=== {label} (cap {cap}) ===")
        rows, st = _process_source(label, loaders[label], cap, rng)
        print(
            f"  loaded {st.loaded}, lang-dropped {st.lang_filtered},"
            f" validated {st.validated}, capped to {len(rows)}"
        )
        all_rows.extend(rows)
        stats[label] = st

    print("\nCross-source dedup ...")
    kept, dropped = dedup(all_rows)
    for r in dropped:
        stats[r["source"]].dedup_dropped += 1
    print(f"  dropped {len(dropped)} duplicate rows")

    print(f"\nLength bound (max {args.max_tokens} tokens) ...")
    final: list[dict[str, Any]] = []
    for r in kept:
        n = count_tokens(tokenizer, template, r["messages"], r["tools"])
        st = stats[r["source"]]
        if n > args.max_tokens:
            st.length_dropped += 1
            continue
        st.token_counts.append(n)
        st.tool_calls += sum(
            len(m.get("tool_calls", []))
            for m in r["messages"]
            if m.get("role") == "assistant"
        )
        final.append(r)
    for st in stats.values():
        st.kept = len(st.token_counts)
    print(f"  kept {len(final)} rows")
    _report(list(stats.values()))

    train_rows, eval_rows = split_by_trajectory(final, rng, SPLIT)
    print(f"\nSplit: {len(train_rows)} train / {len(eval_rows)} eval")

    print("\nRendered example (first kept row):")
    if final:
        print(
            render_row(tokenizer, template, final[0]["messages"], final[0]["tools"])[
                :1500
            ]
        )

    if args.no_push:
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "train.jsonl")
        eval_path = os.path.join(args.output_dir, "eval.jsonl")
        _write_local(train_rows, train_path)
        _write_local(eval_rows, eval_path)
        print(
            f"\nWrote {train_path} ({len(train_rows)})"
            f" and {eval_path} ({len(eval_rows)})"
        )
        return 0

    if not args.namespace:
        print(
            "A HF namespace is required to push"
            " (--namespace or HF_NAMESPACE env var).",
            file=sys.stderr,
        )
        return 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is required to push (write access).", file=sys.stderr)
        return 1
    for repo, rows in ((TRAIN_REPO, train_rows), (EVAL_REPO, eval_rows)):
        repo_id = f"{args.namespace}/{repo}"
        print(f"Pushing {len(rows)} rows → {repo_id} (private) ...")
        _to_dataset(rows).push_to_hub(repo_id, private=True, token=token)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
