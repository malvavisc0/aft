#!/usr/bin/env python
"""Build the reasoning+code GPTQ calibration set for Qwen3.8-9B.

Mixes three public NVIDIA sources into a 256-row JSONL of ``{"messages":
[...]}`` rows, matching the serving distribution of a math/code CoT
distillation:

    math     nvidia/OpenMathReasoning  (split ``cot``)            96 rows
    code     nvidia/OpenCodeReasoning  (config ``split_0``)       96 rows
    agentic  nvidia/Nemotron-Agentic-v1 (``interactive_agent``)   64 rows

Math/code rows become two-turn conversations (user problem, assistant
``<think>`` trace). Agentic rows pass through unchanged. Rows are
exact-hash deduped across the combined set; each source must reach 80% of
its quota or the script fails.

Usage:
    uv run python scripts/prepare_reasoning_calibration.py \
        --output experiments/qwen3.8/calibration_reasoning_code.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from typing import Any

SEED = 20260816

QUOTAS = {"math": 96, "code": 96, "agentic": 64}
MIN_QUOTA_FRACTION = 0.8

# Char caps keep 4096-token truncation loss minimal (~3 chars/token for
# reasoning text with markup).
MAX_MATH_PROBLEM_CHARS = 6000
MAX_MATH_SOLUTION_CHARS = 12000
MAX_CODE_INPUT_CHARS = 6000
MAX_CODE_OUTPUT_CHARS = 12000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_sha(messages: Sequence[dict[str, Any]]) -> str:
    return _sha(json.dumps(list(messages), sort_keys=True, ensure_ascii=False))


def _stream(
    repo: str, *, name: str | None = None, split: str
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    return load_dataset(repo, **kwargs)


def _collect_math(quota: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in _stream("nvidia/OpenMathReasoning", split="cot"):
        problem = str(r.get("problem") or "").strip()
        solution = str(r.get("generated_solution") or "").strip()
        if not problem or "<think>" not in solution:
            continue
        if (
            len(problem) > MAX_MATH_PROBLEM_CHARS
            or len(solution) > MAX_MATH_SOLUTION_CHARS
        ):
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            }
        )
        if len(rows) >= quota:
            break
    return rows


def _collect_code(quota: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in _stream("nvidia/OpenCodeReasoning", name="split_0", split="split_0"):
        problem = str(r.get("input") or "").strip()
        output = str(r.get("output") or "").strip()
        if not problem or not output.startswith("<think>"):
            continue
        if len(problem) > MAX_CODE_INPUT_CHARS or len(output) > MAX_CODE_OUTPUT_CHARS:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": output},
                ]
            }
        )
        if len(rows) >= quota:
            break
    return rows


def _collect_agentic(quota: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in _stream("nvidia/Nemotron-Agentic-v1", split="interactive_agent"):
        messages = r.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        rows.append({"messages": messages})
        if len(rows) >= quota:
            break
    return rows


def _percentiles(values: Sequence[int]) -> str:
    if not values:
        return "n/a"
    s = sorted(values)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return f"p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={s[-1]}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reasoning+code GPTQ calibration set."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    collectors = {
        "math": _collect_math,
        "code": _collect_code,
        "agentic": _collect_agentic,
    }
    combined: list[dict[str, Any]] = []
    per_source: dict[str, list[dict[str, Any]]] = {}
    for label, collect in collectors.items():
        quota = QUOTAS[label]
        print(f"[{label}] collecting up to {quota} rows ...", flush=True)
        rows = collect(quota)
        if len(rows) < int(quota * MIN_QUOTA_FRACTION):
            print(
                f"[{label}] FAIL: got {len(rows)} rows, need >= "
                f"{int(quota * MIN_QUOTA_FRACTION)} (80% of {quota})",
                file=sys.stderr,
            )
            return 1
        rng.shuffle(rows)
        per_source[label] = rows
        combined.extend(rows)
        print(f"[{label}] collected {len(rows)}")

    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in combined:
        digest = _row_sha(row["messages"])
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(row)
    print(f"\nDedup: dropped {dropped}, kept {len(kept)}")

    print("\nPer-source char-length stats (flattened messages):")
    for label, rows in per_source.items():
        lengths = [
            sum(len(str(m.get("content") or "")) for m in r["messages"]) for r in rows
        ]
        print(f"  {label:<8} n={len(rows):>3}  {_percentiles(lengths)}")

    with open(args.output, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} rows → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
