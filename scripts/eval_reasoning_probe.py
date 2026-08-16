#!/usr/bin/env python
"""Behavioral probe for quantized Qwen3.8-9B artifacts served by vLLM.

Sends 5 prompts at the served endpoint and verifies:
  - 3 GSM8K-style math problems: ``<think>`` block present + correct answer
  - 1 competitive-programming prompt: code block produced
  - 1 tool-call prompt: ``<tool_call>`` XML parses (parser reused from
    ``scripts/eval_acceptance.py``)

Usage:
    python scripts/eval_reasoning_probe.py \
        --url http://localhost:8399/v1 \
        --model /path/to/artifact \
        --output experiments/qwen3.8/eval_gptq_int4.json
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Sequence

# Tool-call delimiters in the v22 template, built from chr() to keep this
# file free of literal markers (same convention as eval_acceptance.py).
T_OPEN = chr(60) + "tool_call" + chr(62)
T_CLOSE = chr(60) + "/tool_call" + chr(62)


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Parse ``<tool_call>`` blocks (json or v22 xml format)."""
    calls: list[dict[str, Any]] = []
    idx = 0
    found = False
    while True:
        start = text.find(T_OPEN, idx)
        if start == -1:
            break
        end = text.find(T_CLOSE, start)
        if end == -1:
            return None
        found = True
        body = text[start + len(T_OPEN) : end].strip()
        idx = end + len(T_CLOSE)
        parsed = _parse_block_body(body)
        if parsed is None:
            return None
        calls.extend(parsed)
    return calls if found else None


def _parse_block_body(body: str) -> list[dict[str, Any]] | None:
    """Parse one tool-call block body (json or xml format)."""
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        return [obj]
    lt, gt = chr(60), chr(62)
    fn_re = lt + r"function=(\w+)" + gt + r"(.*?)" + lt + r"/function" + gt
    pm_re = lt + r"parameter=(\w+)" + gt + r"(.*?)" + lt + r"/parameter" + gt
    out: list[dict[str, Any]] = []
    for fn_match in re.finditer(fn_re, body, re.DOTALL):
        args: dict[str, Any] = {}
        for pm in re.finditer(pm_re, fn_match.group(2), re.DOTALL):
            val: Any = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[pm.group(1)] = val
        out.append({"name": fn_match.group(1), "arguments": args})
    return out or None


SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "extra_body": {"top_k": 20},
}

MATH_PROBLEMS = [
    {
        "prompt": (
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        "answer": "72",
    },
    {
        "prompt": (
            "A robe takes 2 bolts of blue fiber and half that much white "
            "fiber. How many bolts in total does it take?"
        ),
        "answer": "3",
    },
    {
        "prompt": (
            "Josh decides to try flipping a house. He buys a house for "
            "$80,000 and then puts in $50,000 in repairs. This increased the "
            "value of the house by 150%. How much profit did he make?"
        ),
        "answer": "70000",
    },
]

CODE_PROMPT = (
    "Write a Python function `max_subarray_sum(nums)` that returns the "
    "maximum sum of any contiguous subarray (Kadane's algorithm). Handle "
    "the all-negative case correctly."
)

TOOL_PROMPT = {
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Paris right now?",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ],
}


def _chat(
    client: Any,
    model: str,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> str:
    """Chat completion, returning the raw generated text.

    vLLM's chat serializer strips ``<think>`` markers from ``content``
    without populating ``reasoning`` (no reasoning parser configured), so
    decode the returned token ids instead — the only unfiltered view of
    what the model generated.
    """
    sampling = dict(SAMPLING)
    sampling["extra_body"] = {**sampling["extra_body"], "return_token_ids": True}
    resp = client.chat.completions.create(
        model=model, messages=messages, **sampling, **extra
    )
    ids = resp.choices[0].token_ids
    if ids:
        return tokenizer.decode(ids, skip_special_tokens=False)
    return resp.choices[0].message.content or ""


def _has_think(text: str) -> bool:
    """True if a think block closed (the prompt supplies the opener)."""
    return "</think>" in text


def _final_number(text: str) -> str | None:
    tail = text[text.rfind("</think>") :] if "</think>" in text else text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", tail)
    return numbers[-1].replace(",", "") if numbers else None


def run_probe(url: str, model: str) -> list[dict[str, Any]]:
    from openai import OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(base_url=url, api_key="x")
    tokenizer = AutoTokenizer.from_pretrained(model)
    results: list[dict[str, Any]] = []

    for i, prob in enumerate(MATH_PROBLEMS):
        text = _chat(
            client, model, tokenizer, [{"role": "user", "content": prob["prompt"]}]
        )
        final = _final_number(text)
        ok = _has_think(text) and final is not None and prob["answer"] in final
        results.append(
            {
                "name": f"math_{i}",
                "pass": ok,
                "think": _has_think(text),
                "expected": prob["answer"],
                "got": final,
                "generation": text,
            }
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"[math_{i}] think={_has_think(text)} expected={prob['answer']}"
            f" got={final} -> {status}"
        )

    text = _chat(client, model, tokenizer, [{"role": "user", "content": CODE_PROMPT}])
    has_code = "```" in text and "def max_subarray_sum" in text
    code_ok = has_code and _has_think(text)
    results.append(
        {
            "name": "code",
            "pass": code_ok,
            "think": _has_think(text),
            "code_block": has_code,
            "generation": text,
        }
    )
    print(
        f"[code] think={_has_think(text)} code_block={has_code}"
        f" -> {'PASS' if code_ok else 'FAIL'}"
    )

    text = _chat(
        client, model, tokenizer, TOOL_PROMPT["messages"], tools=TOOL_PROMPT["tools"]
    )
    calls = parse_tool_calls(text)
    ok = bool(calls) and any(c.get("name") == "get_weather" for c in calls or [])
    results.append(
        {"name": "tool_call", "pass": ok, "parsed_calls": calls, "generation": text}
    )
    print(f"[tool_call] parsed={calls} -> {'PASS' if ok else 'FAIL'}")

    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning probe for quantized Qwen3.8-9B.")
    p.add_argument("--url", required=True, help="vLLM OpenAI base URL.")
    p.add_argument("--model", required=True, help="Served model id/path.")
    p.add_argument("--output", required=True, help="JSON output path.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_probe(args.url, args.model)
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} probes passed")
    with open(args.output, "w") as fh:
        json.dump(
            {
                "model": args.model,
                "url": args.url,
                "passed": passed,
                "total": len(results),
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"Wrote {args.output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
