#!/usr/bin/env python3
"""Experiment 0 acceptance probe (section 16.4).

Serves the quantized (or merged) model with the v22 chat template via
vLLM and generates responses for held-out eval prompts.  Verifies the
model emits parseable tool-call blocks in the trained format and that
tool-results wrappers are handled correctly.

Usage (local, GPU free; run with the vLLM venv's python):

    python scripts/eval_acceptance.py \
        --model models/<run>/gptq-int4 \
        --eval-dataset <namespace>/qwen3.5-9b-agent-smoke-eval \
        --chat-template experiments/qwen3.5/chat_template.jinja \
        --quantization gptq_marlin \
        --num-prompts 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Delimiters in the v22 template, built from chr() to keep this file
# free of literal tool-call markers that confuse editors/formatters.
T_OPEN = chr(60) + "tool_call" + chr(62)
T_CLOSE = chr(60) + "/tool_call" + chr(62)
T_RES_OPEN = chr(60) + "tool_results" + chr(62)
T_RES_CLOSE = chr(60) + "/tool_results" + chr(62)
TH_OPEN = chr(60) + "think" + chr(62)
TH_CLOSE = chr(60) + "/think" + chr(62)


def parse_tool_calls(text: str) -> list[dict] | None:
    """Return parsed tool-call dicts, or None if unparseable.

    For the json format each block body is a single JSON object with
    "name" and "arguments".  For xml the body holds <function=...>
    blocks; those are parsed leniently.
    """
    calls: list[dict] = []
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
    if not found:
        return None
    return calls


def _parse_block_body(body: str) -> list[dict] | None:
    """Parse one tool-call block body (json or xml format)."""
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        return [obj]
    out: list[dict] = []
    for fn_match in re.finditer(
        chr(60)
        + r"function=(\w+)"
        + chr(62)
        + r"(.*?)"
        + chr(60)
        + r"/function"
        + chr(62),
        body,
        re.DOTALL,
    ):
        name = fn_match.group(1)
        args: dict = {}
        for pm in re.finditer(
            chr(60)
            + r"parameter=(\w+)"
            + chr(62)
            + r"(.*?)"
            + chr(60)
            + r"/parameter"
            + chr(62),
            fn_match.group(2),
            re.DOTALL,
        ):
            val = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[pm.group(1)] = val
        out.append({"name": name, "arguments": args})
    return out or None


def has_tool_results(text: str) -> bool:
    """True if the generation contains a tool-results wrapper."""
    return T_RES_OPEN in text and T_RES_CLOSE in text


def split_thinking(text: str) -> tuple[str, str]:
    """Split generation into (thinking, response).

    The v22 template opens `` in the generation prompt, so the
    raw generation starts with thinking content.  The first `...`
    closes it; everything after is the response.  If no `` is
    found the model was still thinking (truncated by max_tokens).
    """
    idx = text.find(TH_CLOSE)
    if idx == -1:
        return text.strip(), ""
    thinking = text[:idx].strip()
    rest = text[idx + len(TH_CLOSE) :].lstrip()
    return thinking, rest


def build_prompt(row_messages: list[dict]) -> list[dict] | None:
    """Drop the trailing assistant turn; return prompt messages.

    Returns None if the row is unusable (e.g. single message).
    """
    if len(row_messages) < 2:
        return None
    if row_messages[-1]["role"] != "assistant":
        return None
    return row_messages[:-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 0 acceptance probe")
    parser.add_argument("--model", required=True, type=Path, help="Path to model dir")
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument(
        "--chat-template",
        default=Path("experiments/qwen3.5/chat_template.jinja"),
        type=Path,
    )
    parser.add_argument(
        "--quantization", default=None, help="vLLM quant arg, e.g. gptq_marlin"
    )
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max generated tokens"
    )
    parser.add_argument(
        "--tool-call-format",
        default="xml",
        choices=["xml", "json"],
        help="Tool-call format the model was trained with",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model for comparison (HF repo id or local path)",
    )
    parser.add_argument(
        "--base-quantization",
        default=None,
        help="Quantization for base model, e.g. bitsandbytes",
    )
    parser.add_argument(
        "--output", default=None, type=Path, help="Write JSON results here"
    )
    args = parser.parse_args()

    template = args.chat_template.read_text()
    print(f"Chat template: {args.chat_template} ({len(template)} chars)")

    from datasets import load_dataset

    ds = load_dataset(args.eval_dataset, split="train")
    print(f"Loaded {len(ds)} eval rows from {args.eval_dataset}")
    rows = [(json.loads(r["messages"]), json.loads(r["tools"])) for r in ds]

    prompts: list[tuple[list[dict], list]] = []
    for msgs, tools in rows:
        built = build_prompt(msgs)
        if built is not None:
            prompts.append((built, tools))
    prompts = prompts[: args.num_prompts]
    print(f"Built {len(prompts)} prompts (requested {args.num_prompts})")

    if not prompts:
        print("No usable prompts found.", file=sys.stderr)
        return 1

    import torch
    from vllm import LLM, SamplingParams

    llm_kwargs: dict = {
        "model": str(args.model),
        "chat_template": template,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 8192,
        "max_num_seqs": 10,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    print(
        f"Loading model: {args.model} (quantization={args.quantization or 'auto'}) ..."
    )
    llm = LLM(**llm_kwargs)

    # Optionally load the base model for comparison
    base_llm: LLM | None = None
    if args.base_model:
        base_kwargs: dict = {
            "model": str(args.base_model),
            "chat_template": template,
            "gpu_memory_utilization": 0.92,
            "max_model_len": 8192,
            "max_num_seqs": 10,
        }
        if args.base_quantization:
            base_kwargs["quantization"] = args.base_quantization
        print(
            f"Loading base model: {args.base_model}"
            f" (quantization={args.base_quantization or 'auto'}) ..."
        )
        # Free the fine-tuned model first to reclaim VRAM
        del llm
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        base_llm = LLM(**base_kwargs)

    active_llm = base_llm or llm
    active_tok = active_llm.get_tokenizer()

    rendered: list[str] = []
    for prompt_msgs, tools in prompts:
        text = active_tok.apply_chat_template(
            prompt_msgs,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
            tool_call_format=args.tool_call_format,
        )
        rendered.append(text)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    label = "base" if base_llm else "fine-tuned"
    print(
        f"Generating {len(rendered)} responses"
        f" ({label}, max_tokens={args.max_tokens}) ..."
    )
    outputs = active_llm.generate(rendered, sampling)

    results = []
    n_parseable = 0
    n_with_tool_calls = 0
    n_with_tool_results = 0
    n_thinking_complete = 0
    for i, (output, (prompt_msgs, tools)) in enumerate(zip(outputs, prompts)):
        gen = output.outputs[0].text
        thinking, response = split_thinking(gen)
        calls = parse_tool_calls(response)
        has_tr = has_tool_results(response)
        parseable = calls is not None
        has_calls = bool(calls)
        thinking_complete = bool(response)
        if parseable:
            n_parseable += 1
        if has_calls:
            n_with_tool_calls += 1
        if has_tr:
            n_with_tool_results += 1
        if thinking_complete:
            n_thinking_complete += 1
        has_tools = bool(tools)
        row = {
            "index": i,
            "label": label,
            "has_tools": has_tools,
            "last_user": next(
                (
                    m["content"][:120]
                    for m in reversed(prompt_msgs)
                    if m["role"] == "user"
                ),
                "",
            ),
            "thinking_complete": thinking_complete,
            "has_tool_calls": has_calls,
            "parseable": parseable,
            "has_tool_results": has_tr,
            "n_tool_calls": len(calls) if calls else 0,
            "tool_call_names": [c["name"] for c in calls] if calls else [],
            "thinking": thinking[:400],
            "response": response[:600],
            "generation": gen[:600],
        }
        results.append(row)
        status = "OK" if parseable else "NOPARSE"
        tc = f" tc={row['n_tool_calls']} {row['tool_call_names']}" if has_calls else ""
        think_stat = "think" if thinking_complete else "TRUNC"
        print(f"  [{i}] {'TOOL' if has_tools else 'TEXT'} {status} {think_stat}{tc}")
        print(f"      last_user: {row['last_user']!r}")
        if thinking_complete:
            print(f"      thinking: {row['thinking'][:120]!r}")
            print(f"      response: {row['response'][:200]!r}")
        else:
            print(f"      thinking (truncated): {row['thinking'][:200]!r}")

    print("\n" + "=" * 60)
    print(f"Model:            {label}")
    print(f"Prompts:          {len(outputs)}")
    print(f"Thinking complete: {n_thinking_complete}/{len(outputs)}")
    print(f"Parseable:        {n_parseable}/{len(outputs)}")
    print(f"Tool calls:       {n_with_tool_calls}/{len(outputs)}")
    print(f"Tool results:     {n_with_tool_results}/{len(outputs)}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nResults written to {args.output}")

    # Acceptance: rows WITH tools must produce parseable tool-call blocks;
    # plain-text rows pass if the response is non-empty (thinking completed).
    n_pass = sum(
        1
        for r in results
        if r["parseable"]
        or (not r["has_tools"] and r["thinking_complete"] and r["response"])
    )
    print(f"\nAcceptance: {n_pass}/{len(results)} pass")
    if n_pass < len(results) * 0.8:
        print("FAIL: fewer than 80% of prompts produced parseable output")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
