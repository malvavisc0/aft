#!/usr/bin/env bash
# Experiment 0: QLoRA SFT of unsloth/Qwen3.5-9B on the smoke agent dataset
# (train -> merge -> GPTQ int4). Runs on the GB10 Spark.
#
# Usage:
#   scripts/train_qwen35_smoke.sh                  # full run
#   scripts/train_qwen35_smoke.sh --skip-quantize  # train + merge only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Reading private datasets needs a token. Non-interactive shells don't source
# ~/.bashrc, so pick up HF_TOKEN from there when unset.
if [[ -z "${HF_TOKEN:-}" ]] && grep -q HF_TOKEN ~/.bashrc 2>/dev/null; then
    export HF_TOKEN
    HF_TOKEN="$(grep HF_TOKEN ~/.bashrc | sed 's/.*HF_TOKEN=//' | tr -d "\"'" | tr -d ' ')"
fi

MODEL="${MODEL:-unsloth/Qwen3.5-9B}"
DATASET="${DATASET:?Set DATASET to the training dataset repo id (namespace/name)}"
RUN_NAME="${RUN_NAME:-qwen35-9b-agent-smoke}"
TEMPLATE="$REPO_ROOT/experiments/qwen3.5/chat_template.jinja"
# Default discovery misses the GatedDeltaNet projections; list them explicitly.
TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,in_proj_qkv,in_proj_z"

# import fla.ops before transformers so the fused Triton GatedDeltaNet
# kernels resolve (fla-core does not expose fla.ops from its __init__;
# without this the model uses the slow pure-torch fallback).
"$REPO_ROOT/.venv/bin/python" -c '
import fla.ops  # noqa: F401
import sys

from aft import main

sys.argv = ["aft", *sys.argv[1:]]
main()
' run \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --run-name "$RUN_NAME" \
    --format messages \
    --chat-template "$TEMPLATE" \
    --mask-strategy full \
    --max-seq-len 8192 \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 8 \
    --learning-rate 1e-4 \
    --target-modules "$TARGETS" \
    --quant-type int4 \
    "$@"
