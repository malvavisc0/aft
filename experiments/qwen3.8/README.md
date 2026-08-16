# Qwen3.8-9B Quantization — GPTQ Int4 (shipped) + FP8 (blocked)

**Date:** 2026-08-16
**Status:** Int4 complete and published; FP8 quantized but blocked at
vLLM serving (loader rejects gptqmodel's fp8 checkpoint format).

Artifact: https://huggingface.co/malvavisc0/Qwen3.8-9B-gptq-int4 (public)

## 1. Objective

Quantize [`empero-ai/Qwen3.8-9B`](https://huggingface.co/empero-ai/Qwen3.8-9B)
(revision `0934f3d2327ff2df2197495278c4c46ae5a56bd9`) — a full-parameter
distillation of Qwen3.8 2.4T A95B into the Qwen3.5-9B architecture — into
GPTQ Int4 (primary) and FP8 (secondary), and publish both publicly.

Same `qwen3_5` hybrid architecture as Qwopus3.5-9B-v3.5: multimodal
`Qwen3_5ForConditionalGeneration`, 32 text layers (24 GatedDeltaNet
`linear_attention` + 8 `full_attention`), 27-block vision tower, MTP
layer, 262k context, no top-level `rope_parameters`.

## 2. Recipe

| | |
|---|---|
| Host | omnitron (NVIDIA GB10, sm_121, aarch64, 128 GB unified) |
| Quantizer | gptqmodel 7.3.4, torch 2.13.0+cu130, transformers 5.15.0 |
| Method | GPTQ Int4, group size 128, `desc_act` |
| Calibration | 256 rows: 96 `nvidia/OpenMathReasoning` (cot), 96 `nvidia/OpenCodeReasoning` (split_0), 64 `nvidia/Nemotron-Agentic-v1` (interactive_agent); `{"messages": [...]}` JSONL flattened by `flatten_row_to_text`, wrapped by the v22 chat template as a single user turn; seq len 4096 |
| Chat template | `experiments/qwen3.5/chat_template.jinja` (v22) for calibration **and** shipped in the artifact |
| Coverage | `--allow-partial-coverage` (vision tower only) |

New script: `scripts/prepare_reasoning_calibration.py` (stdlib +
`datasets`; streams the three sources, enforces `<think>` presence on
math/code, char-caps fields, exact-hash dedups, fails under 80% quota).

## 3. Results

### GPTQ Int4 — shipped

- **Coverage:** 200/359 linear modules quantized (200 text, 159 vision
  skipped by design). Zero text-stack modules unquantized.
- **Quality (quant_log.csv, 200 rows):** mean per-layer loss **2.28e-5**,
  worst **1.09e-4** — better than the Qwopus baseline (4.2e-5 / 1.9e-4).
- **Size:** 8.5 GB (vs 19 GB BF16 source).
- **Sanitizer:** removed spurious top-level `rope_parameters` and
  `offload_to_disk_path` from both config files.
- **Provenance:** `aft_provenance.json` records pinned revision, 256/256
  samples, seq_len 4096, chat_template_applied true.
- **Smoke serve + behavioral probe** (vLLM 0.27.1, `gptq_marlin`,
  `--max-num-seqs 10`, v22 template, tool parser `qwen3_xml`):
  **5/5 passed** — 3 GSM8K-style problems correct with closed ``
  blocks, 1 competitive-programming prompt with code block, 1 tool call
  parsed (`get_weather(city=Paris, unit=celsius)`). Generations:
  `experiments/qwen3.8/eval_gptq_int4.json`.

### FP8 — quantized, serving blocked

- Quantization succeeded (after the device-validation fix in §4):
  200 text modules as `F8_E4M3` + F32 scales, vision tower BF16
  passthrough, 12 GB, configs sanitized.
- **Blocked at vLLM 0.27.1 serving.** gptqmodel 7.3.4's fp8 checkpoint
  format is incompatible with vLLM's fp8 loader for this architecture:
  - The config lacks `activation_scheme` (patched: `"dynamic"`), then
  - weight loading dies in `MergedColumnParallelLinear.weight_loader`
    (`'MergedColumnParallelLinear' object has no attribute 'data'`) —
    vLLM applies `Fp8LinearMethod` to the GDN fused projections
    (`in_proj_qkvz`, `in_proj_ba`) whose shards are mixed precision
    (fp8 `in_proj_qkv`/`in_proj_z`, BF16 `in_proj_a`/`in_proj_b`), and
    `is_layer_skipped` requires uniform precision per fused module, so
    there is no config-only way to exempt them.
- Recorded as blocked-by-platform per the plan's escape hatch. The
  artifact remains at `models/qwen3.8-9b/fp8` on omnitron.

## 4. Fix landed in `aft`

`src/aft/pipeline.py::_patch_fp8_device_validation` — gptqmodel 7.3.4's
FP8 weight-only finalize path passes a raw `torch.device` into
`BaseQuantLinear.validate_device`, which asserts `isinstance(device,
DEVICE)` (an enum) and crashes on any CUDA host. The quantize pipeline
now coerces `torch.device` → `DEVICE` via a narrow monkeypatch applied
only for FP8 runs. Lint + basedpyright clean (no new errors).

## 5. Serving notes (GB10 / vLLM 0.27.1)

- flashinfer JIT needs `ninja` on `PATH` — prepend the venv's `bin`
  (`PATH=/home/malvavisco/.vllm/bin:$PATH`) or EngineCore dies with
  `FileNotFoundError: 'ninja'`.
- Tool use requires `--enable-auto-tool-choice --tool-call-parser
  qwen3_xml`.
- vLLM does **not** auto-load `chat_template.jinja` from the model dir
  in this setup — pass `--chat-template <artifact>/chat_template.jinja`
  explicitly, or the tokenizer's minimal bundled template is used and
  the model emits no `` blocks.
- vLLM's chat serializer strips ``/`` from `content`
  without populating `reasoning` (no reasoning parser configured). For
  eval, decode `choices[0].token_ids` (`return_token_ids: true`) — the
  only unfiltered view. Probe script: `scripts/eval_reasoning_probe.py`.
- Keep `--max-num-seqs 10` for the hybrid Mamba cache. No
  `--max-model-len` clamp needed on 128 GB unified memory.

## 6. Files

- `scripts/prepare_reasoning_calibration.py` — calibration builder
- `scripts/eval_reasoning_probe.py` — 5-prompt behavioral probe
- `experiments/qwen3.8/calibration_reasoning_code.jsonl` — 256 rows
- `experiments/qwen3.8/eval_gptq_int4.json` — probe generations
- `src/aft/pipeline.py` — FP8 device-validation fix
