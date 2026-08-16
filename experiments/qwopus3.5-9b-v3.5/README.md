# Qwopus3.5-9B-v3.5 — GPTQ Int4 Experiment Record

**Date:** 2026-08-13 → 2026-08-16
**Status:** Complete. Qwopus3.5-9B-v3.5 GPTQ Int4 artifact built and
verified structurally; serving acceptance probe pending.

This document consolidates everything worth preserving from the
experiment series: the architecture facts, the failure modes found,
the fixes that landed in `aft`, and the final quantization recipe.
It supersedes the working plans (`plans/finetune-qwen3.8-27b-agent.md`,
`plans/handoff-27b-run.md`, `plans/quantize-qwopus3.5-9b-v3.5.md`),
which contained machine-specific paths and credentials references.

---

## 1. What was done

1. **Step 0 — messages-path SFT in `aft`.** Reworked the trainer from
   flat-text to structured chat (`messages` column with roles,
   `tool_calls`, `reasoning_content`), with manual turn-diff label
   masking against the local v22 chat template. 276 lines of tests.
2. **Dataset prep.** Converters for four public sources (agentic coding
   traces, search-agent traces, cleaned traces, humanizer rewrites) into
   a canonical messages schema, with validation, exact-hash dedup, and a
   trajectory-disjoint 95/5 split. Published as two private HF datasets
   (400 train / 22 eval smoke rows).
3. **Experiment 0 — pipeline validation** on `unsloth/Qwen3.5-9B`:
   QLoRA LoRA train → merge → GPTQ Int4 → vLLM serve → tool-call eval.
   All stages passed; 10-prompt acceptance probe ran on the artifact.
4. **Qwopus quantization (final objective).** Direct base-model GPTQ
   Int4 of `Jackrong/Qwopus3.5-9B-v3.5` — no fine-tuning, since the
   model is already agentic-trained upstream. Artifact: 8.45 GB
   (vs 17.98 GB BF16).

## 2. Architecture facts (verified against live configs/weights)

`qwen3_5` (`Qwen3_5ForConditionalGeneration`) is **multimodal** with a
**hybrid text stack**:

- 32 text layers: 24 `linear_attention` (GatedDeltaNet) + 8
  `full_attention`, interval 4. Hidden 4096, heads 16/4, head_dim 256,
  vocab 248320, `max_position_embeddings=262144`, mrope
  (`rope_theta=1e7`, `mrope_section=[11,11,10]`) under
  `text_config.rope_parameters` — **no** legitimate top-level
  `rope_parameters`.
- 27-block vision tower (hidden 1152), plus an MTP layer
  (`unsloth_fixed_mtp`; 15 `mtp.*` tensors get merged into the state
  dict at quantize time).
- GatedDeltaNet projection names: `in_proj_qkv`, `in_proj_z`,
  `in_proj_b`, `in_proj_a`, `out_proj`, `conv1d`. Default LoRA discovery
  finds `q/k/v/o_proj` + `gate/up/down_proj` + `out_proj` but **misses**
  `in_proj_qkv`/`in_proj_z` → always pass explicit `--target-modules`:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,in_proj_qkv,in_proj_z`.
- Tokenizer: `Qwen2Tokenizer`, eos `` (248046), pad
  `<|vision_pad|>` (harmless quirk); dedicated tokens exist for
  `<tool_call>`/`</tool_call>`, `<tool_response>`/`</tool_response>`,
  `<think>`/`</think>`.
- **Bundled chat templates are minimal** (4 KB, no `reasoning_effort` /
  `preserve_thinking` / `tool_call_format` kwargs). The local v22
  template (`experiments/qwen3.5/chat_template.jinja`, 19 KB) is the
  single source of truth — override at train, calibrate, and serve time.
- gptqmodel 7.3.4's `models/definitions/qwen3_5.py` maps the full text
  stack (`in_proj_qkv`/`in_proj_z`/`out_proj` included), so strict layer
  coverage passes on the text stack; the vision tower is the only
  skipped set (159 modules).

## 3. Failure modes found (and fixes that landed)

| Failure | Root cause | Fix |
|---|---|---|
| DataLoader workers crash after CUDA init | Python 3.14 defaults multiprocessing to **forkserver** | Project moved to **Python 3.12** (fork default); `requires-python >=3.12`, ruff `py312` |
| GPTQ crash when fla imported | gptqmodel's `nogil_patcher` monkey-patches Triton's autotuner; fla's subclass lacks `_cache_lock` | **Train (fla on) and quantize (fla off) as separate processes.** Training script pre-imports `fla.ops` (fla-core's `__init__` doesn't expose it; fused kernels are 2.6× faster than pure-torch fallback) |
| peft AWQ dispatcher error | `optimum` missing when gptqmodel installed | Added `optimum` + `fla-core` deps |
| `--skip-finetune`/`--skip-quantize` ignored | resume logic overwrote CLI flags | Fixed (`bdabaaa`) |
| Calibration on bare web text shifts GPTQ activation stats | `fineweb-edu` default is untemplated web text | `--calibration nemotron-agentic` (agentic tool-use chat) + chat template applied to calibration |
| Calibration used tokenizer's bundled template, not v22 | no CLI flag existed | **`--chat-template` added to the quantize path** (`QuantizeConfig.chat_template`, overrides `tokenizer.chat_template` before calibration) |
| GPTQModel silently skipping hybrid-arch layers | layer-map gaps leave modules in BF16 | `strict_layer_coverage=True` default; coverage report counts quantized vs quantizable linears and hard-fails |
| gptqmodel blemishes in saved configs | spurious top-level `rope_parameters` (contradicts mrope), leaked `/tmp` offload path | `sanitize_saved_config` strips both post-save |
| trl silently truncating pre-tokenized rows | trl `max_length` default 1024 | pass `max_length=None` for the messages path |
| `datasets` 5.0.1 rejects bare local `.jsonl` | requires `namespace/name` repo ids | datasets published to HF hub; calibration JSONL path still supported for quantize |
| vLLM on 16 GB card: KV cache / Mamba cache errors | default 262144 context too big; hybrid Mamba needs fewer seqs | `--max-model-len 8192 --max-num-seqs 10` |
| HF write token shadowed on non-interactive ssh | stale read-only token in `~/.cache/huggingface/token` wins when `HF_TOKEN` unset | scripts extract `HF_TOKEN` from `~/.bashrc` explicitly |

## 4. Final quantization recipe (Qwopus3.5-9B-v3.5)

Source: `Jackrong/Qwopus3.5-9B-v3.5` @ `dc2b00e1b1bc404133e3a3e15e7ddcdff814fd86`
(still HEAD as of 2026-08-16; repo unchanged since 2026-04-16).

```bash
aft quantize \
  --model <local snapshot path> \
  --output models/qwopus3.5-9b-v3.5/gptq-int4 \
  --quant-type int4 \
  --group-size 128 --desc-act \
  --calibration nemotron-agentic \
  --n-calibration-samples 128 --calibration-seq-len 2048 \
  --chat-template experiments/qwen3.5/chat_template.jinja \
  --allow-partial-coverage
```

- `--allow-partial-coverage` is **expected**: the vision tower stays
  BF16 (text-only calibration can't represent it). Text stack: 200/200
  modules quantized. Coverage line reads 200/359.
- Result: 8.45 GB (53% reduction). Per-layer GPTQ loss: mean 4.2e-5,
  worst 1.9e-4 — excellent reconstruction.
- After saving, replace the artifact's `chat_template.jinja` with the
  v22 file (the aux-file copy carries the model's bundled minimal
  template, not the one calibration used).
- Provenance (`aft_provenance.json`) and per-layer error log
  (`quant_log.csv`) ship inside the artifact.

## 5. Serving

```bash
vllm serve <artifact> \
  --quantization gptq_marlin \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

- `qwen3_xml` is the correct parser: the v22 template's XML format
  (`<tool_call><function=name><parameter=key>value</parameter>…`) is
  exactly what vLLM's `Qwen3EngineToolParser` consumes. (`hermes` would
  be for JSON-format tool calls.)
- The artifact ships v22 as `chat_template.jinja`; vLLM picks it up
  automatically — no `--chat-template` flag needed.
- On 12–16 GB cards add `--max-model-len 8192 --max-num-seqs 10`.

## 6. Acceptance status

| Check | Result |
|---|---|
| Text-stack coverage | ✅ 200/200 modules Int4 (g128, desc_act, sym) |
| Quantization error | ✅ mean 4.2e-5, worst 1.9e-4 per layer |
| Config sanitation | ✅ spurious `rope_parameters` + leaked tmp path stripped |
| Size | ✅ 8.45 GB (fits 12–24 GB with KV headroom) |
| Source revision | ✅ `dc2b00e1…` = current HEAD |
| vLLM load + 10-prompt tool-call probe | ⏳ pending |

## 7. Decisions worth remembering

- **No fine-tuning of Qwopus.** It is an SFT continuation with ~2× the
  upstream SFT data; a small LoRA pass cannot add capability and risks
  eroding tool-calling. Leverage was in calibration + serving config.
- **Int4 is a hard requirement**, not a preference: FP8/Int8 (~10 GB)
  leave no KV-cache headroom on 12–24 GB targets. FP8 was demoted to a
  diagnostic; Int8 GPTQ is pointless (same layer-matching path as Int4).
- **Partial coverage is acceptable only for the vision tower.** If
  text-stack projections are ever skipped, the fix is in gptqmodel's
  layer map, not in relaxing the check.

## 8. Artifacts and where they live

- Quantized model: HF hub, private (`Qwopus3.5-9B-v3.5-gptq-int4`),
  pending the §6 serve probe.
- Smoke datasets: HF hub, private (400 train / 22 eval rows).
- Experiment 0 eval generations: `experiments/qwen3.5/*.json` (base vs
  fine-tuned vs GPTQ artifact on the same 10 prompts).
- v22 chat template: `experiments/qwen3.5/chat_template.jinja`.
- Eval probe (reusable for any future artifact):
  `scripts/eval_acceptance.py` — vLLM offline inference, parses XML/JSON
  tool-call blocks, reports per-prompt results.
- Dataset prep: `scripts/prepare_agent_dataset.py` (converters for the
  four sources, validated end-to-end).
