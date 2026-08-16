# aft — Aria Finetuner

**Standalone fine-tuning and quantization pipeline for instruction-following models.**

`aft` takes a base model from HuggingFace, fine-tunes it with QLoRA (4-bit NF4),
merges the adapter, quantizes to GPTQ (Int4/Int8) or FP8, and optionally
publishes the result to HuggingFace Hub — all from a single CLI.

## Features

- **QLoRA SFT** — 4-bit NF4 training with LoRA adapters on all attention + MLP layers
- **Structured chat training** — `--format messages` trains on chat rows with `tool_calls` and `reasoning_content`, with per-turn loss masking rendered through your own chat template (`--chat-template`)
- **Automatic parameter tuning** — detects your GPU and model size, then recommends rank, learning rate, batch size, and sequence length
- **Dataset cleaning** — whitespace normalization, special-character filtering, token-length bounds, language detection, and deduplication
- **Conversational datasets** — auto-flattens chat-format rows (`messages`/`conversations`) from any schema (OpenAI `role`/`content`, ShareGPT `from`/`value`, OpenAI multi-part content blocks)
- **Quantization** — GPTQ Int4/Int8 and FP8 quantization via GPTQModel, producing vLLM-ready checkpoints
- **Domain-matched calibration** — code, agentic, and general-web calibration presets; arbitrary HF datasets accepted; activation order on by default for higher fidelity
- **Clean artifacts** — strips gptqmodel's leaked local temp paths and spurious top-level `rope_parameters` from saved configs
- **Hub integration** — push quantized models directly to HuggingFace Hub
- **Modular pipeline** — run the full stack or individual phases (train, merge, quantize, push)
- **Resumable runs** — automatically detects completed phases and skips them with `--resume`

![Screenshot](https://raw.githubusercontent.com/malvavisc0/aft/refs/heads/master/running.png)

## Requirements

- Python ≥ 3.12, < 3.14 — **3.14 is not supported**: its forkserver
  multiprocessing default crashes DataLoader workers after CUDA init
- NVIDIA GPU with CUDA support
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/malvavisc0/aft.git
cd aft
uv venv
uv sync
```

Or with pip:

```bash
pip install .
```

## Quick Start

```bash
# 1. Get recommended parameters for your hardware + model
aft recommend --model Qwen/Qwen2.5-7B

# 2. Run the full pipeline (train → merge → quantize)
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset teknium/OpenHermes-2.5 \
    --run-name my-run \
    --output ./models

# 3. Push the quantized model to HuggingFace Hub
aft push \
    --model ./models/my-run/gptq-int4 \
    --repo-id myorg/my-model-gptq-int4

# 4. Serve with vLLM
vllm serve ./models/my-run/gptq-int4 --quantization gptq_marlin
```

### Quantize a base model (no fine-tuning)

For models too large to train or merge locally, quantize directly:

```bash
# Download the source model once
hf download empero-ai/Qwable-9B-Claude-Fable-5 --local-dir ./models/qwable-src

# Quantize with domain-matched calibration (agentic tool-use traces)
aft quantize \
    --model ./models/qwable-src \
    --output ./models/qwable-gptq-int4 \
    --calibration nemotron-agentic \
    --calibration-samples 256 \
    --calibration-seq-len 4096

# Serve
vllm serve ./models/qwable-gptq-int4 --quantization gptq_marlin
```

## Commands

### `aft recommend`

Detects your GPU hardware, fetches model metadata from HuggingFace, and outputs recommended QLoRA hyperparameters with reasoning.

```bash
aft recommend --model Qwen/Qwen2.5-7B
aft recommend --model Qwen/Qwen2.5-7B --token hf_xxxx
```

### `aft run`

Runs the full pipeline: QLoRA SFT → LoRA merge → quantization.

```bash
# GPTQ Int4 (default)
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset teknium/OpenHermes-2.5 \
    --run-name my-run \
    --output ./models

# FP8 quantization
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset teknium/OpenHermes-2.5 \
    --run-name my-fp8-run \
    --output ./models \
    --quant-type fp8

# After FP8 quantization, serve with vLLM:
# vllm serve ./models/my-fp8-run/fp8 --quantization fp8

# Agentic model: train on Nemotron tool-use traces, calibrate on the same domain
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset nvidia/Nemotron-Agentic-v1:interactive_agent \
    --calibration nemotron-agentic \
    --run-name my-agentic-run \
    --output ./models
```

#### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `2 × lora-rank` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout rate |
| `--max-seq-len` | `2048` | Maximum sequence length |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--max-samples` | all | Limit number of training samples |
| `--trust-remote-code` | off | Allow loading models with custom code |

#### Dataset cleaning options

| Flag | Default | Description |
|------|---------|-------------|
| `--clean` | off | Enable dataset cleaning |
| `--dedup` | off | Remove exact duplicate texts |
| `--min-tokens` | `10` | Minimum token count |
| `--max-tokens` | `max-seq-len` | Maximum token count |
| `--languages` | all | Comma-separated language codes (requires `langdetect`) |
| `--max-special-ratio` | `0.3` | Max fraction of non-alphanumeric characters |

#### Quantization options

| Flag | Default | Description |
|------|---------|-------------|
| `--quant-type` | `int4` | Quantization type: `int4`, `int8`, or `fp8` |
| `--gptq-group-size` | `128` | GPTQ group size (int4/int8 only) |
| `--desc-act` / `--no-desc-act` | `--desc-act` | Activation order — slower to quantize, higher fidelity (int4/int8 only) |
| `--calibration` | `fineweb-edu` | Calibration source: alias (`fineweb-edu`, `starcoder`, `nemotron-agentic`), HF repo id, or JSONL path |
| `--calibration-samples` | `128` | Number of calibration samples |
| `--calibration-seq-len` | `2048` | Calibration sequence length |
| `--no-chat-template` | off | Calibrate on raw text instead of the model's chat template |
| `--chat-template` | tokenizer's | Path to a local `.jinja` chat template — overrides the tokenizer's bundled one for both training (`--format messages`) and calibration rendering. Use the same file you will pass to vLLM at serve time, so train/calibrate/serve all match |
| `--target-modules` | auto-detected | Comma-separated LoRA target modules. Discovered from the model when omitted |
| `--revision` | latest | Pin the base model to a specific git revision |

#### Pipeline control

| Flag | Description |
|------|-------------|
| `--skip-finetune` | Skip SFT, merge + quantize an existing adapter |
| `--base-model-only` | Quantize the base model directly — no SFT, no adapter, no merge |
| `--skip-quantize` | Stop after merge (produces fp16 model) |
| `--resume` | Auto-detect completed phases and skip them |

### `aft quantize`

Quantize an already-merged fp16 model (GPTQ int4/int8 or FP8).

> **`--model` must be a local directory**, not a HuggingFace repo ID. It
> expects a fully-merged fp16/bf16 checkpoint (safetensors + config) on
> disk — typically the output of `aft run`/`aft merge`, or an existing
> model pulled straight from the Hub. If you want to quantize a model
> that only exists on the Hub (no fine-tuning of your own), download it
> locally first with the `hf` CLI:
>
> ```bash
> hf download <org>/<model> --local-dir ./models/<model>-src
> ```
>
> Then pass that local directory as `--model` below.

```bash
# GPTQ Int4 (default)
aft quantize \
    --model ./models/my-run/merged \
    --output ./models/my-run/gptq-int4

# FP8
aft quantize \
    --model ./models/my-run/merged \
    --output ./models/my-run/fp8 \
    --quant-type fp8

# GPTQ Int8
aft quantize \
    --model ./models/my-run/merged \
    --output ./models/my-run/gptq-int8 \
    --quant-type int8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--quant-type` | `int4` | Quantization type: `int4`, `int8`, or `fp8` |
| `--group-size` | `128` | GPTQ group size (int4/int8 only). Must be 128, 64, 32, or -1 |
| `--desc-act` / `--no-desc-act` | `--desc-act` | Activation order — slower to quantize, higher fidelity (int4/int8 only) |
| `--calibration` | `fineweb-edu` | Calibration source: alias (`fineweb-edu`, `starcoder`, `nemotron-agentic`), HF repo id, or JSONL path |
| `--n-calibration-samples` | `128` | Number of calibration samples |
| `--calibration-seq-len` | `2048` | Calibration sequence length |
| `--no-chat-template` | off | Calibrate on raw text instead of the model's chat template |
| `--chat-template` | tokenizer's | Path to a local `.jinja` chat template — overrides the tokenizer's bundled one when rendering calibration texts. Use the same file you will serve with |
| `--allow-partial-coverage` | off | Permit layers to be left unquantized instead of failing |
| `--revision` | latest | Pin the model to a specific git revision |
| `--trust-remote-code` | off | Allow loading models with custom code |

### `aft push`

Publish a quantized model directory to HuggingFace Hub.

```bash
aft push \
    --model ./models/my-run/gptq-int4 \
    --repo-id myorg/my-model-gptq-int4 \
    --private
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | required | HuggingFace repo ID |
| `--private` | off | Create as private repository |
| `--token` | `HF_TOKEN` env | HuggingFace API token |
| `--message` | `"Upload GPTQ quantized model"` | Commit message |

## Supported Datasets

### Structured chat (`--format messages`)

For agentic/tool-calling SFT, `aft` consumes a `messages` column of
`{role, content}` dicts — assistant turns may carry `tool_calls`
(`function.arguments` as a **dict**) and `reasoning_content`; tool results
use `role: "tool"` with `tool_call_id`. Each row's tool definitions travel
in a sibling `tools` column and are passed to the template's `tools=` kwarg.
Tokenization renders through the template given by `--chat-template`
(required), and labels are built by turn-diff so loss covers assistant spans
(reasoning + content + tool calls) and masks system/user/tool context:

```bash
aft run \
    --model unsloth/Qwen3.5-9B \
    --dataset myorg/my-agent-data \
    --format messages \
    --chat-template experiments/qwen3.5/chat_template.jinja \
    --mask-strategy full \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,in_proj_qkv,in_proj_z \
    --run-name my-agent --output ./models
```

`--mask-strategy full` labels every assistant span (full trajectories);
`cumulative` labels only the final one.

### Flat text (default)

`aft` auto-detects the dataset format when loading from HuggingFace and
flattens each row into a single text string:

| Column(s) | Handling |
|-----------|----------|
| `text`, `content`, `code`, `body` | Used directly as training/calibration text |
| `messages` | OpenAI/HF chat format — `[{role, content}]` flattened to `role: content` lines (supports multi-part `content` blocks) |
| `conversations` | ShareGPT format — `[{from, value}]` flattened the same way |

Datasets that expose named splits instead of `train` (e.g.
`nvidia/Nemotron-Agentic-v1` has `interactive_agent` / `tool_calling`) are
auto-resolved: `aft` queries the available splits and falls back to the first
one when `train` is absent. Append `:split` to pick a specific subset:

```bash
aft run --dataset nvidia/Nemotron-Agentic-v1:tool_calling --model ... --run-name ...
```

Multiple datasets can be passed as a comma-separated list:

```bash
aft run --dataset teknium/OpenHermes-2.5,Open-Orca/SlimOrca --model ... --run-name ...
```

### Calibration presets

`--calibration` accepts short aliases, arbitrary HF repo ids, or a local JSONL
file. Aliases map to a curated registry with the right text field and split:

| Alias | Dataset | Field | Notes |
|-------|---------|-------|-------|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` | `text` | General educational web (default) |
| `fineweb` | `HuggingFaceFW/fineweb` | `text` | General web |
| `c4` | `allenai/c4` | `text` | General web |
| `starcoder` | `bigcode/starcoderdata` | `content` | Code — gated, streams the `python` subdir |
| `nemotron-agentic` | `nvidia/Nemotron-Agentic-v1` | `messages` | Agentic tool-use chat — `interactive_agent` split |

For a coding/agentic model, match the calibration distribution to the serving
traffic rather than defaulting to general web text:

```bash
aft quantize --model ./merged --output ./gptq \
    --calibration nemotron-agentic --calibration-samples 256 --calibration-seq-len 4096
```

Calibration texts are rendered through the chat template before tokenization
(unless `--no-chat-template`), because GPTQ measures activation statistics —
they must match what the model sees at serving time. If you serve with a
custom chat template, pass the same file via `--chat-template` here; the
tokenizer's bundled template is often a minimal one that differs from what
you actually serve.

## Supported Architectures

LoRA target modules are **discovered from the loaded model**, by intersecting its
actual linear layers with a set of known projection names (`q/k/v/o_proj`,
`qkv_proj`, `gate/up/down_proj`, `gate_up_proj`, `in_proj`, `out_proj`). Well
tested on:

- **Llama** family (Llama 2, Llama 3, Code Llama)
- **Qwen** family (Qwen 2, Qwen 2.5, Qwen 3.5 hybrid — see Known Limitations)
- **Mistral** / Mixtral
- **Gemma** / Gemma 2

If no candidate names match, `aft` **fails with an explicit error** rather than
training an adapter that touches almost nothing. Use `--target-modules` to name
the layers yourself for unusual architectures.

**Multimodal models** (`*ForConditionalGeneration`) are loaded with
`AutoModelForImageTextToText` + `AutoProcessor`, and their
`preprocessor_config.json` / `processor_config.json` / `chat_template.jinja`
are carried into every output directory. Note that calibration is text-only, so
the vision tower's activation statistics are not represented — `aft` warns when
this applies.

**Hybrid / MoE architectures** (linear attention, SSM blocks, sparse experts)
often contain modules GPTQModel does not recognise. Those layers would otherwise
be silently left in full precision, so after quantizing `aft` reports layer
coverage and **fails** if any quantizable linear was skipped. Pass
`--allow-partial-coverage` if that is genuinely intended.

## Serving with vLLM

After quantization, the output directory is ready for vLLM. The `--quantization` flag
depends on the quantization type:

```bash
# GPTQ Int4 / Int8
vllm serve ./models/my-run/gptq-int4 --quantization gptq_marlin
vllm serve ./models/my-run/gptq-int8 --quantization gptq_marlin

# FP8
vllm serve ./models/my-run/fp8 --quantization fp8
```

The `gptq_marlin` kernel provides near-native inference speed for GPTQ-quantized
models; FP8 requires Hopper GPUs (H100/H200) or newer for optimal performance.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models and Hub uploads |
| `LOGURU_LEVEL` | Set to `DEBUG` for verbose pipeline logging (default: `INFO`) |

## Output Structure

`--output` is the base directory; `--run-name` creates a subdirectory under it.
The quantized output directory name depends on `--quant-type`:

```
<output>/<run-name>/
├── adapter/          # LoRA adapter weights
├── checkpoints/      # Training checkpoints
├── merged/           # Merged fp16 model
└── gptq-int4/        # GPTQ Int4 model (default, vLLM-ready)
    gptq-int8/        # GPTQ Int8 model (--quant-type int8)
    fp8/              # FP8 model (--quant-type fp8)
```

## Resuming a Run

If a run is interrupted (e.g. OOM during quantization), use `--resume` to pick up from the last completed phase:

```bash
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset teknium/OpenHermes-2.5 \
    --run-name my-run \
    --output ./models \
    --resume
```

The resume logic checks for:
- `adapter/adapter_config.json` → skips training
- `merged/*.safetensors` → skips merge
- `gptq-int4/*.safetensors` → skips quantization

## Example: `aft recommend` Output

> Terminal output is rendered with [Rich](https://github.com/Textualize/rich) styling. The plain-text version below is approximate.

```bash
$ aft recommend --model Qwen/Qwen2.5-7B
```

```
  ╔═╗╔═╗╔╦╗
  ╠═╣╠╣  ║ 
  ╩ ╩╚   ╩  v0.0.1 │ QLoRA fine-tuning → GPTQ quantization

  ⚡ Hardware
    GPU  NVIDIA GeForce RTX 4090 (24.0 GiB VRAM)
    RAM  62.5 GiB  BF16 ✓  OS  Linux

  🧠 Qwen/Qwen2.5-7B
    Params 7.0B  Type qwen2
    Arch   Qwen2ForCausalLM
    Hidden 3584  Layers 28

  ⚙  Recommended Parameters
    lora_rank                  16
    lora_alpha                 32
    max_seq_len                2048
    batch_size                 1
    gradient_accumulation       16
    learning_rate              2e-4
    epochs                     2
    max_memory                 auto

  💡 Reasoning:
    • Model ~7.0B params on 24.0 GiB VRAM
    • Medium model (3-8B) → rank 16, lr 0.0002, 2 epochs
    • QLoRA 4-bit base weights ≈ 3.9 GiB
    • VRAM moderate (16.5 GiB remaining) → seq_len=2048, batch=1
    • Effective batch size: 1 × 16 = 16 (target ~16)
    • BF16 supported ✓ — will use bf16 compute

  📋 Copy-paste command:
    aft run \
      --model Qwen/Qwen2.5-7B \
      --dataset <DATASET> \
      --run-name <RUN_NAME> \
      --output ./models/<RUN_NAME> \
      --lora-rank 16 \
      --max-seq-len 2048 \
      --epochs 2 \
      --batch-size 1 \
      --grad-accum 16 \
      --learning-rate 2e-4
```

## How It Works

### Why QLoRA over full fine-tuning?

Full 16-bit fine-tuning of a 7B model requires ~14 GB of VRAM just for the model weights, plus memory for optimizer states and gradients. QLoRA reduces this to ~4 GB by quantizing the base model to 4-bit NF4, then training only low-rank adapter matrices (~0.5% of total parameters). This makes fine-tuning feasible on consumer GPUs with 8–24 GB VRAM.

### Why GPTQ over AWQ?

GPTQ was chosen because:
- **vLLM integration** — vLLM has first-class support for GPTQ via the `gptq_marlin` kernel, giving near-native inference speed.
- **Calibration flexibility** — GPTQ supports arbitrary calibration datasets (default: `fineweb-edu`), and `aft` ships domain-matched presets (code, agentic) so you can tune quantization quality for your model's serving traffic.
- **Activation order by default** — `--desc-act` is on by default, reordering weight columns by activation magnitude before quantizing for measurably lower loss on instruct/reasoning fine-tunes. The cost is one-time slower quantization; the artifact is permanent.
- **Active development** — GPTQModel (used by this tool) is actively maintained with broad model architecture support.

### Why these LoRA target modules?

The adapter targets attention projections (`q/k/v/o_proj`) plus the feed-forward
projections (`gate/up/down_proj`). Covering both is more comprehensive than
attention-only targeting: the MLP layers often carry domain-specific knowledge,
making them important for instruction-following fine-tuning.

These names are treated as *candidates*, not assumptions — the final list is the
intersection with the modules the loaded model actually has, so a wrong guess
surfaces as an error instead of a quietly ineffective training run.

### Memory estimation

The 4-bit memory footprint is estimated as:

```
base_weights_gib ≈ params_b × 0.55
```

The `0.55` factor accounts for the NF4 quantization (4 bits/param) plus safetensors header overhead and embedding layers that may use higher precision. The remaining VRAM is split between optimizer states, gradients, and activation memory.

## Known Limitations

- **Merging requires the whole model in RAM** — `merge_adapter` materializes the full unquantized model on CPU, so a 35B/65 GiB checkpoint needs ≥65 GiB of system RAM. `aft` warns when the checkpoint clearly won't fit; for large models use `--base-model-only` to quantize the base model directly.
- **No layer-sequential or CPU-offload quantization** — quantization assumes the model can be loaded on the available hardware.
- **Text-only calibration** — even for multimodal models; the vision path is not calibrated.
- **Single-GPU training only** — No FSDP or DeepSpeed support. Multi-GPU is limited to data-parallel via `device_map="auto"`.
- **No eval/validation during training** — Training runs without a validation set. Metrics are training loss only.
- **No early stopping** — The pipeline trains for the specified number of epochs without monitoring validation loss.
- **Gated datasets need manual setup** — presets like `starcoder` require accepting access terms on the dataset's HF page and setting `HF_TOKEN`. Non-gated presets (`fineweb-edu`, `nemotron-agentic`) work out of the box.
- **fla and gptqmodel conflict in one process** — on hybrid-arch models (e.g. `qwen3_5` GatedDeltaNet), pre-importing `fla.ops` gives fused Triton kernels for training, but gptqmodel's Triton autotuner patch then crashes quantization. Train (fla imported) and quantize (no fla) as separate processes.
- **Hybrid-arch LoRA discovery under-covers** — on `qwen3_5`-style models the default discovery misses the GatedDeltaNet `in_proj_qkv`/`in_proj_z` projections; pass `--target-modules` explicitly (see `experiments/qwopus3.5-9b-v3.5/README.md`).

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=aft --cov-report=term-missing

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## License

MIT
