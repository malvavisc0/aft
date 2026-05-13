# aft — Aria Finetuner

**Standalone fine-tuning and GPTQ quantization pipeline for instruction-following models.**

`aft` takes a base model from HuggingFace, fine-tunes it with QLoRA (4-bit NF4), merges the adapter, quantizes to GPTQ Int4, and optionally publishes the result to HuggingFace Hub — all from a single CLI.

## Features

- **QLoRA SFT** — 4-bit NF4 training with LoRA adapters on all attention + MLP layers
- **Automatic parameter tuning** — detects your GPU and model size, then recommends rank, learning rate, batch size, and sequence length
- **Dataset cleaning** — whitespace normalization, special-character filtering, token-length bounds, language detection, and deduplication
- **GPTQ quantization** — Int4 quantization via GPTQModel, producing vLLM-ready checkpoints
- **Hub integration** — push quantized models directly to HuggingFace Hub
- **Modular pipeline** — run the full stack or individual phases (train, merge, quantize, push)
- **Resumable runs** — automatically detects completed phases and skips them with `--resume`

## Requirements

- Python ≥ 3.12
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

## Commands

### `aft recommend`

Detects your GPU hardware, fetches model metadata from HuggingFace, and outputs recommended QLoRA hyperparameters with reasoning.

```bash
aft recommend --model Qwen/Qwen2.5-7B
aft recommend --model Qwen/Qwen2.5-7B --token hf_xxxx
```

### `aft run`

Runs the full pipeline: QLoRA SFT → LoRA merge → GPTQ quantization.

```bash
aft run \
    --model Qwen/Qwen2.5-7B \
    --dataset teknium/OpenHermes-2.5 \
    --run-name my-run \
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
| `--gptq-bits` | `4` | Quantization bit-width |
| `--gptq-group-size` | `128` | GPTQ group size |
| `--calibration` | `wikitext2` | Calibration dataset (`wikitext2` or path to JSONL) |

#### Pipeline control

| Flag | Description |
|------|-------------|
| `--skip-finetune` | Skip SFT, merge + quantize an existing adapter |
| `--skip-quantize` | Stop after merge (produces fp16 model) |
| `--resume` | Auto-detect completed phases and skip them |

### `aft quantize`

Quantize an already-merged fp16 model to GPTQ Int4.

```bash
aft quantize \
    --model ./models/my-run/merged \
    --output ./models/my-run/gptq-int4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bits` | `4` | Quantization bit-width |
| `--group-size` | `128` | GPTQ group size |
| `--desc-act` | off | Use activation order (slower, better quality) |
| `--calibration` | `wikitext2` | Calibration dataset |
| `--n-calibration-samples` | `128` | Number of calibration samples |
| `--calibration-seq-len` | `2048` | Calibration sequence length |
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

`aft` auto-detects the dataset format when loading from HuggingFace:

| Column | Handling |
|--------|----------|
| `text` | Used directly as training text |
| `conversations` | Chat-format rows flattened to `role: content` lines |
| Other | Falls back to the first column |

Multiple datasets can be passed as a comma-separated list:

```bash
aft run --dataset teknium/OpenHermes-2.5,Open-Orca/SlimOrca --model ... --run-name ...
```

## Supported Architectures

LoRA adapters target `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) and `gate_proj`, `up_proj`, `down_proj` (MLP). This covers models that use these layer names, including:

- **Llama** family (Llama 2, Llama 3, Code Llama)
- **Qwen** family (Qwen 2, Qwen 2.5)
- **Mistral** / Mixtral
- **Gemma** / Gemma 2

Models with different layer names (e.g. Mamba, RWKV) will not match these targets and are not currently supported without changes.

## Serving with vLLM

After quantization, the GPTQ output directory is ready for vLLM:

```bash
vllm serve ./models/my-run/gptq-int4 --quantization gptq_marlin
```

The `gptq_marlin` kernel provides near-native inference speed for GPTQ-quantized models.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models and Hub uploads |
| `LOGURU_LEVEL` | Set to `DEBUG` for verbose pipeline logging (default: `INFO`) |

## Output Structure

`--output` is the base directory; `--run-name` creates a subdirectory under it.

```
<output>/<run-name>/
├── adapter/          # LoRA adapter weights
├── checkpoints/      # Training checkpoints
├── merged/           # Merged fp16 model
└── gptq-int4/        # GPTQ quantized model (vLLM-ready)
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
- **Calibration flexibility** — GPTQ supports arbitrary calibration datasets (default: wikitext-2), allowing you to tune quantization quality for your domain.
- **Active development** — GPTQModel (used by this tool) is actively maintained with broad model architecture support.

### Why these LoRA target modules?

The adapter targets `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) plus `gate_proj`, `up_proj`, `down_proj` (MLP). This covers both the attention and feed-forward layers, which is more comprehensive than attention-only targeting. The MLP layers often carry domain-specific knowledge, making them important for instruction-following fine-tuning.

### Memory estimation

The 4-bit memory footprint is estimated as:

```
base_weights_gib ≈ params_b × 0.55
```

The `0.55` factor accounts for the NF4 quantization (4 bits/param) plus safetensors header overhead and embedding layers that may use higher precision. The remaining VRAM is split between optimizer states, gradients, and activation memory.

## Known Limitations

- **Hard-coded target modules** — LoRA targets `q/k/v/o_proj + gate/up/down_proj`, which covers Llama and Qwen-style architectures. Models with different layer names (e.g. Mamba, RWKV) will need manual `target_modules` configuration.
- **Single-GPU training only** — No FSDP or DeepSpeed support. Multi-GPU is limited to data-parallel via `device_map="auto"`.
- **No eval/validation during training** — Training runs without a validation set. Metrics are training loss only.
- **No early stopping** — The pipeline trains for the specified number of epochs without monitoring validation loss.

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
