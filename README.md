# aft

**Standalone fine-tuning and GPTQ quantization pipeline for instruction-following models.**

`aft` takes a base model from HuggingFace, fine-tunes it with QLoRA (4-bit NF4), merges the adapter, quantizes to GPTQ Int4, and optionally publishes the result to HuggingFace Hub — all from a single CLI.

## Features

- **QLoRA SFT** — 4-bit NF4 training with LoRA adapters on all attention + MLP layers
- **Automatic parameter tuning** — detects your GPU and model size, then recommends rank, learning rate, batch size, and sequence length
- **Dataset cleaning** — whitespace normalization, special-character filtering, token-length bounds, language detection, and deduplication
- **GPTQ quantization** — Int4 quantization via GPTQModel, producing vLLM-ready checkpoints
- **Hub integration** — push quantized models directly to HuggingFace Hub
- **Modular pipeline** — run the full stack or individual phases (train, merge, quantize, push)

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
    --output ./models/my-run

# 3. Push the quantized model to HuggingFace Hub
aft push \
    --model ./models/my-run/gptq-int4 \
    --repo-id myorg/my-model-gptq-int4
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
    --output ./models/my-run
```

#### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `2 × lora-rank` | LoRA alpha |
| `--max-seq-len` | `2048` | Maximum sequence length |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--max-samples` | all | Limit number of training samples |

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
| `--calibration` | `wikitext2` | Calibration dataset |
| `--n-calibration-samples` | `128` | Number of calibration samples |

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models and Hub uploads |

## Output Structure

```
models/<run-name>/
├── adapter/          # LoRA adapter weights
├── checkpoints/      # Training checkpoints
├── merged/           # Merged fp16 model
└── gptq-int4/        # GPTQ quantized model (vLLM-ready)
```

## Serving with vLLM

After quantization, serve the model with vLLM:

```bash
vllm serve ./models/my-run/gptq-int4 --quantization gptq_marlin
```

## License

MIT