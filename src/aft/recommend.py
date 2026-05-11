"""Hardware detection and QLoRA parameter recommendation."""

import json
import re
import subprocess

from aft.config import ModelInfo, Recommendation


def detect_gpus() -> list[dict]:
    """Detect NVIDIA GPUs via nvidia-smi. Returns list of {name, vram_mib}."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(", ")
            if len(parts) == 2:
                gpus.append({"name": parts[0], "vram_mib": int(parts[1])})
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_total_vram_mib() -> int:
    """Get total VRAM across all GPUs in MiB."""
    return sum(g["vram_mib"] for g in detect_gpus())


def detect_system_ram_mib() -> int:
    """Detect total system RAM in MiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (FileNotFoundError, ValueError):
        pass
    return 0


def fetch_model_info(repo_id: str, token: str | None = None) -> ModelInfo:
    """Fetch model metadata from HuggingFace Hub.

    Uses the HF API to retrieve parameter count and architecture info.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    info = api.model_info(repo_id)

    params_b = 0.0
    if info.safetensors and info.safetensors.total:
        params_b = info.safetensors.total / 1e9

    architectures = info.config.get("architectures", []) if info.config else []
    model_type = info.config.get("model_type", "unknown") if info.config else "unknown"
    hidden_size = info.config.get("hidden_size") if info.config else None
    num_layers = info.config.get("num_hidden_layers") if info.config else None

    if params_b == 0.0:
        try:
            config_path = hf_hub_download(repo_id, "config.json", token=token)
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", model_type)
            hidden_size = cfg.get("hidden_size", hidden_size)
            num_layers = cfg.get("num_hidden_layers", num_layers)
            architectures = cfg.get("architectures", architectures)
            # Estimate params from architecture dimensions
            if hidden_size and num_layers:
                vocab_size = cfg.get("vocab_size", 32000)
                intermediate = cfg.get("intermediate_size", hidden_size * 4)
                # embed + per-layer (attn + mlp + norms) + final head
                params_b = (
                    vocab_size * hidden_size  # embeddings
                    + num_layers
                    * (
                        4 * hidden_size * hidden_size  # attention QKV+O
                        + 3 * hidden_size * intermediate  # MLP gate/up/down
                        + 2 * hidden_size  # layer norms
                    )
                    + vocab_size * hidden_size  # lm_head
                ) / 1e9
        except Exception:
            pass

    # Last resort: parse size from repo name (e.g. "9B", "7b", "70B")
    if params_b == 0.0:
        match = re.search(r"(\d+(?:\.\d+)?)[Bb]", repo_id)
        if match:
            params_b = float(match.group(1))

    return ModelInfo(
        repo_id=repo_id,
        params_b=params_b,
        model_type=model_type,
        architectures=architectures,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


def recommend(
    model_info: ModelInfo,
    vram_mib: int,
    ram_mib: int,
    bf16_supported: bool,
) -> Recommendation:
    """Compute recommended QLoRA SFT parameters for the given hardware.

    Args:
        model_info: Model metadata from :func:`fetch_model_info`.
        vram_mib: Total GPU VRAM in MiB.
        ram_mib: Total system RAM in MiB.
        bf16_supported: Whether the GPU supports BF16.

    Returns:
        A :class:`Recommendation` with hyper-parameters and reasoning.
    """
    reasoning: list[str] = []
    params_b = model_info.params_b
    vram_gib = vram_mib / 1024
    ram_gib = ram_mib / 1024

    reasoning.append(f"Model ~{params_b:.1f}B params on {vram_gib:.1f} GiB VRAM")

    # ── Size-based LoRA + LR heuristics ────────────────────────────────
    if params_b < 3:
        lora_rank, lr, epochs = 8, 2e-4, 3
        reasoning.append(
            f"Small model (<3B) → rank {lora_rank}, lr {lr}, {epochs} epochs"
        )
    elif params_b < 8:
        lora_rank, lr, epochs = 16, 2e-4, 2
        reasoning.append(
            f"Medium model (3-8B) → rank {lora_rank}, lr {lr}, {epochs} epochs"
        )
    elif params_b < 20:
        lora_rank, lr, epochs = 32, 1e-4, 2
        reasoning.append(
            f"Large model (8-20B) → rank {lora_rank}, lr {lr}, {epochs} epochs"
        )
    elif params_b < 70:
        lora_rank, lr, epochs = 64, 5e-5, 1
        reasoning.append(
            f"Very large model (20-70B) → rank {lora_rank}, lr {lr}, {epochs} epoch"
        )
    else:
        lora_rank, lr, epochs = 64, 2e-5, 1
        reasoning.append(
            f"Massive model (>70B) → rank {lora_rank}, lr {lr}, {epochs} epoch"
        )

    lora_alpha = lora_rank * 2

    # ── QLoRA 4-bit memory estimation ──────────────────────────────────
    base_weights_gib = params_b * 0.55
    overhead_gib = 2.0

    reasoning.append(f"QLoRA 4-bit base weights ≈ {base_weights_gib:.1f} GiB")

    # ── Seq len + batch size tuning ────────────────────────────────────
    available_gib = vram_gib * 0.85
    remaining_gib = available_gib - base_weights_gib - overhead_gib

    if remaining_gib < 1.0:
        max_seq_len = 512
        batch_size = 1
        max_memory = {
            "0": f"{int(vram_gib * 0.9)}GiB",
            "cpu": f"{int(ram_gib * 0.8)}GiB",
        }
        reasoning.append(
            f"⚠ VRAM very tight ({remaining_gib:.1f} GiB after weights) → "
            f"seq_len={max_seq_len}, batch=1, CPU offload enabled"
        )
    elif remaining_gib < 4.0:
        max_seq_len = 1024
        batch_size = 1
        max_memory = None
        reasoning.append(
            f"VRAM tight ({remaining_gib:.1f} GiB remaining) → "
            f"seq_len={max_seq_len}, batch={batch_size}"
        )
    elif remaining_gib < 10.0:
        max_seq_len = 2048
        batch_size = 1
        max_memory = None
        reasoning.append(
            f"VRAM moderate ({remaining_gib:.1f} GiB remaining) → "
            f"seq_len={max_seq_len}, batch={batch_size}"
        )
    else:
        max_seq_len = 2048
        batch_size = 2
        max_memory = None
        reasoning.append(
            f"VRAM comfortable ({remaining_gib:.1f} GiB remaining) → "
            f"seq_len={max_seq_len}, batch={batch_size}"
        )

    # ── Gradient accumulation → target effective batch ≈ 16 ────────────
    target_effective = 16
    grad_accum = max(1, target_effective // batch_size)
    reasoning.append(
        f"Effective batch size: {batch_size} × {grad_accum} = "
        f"{batch_size * grad_accum} (target ~{target_effective})"
    )

    if bf16_supported:
        reasoning.append("BF16 supported ✓ — will use bf16 compute")

    return Recommendation(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        epochs=epochs,
        max_memory=max_memory,
        reasoning=reasoning,
    )
