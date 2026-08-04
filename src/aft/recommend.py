"""Hardware detection and QLoRA parameter recommendation."""

import math
import re
from typing import Any

from loguru import logger

from aft.config import LORA_ALPHA_MULTIPLIER, ModelInfo, Recommendation


def detect_system_ram_mib() -> int | None:
    """Detect total system RAM in MiB from /proc/meminfo.

    Returns ``None`` when detection fails (non-Linux, permissions, corrupt
    content). Callers must handle ``None`` explicitly — returning ``0`` would
    silently disable RAM-based feasibility checks.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (FileNotFoundError, ValueError):
        pass
    return None


#: Bytes per parameter for the dtype names HF reports in ``config.dtype``.
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "int8": 1,
}


def checkpoint_size_gib(params: float, dtype_bytes: int) -> float:
    """Estimate checkpoint size in GiB from parameter count and dtype.

    Shared by :func:`recommend` and
    :func:`aft.pipeline._warn_if_merge_wont_fit` so both use the same math.
    """
    return params * dtype_bytes / 1024**3


def _config_get(
    config: dict[str, Any], key: str, default: Any = None
) -> Any:
    """Read ``key`` from a HF config, falling back to a nested ``text_config``.

    Multimodal models (e.g. ``*ForConditionalGeneration``) put the language
    model hyper-parameters under ``text_config`` rather than at the top level.
    """
    if not config:
        return default
    if key in config and config[key] is not None:
        return config[key]
    text_config = config.get("text_config") or {}
    value = text_config.get(key)
    return value if value is not None else default


def fetch_model_info(repo_id: str, token: str | None = None) -> ModelInfo:
    """Fetch model metadata from HuggingFace Hub.

    Uses the HF API to retrieve parameter count and architecture info,
    including hybrid ``layer_types`` and MoE expert counts, both of which
    determine whether this pipeline can handle the model at all.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.model_info(repo_id)

    # ``safetensors.total`` is a *parameter count*, not a byte count.
    params_b = 0.0
    if info.safetensors and info.safetensors.total:
        params_b = info.safetensors.total / 1e9

    config = info.config or {}
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "unknown")
    hidden_size = _config_get(config, "hidden_size")
    num_layers = _config_get(config, "num_hidden_layers")
    num_experts = _config_get(config, "num_experts")

    raw_layer_types = _config_get(config, "layer_types") or []
    # Preserve first-seen order while de-duplicating.
    layer_types = list(dict.fromkeys(raw_layer_types))

    dtype_name = _config_get(config, "dtype") or _config_get(
        config, "torch_dtype"
    )
    dtype_bytes = _DTYPE_BYTES.get(str(dtype_name), 2)

    if params_b == 0.0:
        # Fallback: parse size from repo name (e.g. "9B", "7b", "70B").
        # This is a guess — warn so the user knows the count is not from
        # safetensors metadata and may be wrong (e.g. "2Bit-7B" → "2B").
        match = re.search(r"(\d+(?:\.\d+)?)[Bb]", repo_id)
        if match:
            params_b = float(match.group(1))
            logger.warning(
                "Parameter count not in safetensors metadata;"
                " guessed {:.1f}B from repo name '{}' — this may be"
                " inaccurate",
                params_b,
                repo_id,
            )

    return ModelInfo(
        repo_id=repo_id,
        params_b=params_b,
        model_type=model_type,
        architectures=architectures,
        hidden_size=hidden_size,
        num_layers=num_layers,
        layer_types=layer_types,
        num_experts=num_experts,
        dtype_bytes=dtype_bytes,
        revision=getattr(info, "sha", None),
    )


#: Data-driven size tiers: (max_params_b, lora_rank, lr, epochs, label).
#: Replaces a long if/elif chain that repeated the same assign+reasoning
#: pattern in every branch.
_TIERS: list[tuple[float, int, float, int, str]] = [
    (3, 8, 2e-4, 3, "Small model (<3B)"),
    (8, 16, 2e-4, 2, "Medium model (3-8B)"),
    (20, 32, 1e-4, 2, "Large model (8-20B)"),
    (70, 64, 5e-5, 1, "Very large model (20-70B)"),
    (math.inf, 64, 2e-5, 1, "Massive model (>70B)"),
]


def recommend(
    model_info: ModelInfo,
    vram_mib: int,
    ram_mib: int | None,
    bf16_supported: bool,
    gpu_vram_mib: list[int] | None = None,
) -> Recommendation:
    """Compute recommended QLoRA SFT parameters for the given hardware.

    Args:
        model_info: Model metadata from :func:`fetch_model_info`.
        vram_mib: Total GPU VRAM in MiB.
        ram_mib: Total system RAM in MiB, or ``None`` if undetectable.
        bf16_supported: Whether the GPU supports BF16.
        gpu_vram_mib: Per-GPU VRAM list in MiB. When provided,
            ``max_memory`` will set per-device limits for each GPU.

    Returns:
        A :class:`Recommendation` with hyper-parameters and reasoning.
    """
    reasoning: list[str] = []
    params_b = model_info.params_b
    vram_gib = vram_mib / 1024
    ram_gib = ram_mib / 1024 if ram_mib is not None else 0

    reasoning.append(
        f"Model ~{params_b:.1f}B params on {vram_gib:.1f} GiB VRAM"
    )

    # ── Architecture warnings ──────────────────────────────────────────
    checkpoint_gib = checkpoint_size_gib(
        params_b * 1e9, model_info.dtype_bytes
    )
    reasoning.append(
        f"Checkpoint ≈ {checkpoint_gib:.0f} GiB"
        f" ({model_info.dtype_bytes} bytes/param);"
        f" a CPU-side merge needs at least that much system RAM"
    )
    if ram_mib is None:
        reasoning.append(
            "⚠ Could not detect system RAM — merge feasibility"
            " cannot be assessed"
        )
    elif checkpoint_gib > ram_gib:
        reasoning.append(
            f"⚠ Checkpoint ({checkpoint_gib:.0f} GiB) exceeds system RAM"
            f" ({ram_gib:.0f} GiB) — the merge phase will not fit"
        )
    if model_info.num_experts:
        reasoning.append(
            f"⚠ Sparse MoE ({model_info.num_experts} experts):"
            f" total-parameter heuristics below are unreliable, and expert"
            f" layers must be verified as quantized"
        )
    if len(model_info.layer_types) > 1:
        reasoning.append(
            f"⚠ Hybrid attention (layer_types: "
            f"{', '.join(model_info.layer_types)}):"
            f" non-Linear modules will not match the default LoRA/GPTQ"
            f" targets"
        )
    if any(
        arch.endswith("ForConditionalGeneration")
        for arch in model_info.architectures
    ):
        reasoning.append(
            "⚠ Multimodal architecture — requires a processor-aware"
            " loader, not AutoModelForCausalLM"
        )

    if gpu_vram_mib and len(gpu_vram_mib) > 1:
        reasoning.append(
            f"Multi-GPU: {len(gpu_vram_mib)} GPUs"
            f" ({', '.join(f'{v / 1024:.0f} GiB' for v in gpu_vram_mib)})"
            f" → model shards via device_map='auto'"
        )

    # ── Size-based LoRA + LR heuristics (data-driven) ──────────────────
    for max_params, lora_rank, lr, epochs, label in _TIERS:
        if params_b < max_params:
            break
    else:  # pragma: no cover - _TIERS ends with inf
        lora_rank, lr, epochs, label = _TIERS[-1][1:]

    epoch_word = "epoch" if epochs == 1 else "epochs"
    reasoning.append(
        f"{label} → rank {lora_rank}, lr {lr}, {epochs} {epoch_word}"
    )

    lora_alpha = lora_rank * LORA_ALPHA_MULTIPLIER

    # ── QLoRA 4-bit memory estimation ──────────────────────────────────
    base_weights_gib = params_b * 0.55
    overhead_gib = 2.0

    reasoning.append(
        f"QLoRA 4-bit base weights ≈ {base_weights_gib:.1f} GiB"
    )

    # ── Seq len + batch size tuning ────────────────────────────────────
    available_gib = vram_gib * 0.85
    remaining_gib = available_gib - base_weights_gib - overhead_gib

    if remaining_gib < 1.0:
        max_seq_len = 512
        batch_size = 1
        max_memory: dict[str, str] = {}
        if gpu_vram_mib:
            for i, vram in enumerate(gpu_vram_mib):
                max_memory[str(i)] = f"{int(vram / 1024 * 0.9)}GiB"
        else:
            max_memory["0"] = f"{int(vram_gib * 0.9)}GiB"
        max_memory["cpu"] = f"{int(ram_gib * 0.8)}GiB"
        reasoning.append(
            f"VRAM very tight ({remaining_gib:.1f} GiB after weights) - "
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