"""Architecture introspection and model loading helpers.

These helpers are what stand between the pipeline and *silently* wrong
artifacts (a merged model that loads at the wrong dtype, a multimodal
checkpoint loaded with the wrong AutoModel class, a LoRA adapter that
touches nothing).  They fail loudly rather than degrade silently.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from aft.errors import AftError

# ── Constants ─────────────────────────────────────────────────────────────

#: Files that are part of the model's input contract but are not weights.
#: They must follow the model through merge and quantization or the output
#: directory will not load.
AUXILIARY_FILES: tuple[str, ...] = (
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "video_preprocessor_config.json",
)

#: Module name suffixes that are commonly quantizable linear projections.
#: Used only as a *filter* over modules actually present on the model, never
#: as a hard-coded assumption about the architecture.
LORA_CANDIDATE_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "gate_up_proj",
)


# ── Dtype resolution ──────────────────────────────────────────────────────


def resolve_dtype(hf_config: Any) -> torch.dtype:
    """Resolve the checkpoint dtype from a HuggingFace config.

    Handles all three shapes seen in the wild:

    * the modern top-level ``dtype`` key,
    * the legacy ``torch_dtype`` key,
    * a nested ``text_config`` on multimodal models,

    and coerces string values (e.g. ``"bfloat16"``) into real ``torch.dtype``
    objects.

    Raises:
        AftError: When no dtype can be resolved.  Loading a model at the
            wrong dtype silently corrupts the weights, so we refuse to guess.
    """
    candidates: list[Any] = [
        getattr(hf_config, "dtype", None),
        getattr(hf_config, "torch_dtype", None),
    ]
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        candidates.append(getattr(text_config, "dtype", None))
        candidates.append(getattr(text_config, "torch_dtype", None))

    for value in candidates:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            resolved = getattr(torch, value.replace("torch.", ""), None)
            if isinstance(resolved, torch.dtype):
                return resolved

    raise AftError(
        "Could not resolve checkpoint dtype from config.\n"
        "  Checked: dtype, torch_dtype, text_config.dtype,"
        " text_config.torch_dtype.\n"
        "  Loading at the wrong dtype silently corrupts weights,"
        " so we refuse to guess.\n"
        "  Pass --dtype explicitly or fix the model config."
    )


def dtype_bytes(dtype: torch.dtype) -> int:
    """Bytes per parameter for a torch dtype.

    Uses ``torch.finfo`` for floating-point and ``torch.iinfo`` for integer
    types, so int8 correctly yields 1 (not a silent fallback of 2).
    """
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


# ── Architecture detection ────────────────────────────────────────────────


def is_multimodal(hf_config: Any) -> bool:
    """Whether the config describes a conditional-generation (multimodal) model."""
    architectures = getattr(hf_config, "architectures", None) or []
    if any(str(a).endswith("ForConditionalGeneration") for a in architectures):
        return True
    return getattr(hf_config, "vision_config", None) is not None


def auto_model_class(hf_config: Any) -> Any:
    """Pick the right AutoModel class for the architecture.

    ``AutoModelForCausalLM`` is wrong for ``*ForConditionalGeneration``
    checkpoints: it either raises or silently discards the vision tower.
    """
    if is_multimodal(hf_config):
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


# ── Config / processor / tokenizer loading ────────────────────────────────


def load_config(
    model_ref: str,
    *,
    trust_remote_code: bool,
    token: str | None,
    revision: str | None = None,
) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        token=token,
        revision=revision,
    )


def load_processor(
    model_ref: str,
    *,
    multimodal: bool,
    trust_remote_code: bool,
    token: str | None,
    revision: str | None = None,
) -> Any:
    """Load an ``AutoProcessor`` for multimodal models, else an ``AutoTokenizer``.

    Multimodal checkpoints ship ``preprocessor_config.json`` /
    ``processor_config.json``; loading only the tokenizer drops the vision
    path and produces an output directory that cannot be served.

    For multimodal models, a failure to load ``AutoProcessor`` is raised —
    silently falling back to the tokenizer would produce a broken artifact.
    For text-only models, ``AutoTokenizer`` is the correct loader and
    ``AutoProcessor`` is never attempted.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = dict(
        trust_remote_code=trust_remote_code,
        token=token,
        revision=revision,
    )
    if multimodal:
        from transformers import AutoProcessor

        try:
            return AutoProcessor.from_pretrained(model_ref, **kwargs)
        except Exception as exc:
            raise AftError(
                f"AutoProcessor failed to load for multimodal model"
                f" '{model_ref}': {exc}\n"
                f"  The vision path cannot be silently dropped — the"
                f" output artifact would be unloadable for serving."
            ) from exc
    return AutoTokenizer.from_pretrained(model_ref, fix_mistral_regex=True, **kwargs)


def tokenizer_of(processor: Any) -> Any:
    """Return the tokenizer belonging to a processor (or the processor itself)."""
    return getattr(processor, "tokenizer", processor)


def load_model_inputs(
    model_ref: str,
    *,
    trust_remote_code: bool,
    token: str | None,
    revision: str | None = None,
) -> tuple[Any, bool, Any, Any]:
    """Load config, detect multimodality, and load the processor/tokenizer.

    Consolidates the config → multimodal → processor → tokenizer sequence
    that was duplicated across ``train``, ``merge_adapter``, and ``quantize``.

    Returns:
        ``(hf_config, multimodal, processor, tokenizer)``
    """
    hf_config = load_config(
        model_ref,
        trust_remote_code=trust_remote_code,
        token=token,
        revision=revision,
    )
    multimodal = is_multimodal(hf_config)
    processor = load_processor(
        model_ref,
        multimodal=multimodal,
        trust_remote_code=trust_remote_code,
        token=token,
        revision=revision,
    )
    tokenizer = tokenizer_of(processor)
    return hf_config, multimodal, processor, tokenizer


# ── LoRA target discovery ─────────────────────────────────────────────────


def _linear_leaf_names(model: torch.nn.Module) -> set[str]:
    """Names of 2-D-weight leaf modules (the quantizable linear projections)."""
    names: set[str] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        if len(list(module.children())) > 0:
            continue
        names.add(name.split(".")[-1])
    return names


def discover_lora_targets(
    model: torch.nn.Module, explicit: list[str] | None = None
) -> list[str]:
    """Determine LoRA target module names for the *actual* loaded model.

    Rather than assuming Llama/Qwen layer names, this inspects the module tree
    and keeps only candidate projection names that really exist.  Fails closed
    with an actionable error when nothing matches, instead of silently
    training an adapter that touches almost nothing.
    """
    if explicit:
        logger.info("Using explicit LoRA target modules: {}", ", ".join(explicit))
        return explicit

    leaf_names = _linear_leaf_names(model)
    targets = sorted(n for n in leaf_names if n in LORA_CANDIDATE_SUFFIXES)
    if not targets:
        raise AftError(
            "Could not discover any LoRA target modules on this model.\n"
            f"  Linear leaf module names found: "
            f"{', '.join(sorted(leaf_names)[:20]) or '(none)'}\n"
            "  This architecture is not covered by the default projection\n"
            "  names. Pass --target-modules with an explicit comma-separated\n"
            "  list to proceed."
        )

    missing = sorted(set(LORA_CANDIDATE_SUFFIXES) - leaf_names)
    logger.info("Discovered LoRA target modules: {}", ", ".join(targets))
    logger.debug("Candidate names not present on this model: {}", ", ".join(missing))
    return targets


# ── Auxiliary file copying ────────────────────────────────────────────────


def copy_auxiliary_files(source: str | Path, destination: Path) -> list[str]:
    """Copy processor/chat-template files from a local model dir to the output.

    ``save_pretrained`` on the model and tokenizer does not emit these, so
    without this step a multimodal quantized artifact is unloadable.
    """
    src = Path(source)
    if not src.is_dir():
        return []

    copied: list[str] = []
    for name in AUXILIARY_FILES:
        candidate = src / name
        if candidate.is_file() and not (destination / name).exists():
            shutil.copy2(candidate, destination / name)
            copied.append(name)
    if copied:
        logger.info(
            "Copied auxiliary files to {}: {}",
            destination,
            ", ".join(copied),
        )
    return copied


# ── Safetensors shard resolution ──────────────────────────────────────────


def _checkpoint_key_candidates(name: str) -> list[str]:
    """Names to try against on-disk checkpoint keys, most specific first.

    Some loaders (e.g. GPTQModel) wrap the HF model in an extra ``.model``
    attribute, so a live parameter's qualified name can carry one more
    leading ``model.`` segment than the key actually stored in the
    safetensors checkpoint. Try the exact name first, then progressively
    strip leading ``model.`` segments.
    """
    candidates = [name]
    stripped = name
    while stripped.startswith("model."):
        stripped = stripped[len("model.") :]
        candidates.append(stripped)
    return candidates


def shards_for(model_path: Path, needed: set[str]) -> list[Path]:
    """Resolve which shard files hold ``needed`` tensors, via the index map.

    A bare ``glob("*.safetensors")`` misses nested layouts and forces every
    shard to be opened.  ``model.safetensors.index.json`` is authoritative, so
    prefer it and only fall back to globbing when it is absent.  A *malformed*
    index is raised on — if the file exists but is corrupt, something is
    seriously wrong with the checkpoint.
    """
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except json.JSONDecodeError as exc:
            raise AftError(
                f"Malformed safetensors index: {index_path}\n"
                f"  The file exists but could not be parsed as JSON.\n"
                f"  This usually means the checkpoint is corrupted."
            ) from exc
        shard_names: set[str] = set()
        missing: set[str] = set()
        for name in needed:
            for cand in _checkpoint_key_candidates(name):
                if cand in weight_map:
                    shard_names.add(weight_map[cand])
                    break
            else:
                missing.add(name)
        if missing:
            logger.warning(
                "{} tensor(s) absent from the safetensors index (e.g. {})",
                len(missing),
                ", ".join(sorted(missing)[:4]),
            )
        return sorted(model_path / n for n in shard_names)

    return sorted(model_path.rglob("*.safetensors"))


# ── Meta-device materialization ───────────────────────────────────────────


def _install_meta_tensor(
    model: torch.nn.Module,
    name: str,
    old_tensor: torch.Tensor,
    found: dict[str, torch.Tensor],
    tied: dict[int, torch.Tensor],
    as_parameter: bool,
) -> int:
    """Replace a single meta tensor with its checkpoint value.

    Returns 1 if a new tensor was materialized (storage not previously seen),
    0 if reusing a tied storage.
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    storage_key = id(found[name])
    value = tied.get(storage_key)
    count = 0
    if value is None:
        target_dtype = (
            old_tensor.dtype if old_tensor.dtype != torch.float32 else found[name].dtype
        )
        value = found[name].to(dtype=target_dtype)
        tied[storage_key] = value
        count = 1

    if as_parameter:
        setattr(
            parent,
            parts[-1],
            torch.nn.Parameter(value, requires_grad=old_tensor.requires_grad),
        )
    else:
        parent.register_buffer(parts[-1], value)
    return count


def _recompute_rope_buffers(model: torch.nn.Module) -> int:
    """Recompute rotary-embedding ``inv_freq`` buffers left on the meta device.

    These are never stored in checkpoints — every HF rotary embedding module
    computes them at ``__init__``, but loaders that construct the model
    lazily/on the meta device skip that computation. Two shapes exist across
    HF architectures:

    * Text/"config-driven" rotary modules (``self.config``, ``self.rope_type``):
      ``inv_freq`` comes from ``ROPE_INIT_FUNCTIONS[rope_type](config, device)``,
      or ``compute_default_rope_parameters`` for ``rope_type == "default"``.
    * Standalone vision rotary modules (``self.dim``, ``self.theta``, no
      config): ``inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)``.
    """
    fixed = 0
    for name, module in model.named_modules():
        buf = getattr(module, "inv_freq", None)
        if buf is None or not buf.is_meta:
            continue

        config = getattr(module, "config", None)
        if config is not None:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_params = getattr(config, "rope_parameters", {})
            rope_type = rope_params.get("rope_type", "default")
            rope_init_fn = (
                module.compute_default_rope_parameters
                if rope_type == "default"
                else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, attention_scaling = rope_init_fn(config, torch.device("cpu"))
            module.attention_scaling = attention_scaling
        elif hasattr(module, "dim") and hasattr(module, "theta"):
            dim, theta = module.dim, module.theta
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
            )
        else:
            raise AftError(
                f"'{name}.inv_freq' is on the meta device and this module has"
                " neither config/rope_type nor dim/theta to recompute it from."
            )

        module.register_buffer("inv_freq", inv_freq, persistent=False)
        if hasattr(module, "original_inv_freq"):
            module.register_buffer(
                "original_inv_freq", inv_freq.clone(), persistent=False
            )
        fixed += 1
    return fixed


def _load_meta_tensors(needed: set[str], shards: list[Path]) -> dict[str, torch.Tensor]:
    """Read the named tensors from safetensors shards, with key-alias fallback."""
    from safetensors.torch import load_file

    found: dict[str, torch.Tensor] = {}
    remaining = set(needed)
    for shard in shards:
        if not remaining:
            break
        weights = load_file(str(shard), device="cpu")
        for name in list(remaining):
            for cand in _checkpoint_key_candidates(name):
                if cand in weights:
                    found[name] = weights[cand]
                    break
        remaining -= found.keys()
    return found


def _install_meta_tensors(
    model: torch.nn.Module,
    meta_params: dict[str, torch.Tensor],
    meta_buffers: dict[str, torch.Tensor],
    found: dict[str, torch.Tensor],
) -> int:
    """Install loaded weights for every meta param/buffer, preserving ties."""
    tied: dict[int, torch.Tensor] = {}
    materialized = 0
    for name, param in meta_params.items():
        materialized += _install_meta_tensor(
            model, name, param, found, tied, as_parameter=True
        )
    for name, buffer in meta_buffers.items():
        materialized += _install_meta_tensor(
            model, name, buffer, found, tied, as_parameter=False
        )
    return materialized


def _collect_meta(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``(meta_params, meta_buffers)`` still on the meta device."""
    meta_params = {name: p for name, p in model.named_parameters() if p.is_meta}
    meta_buffers = {name: b for name, b in model.named_buffers() if b.is_meta}
    return meta_params, meta_buffers


def _resolve_meta_weights(
    model_path: Path, all_names: list[str]
) -> dict[str, torch.Tensor]:
    """Locate and load the named tensors from the checkpoint shards."""
    shards = shards_for(model_path, set(all_names))
    if not shards:
        raise AftError(
            f"{len(all_names)} tensors are still on the meta device but"
            f" no safetensors shards were found in {model_path}.\n"
            "  Quantizing now would silently produce garbage weights."
        )
    found = _load_meta_tensors(set(all_names), shards)
    missing = set(all_names) - found.keys()
    if missing:
        raise AftError(
            f"{len(missing)} meta tensor(s) have no value in the checkpoint:\n"
            f"  {', '.join(sorted(missing)[:8])}\n"
            "  Refusing to quantize an incompletely materialized model."
        )
    return found


def materialize_meta_params(model: torch.nn.Module, model_path: Path) -> int:
    """Load real weights for any parameters/buffers stuck on the meta device.

    GPTQModel's loader does not materialize every architecture (sparse MoE
    experts in particular).  This resolves the leftover meta tensors through
    the safetensors *index map*, covers buffers as well as parameters, and
    preserves tied-weight identity so that two names pointing at one storage
    stay tied after replacement.  Rotary-embedding ``inv_freq`` buffers are
    never in the checkpoint and are recomputed directly instead.

    Returns the number of tensors that were materialized.
    """
    _recompute_rope_buffers(model)

    meta_params, meta_buffers = _collect_meta(model)
    if not meta_params and not meta_buffers:
        return 0

    all_names = list(meta_params) + list(meta_buffers)
    logger.warning(
        "Found {} parameters and {} buffers on the meta device after load: {}",
        len(meta_params),
        len(meta_buffers),
        ", ".join(all_names[:8]) + ("..." if len(all_names) > 8 else ""),
    )

    found = _resolve_meta_weights(model_path, all_names)
    materialized = _install_meta_tensors(model, meta_params, meta_buffers, found)

    remaining = [n for n, p in model.named_parameters() if p.is_meta]
    if remaining:
        raise AftError(
            f"{len(remaining)} parameter(s) are still on the meta device"
            f" after materialization: {', '.join(remaining[:8])}"
        )

    logger.info("Materialized {} meta tensors from checkpoint", materialized)
    return materialized


# ── Version reporting ─────────────────────────────────────────────────────


def library_versions() -> dict[str, str]:
    """Collect version strings for provenance, using importlib.metadata."""
    import platform

    from aft import __version__

    versions: dict[str, str] = {
        "aft": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for mod in ("transformers", "gptqmodel"):
        try:
            versions[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            versions[mod] = "unknown"
    return versions
