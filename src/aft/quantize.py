"""GPTQ / FP8 quantization helpers.

Extracted from the original monolithic ``pipeline.py`` so each concern
(validation, calibration, coverage reporting, provenance, saving) is a
small, testable function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from aft.cleaning import flatten_row_to_text, resolve_dataset_split
from aft.config import QuantizeConfig
from aft.errors import AftError
from aft.model_utils import (
    copy_auxiliary_files,
    library_versions,
    load_model_inputs,
    resolve_dtype,
)
from aft.ui import console

# Minimum character length for calibration text samples.
MIN_CALIBRATION_TEXT_LEN = 100

#: Maps well-known short names to HuggingFace dataset specs.
#:   ``repo``     — HF dataset repo id (required).
#:   ``field``    — row key holding the raw text or message list (optional;
#:                  falls back through flat/message fields when absent).
#:   ``config``   — HF dataset config name (optional, passed as ``name=``).
#:   ``data_dir`` — load only a subdirectory of a sharded dataset (optional).
#:   ``split``    — split name (optional; defaults to ``train`` with fallback).
#:   ``gated``    — when truthy, the dataset requires agreeing to access terms
#:                  on the HF web page and a ``HF_TOKEN``; a clear error is
#:                  raised on a load failure instead of a raw traceback.
#: Code- and agentic-flavored aliases exist so models can calibrate on a
#: distribution closer to their serving traffic instead of general web text.
_HF_CALIBRATION_DATASETS: dict[str, dict[str, str | None]] = {
    "fineweb": {"repo": "HuggingFaceFW/fineweb", "field": "text"},
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "field": "text"},
    "c4": {"repo": "allenai/c4", "field": "text"},
    # Code calibration from the StarCoder training set. The full dataset is
    # 783 GB and cannot be loaded at once; ``data_dir="python"`` streams one
    # language. It is gated — the user must accept the terms on the dataset
    # page and supply HF_TOKEN, or the load fails.
    "starcoder": {
        "repo": "bigcode/starcoderdata",
        "field": "content",
        "data_dir": "python",
        "gated": True,
    },
    # Agentic tool-use calibration from NVIDIA's Nemotron-Agentic-v1.  The
    # dataset has two named splits (no ``train``): ``interactive_agent``
    # (19k judged trajectories) and ``tool_calling`` (316k).  Rows are
    # conversational (``messages`` list), flattened by ``flatten_row_to_text``.
    # Public (CC-BY-4.0), not gated.
    "nemotron-agentic": {
        "repo": "nvidia/Nemotron-Agentic-v1",
        "field": "messages",
        "split": "interactive_agent",
    },
}


# ── Chat template ─────────────────────────────────────────────────────────


def apply_chat_template(tokenizer: Any, text: str) -> str:
    """Wrap raw calibration text in the model's own chat template.

    Instruction-tuned models see templated input at serving time; calibrating
    on bare web text shifts the activation statistics GPTQ measures.

    If the template fails to apply, the exception propagates — the whole
    point of ``use_chat_template=True`` is that the user *wants* the template
    applied, and silently falling back to raw text would produce wrong
    activation statistics with only a debug-level log.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Calibration data ──────────────────────────────────────────────────────


def _collect_texts(data: Any, n_samples: int, preferred_field: str | None) -> list[str]:
    """Pull up to ``n_samples`` non-empty texts from a streaming dataset."""
    texts: list[str] = []
    for row in data:
        raw = flatten_row_to_text(row, preferred_field)
        if raw is None:
            continue
        text = raw.strip()
        if len(text) > MIN_CALIBRATION_TEXT_LEN:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


def _load_registry_calibration(
    hf_datasets: Any, dataset_name: str, n_samples: int
) -> list[str]:
    """Load a named calibration preset from the registry."""
    spec = _HF_CALIBRATION_DATASETS[dataset_name]
    hf_repo = spec["repo"]
    preferred_field = spec.get("field")
    config_name = spec.get("config")
    data_dir = spec.get("data_dir")
    split = spec.get("split") or "train"
    is_gated = bool(spec.get("gated"))
    where = f"{hf_repo}/{data_dir}" if data_dir else hf_repo
    console.print(
        f"[cyan]Loading {where}"
        f"{f' ({config_name})' if config_name else ''}"
        f" split {split}"
        f" for calibration ({n_samples} samples)...[/cyan]"
    )

    load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config_name:
        load_kwargs["name"] = config_name
    if data_dir:
        load_kwargs["data_dir"] = data_dir
    try:
        data = hf_datasets.load_dataset(hf_repo, **load_kwargs)
    except Exception as exc:
        if is_gated:
            raise AftError(
                f"Could not load gated dataset '{hf_repo}': {exc}\n"
                f"  Gated datasets require you to accept the access"
                f" terms on the dataset's HF page, and to supply a"
                f" token (HF_TOKEN env var or --token).\n"
                f"  Open: https://huggingface.co/datasets/{hf_repo}"
            ) from exc
        raise
    return _collect_texts(data, n_samples, preferred_field)


def _load_jsonl_calibration(path: Path) -> list[str]:
    """Load calibration texts from a local JSONL file."""
    from aft.cleaning import supported_text_columns

    texts: list[str] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            raw = flatten_row_to_text(row, None)
            if raw is None:
                raise AftError(
                    f"Calibration JSONL line {i}: no supported text"
                    f" column in {path}.\n"
                    f"  Expected {supported_text_columns()}.\n"
                    f"  Found keys: {', '.join(list(row)[:10])}"
                )
            texts.append(raw)
        except json.JSONDecodeError as exc:
            raise AftError(
                f"Calibration JSONL line {i}: invalid JSON in {path}"
            ) from exc
    return texts


def _load_hf_calibration(
    hf_datasets: Any, dataset_name: str, n_samples: int
) -> list[str]:
    """Load calibration texts from an arbitrary HF dataset repo id."""
    console.print(
        f"[cyan]'{dataset_name}' is not a known alias or local file"
        f" — trying it as a HuggingFace dataset"
        f" ({n_samples} samples)...[/cyan]"
    )
    split = resolve_dataset_split(dataset_name, "train")
    data = hf_datasets.load_dataset(dataset_name, split=split, streaming=True)
    return _collect_texts(data, n_samples, None)


def _resolve_calibration_texts(
    tokenizer: Any,
    dataset_name: str,
    n_samples: int,
) -> list[str]:
    """Resolve a calibration source name into raw text samples."""
    import datasets as hf_datasets

    if dataset_name in _HF_CALIBRATION_DATASETS:
        return _load_registry_calibration(hf_datasets, dataset_name, n_samples)

    p = Path(dataset_name)
    if p.exists():
        return _load_jsonl_calibration(p)
    return _load_hf_calibration(hf_datasets, dataset_name, n_samples)


def _apply_chat_templates(tokenizer: Any, texts: list[str]) -> list[str]:
    """Wrap each text in the model's chat template, when available."""
    if not getattr(tokenizer, "chat_template", None):
        logger.warning(
            "use_chat_template requested but the tokenizer has"
            " no chat_template; calibrating on raw text"
        )
        return texts
    console.print("[cyan]Applying the model's chat template.[/cyan]")
    return [apply_chat_template(tokenizer, t) for t in texts]


def _tokenize_calibration(
    tokenizer: Any, texts: list[str], seq_len: int
) -> list[dict[str, Any]]:
    """Tokenize texts into per-sample input tensors."""
    samples: list[dict[str, Any]] = []
    skipped = 0
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        if enc["input_ids"].numel() == 0:
            skipped += 1
            continue
        samples.append({k: v.squeeze(0) for k, v in enc.items()})

    if skipped:
        logger.warning(
            "Skipped {} calibration samples that tokenized empty",
            skipped,
        )
    if not samples:
        raise AftError("All calibration samples tokenized to zero tokens.")
    console.print(f"[cyan]Prepared {len(samples)} calibration samples.[/cyan]")
    return samples


def get_calibration_data(
    tokenizer: Any,
    dataset_name: str,
    n_samples: int,
    seq_len: int,
    *,
    use_chat_template: bool = False,
) -> list[dict[str, Any]]:
    """Build tokenized calibration samples for GPTQ / FP8 quantization."""
    texts = _resolve_calibration_texts(tokenizer, dataset_name, n_samples)

    if not texts:
        raise AftError(
            f"Calibration source '{dataset_name}' produced no usable"
            f" samples.\n"
            f"  Every row was empty or shorter than"
            f" {MIN_CALIBRATION_TEXT_LEN} characters."
        )
    if len(texts) < n_samples:
        logger.warning(
            "Only {} of the requested {} calibration samples were available",
            len(texts),
            n_samples,
        )

    if use_chat_template:
        texts = _apply_chat_templates(tokenizer, texts)

    return _tokenize_calibration(tokenizer, texts, seq_len)


# ── Layer coverage reporting ──────────────────────────────────────────────


def report_layer_coverage(model: Any, *, strict: bool) -> None:
    """Report which linear layers GPTQ actually quantized.

    Relaxing ``layer_modules_strict`` lets GPTQModel skip modules it does not
    recognise — on hybrid architectures that silently leaves most layers in
    BF16.  Counting quantized versus quantizable modules turns that silent
    quality loss into a visible, actionable failure.
    """
    inner = getattr(model, "model", model)

    quantized: list[str] = []
    unquantized: list[str] = []
    for name, module in inner.named_modules():
        cls = type(module).__name__
        if "QuantLinear" in cls or hasattr(module, "qweight"):
            quantized.append(name)
        elif isinstance(module, torch.nn.Linear):
            unquantized.append(name)

    total = len(quantized) + len(unquantized)
    if total == 0:
        logger.warning("Found no linear modules to report coverage for")
        return

    pct = 100.0 * len(quantized) / total
    console.print(
        f"[cyan]Layer coverage: {len(quantized)}/{total} linear modules"
        f" quantized ({pct:.1f}%)[/cyan]"
    )
    if unquantized:
        preview = ", ".join(unquantized[:10])
        logger.warning(
            "{} linear modules were NOT quantized (e.g. {})",
            len(unquantized),
            preview,
        )
        if strict:
            raise AftError(
                f"{len(unquantized)} of {total} linear modules were"
                f" left unquantized:\n  {preview}\n"
                "  This usually means the architecture has module types\n"
                "  GPTQModel does not recognise (hybrid attention, MoE\n"
                "  experts, MTP heads). Verify support for this"
                " architecture,\n"
                "  or re-run with strict layer coverage disabled if"
                " leaving\n"
                "  these layers in full precision is intentional."
            )


# ── Provenance ────────────────────────────────────────────────────────────


def write_provenance(
    output: Path,
    *,
    source: str,
    config: QuantizeConfig,
    quant_label: str,
    revision: str | None,
    n_calibration_samples: int,
) -> None:
    """Record how this artifact was produced, next to the weights."""
    versions = library_versions()

    provenance = {
        "source_model": str(source),
        "source_revision": revision,
        "quantization": {
            "label": quant_label,
            "format": config.format,
            "bits": config.bits,
            "group_size": config.group_size,
            "desc_act": config.desc_act,
        },
        "calibration": {
            "dataset": config.calibration_dataset,
            "requested_samples": config.n_calibration_samples,
            "actual_samples": n_calibration_samples,
            "seq_len": config.calibration_seq_len,
            "chat_template_applied": config.use_chat_template,
        },
        "versions": versions,
        "cuda_capability": (
            ".".join(map(str, torch.cuda.get_device_capability()))
            if torch.cuda.is_available()
            else None
        ),
    }
    path = output / "aft_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("Wrote provenance to {}", path)


# ── Post-save config sanitization ──────────────────────────────────────────

#: Machine-local runtime fields gptqmodel writes into its quantization-config
#: ``meta`` that must never reach a published artifact: they either leak an
#: absolute temp path from the build host or record a host-specific VRAM
#: decision that is meaningless to consumers.
_LEAKED_QUANT_META_KEYS: tuple[str, ...] = ("offload_to_disk_path",)


def _sanitize_json_file(path: Path, mutate) -> bool:
    """Load a JSON file, apply ``mutate`` in place, rewrite if it changed.

    Returns whether the file was rewritten.  All-or-nothing: if ``mutate``
    raises or the result is not JSON-serializable, the original file is left
    untouched and the error propagates.
    """
    if not path.is_file():
        return False
    raw = path.read_text()
    data = json.loads(raw)
    mutate(data)
    new_raw = json.dumps(data, indent=2) + "\n"
    if new_raw == raw:
        return False
    path.write_text(new_raw)
    return True


def sanitize_saved_config(output: Path) -> None:
    """Clean gptqmodel-injected blemishes from the saved config files.

    Two classes of problem are corrected:

    * **Spurious top-level ``rope_parameters``.**  gptqmodel's
      ``_normalize_rope_parameters_config_compat`` shim synthesizes a
      top-level ``rope_parameters`` on multimodal configs (e.g. Qwen3.5)
      whose language-model RoPE actually lives under ``text_config``.  The
      synthesized dict falls back to ``rope_theta=10000`` and a plain
      ``default`` rope, silently contradicting the real ``mrope`` config
      under ``text_config``.  It is removed when a legitimate
      ``text_config.rope_parameters`` is present.

    * **Leaked local temp paths.**  The quantization-config ``meta`` records
      ``offload_to_disk_path`` (a ``/tmp/...`` scratch dir from the build
      host).  It is stripped from both ``config.json`` (under
      ``quantization_config.meta``) and ``quantize_config.json`` (under
      ``meta``).

    The original source model's ``rope_parameters`` (which is absent at the
    top level for these architectures) is the intended shape, so this only
    ever removes gptqmodel's additions — it never invents new fields.
    """
    changes: list[str] = []

    def _mutate_main(cfg: dict[str, Any]) -> None:
        nonlocal changes
        text_cfg = cfg.get("text_config")
        if (
            isinstance(text_cfg, dict)
            and isinstance(text_cfg.get("rope_parameters"), dict)
            and isinstance(cfg.get("rope_parameters"), dict)
        ):
            del cfg["rope_parameters"]
            changes.append("removed spurious top-level rope_parameters")

        qcfg = cfg.get("quantization_config")
        if isinstance(qcfg, dict):
            meta = qcfg.get("meta")
            if isinstance(meta, dict):
                for key in _LEAKED_QUANT_META_KEYS:
                    if key in meta:
                        del meta[key]
                        changes.append(f"removed {key} from config.json")

    def _mutate_quant(qcfg: dict[str, Any]) -> None:
        nonlocal changes
        meta = qcfg.get("meta")
        if isinstance(meta, dict):
            for key in _LEAKED_QUANT_META_KEYS:
                if key in meta:
                    del meta[key]
                    changes.append(f"removed {key} from quantize_config.json")

    _sanitize_json_file(output / "config.json", _mutate_main)
    _sanitize_json_file(output / "quantize_config.json", _mutate_quant)

    if changes:
        logger.info("Sanitized saved config: {}", "; ".join(changes))


# ── Quantization sub-steps ────────────────────────────────────────────────


def validate_quant_config(
    config: QuantizeConfig,
) -> tuple[bool, str, str]:
    """Validate format, group size, and FP8 capability.

    Returns ``(is_fp8, quant_label, vllm_quant_arg)``.
    """
    _VALID_FORMATS = {"gptq", "fp8"}
    if config.format not in _VALID_FORMATS:
        raise AftError(
            f"Unknown quantization format '{config.format}'.\n"
            f"  Valid options: {', '.join(sorted(_VALID_FORMATS))}"
        )

    is_fp8 = config.format == "fp8"
    quant_label = "FP8" if is_fp8 else f"GPTQ Int{config.bits}"
    vllm_quant_arg = "fp8" if is_fp8 else "gptq_marlin"

    if not is_fp8 and config.group_size not in (-1, 32, 64, 128):
        raise AftError(
            f"Unsupported GPTQ group size {config.group_size}.\n"
            "  Valid options: 128 (recommended), 64, 32, or -1"
            " (per-column).\n"
            "  vLLM's gptq_marlin kernel expects 128 or -1."
        )

    if is_fp8 and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (8, 9):
            console.print(
                f"[yellow]⚠ This GPU is sm_{major}{minor}; FP8 kernels"
                f" need sm_89+. The artifact can be written here but"
                f" cannot be benchmarked or validated on this machine."
                f" Prefer GPTQ Int4 for a portable artifact.[/yellow]"
            )

    return is_fp8, quant_label, vllm_quant_arg


def load_model_for_quantization(
    model_path: Path,
    config: QuantizeConfig,
    *,
    is_fp8: bool,
    hf_token: str | None,
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    """Load config, processor, calibration data, and the GPTQModel.

    Returns ``(model, processor, hf_config, calibration)``.
    """
    from gptqmodel import GPTQModel
    from gptqmodel import QuantizeConfig as GptqCfg

    hf_config, multimodal, processor, tokenizer = load_model_inputs(
        str(model_path),
        trust_remote_code=config.trust_remote_code,
        token=hf_token,
        revision=config.revision,
    )
    if multimodal:
        console.print(
            "[yellow]⚠ Multimodal checkpoint: calibration is"
            " text-only, so the vision tower's activation statistics"
            " are not represented.[/yellow]"
        )

    if config.chat_template:
        template_path = Path(config.chat_template)
        if not template_path.is_file():
            raise AftError(f"Chat template file not found: {template_path}")
        tokenizer.chat_template = template_path.read_text()
        console.print(f"[cyan]Chat template override: {template_path}[/cyan]")

    console.print("[cyan]Building calibration dataset...[/cyan]")
    calibration = get_calibration_data(
        tokenizer,
        config.calibration_dataset,
        config.n_calibration_samples,
        config.calibration_seq_len,
        use_chat_template=config.use_chat_template,
    )

    if is_fp8:
        quant_cfg = GptqCfg(bits=8, format="fp8")
    else:
        quant_cfg = GptqCfg(
            bits=config.bits,
            group_size=config.group_size,
            desc_act=config.desc_act,
        )

    console.print(f"[cyan]Loading model for quantization: {model_path}[/cyan]")
    model_dtype = resolve_dtype(hf_config)
    logger.info("Quantizing from dtype {}", model_dtype)
    try:
        model = GPTQModel.from_pretrained(
            str(model_path),
            quantize_config=quant_cfg,
            torch_dtype=model_dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as e:
        model_type = getattr(hf_config, "model_type", "unknown")
        logger.error("GPTQModel could not load {}: {}", model_path, e)
        raise AftError(
            f"GPTQModel could not load this model.\n"
            f"  Model type:    {model_type}\n"
            f"  Architectures: "
            f"{', '.join(getattr(hf_config, 'architectures', []) or ['?'])}\n"
            f"  This usually means the installed gptqmodel does not\n"
            f"  support the architecture yet. Check that gptqmodel and\n"
            f"  transformers are new enough for '{model_type}'."
        ) from e

    return model, processor, hf_config, calibration


def save_quantized_artifact(
    model: Any,
    processor: Any,
    output: Path,
    model_path: Path,
    config: QuantizeConfig,
    *,
    quant_label: str,
    vllm_quant_arg: str,
    n_calibration_samples: int,
) -> None:
    """Save the quantized model, processor, aux files, and provenance."""
    model.save_quantized(str(output))
    processor.save_pretrained(str(output))
    copy_auxiliary_files(model_path, output)
    sanitize_saved_config(output)
    write_provenance(
        output,
        source=str(model_path),
        config=config,
        quant_label=quant_label,
        revision=config.revision,
        n_calibration_samples=n_calibration_samples,
    )

    logger.info("{} model saved to {}", quant_label, output)
    console.print(f"[green]✓ {quant_label} → {output}[/green]")
    console.print(
        f"[dim]  vLLM: --model {output} --quantization {vllm_quant_arg}[/dim]"
    )
