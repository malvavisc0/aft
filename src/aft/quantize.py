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

#: Maps well-known short names to HuggingFace dataset IDs.
_HF_CALIBRATION_DATASETS: dict[str, str] = {
    "fineweb": "HuggingFaceFW/fineweb",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "c4": "allenai/c4",
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


def get_calibration_data(
    tokenizer: Any,
    dataset_name: str,
    n_samples: int,
    seq_len: int,
    *,
    use_chat_template: bool = False,
) -> list[dict[str, Any]]:
    """Build tokenized calibration samples for GPTQ / FP8 quantization."""
    import datasets as hf_datasets

    if dataset_name in _HF_CALIBRATION_DATASETS:
        hf_repo = _HF_CALIBRATION_DATASETS[dataset_name]
        console.print(
            f"[cyan]Loading {hf_repo} for calibration ({n_samples} samples)...[/cyan]"
        )

        data = hf_datasets.load_dataset(
            hf_repo,
            split="train",
            streaming=True,
        )

        texts: list[str] = []
        for row in data:
            text = row["text"].strip()
            if len(text) > MIN_CALIBRATION_TEXT_LEN:
                texts.append(text)
            if len(texts) >= n_samples:
                break

    else:
        # Local JSONL fallback
        p = Path(dataset_name)
        if not p.exists():
            raise AftError(f"Calibration JSONL not found: {p}")

        texts = []
        for i, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "text" not in row:
                    raise AftError(
                        f"Calibration JSONL line {i}: missing 'text' key in {p}"
                    )
                texts.append(row["text"])
            except json.JSONDecodeError as exc:
                raise AftError(
                    f"Calibration JSONL line {i}: invalid JSON in {p}"
                ) from exc

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
        if getattr(tokenizer, "chat_template", None):
            console.print("[cyan]Applying the model's chat template.[/cyan]")
            texts = [apply_chat_template(tokenizer, t) for t in texts]
        else:
            logger.warning(
                "use_chat_template requested but the tokenizer has"
                " no chat_template; calibrating on raw text"
            )

    # Tokenize
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
