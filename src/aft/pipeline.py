"""Core pipeline: QLoRA SFT → Merge → GPTQ Quantize → Push to Hub.

This module is a thin orchestrator.  All architecture introspection,
model loading, and quantization logic lives in :mod:`aft.model_utils`
and :mod:`aft.quantize` so each concern is small and testable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from aft.cleaning import (
    flatten_row_to_text,
    parse_dataset_spec,
    resolve_dataset_split,
    supported_text_columns,
)
from aft.config import QuantizeConfig, TrainConfig
from aft.dataprep import stack_messages, tokenize_dataset
from aft.errors import AftError
from aft.model_utils import (
    auto_model_class,
    copy_auxiliary_files,
    discover_lora_targets,
    dtype_bytes,
    load_model_inputs,
    materialize_meta_params,
    resolve_dtype,
)
from aft.quantize import (
    load_model_for_quantization,
    report_layer_coverage,
    save_quantized_artifact,
    validate_quant_config,
)
from aft.recommend import checkpoint_size_gib
from aft.ui import console, silence_noisy_loggers


def _hf_token() -> str | None:
    """Resolve HuggingFace token from environment."""
    return os.getenv("HF_TOKEN") or None


_SUPPORTED_TEXT_COLUMNS: frozenset[str] = frozenset(
    {"text", "content", "code", "body", "messages", "conversations"}
)


def _load_training_texts(datasets: list[str], hf_token: str | None) -> list[str]:
    """Load and flatten every named dataset into a single text list."""
    import datasets as hf_datasets

    console.print(f"[cyan]Loading datasets: {datasets}[/cyan]")
    all_texts: list[str] = []
    for ds_spec in datasets:
        ds_id, requested_split = parse_dataset_spec(ds_spec)
        split = resolve_dataset_split(ds_id, requested_split, token=hf_token)
        ds = hf_datasets.load_dataset(ds_id, split=split, token=hf_token)
        if not (set(ds.column_names) & _SUPPORTED_TEXT_COLUMNS):
            raise AftError(
                f"Dataset {ds_id} has no supported text column.\n"
                f"  Expected {supported_text_columns()}.\n"
                f"  Columns found: {', '.join(ds.column_names)}"
            )
        for row in ds:
            text = flatten_row_to_text(row)
            if text:
                all_texts.append(text)
    return all_texts


def _load_training_messages(
    datasets: list[str], hf_token: str | None
) -> dict[str, list[Any]]:
    """Load every dataset's structured rows into ``messages``/``tools`` columns."""
    import datasets as hf_datasets

    console.print(f"[cyan]Loading message datasets: {datasets}[/cyan]")
    columns: dict[str, list[Any]] = {"messages": [], "tools": []}
    for ds_spec in datasets:
        ds_id, requested_split = parse_dataset_spec(ds_spec)
        split = resolve_dataset_split(ds_id, requested_split, token=hf_token)
        ds = hf_datasets.load_dataset(ds_id, split=split, token=hf_token)
        if "messages" not in ds.column_names and "conversations" not in ds.column_names:
            raise AftError(
                f"Dataset {ds_id} has no messages column for format='messages'.\n"
                f"  Columns found: {', '.join(ds.column_names)}"
            )
        for row in stack_messages(ds):
            columns["messages"].append(row["messages"])
            columns["tools"].append(row["tools"])
    return columns


def _load_chat_template(config: TrainConfig) -> str:
    """Return the local chat template content, validating it exists."""
    if not config.chat_template:
        raise AftError(
            "format='messages' requires --chat-template pointing at the local"
            " v22 .jinja file."
        )
    path = Path(config.chat_template)
    if not path.is_file():
        raise AftError(f"Chat template file not found: {path}")
    return path.read_text()


def _build_train_dataset(
    config: TrainConfig, tokenizer: Any, hf_token: str | None
) -> tuple[Any, dict[str, Any]]:
    """Build the train dataset and the dataset-specific ``SFTConfig`` kwargs."""
    import datasets as hf_datasets

    from aft.cleaning import clean_dataset

    if config.format not in ("text", "messages"):
        raise AftError(
            f"Unknown dataset format: {config.format!r}. Expected 'text' or 'messages'."
        )

    if config.format == "messages":
        if config.clean:
            raise AftError(
                "format='messages' does not support --clean yet; the text"
                " path is the structured-clean route."
            )
        chat_template = _load_chat_template(config)
        columns = _load_training_messages(config.datasets, hf_token)
        dataset = hf_datasets.Dataset.from_dict(columns)
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        dataset = tokenize_dataset(
            dataset,
            tokenizer,
            max_seq_len=config.max_seq_len,
            mask_strategy=config.mask_strategy,
            chat_template=chat_template,
            enable_thinking=config.enable_thinking,
            reasoning_effort=config.reasoning_effort,
            tool_call_format=config.tool_call_format,
        )
        # Rows are pre-tokenized and pre-truncated by dataprep: trl must not
        # re-truncate (its max_length default would silently cut to 1024).
        return dataset, {"max_length": None}

    all_texts = _load_training_texts(config.datasets, hf_token)

    dataset = hf_datasets.Dataset.from_dict({"text": all_texts})
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    if config.clean:
        dataset = clean_dataset(
            dataset,
            tokenizer,
            dedup=config.dedup,
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens or config.max_seq_len,
            languages=config.languages,
            max_special_ratio=config.max_special_ratio,
        )
    return dataset, {
        "dataset_text_field": "text",
        "max_length": config.max_seq_len,
    }


def train(config: TrainConfig) -> Path:
    """Run QLoRA supervised fine-tuning."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    silence_noisy_loggers()

    hf_token = _hf_token()
    out = (
        Path(config.output_dir)
        if config.output_dir
        else Path("models") / config.run_name
    )
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    hf_config, multimodal, processor, tokenizer = load_model_inputs(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        token=hf_token,
        revision=config.revision,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise AftError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.print("[cyan]Loading base model in 4-bit NF4 (QLoRA)...[/cyan]")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs: dict[str, Any] = dict(
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        attn_implementation="sdpa",
        token=hf_token,
        revision=config.revision,
    )
    if config.max_memory is not None:
        model_kwargs["max_memory"] = config.max_memory
    auto_cls = auto_model_class(hf_config)
    model = auto_cls.from_pretrained(config.base_model, **model_kwargs)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=discover_lora_targets(model, config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset, sft_dataset_kwargs = _build_train_dataset(config, tokenizer, hf_token)

    _total_steps = max(
        1,
        len(dataset)
        * config.num_epochs
        // (config.per_device_batch_size * config.gradient_accumulation_steps),
    )
    _warmup_steps = max(1, int(_total_steps * config.warmup_ratio))
    bf16_ok = torch.cuda.is_bf16_supported()

    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=_warmup_steps,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        run_name=config.run_name,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=0,
        packing=False,
        **sft_dataset_kwargs,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, args=args
    )

    console.print("[cyan]Training started...[/cyan]")
    try:
        trainer.train()
    except Exception as e:
        raise AftError("QLoRA training failed.") from e
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    console.print(f"[green]✓ Adapter → {adapter_dir}[/green]")
    return adapter_dir


def _warn_if_merge_wont_fit(hf_config: Any, base_model: str) -> None:
    from aft.recommend import detect_system_ram_mib

    num_params = getattr(hf_config, "num_parameters", None)
    if not num_params:
        return
    dtype = resolve_dtype(hf_config)
    needed_gib = checkpoint_size_gib(num_params, dtype_bytes(dtype))
    ram_mib = detect_system_ram_mib()
    if ram_mib is None:
        logger.warning(
            "Could not detect system RAM; cannot assess merge"
            " feasibility for {} (needs ≈{:.0f} GiB)",
            base_model,
            needed_gib,
        )
        return
    ram_gib = ram_mib / 1024
    if needed_gib > ram_gib * 0.8:
        console.print(
            f"[yellow]⚠ {base_model} needs ≈{needed_gib:.0f} GiB of RAM to"
            f" merge on CPU but only {ram_gib:.0f} GiB is available."
            f" Consider --base-model-only instead.[/yellow]"
        )


def merge_adapter(
    base_model: str,
    adapter_path: Path,
    output: Path,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> Path:
    """Merge LoRA adapter into the base model as safetensors."""
    from peft import PeftModel

    hf_token = _hf_token()
    output.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Loading base model on CPU for merge: {base_model}[/cyan]")
    hf_config, multimodal, processor, _ = load_model_inputs(
        base_model,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        revision=revision,
    )
    _warn_if_merge_wont_fit(hf_config, base_model)
    model_dtype = resolve_dtype(hf_config)
    auto_cls = auto_model_class(hf_config)
    model = auto_cls.from_pretrained(
        base_model,
        dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        token=hf_token,
        revision=revision,
    )
    console.print("[cyan]Merging LoRA weights...[/cyan]")
    try:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    except Exception as e:
        raise AftError(f"LoRA merge failed. Adapter: {adapter_path}") from e
    model.save_pretrained(str(output), safe_serialization=True)
    processor.save_pretrained(str(output))
    copy_auxiliary_files(adapter_path, output)
    console.print(f"[green]✓ Merged → {output}[/green]")
    return output


def quantize(
    model_path: Path, output: Path, config: QuantizeConfig, *, token: str | None = None
) -> Path:
    """Quantize a merged model using GPTQModel."""
    silence_noisy_loggers()
    # Resolve to absolute paths *before* chdir-ing into the tempdir below —
    # otherwise a relative --model/--output/JSONL-calibration path silently
    # breaks once cwd moves.
    model_path = Path(model_path).resolve()
    output = Path(output).resolve()
    from aft.quantize import _HF_CALIBRATION_DATASETS

    if config.calibration_dataset not in _HF_CALIBRATION_DATASETS:
        config.calibration_dataset = str(Path(config.calibration_dataset).resolve())
    is_fp8, quant_label, vllm_quant_arg = validate_quant_config(config)
    prev_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="gptq_") as tmp_dir:
            os.chdir(tmp_dir)
            hf_token = token or _hf_token()
            output.mkdir(parents=True, exist_ok=True)
            model, processor, _hf_config, calibration = load_model_for_quantization(
                model_path, config, is_fp8=is_fp8, hf_token=hf_token
            )
            materialize_meta_params(model, model_path)
            gptq_model = getattr(model, "gptq_model", None)
            if gptq_model is not None:
                gptq_model.layer_modules_strict = False
            else:
                logger.warning(
                    "Model has no gptq_model attribute;"
                    " cannot relax layer_modules_strict."
                )
            extra_info = f"(group_size={config.group_size})" if not is_fp8 else ""
            console.print(f"[cyan]Quantizing → {quant_label} {extra_info}...[/cyan]")
            try:
                model.quantize(calibration)
            except Exception as e:
                raise AftError(f"{quant_label} quantization failed.") from e
            report_layer_coverage(model, strict=config.strict_layer_coverage)
            save_quantized_artifact(
                model,
                processor,
                output,
                model_path,
                config,
                quant_label=quant_label,
                vllm_quant_arg=vllm_quant_arg,
                n_calibration_samples=len(calibration),
            )
            return output
    finally:
        os.chdir(prev_cwd)


def push_to_hub(
    model_path: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload quantized model",
) -> str:
    """Push a quantized model directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    resolved_token = token or _hf_token()
    api = HfApi(token=resolved_token)
    console.print(f"[cyan]Creating/verifying repo: {repo_id}[/cyan]")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    console.print(f"[cyan]Uploading {model_path} → {repo_id}...[/cyan]")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    console.print(f"[bold green]✓ Published → {url}[/bold green]")
    return url
