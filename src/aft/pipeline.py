"""Core pipeline: QLoRA SFT → Merge → GPTQ Quantize → Push to Hub."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from aft.config import QuantizeConfig, TrainConfig
from aft.ui import console, silence_noisy_loggers


class AftError(RuntimeError):
    """User-facing pipeline error (caught by CLI for clean output)."""


def _hf_token() -> str | None:
    """Resolve HuggingFace token from environment.

    Returns ``None`` when the variable is absent *or* set to an empty string.
    """
    return os.getenv("HF_TOKEN") or None


# ── Phase 1: QLoRA SFT ────────────────────────────────────────────────────


def train(config: TrainConfig) -> Path:
    """Run QLoRA supervised fine-tuning.

    Loads the base model in 4-bit (NF4), applies LoRA adapters targeting all
    attention + MLP projection layers, trains with SFTTrainer, and saves
    the adapter weights.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    import datasets as hf_datasets
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from aft.cleaning import clean_dataset

    silence_noisy_loggers()

    hf_token = _hf_token()
    out = (
        Path(config.output_dir)
        if config.output_dir
        else Path("models") / config.run_name
    )
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Loading tokenizer: {config.base_model}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        token=hf_token,
        fix_mistral_regex=True,
    )
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
    )
    if config.max_memory is not None:
        model_kwargs["max_memory"] = config.max_memory
        logger.debug("Using max_memory: {}", config.max_memory)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    # Disable gradient checkpointing here; SFTConfig will handle it via
    # the ``gradient_checkpointing`` kwarg to avoid double-wrapping.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load and format datasets ──────────────────────────────────────
    console.print(f"[cyan]Loading datasets: {config.datasets}[/cyan]")
    all_texts: list[str] = []
    for ds_id in config.datasets:
        ds = hf_datasets.load_dataset(ds_id, split="train", token=hf_token)
        if "text" in ds.column_names:
            all_texts.extend(ds["text"])
        elif "conversations" in ds.column_names:
            for row in ds:
                parts = [
                    f"{msg['role']}: {msg['content']}" for msg in row["conversations"]
                ]
                all_texts.append("\n".join(parts))
        else:
            cols = ds.column_names
            raise AftError(
                f"Dataset '{ds_id}' has no 'text' or 'conversations' column.\n"
                f"  Available columns: {cols}\n"
                f"  Ensure the dataset has a 'text' column or a"
                f" 'conversations' column with role/content fields."
            )

    dataset = hf_datasets.Dataset.from_dict({"text": all_texts})
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    console.print(f"[cyan]Dataset: {len(dataset)} samples[/cyan]")

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

    _total_steps = max(
        1,
        len(dataset)
        * config.num_epochs
        // (config.per_device_batch_size * config.gradient_accumulation_steps),
    )
    _warmup_steps = max(1, int(_total_steps * config.warmup_ratio))

    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=_warmup_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        run_name=config.run_name,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=4,
        dataset_text_field="text",
        max_length=config.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=args,
    )

    console.print("[cyan]Training started...[/cyan]")
    try:
        trainer.train()
    except Exception as e:
        logger.error(
            "Training failed at adapter dir {}: {}",
            adapter_dir,
            e,
        )
        raise AftError(
            f"QLoRA training failed. Check logs above for details.\n"
            f"  Adapter dir: {adapter_dir}\n"
            f"  To resume, fix the issue and re-run."
        ) from e
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to {}", adapter_dir)
    console.print(f"[green]✓ Adapter → {adapter_dir}[/green]")
    return adapter_dir


# ── Phase 2: Merge LoRA ───────────────────────────────────────────────────


def merge_adapter(
    base_model: str,
    adapter_path: Path,
    output: Path,
    *,
    trust_remote_code: bool = False,
) -> Path:
    """Merge LoRA adapter into the base model as safetensors.

    The merged model is saved using the base model's native ``torch_dtype``
    (from its config), falling back to bfloat16 when unspecified.

    Returns:
        Path to the merged model directory.
    """
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_token = _hf_token()
    output.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Loading base model on CPU for merge: {base_model}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=trust_remote_code,
        token=hf_token,
        fix_mistral_regex=True,
    )
    hf_config = AutoConfig.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    model_dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    console.print("[cyan]Merging LoRA weights...[/cyan]")
    try:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    except Exception as e:
        logger.error("Merge failed for adapter {}: {}", adapter_path, e)
        raise AftError(
            f"LoRA merge failed.\n"
            f"  Adapter: {adapter_path}\n"
            f"  Output:  {output}\n"
            f"  Check that the adapter is compatible with the base model."
        ) from e
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    logger.info("Merged model saved to {}", output)
    console.print(f"[green]✓ Merged → {output}[/green]")
    return output


# ── Phase 3: Quantization (GPTQ / FP8) ───────────────────────────────────


_MIN_CALIBRATION_TEXT_LEN = 100
"""Minimum character length for calibration text samples."""


def _get_calibration_data(
    tokenizer: Any,
    dataset_name: str,
    n_samples: int,
    seq_len: int,
) -> list[dict[str, Any]]:
    """Build tokenized calibration samples for GPTQ / FP8 quantization."""
    import datasets as hf_datasets

    # Map well-known short names to HuggingFace dataset IDs.
    _HF_CALIBRATION_DATASETS: dict[str, str] = {
        "fineweb": "HuggingFaceFW/fineweb",
        "fineweb-edu": "HuggingFaceFW/fineweb-edu",
        "c4": "allenai/c4",
    }

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
            if len(text) > _MIN_CALIBRATION_TEXT_LEN:
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

    # Tokenize
    samples: list[dict[str, Any]] = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        samples.append({k: v.squeeze(0) for k, v in enc.items()})

    console.print(f"[cyan]Prepared {len(samples)} calibration samples.[/cyan]")
    return samples


def _materialize_meta_params(model: torch.nn.Module, model_path: Path) -> int:
    """Load real weights for any parameters stuck on the meta device.

    GPTQModel's built-in materialization doesn't handle every architecture
    (e.g. sparse MoE experts in Mellum).  This detects leftover meta tensors
    and loads their values from the safetensors checkpoint files.

    Returns the number of parameters that were materialized.
    """
    from safetensors.torch import load_file

    meta_params = {
        name: param for name, param in model.named_parameters() if param.is_meta
    }
    if not meta_params:
        return 0

    meta_names = list(meta_params)
    logger.warning(
        "Found {} parameters still on meta device after load: {}",
        len(meta_params),
        ", ".join(meta_names[:8]) + ("..." if len(meta_names) > 8 else ""),
    )

    safetensor_files = sorted(Path(model_path).glob("*.safetensors"))
    if not safetensor_files:
        logger.error(
            "No .safetensors files found in {} -- cannot materialize", model_path
        )
        return 0

    # Search shards lazily: load one at a time and extract only the weights we
    # still need, so peak memory stays proportional to the largest single shard
    # rather than the whole (already-loaded) model.
    needed = set(meta_names)
    found: dict[str, torch.Tensor] = {}
    for shard in safetensor_files:
        if not needed:
            break
        weights = load_file(str(shard), device="cpu")
        for key in needed & weights.keys():
            found[key] = weights[key]
        needed -= found.keys()

    materialized = 0
    for name in meta_names:
        param = meta_params[name]
        if name not in found:
            logger.warning("Meta param '{}' not found in checkpoint -- skipping", name)
            continue

        # Walk the module tree to replace the parameter in-place.
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # A meta tensor's reported dtype can be an unreliable default (often
        # float32); trust the checkpoint's dtype unless the model explicitly
        # asks for something other than float32.
        target_dtype = (
            param.dtype if param.dtype != torch.float32 else found[name].dtype
        )
        new_param = torch.nn.Parameter(
            found[name].to(dtype=target_dtype),
            requires_grad=param.requires_grad,
        )
        setattr(parent, parts[-1], new_param)
        materialized += 1

    if materialized:
        logger.info("Materialized {} meta parameters from checkpoint", materialized)
    return materialized


def quantize(
    model_path: Path, output: Path, config: QuantizeConfig, *, token: str | None = None
) -> Path:
    """Quantize a merged model using GPTQModel.

    Supports GPTQ (Int4 / Int8) and FP8 via ``config.format``.

    Runs quantization inside a temporary directory (for gptqmodel temp files)
    with automatic cleanup on exit.

    Returns:
        Path to the quantized model directory (vLLM-ready).
    """
    from gptqmodel import GPTQModel
    from gptqmodel import QuantizeConfig as GptqCfg
    from transformers import AutoConfig, AutoTokenizer

    silence_noisy_loggers()

    _VALID_FORMATS = {"gptq", "fp8"}
    if config.format not in _VALID_FORMATS:
        raise AftError(
            f"Unknown quantization format '{config.format}'.\n"
            f"  Valid options: {', '.join(sorted(_VALID_FORMATS))}"
        )

    is_fp8 = config.format == "fp8"
    quant_label = "FP8" if is_fp8 else f"GPTQ Int{config.bits}"
    vllm_quant_arg = "fp8" if is_fp8 else "gptq_marlin"

    # gptqmodel writes intermediate files (logs, temp weights) to the current
    # working directory.  We chdir into a disposable temp dir so those artefacts
    # don't pollute the user's project root.  The TemporaryDirectory context
    # manager guarantees cleanup, and the finally block restores the original cwd
    # even if quantization raises.
    prev_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="gptq_") as tmp_dir:
            os.chdir(tmp_dir)
            logger.debug("Changed working directory to {} for quantization", tmp_dir)

            hf_token = token or _hf_token()
            output.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=config.trust_remote_code,
                token=hf_token,
                fix_mistral_regex=True,
            )

            console.print("[cyan]Building calibration dataset...[/cyan]")
            calibration = _get_calibration_data(
                tokenizer,
                config.calibration_dataset,
                config.n_calibration_samples,
                config.calibration_seq_len,
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
            hf_config = AutoConfig.from_pretrained(
                str(model_path),
                trust_remote_code=config.trust_remote_code,
                token=hf_token,
            )
            model_dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16
            model = GPTQModel.from_pretrained(
                str(model_path),
                quantize_config=quant_cfg,
                torch_dtype=model_dtype,
                trust_remote_code=config.trust_remote_code,
            )

            # Materialize any parameters GPTQModel left on the meta device
            # (e.g. sparse MoE experts that its loader doesn't handle).
            _materialize_meta_params(model, model_path)

            # Disable strict layer matching for hybrid architectures with modules
            # that gptqmodel doesn't recognize (e.g. linear_attn.conv1d)
            if hasattr(model, "gptq_model") and model.gptq_model is not None:
                model.gptq_model.layer_modules_strict = False

            extra_info = f"(group_size={config.group_size})" if not is_fp8 else ""
            console.print(f"[cyan]Quantizing → {quant_label} {extra_info}...[/cyan]")
            try:
                model.quantize(calibration)
            except Exception as e:
                logger.error("{} quantization failed: {}", quant_label, e)
                raise AftError(
                    f"{quant_label} quantization failed.\n"
                    f"  Model:   {model_path}\n"
                    f"  Output:  {output}\n"
                    "  Try reducing n_calibration_samples or"
                    " using a different calibration dataset."
                ) from e
            model.save_quantized(str(output))
            tokenizer.save_pretrained(str(output))

            logger.info("{} model saved to {}", quant_label, output)
            console.print(f"[green]✓ {quant_label} → {output}[/green]")
            console.print(
                f"[dim]  vLLM: --model {output} --quantization {vllm_quant_arg}[/dim]"
            )
            return output
    finally:
        os.chdir(prev_cwd)


# ── Publish to HuggingFace Hub ────────────────────────────────────────────


def push_to_hub(
    model_path: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload quantized model",
) -> str:
    """Push a quantized model directory to HuggingFace Hub.

    Creates the repo if it does not exist, then uploads the full directory.

    Returns:
        HTTPS URL of the published repository.
    """
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
    logger.info("Model published to {}", url)
    console.print(f"[bold green]✓ Published → {url}[/bold green]")
    return url
