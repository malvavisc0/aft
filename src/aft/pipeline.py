"""Core pipeline: QLoRA SFT → Merge → GPTQ Quantize → Push to Hub."""

import json
import logging
import os
import tempfile
from pathlib import Path

import torch
from loguru import logger
from rich.console import Console

from aft.config import QuantizeConfig, TrainConfig

console = Console(width=200)


def _hf_token() -> str | None:
    """Resolve HuggingFace token from environment."""
    return os.getenv("HF_TOKEN") or None


def _silence_debug_loggers() -> None:
    """Suppress noisy DEBUG logs from HF/HTTP libraries."""
    for name in (
        "httpcore",
        "httpx",
        "datasets",
        "torchao",
        "filelock",
        "fsspec",
        "huggingface_hub",
        "bitsandbytes",
        "urllib3",
        "gptqmodel",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


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

    _silence_debug_loggers()

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
        config.base_model, trust_remote_code=True, token=hf_token
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
    model_kwargs: dict = dict(
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        token=hf_token,
    )
    if config.max_memory is not None:
        model_kwargs["max_memory"] = config.max_memory
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
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
            logger.warning("Dataset {} has no 'text' column. Columns: {}", ds_id, cols)
            all_texts.extend(ds[cols[0]])

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

    console.print("[green]Training started...[/green]")
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to {}", adapter_dir)
    console.print(f"[green]✓ Adapter → {adapter_dir}[/green]")
    return adapter_dir


# ── Phase 2: Merge LoRA ───────────────────────────────────────────────────


def merge_adapter(base_model: str, adapter_path: Path, output: Path) -> Path:
    """Merge LoRA adapter into the base model as fp16 safetensors.

    Returns:
        Path to the merged model directory.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = _hf_token()
    output.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Loading base model on CPU for merge: {base_model}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token,
    )
    console.print("[cyan]Merging LoRA weights...[/cyan]")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    logger.info("Merged model saved to {}", output)
    console.print(f"[green]✓ Merged → {output}[/green]")
    return output


# ── Phase 3: GPTQ Int4 Quantization ──────────────────────────────────────


def _get_calibration_data(
    tokenizer,
    dataset_name: str,
    n_samples: int,
    seq_len: int,
) -> list[dict]:
    """Build tokenized calibration samples for GPTQ quantization."""
    import datasets as hf_datasets

    if dataset_name == "wikitext2":
        data = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [row["text"] for row in data if len(row["text"]) > 50]
    else:
        p = Path(dataset_name)
        texts = [
            json.loads(line)["text"]
            for line in p.read_text().splitlines()
            if line.strip()
        ]

    samples = []
    for text in texts[:n_samples]:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        samples.append({k: v.squeeze(0) for k, v in enc.items()})
    return samples


def quantize(model_path: Path, output: Path, config: QuantizeConfig) -> Path:
    """Quantize a merged fp16 model to GPTQ Int4 using GPTQModel.

    Returns:
        Path to the quantized model directory (vLLM-ready).
    """
    from gptqmodel import GPTQModel
    from gptqmodel import QuantizeConfig as GptqCfg
    from transformers import AutoTokenizer

    _silence_debug_loggers()

    # Run from /tmp so gptqmodel writes logs/temp files there
    prev_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix="gptq_")
    os.chdir(tmp_dir)
    logger.debug("Changed working directory to {} for quantization", tmp_dir)

    try:
        hf_token = _hf_token()
        output.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, token=hf_token
        )

        console.print("[cyan]Building calibration dataset...[/cyan]")
        calibration = _get_calibration_data(
            tokenizer,
            config.calibration_dataset,
            config.n_calibration_samples,
            config.calibration_seq_len,
        )

        quant_cfg = GptqCfg(
            bits=config.bits,
            group_size=config.group_size,
            desc_act=config.desc_act,
        )
        console.print(f"[cyan]Loading model for quantization: {model_path}[/cyan]")
        model = GPTQModel.from_pretrained(
            str(model_path),
            quantize_config=quant_cfg,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Disable strict layer matching for hybrid architectures with modules
        # that gptqmodel doesn't recognize (e.g. linear_attn.conv1d)
        if hasattr(model, "gptq_model") and model.gptq_model is not None:
            model.gptq_model.layer_modules_strict = False

        console.print(
            f"[green]Quantizing → GPTQ Int{config.bits} "
            f"(group_size={config.group_size})...[/green]"
        )
        model.quantize(calibration)
        model.save_quantized(str(output))
        tokenizer.save_pretrained(str(output))

        logger.info("GPTQ Int{} model saved to {}", config.bits, output)
        console.print(f"[green]✓ GPTQ Int{config.bits} → {output}[/green]")
        console.print(f"[dim]  vLLM: --model {output} --quantization gptq_marlin[/dim]")
        return output
    finally:
        os.chdir(prev_cwd)


# ── Publish to HuggingFace Hub ────────────────────────────────────────────


def push_to_hub(
    model_path: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload GPTQ quantized model",
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
