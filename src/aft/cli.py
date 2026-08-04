"""CLI commands for aft (entry point: `aft`)."""

import platform
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from aft.ui import console

if TYPE_CHECKING:
    from aft.config import QuantizeConfig

app = typer.Typer(
    name="aft",
    help=(
        "[bold cyan]Fine-tune and quantize models"
        " for instruction following.[/bold cyan]"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _banner() -> None:
    from aft import __version__

    logo = "[bold cyan]  ╔═╗╔═╗╔╦╗\n  ╠═╣╠╣  ║ \n  ╩ ╩╚   ╩ [/bold cyan]"
    console.print(logo)
    console.print(
        f"  [dim]v{__version__}[/dim] [dim]│[/dim]"
        " [bold green]QLoRA fine-tuning[/bold green]"
        " [dim]→[/dim] [bold magenta]GPTQ quantization[/bold magenta]"
    )
    console.print()


#: Maps `--quant-type` to (bits, format, label, output subdirectory).
_QUANT_TYPES: dict[str, tuple[int, str, str, str]] = {
    "int4": (4, "gptq", "GPTQ Int4", "gptq-int4"),
    "int8": (8, "gptq", "GPTQ Int8", "gptq-int8"),
    "fp8": (8, "fp8", "FP8", "fp8"),
}


def _build_quant_config(
    *,
    bits: int,
    format: str,
    group_size: int,
    calibration_dataset: str,
    trust_remote_code: bool,
    revision: str | None,
    desc_act: bool = False,
    n_calibration_samples: int = 128,
    calibration_seq_len: int = 2048,
    use_chat_template: bool = True,
    strict_layer_coverage: bool = True,
) -> "QuantizeConfig":
    """Map CLI args to a QuantizeConfig (shared by run_cmd and quantize_cmd)."""
    from aft.config import QuantizeConfig

    return QuantizeConfig(
        bits=bits,
        format=format,
        group_size=group_size,
        desc_act=desc_act,
        calibration_dataset=calibration_dataset,
        n_calibration_samples=n_calibration_samples,
        calibration_seq_len=calibration_seq_len,
        trust_remote_code=trust_remote_code,
        revision=revision,
        use_chat_template=use_chat_template,
        strict_layer_coverage=strict_layer_coverage,
    )


def _resolve_quant_type(quant_type: str) -> tuple[int, str, str, str]:
    """Validate `--quant-type` instead of silently falling back to Int4."""
    try:
        return _QUANT_TYPES[quant_type.lower()]
    except KeyError as exc:
        console.print(
            f"[red]Unknown --quant-type '{quant_type}'."
            f" Valid options: {', '.join(_QUANT_TYPES)}[/red]"
        )
        raise typer.Exit(1) from exc


def _verify_resume_compatibility(
    quant_dir: Path,
    model: str,
    quant_cfg: object,
    revision: str | None,
) -> None:
    """Refuse to resume into an output directory built from different inputs.

    File existence alone says nothing about *what* produced those files.
    `aft_provenance.json` (written by the quantize phase) lets us compare the
    source model, revision, and quantization settings before reusing them.
    """
    import json

    provenance_path = quant_dir / "aft_provenance.json"
    if not provenance_path.is_file():
        if quant_dir.exists() and any(quant_dir.glob("*.safetensors")):
            console.print(
                f"[yellow]⚠ Resume: {quant_dir} has weights but no"
                f" aft_provenance.json, so it cannot be verified. It may have"
                f" been produced with different settings.[/yellow]"
            )
        return

    try:
        prov = json.loads(provenance_path.read_text())
    except json.JSONDecodeError:
        console.print(f"[yellow]⚠ Resume: {provenance_path} is unreadable[/yellow]")
        return

    mismatches: list[str] = []
    prev_quant = prov.get("quantization", {})
    for label, previous, current in (
        ("bits", prev_quant.get("bits"), getattr(quant_cfg, "bits", None)),
        ("format", prev_quant.get("format"), getattr(quant_cfg, "format", None)),
        (
            "group_size",
            prev_quant.get("group_size"),
            getattr(quant_cfg, "group_size", None),
        ),
    ):
        if previous is not None and previous != current:
            mismatches.append(f"{label}: existing={previous}, requested={current}")

    prev_revision = prov.get("source_revision")
    if revision and prev_revision and prev_revision != revision:
        mismatches.append(f"revision: existing={prev_revision}, requested={revision}")

    if mismatches:
        console.print(
            "[red]✗ Refusing to resume: the existing artifact was built with"
            " different settings.[/red]"
        )
        for mismatch in mismatches:
            console.print(f"[red]    {mismatch}[/red]")
        console.print(
            f"[red]  Source model on disk: {prov.get('source_model')}[/red]\n"
            f"[red]  Delete {quant_dir} or drop --resume to rebuild.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[green]✓ Resume: verified existing artifact in {quant_dir}[/green]")


def _step_bar(active: int, steps: list[str]) -> None:
    """Render a pipeline step indicator."""
    parts: list[str] = []
    for i, step in enumerate(steps):
        if i < active:
            parts.append(f"[bold green]●[/bold green] [green]{step}[/green]")
        elif i == active:
            parts.append(
                f"[bold yellow]●[/bold yellow] [bold yellow]{step}[/bold yellow]"
            )
        else:
            parts.append(f"[dim]○ {step}[/dim]")
    console.print("    ".join(parts))
    console.print()


@app.command("recommend")
def recommend_cmd(
    model: Annotated[str, typer.Option("--model", help="HF repo ID of the model.")],
    token: Annotated[
        str | None,
        typer.Option("--token", help="HF API token. Falls back to HF_TOKEN."),
    ] = None,
) -> None:
    """Suggest QLoRA fine-tuning parameters for your hardware and model."""
    import torch

    _banner()

    from aft.recommend import detect_system_ram_mib, fetch_model_info, recommend

    with console.status("[bold cyan]Detecting hardware...[/bold cyan]", spinner="dots"):
        # Use torch.cuda as the single source of truth for GPU detection
        gpus: list[dict] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
                gpus.append({"name": name, "vram_mib": int(mem)})
        total_vram_mib = sum(g["vram_mib"] for g in gpus)
        total_ram_mib = detect_system_ram_mib()
        bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    console.print("  [bold cyan]⚡ Hardware[/bold cyan]")
    if gpus:
        for i, gpu in enumerate(gpus):
            label = "GPU " if len(gpus) == 1 else f"GPU {i}"
            gpu_str = f"{gpu['name']} [dim]({gpu['vram_mib'] / 1024:.1f} GiB)[/dim]"
            console.print(
                f"    [bold cyan]{label}[/bold cyan] [green]{gpu_str}[/green]"
            )
        if len(gpus) > 1:
            console.print(
                f"    [bold cyan]Total VRAM[/bold cyan]  "
                f"[green]{total_vram_mib / 1024:.1f} GiB across {len(gpus)} "
                "GPUs[/green]"
            )
    else:
        console.print("    [bold cyan]GPU[/bold cyan]  [red]None detected[/red]")

    if total_ram_mib is not None:
        ram_str = f"{total_ram_mib / 1024:.1f} GiB"
    else:
        ram_str = "[red]undetectable[/red]"
    bf16_str = "[green]✓[/green]" if bf16 else "[red]✗[/red]"
    console.print(
        f"    [bold cyan]RAM[/bold cyan]  [green]{ram_str}[/green]"
        f"  [bold cyan]BF16[/bold cyan] {bf16_str}"
        f"  [bold cyan]OS[/bold cyan]  [green]{platform.system()}[/green]"
    )
    console.print()

    if not gpus:
        console.print(
            "[yellow]⚠ No NVIDIA GPU detected — recommendations will be "
            "model-only (hardware-specific tuning unavailable).[/yellow]"
        )
        console.print()

    with console.status(
        f"[bold cyan]Fetching model info: {model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            model_info = fetch_model_info(model, token=token)
        except Exception as exc:
            console.print(f"[red]Failed to fetch model info: {exc}[/red]")
            raise typer.Exit(1) from exc

    console.print(f"  [bold cyan]🧠 {model}[/bold cyan]")
    arch_str = ", ".join(model_info.architectures) or "unknown"
    console.print(
        f"    [bold cyan]Params[/bold cyan] [green]{model_info.params_b:.1f}B[/green]"
        f"  [bold cyan]Type[/bold cyan] [green]{model_info.model_type}[/green]"
    )
    console.print(f"    [bold cyan]Arch[/bold cyan]   [green]{arch_str}[/green]")
    if model_info.hidden_size or model_info.num_layers:
        parts = []
        if model_info.hidden_size:
            parts.append(
                f"[bold cyan]Hidden[/bold cyan] [green]{model_info.hidden_size}[/green]"
            )
        if model_info.num_layers:
            parts.append(
                f"[bold cyan]Layers[/bold cyan] [green]{model_info.num_layers}[/green]"
            )
        console.print(f"    {'  '.join(parts)}")
    console.print()

    # When no GPU is detected, use a hypothetical 24 GiB VRAM but make the
    # assumption explicit to the user instead of burying it in an `or`.
    if not gpus:
        hypothetical_vram_mib = 24 * 1024
        console.print(
            f"[yellow]⚠ No GPU detected — computing a hypothetical"
            f" recommendation assuming {hypothetical_vram_mib / 1024:.0f}"
            f" GiB VRAM. Do not rely on these settings for actual"
            f" training.[/yellow]"
        )
        console.print()
        rec_vram_mib = hypothetical_vram_mib
    else:
        rec_vram_mib = total_vram_mib

    with console.status(
        "[bold cyan]Computing recommendations...[/bold cyan]", spinner="dots"
    ):
        rec = recommend(
            model_info=model_info,
            vram_mib=rec_vram_mib,
            ram_mib=total_ram_mib,
            bf16_supported=bf16,
            gpu_vram_mib=[g["vram_mib"] for g in gpus] or None,
        )

    rec_table = Table(
        show_header=False,
        title="[bold]⚙  Recommended Parameters[/bold]",
        title_style="bold cyan",
        box=box.ROUNDED,
        border_style="cyan",
        pad_edge=True,
    )
    rec_table.add_column("Parameter", style="bold cyan", width=30)
    rec_table.add_column("Value", style="bold green")
    rec_table.add_row("lora_rank", str(rec.lora_rank))
    rec_table.add_row("lora_alpha", str(rec.lora_alpha))
    rec_table.add_row("max_seq_len", str(rec.max_seq_len))
    rec_table.add_row("batch_size", str(rec.batch_size))
    rec_table.add_row("gradient_accumulation", str(rec.gradient_accumulation_steps))
    rec_table.add_row("learning_rate", f"{rec.learning_rate:.0e}")
    rec_table.add_row("epochs", str(rec.epochs))
    if rec.max_memory:
        mem_str = ", ".join(f"{k}: {v}" for k, v in rec.max_memory.items())
        rec_table.add_row("max_memory", mem_str)
    else:
        rec_table.add_row("max_memory", "auto")
    console.print(Panel(rec_table, border_style="green", expand=False))
    console.print()

    console.print("[bold]  💡 Reasoning:[/bold]")
    for reason in rec.reasoning:
        console.print(f"    [dim]•[/dim] {reason}")
    console.print()

    cmd_lines = [
        "aft run \\",
        f"  --model {model} \\",
        "  --dataset <DATASET> \\",
        "  --run-name <RUN_NAME> \\",
        "  --output ./models/<RUN_NAME> \\",
        f"  --lora-rank {rec.lora_rank} \\",
        f"  --max-seq-len {rec.max_seq_len} \\",
        f"  --epochs {rec.epochs} \\",
        f"  --batch-size {rec.batch_size} \\",
        f"  --grad-accum {rec.gradient_accumulation_steps} \\",
        f"  --learning-rate {rec.learning_rate:.0e}",
    ]
    from rich.markup import escape

    console.print("[bold]📋 Copy-paste command:[/bold]")
    for line in cmd_lines:
        console.print(f"  [dim]{escape(line)}[/dim]")
    console.print()


@app.command("run")
def run_cmd(
    model: Annotated[str, typer.Option("--model", help="Base model HF repo ID.")],
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="HF dataset repo ID(s), comma-separated."),
    ],
    run_name: Annotated[
        str, typer.Option("--run-name", help="Run name for output subdirectory.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Base output directory.")
    ] = Path("./models"),
    lora_rank: Annotated[int, typer.Option(help="LoRA rank.")] = 32,
    lora_alpha: Annotated[
        int | None, typer.Option(help="LoRA alpha. Defaults to 2x lora_rank.")
    ] = None,
    lora_dropout: Annotated[float, typer.Option(help="LoRA dropout rate.")] = 0.05,
    max_seq_len: Annotated[int, typer.Option(help="Max sequence length.")] = 2048,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 1,
    batch_size: Annotated[int, typer.Option(help="Per-device batch size.")] = 2,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps.")] = 8,
    learning_rate: Annotated[float, typer.Option(help="Learning rate.")] = 2e-4,
    max_samples: Annotated[
        int | None, typer.Option(help="Limit dataset samples.")
    ] = None,
    clean: Annotated[bool, typer.Option("--clean", help="Enable cleaning.")] = False,
    dedup: Annotated[bool, typer.Option("--dedup", help="Remove duplicates.")] = False,
    min_tokens: Annotated[int, typer.Option("--min-tokens")] = 10,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    languages: Annotated[
        str | None, typer.Option("--languages", help="Comma-separated lang codes.")
    ] = None,
    max_special_ratio: Annotated[float, typer.Option("--max-special-ratio")] = 0.3,
    skip_finetune: Annotated[
        bool, typer.Option("--skip-finetune", help="Skip SFT, go to merge + quantize.")
    ] = False,
    skip_quantize: Annotated[
        bool, typer.Option("--skip-quantize", help="Stop after merge.")
    ] = False,
    quant_type: Annotated[
        str,
        typer.Option(
            "--quant-type",
            help="Quantization type: int4, int8, or fp8.",
        ),
    ] = "int4",
    gptq_group_size: Annotated[int, typer.Option(help="GPTQ group size.")] = 128,
    calibration: Annotated[
        str, typer.Option(help="'fineweb-edu' or path to JSONL.")
    ] = "fineweb-edu",
    target_modules: Annotated[
        str | None,
        typer.Option(
            "--target-modules",
            help="Comma-separated LoRA target module names. Auto-detected"
            " from the model when omitted.",
        ),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Pin the base model to a git revision."),
    ] = None,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow loading models with custom code.")
    ] = False,
    base_model_only: Annotated[
        bool,
        typer.Option(
            "--base-model-only",
            help="Quantize the base model directly — no SFT, no adapter, no merge.",
        ),
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip phases whose output exists.")
    ] = False,
) -> None:
    """Full pipeline: QLoRA SFT → merge LoRA → quantize."""
    from aft.config import LORA_ALPHA_MULTIPLIER, TrainConfig
    from aft.errors import AftError
    from aft.pipeline import merge_adapter, quantize, train

    _banner()

    q_bits, q_format, q_label, q_dir_name = _resolve_quant_type(quant_type)

    steps = ["QLoRA SFT", "Merge LoRA", q_label]
    if base_model_only:
        steps = [q_label]

    dataset_ids = [d.strip() for d in dataset.split(",") if d.strip()]
    lang_list = (
        [lang.strip() for lang in languages.split(",") if lang.strip()]
        if languages
        else None
    )

    target_module_list = (
        [t.strip() for t in target_modules.split(",") if t.strip()]
        if target_modules
        else None
    )

    base_dir = output / run_name
    adapter_dir = base_dir / "adapter"
    merged_dir = base_dir / "merged"
    gptq_dir = base_dir / q_dir_name

    quant_cfg = _build_quant_config(
        bits=q_bits,
        format=q_format,
        group_size=gptq_group_size,
        calibration_dataset=calibration,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    # ── Direct base-model quantization (no SFT, no merge) ──────────────
    # The only viable route for models too large to train or merge locally.
    if base_model_only:
        console.print(
            "[yellow]⚠ --base-model-only: quantizing the base model"
            " directly, skipping SFT and merge.[/yellow]"
        )
        _step_bar(0, steps)
        try:
            quantize(Path(model), gptq_dir, quant_cfg)
        except AftError as exc:
            console.print(f"\n[red]✗ {exc}[/red]")
            raise typer.Exit(1) from exc
        _step_bar(len(steps), steps)
        console.print(f"[bold green]✓ {q_label} → {gptq_dir}[/bold green]")
        return

    # --resume: auto-detect what can be skipped
    skip_merge = False
    if resume:
        _verify_resume_compatibility(gptq_dir, model, quant_cfg, revision)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            skip_finetune = True
            console.print(
                "[yellow]⚠ Resume: adapter exists → skipping training[/yellow]"
            )
        if merged_dir.exists() and any(merged_dir.glob("*.safetensors")):
            skip_merge = True
            console.print(
                "[yellow]⚠ Resume: merged model exists → skipping merge[/yellow]"
            )
        if gptq_dir.exists() and any(gptq_dir.glob("*.safetensors")):
            skip_quantize = True
            console.print(
                "[yellow]⚠ Resume: quantized model exists → skipping quantize[/yellow]"
            )

    try:
        # ── Phase 1: QLoRA SFT ─────────────────────────────────────────
        _step_bar(0, steps)

        if not skip_finetune:
            cfg = TrainConfig(
                base_model=model,
                datasets=dataset_ids,
                run_name=run_name,
                output_dir=str(base_dir),
                lora_rank=lora_rank,
                lora_alpha=(
                    lora_alpha
                    if lora_alpha is not None
                    else lora_rank * LORA_ALPHA_MULTIPLIER
                ),
                lora_dropout=lora_dropout,
                max_seq_len=max_seq_len,
                num_epochs=epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                max_samples=max_samples,
                clean=clean,
                dedup=dedup,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                languages=lang_list,
                max_special_ratio=max_special_ratio,
                trust_remote_code=trust_remote_code,
                revision=revision,
                target_modules=target_module_list,
            )
            train(cfg)
        else:
            if not adapter_dir.exists():
                console.print(
                    f"[red]✗ --skip-finetune set but adapter not found:"
                    f" {adapter_dir}[/red]"
                )
                raise typer.Exit(1)
            console.print(
                f"[yellow]⚠ Skipping SFT — using existing adapter:"
                f" {adapter_dir}[/yellow]"
            )

        # ── Phase 2: Merge LoRA ────────────────────────────────────────
        _step_bar(1, steps)

        if skip_merge:
            console.print(
                f"[yellow]⚠ Skipping merge — using existing: {merged_dir}[/yellow]"
            )
        else:
            merge_adapter(
                model,
                adapter_dir,
                merged_dir,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )

        if skip_quantize:
            console.print(
                f"[yellow]⚠ Skipping quantization. Merged model: {merged_dir}[/yellow]"
            )
            return

        # ── Phase 3: GPTQ Quantize ────────────────────────────────────
        _step_bar(2, steps)

        quantize(merged_dir, gptq_dir, quant_cfg)

        _step_bar(len(steps), steps)

    except AftError as exc:
        console.print(f"\n[red]✗ {exc}[/red]")
        raise typer.Exit(1) from exc

    summary = Table(
        show_header=False,
        title="[bold green]✓ Pipeline complete[/bold green]",
        box=box.ROUNDED,
        border_style="green",
        pad_edge=True,
    )
    summary.add_column("Phase", style="bold cyan", width=12)
    summary.add_column("Path", style="white")
    summary.add_row("Adapter", str(adapter_dir))
    summary.add_row("Merged", str(merged_dir))
    summary.add_row(q_label, str(gptq_dir))
    console.print(Panel(summary, border_style="green", expand=False))


@app.command("quantize")
def quantize_cmd(
    merged_model: Annotated[
        Path, typer.Option("--model", help="Path to merged fp16 model directory.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Output directory for quantized model.")
    ],
    quant_type: Annotated[
        str,
        typer.Option(
            "--quant-type",
            help="Quantization type: int4, int8, or fp8.",
        ),
    ] = "int4",
    group_size: Annotated[int, typer.Option(help="GPTQ group size.")] = 128,
    desc_act: Annotated[
        bool, typer.Option("--desc-act", help="Use activation order (slower, better).")
    ] = False,
    calibration: Annotated[
        str, typer.Option(help="'fineweb-edu' or path to JSONL.")
    ] = "fineweb-edu",
    n_calibration_samples: Annotated[
        int, typer.Option(help="Number of calibration samples.")
    ] = 128,
    calibration_seq_len: Annotated[
        int, typer.Option(help="Calibration sequence length.")
    ] = 2048,
    no_chat_template: Annotated[
        bool,
        typer.Option(
            "--no-chat-template",
            help="Calibrate on raw text instead of applying the model's chat template.",
        ),
    ] = False,
    allow_partial_coverage: Annotated[
        bool,
        typer.Option(
            "--allow-partial-coverage",
            help="Permit layers to be left unquantized instead of failing.",
        ),
    ] = False,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Pin the model to a git revision."),
    ] = None,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow loading models with custom code.")
    ] = False,
    token: Annotated[
        str | None,
        typer.Option("--token", help="HF API token. Falls back to HF_TOKEN."),
    ] = None,
) -> None:
    """Quantize an already-merged model (GPTQ int4/int8 or FP8)."""
    from aft.errors import AftError
    from aft.pipeline import quantize

    _banner()

    q_bits, q_format, _, _ = _resolve_quant_type(quant_type)
    vllm_flag = "fp8" if q_format == "fp8" else "gptq_marlin"

    cfg = _build_quant_config(
        bits=q_bits,
        format=q_format,
        group_size=group_size,
        desc_act=desc_act,
        calibration_dataset=calibration,
        n_calibration_samples=n_calibration_samples,
        calibration_seq_len=calibration_seq_len,
        trust_remote_code=trust_remote_code,
        revision=revision,
        use_chat_template=not no_chat_template,
        strict_layer_coverage=not allow_partial_coverage,
    )
    try:
        quantize(merged_model, output, cfg, token=token)
    except AftError as exc:
        console.print(f"\n[red]✗ {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        Panel(
            f"[dim]vllm serve {output} --quantization {vllm_flag}[/dim]",
            title="[bold]🚀 Serve with vLLM[/bold]",
            border_style="yellow",
            expand=False,
        )
    )


@app.command("push")
def push_cmd(
    model: Annotated[
        Path, typer.Option("--model", help="Local path to quantized model directory.")
    ],
    repo_id: Annotated[str, typer.Option("--repo-id", help="HuggingFace repo ID.")],
    private: Annotated[
        bool, typer.Option("--private", help="Create as private.")
    ] = False,
    token: Annotated[str | None, typer.Option("--token", help="HF API token.")] = None,
    message: Annotated[
        str, typer.Option("--message", help="Commit message.")
    ] = "Upload GPTQ quantized model",
) -> None:
    """Publish a quantized model to HuggingFace Hub."""
    from aft.pipeline import push_to_hub

    _banner()

    if not model.exists():
        console.print(f"[red]Model directory not found: {model}[/red]")
        raise typer.Exit(1)

    url = push_to_hub(
        model_path=model,
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=message,
    )
    console.print(
        Panel(
            f"[bold green]{url}[/bold green]",
            title="[bold]📦 Published to HuggingFace Hub[/bold]",
            border_style="green",
            expand=False,
        )
    )
