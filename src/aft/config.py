"""Configuration dataclasses for the fine-tuning pipeline."""

from dataclasses import dataclass, field

#: Convention: LoRA alpha is 2× the rank. Centralised so CLI, recommendation,
#: and config all agree on the same multiplier.
LORA_ALPHA_MULTIPLIER: int = 2


@dataclass
class TrainConfig:
    """QLoRA SFT phase configuration.

    Core fields (``base_model``, ``datasets``, ``run_name``) are required;
    training hyper-parameters have sensible defaults.
    """

    base_model: str
    datasets: list[str] = field(default_factory=list)
    run_name: str = ""
    output_dir: str | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_len: int = 2048
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_samples: int | None = None
    max_memory: dict | None = None
    # Dataset cleaning
    clean: bool = False
    dedup: bool = False
    min_tokens: int = 10
    max_tokens: int | None = None
    languages: list[str] | None = None
    max_special_ratio: float = 0.3
    trust_remote_code: bool = False
    revision: str | None = None
    #: Explicit LoRA target module names. When ``None`` the modules are
    #: discovered from the loaded model instead of being hard-coded.
    target_modules: list[str] | None = None


@dataclass
class QuantizeConfig:
    """GPTQ / FP8 quantization phase configuration."""

    bits: int = 4
    #: 128 is what vLLM's ``gptq_marlin`` kernel expects (or -1 per-column).
    group_size: int = 128
    desc_act: bool = False
    format: str = "gptq"  # "gptq" | "fp8"
    calibration_dataset: str = "fineweb-edu"
    n_calibration_samples: int = 128
    calibration_seq_len: int = 2048
    trust_remote_code: bool = False
    revision: str | None = None
    #: Apply the model's chat template to calibration texts when available.
    use_chat_template: bool = True
    #: Fail if any quantizable layer is unexpectedly skipped by GPTQModel.
    strict_layer_coverage: bool = True


@dataclass
class ModelInfo:
    """Metadata fetched from HuggingFace Hub for a model."""

    repo_id: str
    params_b: float
    model_type: str
    architectures: list[str]
    hidden_size: int | None = None
    num_layers: int | None = None
    #: Distinct values from ``layer_types`` (hybrid attention models).
    layer_types: list[str] = field(default_factory=list)
    #: Number of MoE experts, when the model is sparse.
    num_experts: int | None = None
    #: Bytes per parameter of the checkpoint dtype (for size estimates).
    dtype_bytes: int = 2
    #: Resolved commit SHA of the fetched revision.
    revision: str | None = None


@dataclass
class Recommendation:
    """Recommended QLoRA SFT hyper-parameters for a given hardware + model."""

    lora_rank: int
    lora_alpha: int
    max_seq_len: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    max_memory: dict | None
    reasoning: list[str]
