"""Configuration dataclasses for the fine-tuning pipeline."""

from dataclasses import dataclass, field


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


@dataclass
class QuantizeConfig:
    """GPTQ quantization phase configuration."""

    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    calibration_dataset: str = "wikitext2"
    n_calibration_samples: int = 128
    calibration_seq_len: int = 2048


@dataclass
class ModelInfo:
    """Metadata fetched from HuggingFace Hub for a model."""

    repo_id: str
    params_b: float
    model_type: str
    architectures: list[str]
    hidden_size: int | None = None
    num_layers: int | None = None


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
