"""Unit tests for aft.config — configuration dataclasses."""

from __future__ import annotations

from aft.config import ModelInfo, QuantizeConfig, Recommendation, TrainConfig


class TestTrainConfigDefaults:
    def test_required_fields(self) -> None:
        cfg = TrainConfig(base_model="meta-llama/Llama-2-7b")
        assert cfg.base_model == "meta-llama/Llama-2-7b"
        assert cfg.datasets == []

    def test_default_lora(self) -> None:
        cfg = TrainConfig(base_model="test")
        assert cfg.lora_rank == 32
        assert cfg.lora_alpha == 64
        assert cfg.lora_dropout == 0.05

    def test_default_training(self) -> None:
        cfg = TrainConfig(base_model="test")
        assert cfg.max_seq_len == 2048
        assert cfg.num_epochs == 1
        assert cfg.per_device_batch_size == 2
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.learning_rate == 2e-4
        assert cfg.warmup_ratio == 0.03

    def test_default_cleaning(self) -> None:
        cfg = TrainConfig(base_model="test")
        assert cfg.clean is False
        assert cfg.dedup is False
        assert cfg.min_tokens == 10
        assert cfg.max_tokens is None
        assert cfg.languages is None
        assert cfg.max_special_ratio == 0.3

    def test_default_trust_remote_code(self) -> None:
        cfg = TrainConfig(base_model="test")
        assert cfg.trust_remote_code is False

    def test_custom_values(self) -> None:
        cfg = TrainConfig(
            base_model="test",
            lora_rank=16,
            learning_rate=1e-4,
            trust_remote_code=True,
        )
        assert cfg.lora_rank == 16
        assert cfg.learning_rate == 1e-4
        assert cfg.trust_remote_code is True


class TestQuantizeConfigDefaults:
    def test_defaults(self) -> None:
        cfg = QuantizeConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.desc_act is False
        assert cfg.calibration_dataset == "wikitext2"
        assert cfg.n_calibration_samples == 128
        assert cfg.calibration_seq_len == 2048
        assert cfg.trust_remote_code is False

    def test_custom_values(self) -> None:
        cfg = QuantizeConfig(bits=8, trust_remote_code=True)
        assert cfg.bits == 8
        assert cfg.trust_remote_code is True


class TestModelInfo:
    def test_creation(self) -> None:
        info = ModelInfo(
            repo_id="test/model",
            params_b=7.0,
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=4096,
            num_layers=32,
        )
        assert info.repo_id == "test/model"
        assert info.params_b == 7.0
        assert info.hidden_size == 4096

    def test_optional_fields(self) -> None:
        info = ModelInfo(
            repo_id="test/model",
            params_b=0.0,
            model_type="unknown",
            architectures=[],
        )
        assert info.hidden_size is None
        assert info.num_layers is None


class TestRecommendation:
    def test_creation(self) -> None:
        rec = Recommendation(
            lora_rank=32,
            lora_alpha=64,
            max_seq_len=2048,
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            epochs=1,
            max_memory=None,
            reasoning=["test reason"],
        )
        assert rec.lora_rank == 32
        assert rec.reasoning == ["test reason"]
        assert rec.max_memory is None

    def test_with_max_memory(self) -> None:
        rec = Recommendation(
            lora_rank=8,
            lora_alpha=16,
            max_seq_len=512,
            batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            epochs=3,
            max_memory={"0": "6GiB", "cpu": "24GiB"},
            reasoning=[],
        )
        assert rec.max_memory is not None
        assert rec.max_memory["0"] == "6GiB"
