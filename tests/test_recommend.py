"""Unit tests for aft.recommend — hardware detection and parameter recommendation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aft.config import ModelInfo
from aft.recommend import detect_system_ram_mib, fetch_model_info, recommend

# ── detect_system_ram_mib ──────────────────────────────────────────────────


class TestDetectSystemRam:
    def test_valid_meminfo(self) -> None:
        fake_content = "MemTotal:       32768000 kB\nMemFree:        16384000 kB\n"
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: iter(fake_content.splitlines(True))
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=mock_file):
            ram = detect_system_ram_mib()
        assert ram == 32768000 // 1024  # 32000 MiB

    def test_file_not_found(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            ram = detect_system_ram_mib()
        assert ram == 0

    def test_invalid_content(self) -> None:
        fake_content = "MemTotal:       not-a-number kB\n"
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: iter(fake_content.splitlines(True))
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=mock_file):
            ram = detect_system_ram_mib()
        assert ram == 0


# ── fetch_model_info ───────────────────────────────────────────────────────


class TestFetchModelInfo:
    def test_from_safetensors(self) -> None:
        mock_info = MagicMock()
        mock_info.safetensors.total = 7_000_000_000
        mock_info.config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_info
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            info = fetch_model_info("meta-llama/Llama-2-7b")
        assert info.params_b == pytest.approx(7.0)
        assert info.architectures == ["LlamaForCausalLM"]
        assert info.model_type == "llama"
        assert info.hidden_size == 4096
        assert info.num_layers == 32

    def test_regex_fallback_when_no_safetensors(self) -> None:
        """When safetensors is missing, falls through to regex on repo name."""
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.config = None
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_info

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            info = fetch_model_info("Qwen/Qwen2.5-7B")
        # Falls through to regex: "7B" in repo name
        assert info.params_b == 7.0

    def test_from_repo_name_regex(self) -> None:
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.config = None
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_info
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            info = fetch_model_info("my-org/SomeModel-13B-chat")
        assert info.params_b == 13.0

    def test_unknown_model(self) -> None:
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.config = None
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_info
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            info = fetch_model_info("org/unknown-model")
        assert info.params_b == 0.0
        assert info.model_type == "unknown"


# ── recommend ──────────────────────────────────────────────────────────────


class TestRecommend:
    def _make_info(self, params_b: float) -> ModelInfo:
        return ModelInfo(
            repo_id="test/model",
            params_b=params_b,
            model_type="llama",
            architectures=["LlamaForCausalLM"],
        )

    def test_small_model_8gb(self) -> None:
        info = self._make_info(1.0)
        rec = recommend(info, vram_mib=8192, ram_mib=32768, bf16_supported=True)
        assert rec.lora_rank == 8
        assert rec.learning_rate == pytest.approx(2e-4)
        assert rec.epochs == 3
        assert rec.lora_alpha == 16

    def test_medium_model_16gb(self) -> None:
        info = self._make_info(7.0)
        rec = recommend(info, vram_mib=16384, ram_mib=32768, bf16_supported=True)
        assert rec.lora_rank == 16
        assert rec.learning_rate == pytest.approx(2e-4)
        assert rec.epochs == 2

    def test_large_model_24gb(self) -> None:
        info = self._make_info(13.0)
        rec = recommend(info, vram_mib=24576, ram_mib=65536, bf16_supported=True)
        assert rec.lora_rank == 32
        assert rec.learning_rate == pytest.approx(1e-4)
        assert rec.epochs == 2

    def test_xlarge_model_80gb(self) -> None:
        info = self._make_info(70.0)
        rec = recommend(info, vram_mib=81920, ram_mib=131072, bf16_supported=True)
        assert rec.lora_rank == 64
        assert rec.learning_rate == pytest.approx(2e-5)
        assert rec.epochs == 1

    def test_massive_model(self) -> None:
        info = self._make_info(100.0)
        rec = recommend(info, vram_mib=81920, ram_mib=131072, bf16_supported=True)
        assert rec.lora_rank == 64
        assert rec.epochs == 1

    def test_tight_vram_cpu_offload(self) -> None:
        # 7B on 8GB: tight VRAM → CPU offload
        info = self._make_info(7.0)
        rec2 = recommend(info, vram_mib=8192, ram_mib=32768, bf16_supported=False)
        # 7B on 8GB: weights ~3.85 GiB, available 6.8, remaining ~0.95 → tight
        assert rec2.batch_size == 1
        assert rec2.max_seq_len == 512
        assert rec2.max_memory is not None

    def test_0_vram(self) -> None:
        info = self._make_info(1.0)
        rec = recommend(info, vram_mib=0, ram_mib=32768, bf16_supported=False)
        # 0 VRAM: available = 0, remaining = -0.55 - 2.0 = negative → tight
        assert rec.batch_size == 1
        assert rec.max_seq_len == 512

    def test_bf16_supported_in_reasoning(self) -> None:
        info = self._make_info(1.0)
        rec = recommend(info, vram_mib=8192, ram_mib=32768, bf16_supported=True)
        bf16_reasons = [r for r in rec.reasoning if "BF16" in r]
        assert len(bf16_reasons) == 1

    def test_bf16_not_supported(self) -> None:
        info = self._make_info(1.0)
        rec = recommend(info, vram_mib=8192, ram_mib=32768, bf16_supported=False)
        bf16_reasons = [r for r in rec.reasoning if "BF16" in r]
        assert len(bf16_reasons) == 0

    def test_grad_accum_target_16(self) -> None:
        info = self._make_info(7.0)
        rec = recommend(info, vram_mib=24576, ram_mib=65536, bf16_supported=True)
        effective = rec.batch_size * rec.gradient_accumulation_steps
        assert effective >= 16
