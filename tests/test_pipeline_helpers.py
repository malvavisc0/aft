"""Unit tests for aft.pipeline architecture-introspection helpers.

These helpers are what stand between the pipeline and *silently* wrong
artifacts (unquantized layers, meta-device garbage weights, a merged model
that loads at the wrong dtype), so they are tested directly rather than only
through the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aft.errors import AftError
from aft.model_utils import (
    discover_lora_targets,
    is_multimodal,
    materialize_meta_params,
    resolve_dtype,
    shards_for,
)
from aft.quantize import report_layer_coverage


class TestResolveDtype:
    def test_reads_modern_dtype_key(self) -> None:
        cfg = SimpleNamespace(dtype="bfloat16")
        assert resolve_dtype(cfg) is torch.bfloat16

    def test_reads_legacy_torch_dtype_key(self) -> None:
        cfg = SimpleNamespace(torch_dtype="float16")
        assert resolve_dtype(cfg) is torch.float16

    def test_accepts_real_dtype_objects(self) -> None:
        cfg = SimpleNamespace(dtype=torch.float32)
        assert resolve_dtype(cfg) is torch.float32

    def test_prefers_dtype_over_torch_dtype(self) -> None:
        cfg = SimpleNamespace(dtype="float16", torch_dtype="float32")
        assert resolve_dtype(cfg) is torch.float16

    def test_falls_back_to_nested_text_config(self) -> None:
        """Multimodal configs put the LM dtype under text_config."""
        cfg = SimpleNamespace(
            dtype=None,
            torch_dtype=None,
            text_config=SimpleNamespace(dtype="bfloat16"),
        )
        assert resolve_dtype(cfg) is torch.bfloat16

    def test_raises_when_dtype_absent(self) -> None:
        """No dtype to resolve → raise, don't silently guess."""
        with pytest.raises(AftError, match="Could not resolve"):
            resolve_dtype(SimpleNamespace())

    def test_raises_on_unrecognized_dtype_string(self) -> None:
        cfg = SimpleNamespace(dtype="not_a_real_dtype")
        with pytest.raises(AftError, match="Could not resolve"):
            resolve_dtype(cfg)


class TestIsMultimodal:
    def test_detects_conditional_generation_architecture(self) -> None:
        cfg = SimpleNamespace(architectures=["NexN2ForConditionalGeneration"])
        assert is_multimodal(cfg) is True

    def test_detects_vision_config(self) -> None:
        cfg = SimpleNamespace(architectures=[], vision_config=SimpleNamespace())
        assert is_multimodal(cfg) is True

    def test_plain_causal_lm_is_not_multimodal(self) -> None:
        cfg = SimpleNamespace(architectures=["LlamaForCausalLM"], vision_config=None)
        assert is_multimodal(cfg) is False


class _TinyModel(torch.nn.Module):
    """Model whose projection names mimic a real transformer block."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.router = torch.nn.Linear(4, 2)  # not a LoRA candidate


class TestDiscoverLoraTargets:
    def test_discovers_only_present_projections(self) -> None:
        targets = discover_lora_targets(_TinyModel())
        assert targets == ["q_proj", "v_proj"]

    def test_excludes_non_candidate_linears(self) -> None:
        assert "router" not in discover_lora_targets(_TinyModel())

    def test_explicit_list_wins(self) -> None:
        assert discover_lora_targets(_TinyModel(), ["custom_proj"]) == ["custom_proj"]

    def test_raises_when_nothing_matches(self) -> None:
        """Better to fail loudly than train an adapter that touches nothing."""

        class Unknown(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weird_name = torch.nn.Linear(4, 4)

        with pytest.raises(AftError, match="Could not discover any LoRA target"):
            discover_lora_targets(Unknown())


class _FakeQuantLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qweight = torch.zeros(2, 2, dtype=torch.int32)


class TestReportLayerCoverage:
    def test_strict_mode_raises_on_partial_coverage(self) -> None:
        model = torch.nn.Module()
        model.quantized = _FakeQuantLinear()
        model.skipped = torch.nn.Linear(4, 4)

        with pytest.raises(AftError, match="unquantized"):
            report_layer_coverage(model, strict=True)

    def test_non_strict_mode_only_warns(self) -> None:
        model = torch.nn.Module()
        model.quantized = _FakeQuantLinear()
        model.skipped = torch.nn.Linear(4, 4)

        report_layer_coverage(model, strict=False)  # must not raise

    def test_full_coverage_passes_strict(self) -> None:
        model = torch.nn.Module()
        model.a = _FakeQuantLinear()
        model.b = _FakeQuantLinear()

        report_layer_coverage(model, strict=True)


class TestShardsFor:
    def test_uses_index_map_to_select_only_needed_shards(self, tmp_path: Path) -> None:
        (tmp_path / "model-00001.safetensors").touch()
        (tmp_path / "model-00002.safetensors").touch()
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a.weight": "model-00001.safetensors",
                        "b.weight": "model-00002.safetensors",
                    }
                }
            )
        )

        shards = shards_for(tmp_path, {"a.weight"})
        assert shards == [tmp_path / "model-00001.safetensors"]

    def test_falls_back_to_glob_without_index(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").touch()
        assert shards_for(tmp_path, {"a.weight"}) == [tmp_path / "model.safetensors"]

    def test_raises_on_malformed_index(self, tmp_path: Path) -> None:
        """A corrupt index file should raise, not silently fall back."""
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "model.safetensors.index.json").write_text("this is not valid json")
        with pytest.raises(AftError, match="Malformed safetensors index"):
            shards_for(tmp_path, {"a.weight"})


class TestMaterializeMetaParams:
    def test_no_op_when_nothing_on_meta(self, tmp_path: Path) -> None:
        assert materialize_meta_params(_TinyModel(), tmp_path) == 0

    def test_loads_meta_params_from_checkpoint(self, tmp_path: Path) -> None:
        from safetensors.torch import save_file

        expected = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        save_file({"q_proj.weight": expected}, str(tmp_path / "model.safetensors"))

        model = _TinyModel()
        with torch.device("meta"):
            model.q_proj.weight = torch.nn.Parameter(torch.empty(4, 4))
        assert model.q_proj.weight.is_meta

        count = materialize_meta_params(model, tmp_path)

        assert count == 1
        assert not model.q_proj.weight.is_meta
        torch.testing.assert_close(model.q_proj.weight.data, expected)

    def test_raises_when_checkpoint_lacks_the_weight(self, tmp_path: Path) -> None:
        """Quantizing an incompletely materialized model yields garbage."""
        from safetensors.torch import save_file

        save_file(
            {"unrelated.weight": torch.zeros(2, 2)},
            str(tmp_path / "model.safetensors"),
        )

        model = _TinyModel()
        with torch.device("meta"):
            model.q_proj.weight = torch.nn.Parameter(torch.empty(4, 4))

        with pytest.raises(AftError, match="no value in the checkpoint"):
            materialize_meta_params(model, tmp_path)

    def test_raises_when_no_shards_exist(self, tmp_path: Path) -> None:
        model = _TinyModel()
        with torch.device("meta"):
            model.q_proj.weight = torch.nn.Parameter(torch.empty(4, 4))

        with pytest.raises(AftError, match="no"):
            materialize_meta_params(model, tmp_path)
