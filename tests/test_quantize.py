"""Unit tests for aft.quantize post-save sanitization and calibration helpers.

The sanitizer exists because gptqmodel injects two kinds of blemishes into
the saved ``config.json`` / ``quantize_config.json`` that must never reach a
published artifact: a spurious top-level ``rope_parameters`` (contradicting
the real one under ``text_config``) and a leaked ``/tmp/...`` offload path.

Row-flattening behavior (flat text vs message lists) is tested via
``aft.cleaning.flatten_row_to_text`` in ``tests/test_cleaning.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aft.quantize import sanitize_saved_config


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


class TestSanitizeRopeParameters:
    def test_removes_spurious_top_level_rope_when_text_config_has_one(
        self, tmp_path: Path
    ) -> None:
        cfg = {
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "text_config": {
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "rope_type": "default",
                    "mrope_section": [11, 11, 10],
                }
            },
        }
        _write_json(tmp_path / "config.json", cfg)
        (tmp_path / "quantize_config.json").write_text("{}")

        sanitize_saved_config(tmp_path)

        out = _read_json(tmp_path / "config.json")
        assert "rope_parameters" not in out
        # The legitimate nested RoPE config is untouched.
        assert out["text_config"]["rope_parameters"]["rope_theta"] == 10000000

    def test_keeps_top_level_rope_for_text_only_models(self, tmp_path: Path) -> None:
        """A plain causal-LM has no text_config; its top-level rope is real."""
        cfg = {"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}}
        _write_json(tmp_path / "config.json", cfg)
        (tmp_path / "quantize_config.json").write_text("{}")

        sanitize_saved_config(tmp_path)

        out = _read_json(tmp_path / "config.json")
        assert out["rope_parameters"]["rope_theta"] == 1000000.0

    def test_keeps_top_level_when_text_config_lacks_rope(self, tmp_path: Path) -> None:
        cfg = {
            "rope_parameters": {"rope_theta": 10000.0},
            "text_config": {"hidden_size": 4096},
        }
        _write_json(tmp_path / "config.json", cfg)
        (tmp_path / "quantize_config.json").write_text("{}")

        sanitize_saved_config(tmp_path)

        out = _read_json(tmp_path / "config.json")
        assert "rope_parameters" in out


class TestSanitizeLeakedPaths:
    def test_strips_offload_path_from_config_json_meta(self, tmp_path: Path) -> None:
        cfg = {
            "quantization_config": {
                "bits": 4,
                "meta": {
                    "offload_to_disk_path": "/tmp/gptqmodel_cfwi2by5",
                    "offload_to_disk": True,
                    "damp_percent": 0.05,
                },
            }
        }
        _write_json(tmp_path / "config.json", cfg)
        _write_json(
            tmp_path / "quantize_config.json",
            {"meta": {"offload_to_disk_path": "/tmp/gptqmodel_cfwi2by5"}},
        )

        sanitize_saved_config(tmp_path)

        main = _read_json(tmp_path / "config.json")
        qcfg = _read_json(tmp_path / "quantize_config.json")
        assert "offload_to_disk_path" not in main["quantization_config"]["meta"]
        assert "offload_to_disk_path" not in qcfg["meta"]
        # The non-path meta is preserved.
        assert main["quantization_config"]["meta"]["offload_to_disk"] is True
        assert main["quantization_config"]["meta"]["damp_percent"] == 0.05

    def test_noop_when_already_clean(self, tmp_path: Path) -> None:
        cfg = {"text_config": {"rope_parameters": {"rope_theta": 1e7}}}
        raw = json.dumps(cfg, indent=2) + "\n"
        (tmp_path / "config.json").write_text(raw)
        (tmp_path / "quantize_config.json").write_text(
            json.dumps({"meta": {"damp_percent": 0.05}}, indent=2) + "\n"
        )

        sanitize_saved_config(tmp_path)

        assert (tmp_path / "config.json").read_text() == raw

    def test_missing_files_are_noop(self, tmp_path: Path) -> None:
        # Must not raise when the config files were never written.
        sanitize_saved_config(tmp_path)
