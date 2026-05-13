"""Smoke tests for the aft CLI — verify commands parse and --help works."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from aft.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes so we can assert on plain text."""
    return _ANSI_RE.sub("", text)


class TestCliHelp:
    def test_aft_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune and quantize" in _plain(result.output)

    def test_recommend_help(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "--model" in _plain(result.output)

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--model" in out
        assert "--dataset" in out

    def test_quantize_help(self) -> None:
        result = runner.invoke(app, ["quantize", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--model" in out
        assert "--bits" in out

    def test_push_help(self) -> None:
        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0
        assert "--repo-id" in _plain(result.output)

    def test_no_args_shows_help(self) -> None:
        """no_args_is_help=True makes typer exit with code 0."""
        result = runner.invoke(app, [])
        # typer with no_args_is_help=True uses exit code 0
        # but some versions use 2; accept either
        assert result.exit_code in (0, 2)
        out = _plain(result.output)
        assert "Fine-tune" in out or "Usage" in out
