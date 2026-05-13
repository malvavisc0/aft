"""Smoke tests for the aft CLI — verify commands parse and --help works."""

from __future__ import annotations

from typer.testing import CliRunner

from aft.cli import app

runner = CliRunner()


class TestCliHelp:
    def test_aft_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune and quantize" in result.output

    def test_recommend_help(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--dataset" in result.output

    def test_quantize_help(self) -> None:
        result = runner.invoke(app, ["quantize", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--bits" in result.output

    def test_push_help(self) -> None:
        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0
        assert "--repo-id" in result.output

    def test_no_args_shows_help(self) -> None:
        """no_args_is_help=True makes typer exit with code 0."""
        result = runner.invoke(app, [])
        # typer with no_args_is_help=True uses exit code 0
        # but some versions use 2; accept either
        assert result.exit_code in (0, 2)
        assert "Fine-tune" in result.output or "Usage" in result.output
