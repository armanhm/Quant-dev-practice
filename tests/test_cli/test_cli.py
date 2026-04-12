# tests/test_cli/test_cli.py
import pytest
from click.testing import CliRunner
from quantflow.cli.main import cli


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "QuantFlow" in result.output

    def test_strategy_list(self):
        result = self.runner.invoke(cli, ["strategy", "list"])
        assert result.exit_code == 0
        assert "sma_crossover" in result.output
        assert "mean_reversion" in result.output

    def test_data_list_empty(self):
        result = self.runner.invoke(cli, ["data", "list"])
        assert result.exit_code == 0
