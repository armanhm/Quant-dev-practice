import pytest
from quantflow.assistant.tools import execute_tool, TOOL_DEFINITIONS


class TestTools:
    def test_list_strategies_tool(self):
        result = execute_tool("list_strategies", {})
        assert "sma_crossover" in result
        assert "mean_reversion" in result

    def test_tool_definitions_valid(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_explain_concept(self):
        result = execute_tool("explain_concept", {"topic": "Sharpe ratio"})
        assert "Sharpe" in result

    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {})
        assert "Unknown" in result
