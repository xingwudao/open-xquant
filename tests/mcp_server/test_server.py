from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture()
def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=PROJECT_ROOT,
    )


async def _list_tools(server_params: StdioServerParameters) -> list[str]:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


def test_server_lists_data_tools(server_params: StdioServerParameters) -> None:
    tool_names = asyncio.run(_list_tools(server_params))
    assert "data_load_symbols" in tool_names
    assert "data_list_symbols" in tool_names
    assert "data_inspect" in tool_names


def test_server_lists_universe_tools(server_params: StdioServerParameters) -> None:
    tool_names = asyncio.run(_list_tools(server_params))
    assert "universe_set" in tool_names
    assert "universe_list_indexes" in tool_names
    assert "universe_inspect" in tool_names
    assert "universe_history" in tool_names


def test_server_lists_factor_tools(server_params: StdioServerParameters) -> None:
    tool_names = asyncio.run(_list_tools(server_params))
    assert "factor_download" in tool_names
    assert "factor_list" in tool_names
    assert "factor_inspect" in tool_names


def test_server_lists_mcp_only_tools(server_params: StdioServerParameters) -> None:
    tool_names = asyncio.run(_list_tools(server_params))
    assert "get_current_date" in tool_names


async def _call_list_symbols(
    server_params: StdioServerParameters, data_dir: str,
) -> dict:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "data_list_symbols", {"data_dir": data_dir},
            )
            return json.loads(result.content[0].text)


def test_server_call_list_symbols(
    server_params: StdioServerParameters, tmp_path: Path,
) -> None:
    result = asyncio.run(_call_list_symbols(server_params, str(tmp_path)))
    assert result["count"] == 0
    assert result["symbols"] == []
