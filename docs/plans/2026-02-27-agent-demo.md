# Agent Demo (MCP Tools + Skill + Streamlit) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete Agent testing/teaching pipeline: MCP data tools → MCP Server (stdio) → Streamlit Agent page with OpenAI-compatible LLM + Data Explorer skill.

**Architecture:** MCP Server runs as a subprocess with stdio transport. Streamlit page acts as MCP Client, connecting via `stdio_client`. LLM uses OpenAI function calling to invoke tools through the MCP protocol. Skill is injected as system prompt.

**Tech Stack:** Python 3.12+, mcp SDK (server + client), openai SDK, streamlit, oxq.data

---

### Task 1: MCP Data Tools

**Files:**
- Modify: `mcp_server/tools/data_tools.py`
- Create: `tests/mcp_server/test_data_tools.py`
- Create: `tests/mcp_server/__init__.py`

**Context:** These are thin wrapper functions that call `oxq.data` SDK. They receive a `MCPServer` instance and register 3 tools via `@mcp.tool()`. Each tool: parse params → call SDK → return dict.

**Step 1: Write the failing tests**

```python
# tests/mcp_server/__init__.py
# (empty)
```

```python
# tests/mcp_server/test_data_tools.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mcp.server.mcpserver import MCPServer

from mcp_server.tools.data_tools import register


@pytest.fixture()
def mcp_app() -> MCPServer:
    app = MCPServer("test")
    register(app)
    return app


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")
    df.to_parquet(tmp_path / "MSFT.parquet")
    return tmp_path


def test_register_adds_three_tools(mcp_app: MCPServer) -> None:
    # MCPServer stores tools internally; verify by listing tool names
    assert mcp_app._tool_manager is not None
    tool_names = list(mcp_app._tool_manager._tools.keys())
    assert "data_load_symbols" in tool_names
    assert "data_list_symbols" in tool_names
    assert "data_inspect" in tool_names


def test_list_symbols(sample_data_dir: Path) -> None:
    from mcp_server.tools.data_tools import _list_symbols

    result = _list_symbols(data_dir=str(sample_data_dir))
    assert set(result["symbols"]) == {"AAPL", "MSFT"}
    assert result["count"] == 2


def test_list_symbols_empty(tmp_path: Path) -> None:
    from mcp_server.tools.data_tools import _list_symbols

    result = _list_symbols(data_dir=str(tmp_path))
    assert result["symbols"] == []
    assert result["count"] == 0


def test_inspect_symbol(sample_data_dir: Path) -> None:
    from mcp_server.tools.data_tools import _inspect

    result = _inspect(symbol="AAPL", data_dir=str(sample_data_dir))
    assert result["symbol"] == "AAPL"
    assert result["rows"] == 5
    assert result["columns"] == ["open", "high", "low", "close", "volume"]
    assert "date_range" in result


def test_inspect_missing_symbol(tmp_path: Path) -> None:
    from mcp_server.tools.data_tools import _inspect

    result = _inspect(symbol="UNKNOWN", data_dir=str(tmp_path))
    assert result["error"] is not None
    assert "UNKNOWN" in result["error"]


def test_load_symbols_yfinance(tmp_path: Path) -> None:
    from mcp_server.tools.data_tools import _load_symbols

    mock_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
    )
    with patch("oxq.data.loaders.yfinance") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        result = _load_symbols(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-12-31",
            source="yfinance",
            data_dir=str(tmp_path),
        )

    assert "AAPL" in result["rows"]
    assert result["rows"]["AAPL"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/mcp_server/test_data_tools.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

```python
# mcp_server/tools/data_tools.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oxq.core.errors import DownloadError, SymbolNotFoundError
from oxq.data.loaders import resolve_data_dir


def _list_symbols(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available symbols."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    if not path.exists():
        return {"symbols": [], "count": 0, "data_dir": str(path)}
    symbols = sorted(p.stem for p in path.glob("*.parquet"))
    return {"symbols": symbols, "count": len(symbols), "data_dir": str(path)}


def _inspect(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
    """Inspect a symbol's local data."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    parquet_path = path / f"{symbol}.parquet"
    if not parquet_path.exists():
        return {"symbol": symbol, "error": f"No data for '{symbol}'. Run data_load_symbols first."}
    df = pd.read_parquet(parquet_path)
    return {
        "symbol": symbol,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "missing_values": int(df.isna().sum().sum()),
        "sample_head": df.head(3).to_dict(orient="index"),
    }


def _load_symbols(
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download symbols from an external source."""
    dest = Path(data_dir) if data_dir else None

    if source == "yfinance":
        from oxq.data import YFinanceDownloader
        dl = YFinanceDownloader()
    elif source == "akshare":
        from oxq.data import AkShareDownloader
        dl = AkShareDownloader()
    else:
        return {"error": f"Unknown source '{source}'. Use 'yfinance' or 'akshare'."}

    rows: dict[str, int] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            dl.download(sym, start, end, dest_dir=dest)
            df = pd.read_parquet(resolve_data_dir(dest) / f"{sym}.parquet")
            rows[sym] = len(df)
        except (DownloadError, Exception) as e:
            errors[sym] = str(e)

    result: dict[str, Any] = {
        "symbols": list(rows.keys()),
        "rows": rows,
        "data_dir": str(resolve_data_dir(dest)),
    }
    if errors:
        result["errors"] = errors
    return result


def register(mcp: Any) -> None:
    """Register data tools on the MCP server instance."""

    @mcp.tool(name="data_load_symbols", description="Download market data for given symbols")
    def data_load_symbols(
        symbols: list[str],
        start: str,
        end: str,
        source: str = "yfinance",
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        return _load_symbols(symbols, start, end, source, data_dir)

    @mcp.tool(name="data_list_symbols", description="List locally available market data symbols")
    def data_list_symbols(data_dir: str | None = None) -> dict[str, Any]:
        return _list_symbols(data_dir)

    @mcp.tool(name="data_inspect", description="Inspect data summary for a symbol (rows, date range, missing values)")
    def data_inspect(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
        return _inspect(symbol, data_dir)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/mcp_server/test_data_tools.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add mcp_server/tools/data_tools.py tests/mcp_server/__init__.py tests/mcp_server/test_data_tools.py
git commit -m "feat(mcp): implement 3 data tools (load_symbols, list_symbols, inspect)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: MCP Server Entry Point

**Files:**
- Modify: `mcp_server/server.py`
- Create: `tests/mcp_server/test_server.py`

**Context:** The server creates an `MCPServer` instance, registers tools from `data_tools`, and runs with stdio transport. Test verifies the server can start and list tools via MCP client protocol.

**Step 1: Write the failing test**

```python
# tests/mcp_server/test_server.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters, stdio_client


SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server" / "server.py")


@pytest.fixture()
def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        cwd=str(Path(__file__).resolve().parents[2]),
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


async def _call_list_symbols(
    server_params: StdioServerParameters, data_dir: str,
) -> dict:
    import json

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp_server/test_server.py -v`
Expected: FAIL (server.py is empty)

**Step 3: Write implementation**

```python
# mcp_server/server.py
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_server.tools.data_tools import register as register_data_tools

mcp = MCPServer("open-xquant")
register_data_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/mcp_server/test_server.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_server.py
git commit -m "feat(mcp): add MCP server entry point with stdio transport

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Data Explorer Skill

**Files:**
- Modify: `skills/data-explorer.md`

**Context:** A markdown workflow guide injected as system prompt. No tests needed — this is pure documentation.

**Step 1: Write the skill**

```markdown
# skills/data-explorer.md

---
name: data-explorer
description: 指导 Agent 探索和准备市场数据
tools_required: [data_load_symbols, data_list_symbols, data_inspect]
---

## 你的角色

你是一个数据助手，帮助用户获取和探索市场数据。你可以使用以下工具：

- **data_list_symbols**: 查看本地已有哪些标的的数据
- **data_load_symbols**: 从外部数据源下载市场数据
- **data_inspect**: 查看标的数据的详细信息（行数、时间范围、缺失值等）

## 工作流

### 1. 了解用户需求

- 用户想看什么市场？（美股 / A股）
- 关注哪些标的？
- 什么时间范围？

### 2. 检查本地数据

- 调用 data_list_symbols 查看已有数据
- 如果目标标的已存在，告知用户并询问是否需要更新

### 3. 下载数据

- 美股 → source 使用 "yfinance"
- A股 → source 使用 "akshare"，symbol 使用纯数字代码（如 "600519"）
- 调用 data_load_symbols 下载
- 报告下载结果（条数、时间范围）

### 4. 数据质量检查

- 调用 data_inspect 查看数据摘要
- 检查时间范围是否完整
- 检查是否有缺失值
- 向用户报告数据状况

## 错误处理

- **SymbolNotFoundError**: 本地无数据。引导用户先用 data_load_symbols 下载。
- **DownloadError**: 下载失败。告知用户可能的原因（网络问题、标的代码错误、数据源不可用），建议检查标的代码或更换 source。
- **不要重试超过 1 次**。如果同一操作连续失败，告知用户错误信息并停止。
```

**Step 2: Commit**

```bash
git add skills/data-explorer.md
git commit -m "docs: add data-explorer skill for Agent workflow guidance

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add to `[project.optional-dependencies]`:

```toml
agent = [
    "streamlit>=1.30",
    "openai>=1.0",
    "mcp>=1.0",
]
```

Ensure `mcp` group still exists separately (for users who only want MCP server without Streamlit):

```toml
mcp = [
    "mcp>=1.0",
]
```

**Step 2: Install and verify**

Run: `uv pip install -e ".[agent,yfinance,akshare,dev]"`
Expected: All dependencies install successfully

**Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest -v`
Expected: All 25+ tests pass

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add agent optional deps (streamlit, openai, mcp)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Streamlit Agent Demo Page

**Files:**
- Create: `app/agent_demo.py`

**Context:** This is the main deliverable — a Streamlit chat app that acts as MCP Client, connects to the MCP Server subprocess, and uses OpenAI-compatible function calling to let users interact with data tools. No automated tests for the UI; manual verification.

**Step 1: Create the app directory**

```bash
mkdir -p app
```

**Step 2: Write the Streamlit app**

```python
# app/agent_demo.py
"""open-xquant Agent Demo — test MCP tools and skills via chat interface."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = str(PROJECT_ROOT / "mcp_server" / "server.py")
SKILLS_DIR = PROJECT_ROOT / "skills"


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="open-xquant Agent", layout="wide")

with st.sidebar:
    st.title("open-xquant Agent")
    st.markdown("---")

    base_url = st.text_input("API Base URL", value="https://api.openai.com/v1")
    api_key = st.text_input("API Key", type="password")
    model = st.text_input("Model", value="gpt-4o")

    st.markdown("---")

    # Skill loader
    skill_files = sorted(SKILLS_DIR.glob("*.md"))
    skill_names = [f.stem for f in skill_files if f.stat().st_size > 0]
    selected_skill = st.selectbox(
        "Skill",
        options=["(无)"] + skill_names,
    )
    if selected_skill != "(无)":
        skill_content = (SKILLS_DIR / f"{selected_skill}.md").read_text()
        st.success(f"已加载 Skill: {selected_skill}")
    else:
        skill_content = ""

    if st.button("清除对话"):
        st.session_state.messages = []
        st.rerun()


# ── MCP helpers ──────────────────────────────────────────────────────────────

def get_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        cwd=str(PROJECT_ROOT),
    )


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format."""
    result = []
    for tool in mcp_tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return result


async def run_agent_turn(
    messages: list[dict],
    openai_client: OpenAI,
    model_name: str,
) -> list[dict]:
    """Execute one full agent turn, potentially with multiple tool calls.

    Returns a list of new message dicts to append to history.
    """
    server_params = get_server_params()
    new_messages: list[dict] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tools_result.tools)

            current_messages = list(messages)

            while True:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=current_messages,
                    tools=openai_tools if openai_tools else None,
                )
                choice = response.choices[0]
                msg = choice.message

                if msg.tool_calls:
                    # Add assistant message with tool calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                    new_messages.append(assistant_msg)
                    current_messages.append(assistant_msg)

                    # Execute each tool call via MCP
                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)

                        try:
                            result = await session.call_tool(tool_name, tool_args)
                            tool_content = result.content[0].text if result.content else "{}"
                        except Exception as e:
                            tool_content = json.dumps({"error": str(e)})

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_content,
                        }
                        new_messages.append(tool_msg)
                        current_messages.append(tool_msg)
                else:
                    # Final response (no more tool calls)
                    new_messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                    })
                    break

    return new_messages


# ── Main chat UI ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    st.info(f"🔧 调用工具: **{tc['function']['name']}**\n```json\n{tc['function']['arguments']}\n```")
            if msg.get("content"):
                st.markdown(msg["content"])
    elif msg["role"] == "tool":
        with st.chat_message("assistant"):
            try:
                parsed = json.loads(msg["content"])
                st.success(f"📋 工具返回:\n```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```")
            except json.JSONDecodeError:
                st.success(f"📋 工具返回: {msg['content']}")

# Chat input
if prompt := st.chat_input("输入消息..."):
    if not api_key:
        st.error("请在侧边栏填入 API Key")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build messages for API call
    api_messages = []
    if skill_content:
        api_messages.append({"role": "system", "content": skill_content})
    api_messages.extend(st.session_state.messages)

    # Run agent turn
    openai_client = OpenAI(base_url=base_url, api_key=api_key)

    with st.spinner("思考中..."):
        try:
            new_messages = asyncio.run(
                run_agent_turn(api_messages, openai_client, model)
            )
        except Exception as e:
            st.error(f"错误: {e}")
            st.stop()

    # Display and store new messages
    for msg in new_messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        st.info(f"🔧 调用工具: **{tc['function']['name']}**\n```json\n{tc['function']['arguments']}\n```")
                if msg.get("content"):
                    st.markdown(msg["content"])
        elif msg["role"] == "tool":
            with st.chat_message("assistant"):
                try:
                    parsed = json.loads(msg["content"])
                    st.success(f"📋 工具返回:\n```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```")
                except json.JSONDecodeError:
                    st.success(f"📋 工具返回: {msg['content']}")

    st.session_state.messages.extend(new_messages)
```

**Step 3: Manual verification**

Run: `uv run streamlit run app/agent_demo.py`

Verify:
1. Page loads with sidebar (API config + skill selector)
2. Entering API key and sending "列出本地数据" triggers `data_list_symbols` via MCP
3. Tool call and result are visible in chat
4. Loading data-explorer skill changes agent behavior

**Step 4: Commit**

```bash
git add app/agent_demo.py
git commit -m "feat: add Streamlit Agent demo page (MCP client + OpenAI function calling)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Lint + Verify

**Step 1: Run ruff on all new files**

Run: `uv run ruff check mcp_server/ app/ tests/mcp_server/`
Fix any issues.

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (25 existing + 8 new MCP tests)

**Step 3: Commit any fixes**

```bash
git add -A && git commit -m "chore: fix lint issues

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

(Skip if no fixes needed.)
