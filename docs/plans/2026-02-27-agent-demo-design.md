# Agent Demo Page + MCP Tools + Skill — Design Document

Date: 2026-02-27

## Overview

构建完整的 Agent 测试/教学链路：MCP Tools → MCP Server → Streamlit Agent 页面 + Data Explorer Skill。

目的：
1. 教用户如何在自己的 Agent 中使用 open-xquant 的 MCP server 和 skill
2. 作为开发者测试 MCP 协议路径的工具

## Architecture

```
┌──────────────────────────────────┐
│  Streamlit Agent 页面             │  ← 用户界面
│  app/agent_demo.py               │
└──────────┬───────────────────────┘
           │ MCP 协议（stdio transport）
           ▼
┌──────────────────────────────────┐
│  MCP Server（独立子进程）          │
│  mcp_server/server.py             │
│  mcp_server/tools/data_tools.py   │
└──────────┬───────────────────────┘
           │ 调用 oxq SDK
           ▼
┌──────────────────────────────────┐
│  oxq.data（已实现）                │
│  Downloader / Provider            │
└──────────────────────────────────┘
```

Streamlit 作为 MCP Client，通过 stdio transport 启动 MCP Server 子进程，走完整 MCP 协议路径。

## Design Decisions

- **MCP 协议必须走真实路径**：不走捷径，确保测试覆盖序列化/反序列化、错误传播等
- **LLM 支持**：仅 OpenAI 兼容 API（base_url + api_key + model），覆盖大多数提供商
- **MCP Tools 范围**：3 个 data tools（load_symbols, list_symbols, inspect）
- **Skill**：独立的 `data-explorer.md`，引导 Agent 完成数据探索工作流

## MCP Tools

### data.load_symbols

- **输入**: `symbols` (list[str]), `start` (str), `end` (str), `source` ("yfinance"|"akshare")
- **行为**: 调用 Downloader 下载数据
- **返回**: `{ symbols: [...], rows: {AAPL: 251, ...}, path: "~/.oxq/data/market/" }`

### data.list_symbols

- **输入**: 无（或 `data_dir` 可选）
- **行为**: 扫描本地 Parquet 文件，列出已有标的
- **返回**: `{ symbols: ["AAPL", "600519", ...], count: 2 }`

### data.inspect

- **输入**: `symbol` (str)
- **行为**: 读取 Parquet，返回数据摘要
- **返回**: `{ symbol, rows, date_range: [start, end], columns, missing_values, sample_head }`

每个 tool 只做：解析参数 → 调用 oxq SDK → 格式化返回。不在 tool 层放业务逻辑。

## MCP Server

- 使用 mcp Python SDK 的标准模式
- stdio transport
- 注册 data_tools 中的 3 个 tools
- 启动方式: `python -m mcp_server.server`

### 错误处理

- `SymbolNotFoundError` → MCP error response + 提示 "Run data.load_symbols first"
- `DownloadError` → MCP error response + 原始错误信息
- 未知异常 → 统一包装为 MCP internal error

## Streamlit Agent 页面

### 页面布局

```
┌─ 侧边栏 ──────────────┐  ┌─ 主区域 ──────────────────┐
│                         │  │                            │
│  API Base URL           │  │  open-xquant Agent Demo    │
│  [https://api.openai..] │  │                            │
│                         │  │  ┌── chat message ───────┐ │
│  API Key                │  │  │ 用户: 帮我下载苹果...   │ │
│  [sk-xxxxx]             │  │  └───────────────────────┘ │
│                         │  │  ┌── assistant ──────────┐ │
│  Model Name             │  │  │ [调用 data.load...]    │ │
│  [gpt-4o]               │  │  │ 已下载 251 条数据      │ │
│                         │  │  └───────────────────────┘ │
│  [连接测试]              │  │                            │
│                         │  │  [输入框]                  │
│  ─────────────────────  │  │                            │
│  Skill: data-explorer   │  │                            │
│  [加载 Skill]            │  │                            │
└─────────────────────────┘  └────────────────────────────┘
```

### 核心流程

1. 用户在侧边栏填入 API 配置（base_url、api_key、model）
2. 页面启动 MCP Server 子进程（stdio transport）
3. 用户发消息 → Streamlit 将消息 + tool definitions 发给 LLM
4. LLM 返回 tool call → Streamlit 通过 MCP 协议调用 tool → 结果返回给 LLM
5. LLM 生成最终回复 → 展示在聊天界面

### 实现细节

- 使用 `openai` SDK 的 function calling
- Tool definitions 从 MCP Server 获取（`tools/list`），不硬编码
- 聊天历史保持在 `st.session_state` 中
- Tool call 的中间过程在聊天界面可见

## Skill: data-explorer

```markdown
---
name: data-explorer
description: 指导 Agent 探索和准备市场数据
tools_required: [data.load_symbols, data.list_symbols, data.inspect]
---

## 工作流

1. 了解用户需求（市场、标的、时间范围）
2. 检查本地数据（data.list_symbols）
3. 下载数据（data.load_symbols，根据市场选择 source）
4. 数据质量检查（data.inspect）

## 错误处理

- SymbolNotFoundError：引导用户先下载
- DownloadError：告知可能原因，建议检查标的代码或更换 source
- 不要重试超过 1 次，连续失败则告知用户并停止
```

## File Manifest

| 文件 | 内容 |
|------|------|
| `mcp_server/server.py` | MCP Server 入口 |
| `mcp_server/tools/data_tools.py` | 3 个 data tools |
| `skills/data-explorer.md` | Data Explorer skill |
| `app/agent_demo.py` | Streamlit Agent 页面 |
| `pyproject.toml` | 新增 `[agent]` optional deps |

## Dependencies

```toml
[project.optional-dependencies]
agent = [
    "streamlit>=1.30",
    "openai>=1.0",
    "mcp>=1.0",
]
```

安装: `pip install open-xquant[agent]`

启动: `streamlit run app/agent_demo.py`
