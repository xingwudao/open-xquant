"""open-xquant Agent Demo — test MCP tools and skills via chat interface."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "skills"


# -- Sidebar -------------------------------------------------------------------

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
        options=["(none)"] + skill_names,
    )
    if selected_skill != "(none)":
        skill_content = (SKILLS_DIR / f"{selected_skill}.md").read_text()
        st.success(f"Loaded Skill: {selected_skill}")
    else:
        skill_content = ""

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# -- MCP helpers ---------------------------------------------------------------


def get_server_params() -> StdioServerParameters:
    """Return MCP server parameters for stdio transport."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format."""
    result = []
    for tool in mcp_tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return result


async def run_agent_turn(
    messages: list[dict],
    openai_client: OpenAI,
    model_name: str,
) -> list[dict]:
    """Execute one full agent turn with tool calling loop.

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
                        except Exception as e:  # noqa: BLE001
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
                    new_messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                        }
                    )
                    break

    return new_messages


# -- Main chat UI --------------------------------------------------------------

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
                    st.info(
                        f"Tool call: **{tc['function']['name']}**\n"
                        f"```json\n{tc['function']['arguments']}\n```"
                    )
            if msg.get("content"):
                st.markdown(msg["content"])
    elif msg["role"] == "tool":
        with st.chat_message("assistant"):
            try:
                parsed = json.loads(msg["content"])
                st.success(
                    f"Tool result:\n```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
                )
            except json.JSONDecodeError:
                st.success(f"Tool result: {msg['content']}")

# Chat input
if prompt := st.chat_input("Type a message..."):
    if not api_key:
        st.error("Please enter your API Key in the sidebar")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build messages for API call
    api_messages: list[dict] = []
    if skill_content:
        api_messages.append({"role": "system", "content": skill_content})
    api_messages.extend(st.session_state.messages)

    # Run agent turn
    openai_client = OpenAI(base_url=base_url, api_key=api_key)

    with st.spinner("Thinking..."):
        try:
            new_messages = asyncio.run(run_agent_turn(api_messages, openai_client, model))
        except Exception as e:  # noqa: BLE001
            st.error(f"Error: {e}")
            st.stop()

    # Display and store new messages
    for msg in new_messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        st.info(
                            f"Tool call: **{tc['function']['name']}**\n"
                            f"```json\n{tc['function']['arguments']}\n```"
                        )
                if msg.get("content"):
                    st.markdown(msg["content"])
        elif msg["role"] == "tool":
            with st.chat_message("assistant"):
                try:
                    parsed = json.loads(msg["content"])
                    st.success(
                        f"Tool result:\n```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
                    )
                except json.JSONDecodeError:
                    st.success(f"Tool result: {msg['content']}")

    st.session_state.messages.extend(new_messages)
