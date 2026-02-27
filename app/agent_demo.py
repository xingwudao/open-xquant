"""open-xquant Agent Demo — test MCP tools and skills via chat interface."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import nest_asyncio
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

nest_asyncio.apply()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "skills"
MAX_TOOL_ROUNDS = 15


# -- Sidebar -------------------------------------------------------------------

st.set_page_config(page_title="open-xquant Agent", layout="wide")

with st.sidebar:
    st.title("open-xquant Agent")
    st.markdown("---")

    base_url = st.text_input("API Base URL", value="https://api.deepseek.com/v1")
    api_key = st.text_input("API Key", type="password")
    model = st.text_input("Model", value="deepseek-chat")

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


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
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


def render_message(msg: dict[str, Any]) -> None:
    """Render a single chat message in the Streamlit UI."""
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


async def stream_agent_turn(
    messages: list[dict[str, Any]],
    openai_client: OpenAI,
    model_name: str,
) -> list[dict[str, Any]]:
    """Execute one full agent turn with streaming and real-time UI updates.

    Renders messages in Streamlit as they arrive (text token-by-token,
    tool calls and results as they complete).
    Returns a list of new message dicts to append to history.
    """
    server_params = get_server_params()
    new_messages: list[dict[str, Any]] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tools_result.tools)

            current_messages = list(messages)

            for _round in range(MAX_TOOL_ROUNDS):
                # Stream LLM response
                stream = openai_client.chat.completions.create(
                    model=model_name,
                    messages=current_messages,
                    tools=openai_tools if openai_tools else None,
                    stream=True,
                )

                content_parts: list[str] = []
                tool_calls_acc: dict[int, dict[str, str]] = {}

                with st.chat_message("assistant"):
                    text_placeholder = st.empty()

                    for chunk in stream:
                        delta = chunk.choices[0].delta

                        # Stream text content token-by-token
                        if delta.content:
                            content_parts.append(delta.content)
                            text_placeholder.markdown(
                                "".join(content_parts) + "▌"
                            )

                        # Accumulate tool call deltas
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc_delta.id:
                                    tool_calls_acc[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_calls_acc[idx][
                                            "name"
                                        ] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_calls_acc[idx][
                                            "arguments"
                                        ] += tc_delta.function.arguments

                    full_content = "".join(content_parts)

                    if tool_calls_acc:
                        # Finalize or clear text placeholder
                        if full_content:
                            text_placeholder.markdown(full_content)
                        else:
                            text_placeholder.empty()

                        # Build tool_calls list
                        tool_calls = [
                            {
                                "id": tool_calls_acc[idx]["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_calls_acc[idx]["name"],
                                    "arguments": tool_calls_acc[idx]["arguments"],
                                },
                            }
                            for idx in sorted(tool_calls_acc)
                        ]

                        # Display tool calls
                        for tc in tool_calls:
                            st.info(
                                f"Tool call: **{tc['function']['name']}**\n"
                                f"```json\n{tc['function']['arguments']}\n```"
                            )

                        # Store assistant message
                        assistant_msg: dict[str, Any] = {
                            "role": "assistant",
                            "content": full_content,
                            "tool_calls": tool_calls,
                        }
                        new_messages.append(assistant_msg)
                        current_messages.append(assistant_msg)

                    else:
                        # Final text response — remove cursor
                        text_placeholder.markdown(full_content)
                        new_messages.append(
                            {"role": "assistant", "content": full_content}
                        )
                        break

                # Execute tool calls via MCP (outside the chat bubble above)
                if tool_calls_acc:
                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        try:
                            tool_args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            tool_args = {}

                        with st.chat_message("assistant"):
                            with st.spinner(f"Running {tool_name}..."):
                                try:
                                    result = await session.call_tool(
                                        tool_name, tool_args
                                    )
                                    tool_content = (
                                        result.content[0].text
                                        if result.content
                                        else "{}"
                                    )
                                except Exception as e:  # noqa: BLE001
                                    tool_content = json.dumps({"error": str(e)})

                            try:
                                parsed = json.loads(tool_content)
                                st.success(
                                    f"Tool result:\n```json\n"
                                    f"{json.dumps(parsed, indent=2, ensure_ascii=False)}"
                                    f"\n```"
                                )
                            except json.JSONDecodeError:
                                st.success(f"Tool result: {tool_content}")

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_content,
                        }
                        new_messages.append(tool_msg)
                        current_messages.append(tool_msg)
            else:
                with st.chat_message("assistant"):
                    st.warning("Reached maximum tool-call rounds. Stopping.")
                new_messages.append(
                    {
                        "role": "assistant",
                        "content": "Reached maximum tool-call rounds. Stopping.",
                    }
                )

    return new_messages


# -- Main chat UI --------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for _msg in st.session_state.messages:
    render_message(_msg)

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
    api_messages: list[dict[str, Any]] = []
    system_parts = [f"Today's date is {date.today().isoformat()}."]
    if skill_content:
        system_parts.append(skill_content)
    api_messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    api_messages.extend(st.session_state.messages)

    # Stream agent turn (renders in real-time)
    openai_client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        new_messages = asyncio.run(
            stream_agent_turn(api_messages, openai_client, model)
        )
    except BaseException as e:  # noqa: BLE001
        # Unwrap anyio TaskGroup / ExceptionGroup to show the real error
        root = e
        while hasattr(root, "exceptions") and root.exceptions:
            root = root.exceptions[0]
        st.error(f"Error: {root}")
        st.stop()

    st.session_state.messages.extend(new_messages)
