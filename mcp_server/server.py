from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from oxq.tools import registry

mcp = FastMCP("open-xquant")

# Auto-register all SDK tools from the registry
for tool_def in registry.all_tools():
    mcp.tool(name=tool_def.name, description=tool_def.description)(tool_def.fn)


# MCP-only tool (not part of oxq SDK)
@mcp.tool(
    name="get_current_date",
    description=(
        "Get today's date. Call this FIRST when the user mentions "
        "relative dates like 'last 6 months', 'recent year', etc."
    ),
)
def get_current_date() -> dict[str, str]:
    from datetime import date

    today = date.today()
    return {"today": today.isoformat(), "weekday": today.strftime("%A")}


if __name__ == "__main__":
    mcp.run(transport="stdio")
