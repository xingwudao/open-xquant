from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.data_tools import register as register_data_tools
from mcp_server.tools.universe_tools import register as register_universe_tools

mcp = FastMCP("open-xquant")
register_data_tools(mcp)
register_universe_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
