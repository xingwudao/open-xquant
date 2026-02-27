# mcp_server/server.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `mcp_server` is importable
# when this script is executed directly (e.g. `python mcp_server/server.py`).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.data_tools import register as register_data_tools

mcp = FastMCP("open-xquant")
register_data_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
