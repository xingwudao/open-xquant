from __future__ import annotations

import pytest

from oxq.tools import registry


def test_report_write_tool_is_not_registered() -> None:
    names = {tool.name for tool in registry.all_tools()}

    assert "report_write" not in names
    assert "experiment_add" in names
    with pytest.raises(KeyError):
        registry.get("report_write")
