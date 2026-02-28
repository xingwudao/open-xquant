"""Tests for ToolRegistry."""

from oxq.tools.registry import ToolDef, ToolRegistry


def test_tool_decorator_registers() -> None:
    reg = ToolRegistry()

    @reg.tool(name="test_tool", description="A test tool")
    def my_tool() -> dict:
        return {"ok": True}

    assert len(reg.all_tools()) == 1
    t = reg.get("test_tool")
    assert t.name == "test_tool"
    assert t.description == "A test tool"
    assert t.fn() == {"ok": True}


def test_all_tools_returns_list() -> None:
    reg = ToolRegistry()

    @reg.tool(name="a", description="A")
    def tool_a() -> dict:
        return {}

    @reg.tool(name="b", description="B")
    def tool_b() -> dict:
        return {}

    tools = reg.all_tools()
    assert len(tools) == 2
    assert all(isinstance(t, ToolDef) for t in tools)


def test_get_raises_on_missing() -> None:
    reg = ToolRegistry()
    try:
        reg.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_global_registry_has_all_tools() -> None:
    from oxq.tools import registry

    names = [t.name for t in registry.all_tools()]
    assert "data_list_symbols" in names
    assert "data_inspect" in names
    assert "data_load_symbols" in names
    assert "universe_set" in names
    assert "universe_list_indexes" in names
    assert "universe_inspect" in names
    assert "universe_history" in names
    assert "factor_download" in names
    assert "factor_list" in names
    assert "factor_inspect" in names
