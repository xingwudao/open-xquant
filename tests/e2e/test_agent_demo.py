"""End-to-end Playwright tests for the agent_demo Streamlit app."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import API_KEY


# ---------------------------------------------------------------------------
# 1. Basic UI rendering
# ---------------------------------------------------------------------------

class TestPageLayout:
    """Verify the page loads with all expected UI elements."""

    def test_page_title(self, app_page: Page) -> None:
        """Page title should be set."""
        expect(app_page).to_have_title(re.compile("open-xquant"))

    def test_sidebar_title(self, app_page: Page) -> None:
        """Sidebar should show the app title."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_contain_text("open-xquant Agent")

    def test_sidebar_has_api_inputs(self, app_page: Page) -> None:
        """Sidebar should have API Base URL, API Key, and Model inputs."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_label("API Base URL")).to_be_visible()
        expect(sidebar.get_by_label("API Key")).to_be_visible()
        expect(sidebar.get_by_label("Model")).to_be_visible()

    def test_sidebar_default_values(self, app_page: Page) -> None:
        """Sidebar inputs should have correct default values."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_label("API Base URL")).to_have_value(
            "https://api.deepseek.com/v1"
        )
        expect(sidebar.get_by_label("Model")).to_have_value("deepseek-chat")

    def test_sidebar_has_skill_selector(self, app_page: Page) -> None:
        """Sidebar should have a Skill dropdown."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_contain_text("Skill")

    def test_sidebar_has_clear_button(self, app_page: Page) -> None:
        """Sidebar should have a Clear Chat button."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_text("Clear Chat")).to_be_visible()

    def test_chat_input_visible(self, app_page: Page) -> None:
        """Main area should have a chat input box."""
        chat_input = app_page.locator('[data-testid="stChatInput"]')
        expect(chat_input).to_be_visible()


# ---------------------------------------------------------------------------
# 2. API Key validation
# ---------------------------------------------------------------------------

class TestApiKeyValidation:
    """Verify the app requires an API key before sending messages."""

    def test_no_api_key_shows_error(self, app_page: Page) -> None:
        """Sending a message without API key should show an error."""
        chat_input = app_page.locator(
            '[data-testid="stChatInput"] textarea'
        )
        chat_input.fill("hello")
        chat_input.press("Enter")

        # Should show error about API Key
        error = app_page.locator('[data-testid="stAlert"]').first
        expect(error).to_contain_text("API Key", timeout=10000)


# ---------------------------------------------------------------------------
# 3. Full agent conversation (requires network)
# ---------------------------------------------------------------------------

class TestAgentConversation:
    """Test a real conversation with the LLM via the agent demo."""

    def _fill_api_key(self, page: Page) -> None:
        """Fill in the API key in the sidebar."""
        sidebar = page.locator('[data-testid="stSidebar"]')
        api_key_input = sidebar.get_by_label("API Key")
        api_key_input.fill(API_KEY)

    def test_simple_chat_response(self, app_page: Page) -> None:
        """Send a simple greeting and verify the assistant responds."""
        self._fill_api_key(app_page)

        # Send a simple message that should NOT trigger tool calls
        chat_input = app_page.locator(
            '[data-testid="stChatInput"] textarea'
        )
        chat_input.fill("请用一句话回答：1+1等于几？")
        chat_input.press("Enter")

        # Wait for user message to appear
        user_msgs = app_page.locator(
            '[data-testid="stChatMessage"]:has-text("1+1")'
        )
        expect(user_msgs.first).to_be_visible(timeout=10000)

        # Wait for assistant response (any new chat message after the user's)
        assistant_msgs = app_page.locator('[data-testid="stChatMessage"]')
        # There should be at least 2 messages (user + assistant)
        expect(assistant_msgs.nth(1)).to_be_visible(timeout=30000)
        # The assistant should mention "2" somewhere
        expect(assistant_msgs.nth(1)).to_contain_text("2", timeout=30000)

    def test_tool_call_get_date(self, app_page: Page) -> None:
        """Ask about today's date to trigger the get_current_date tool."""
        self._fill_api_key(app_page)

        chat_input = app_page.locator(
            '[data-testid="stChatInput"] textarea'
        )
        chat_input.fill("请调用工具告诉我今天是几号？")
        chat_input.press("Enter")

        # Wait for tool call info box to appear (blue info box with tool name)
        tool_call_info = app_page.locator(
            '[data-testid="stAlert"]:has-text("get_current_date")'
        )
        expect(tool_call_info.first).to_be_visible(timeout=30000)

        # Wait for tool result (green success box)
        tool_result = app_page.locator(
            '[data-testid="stAlert"]:has-text("today")'
        )
        expect(tool_result.first).to_be_visible(timeout=30000)

        # Wait for the final assistant response with the date
        # The assistant should eventually produce a text response mentioning a date
        assistant_msgs = app_page.locator('[data-testid="stChatMessage"]')
        # Wait for enough messages to appear (user + tool call + tool result + assistant)
        expect(assistant_msgs.last).to_be_visible(timeout=45000)

    def test_tool_call_list_symbols(self, app_page: Page) -> None:
        """Ask to list symbols to trigger the data_list_symbols tool."""
        self._fill_api_key(app_page)

        chat_input = app_page.locator(
            '[data-testid="stChatInput"] textarea'
        )
        chat_input.fill("请列出本地已有的所有标的数据")
        chat_input.press("Enter")

        # Wait for tool call to data_list_symbols
        tool_call = app_page.locator(
            '[data-testid="stAlert"]:has-text("data_list_symbols")'
        )
        expect(tool_call.first).to_be_visible(timeout=30000)

        # Wait for tool result with symbols info
        tool_result = app_page.locator(
            '[data-testid="stAlert"]:has-text("symbols")'
        )
        expect(tool_result.first).to_be_visible(timeout=30000)


# ---------------------------------------------------------------------------
# 4. Clear chat
# ---------------------------------------------------------------------------

class TestClearChat:
    """Test the Clear Chat button functionality."""

    def test_clear_chat_removes_messages(self, app_page: Page) -> None:
        """Clear Chat button should remove all messages."""
        # First, send a message without API key to get an error displayed
        chat_input = app_page.locator(
            '[data-testid="stChatInput"] textarea'
        )
        chat_input.fill("test message")
        chat_input.press("Enter")
        app_page.wait_for_timeout(2000)

        # Click Clear Chat
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        sidebar.get_by_text("Clear Chat").click()

        # After reload, no chat messages should be visible
        app_page.wait_for_selector(
            '[data-testid="stChatInput"]', timeout=10000
        )
        messages = app_page.locator('[data-testid="stChatMessage"]')
        expect(messages).to_have_count(0, timeout=10000)
