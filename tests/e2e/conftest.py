"""Fixtures for agent_demo E2E tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "examples" / "app" / "agent_demo.py"
STREAMLIT_PORT = 8599  # Use a non-default port to avoid conflicts

API_KEY = "sk-d820e7903402490a9a8a76d3371d1bf4"


@pytest.fixture(scope="session")
def streamlit_server():
    """Start and stop the Streamlit server for the test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(APP_PATH),
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to be ready
    url = f"http://localhost:{STREAMLIT_PORT}"
    for _ in range(30):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("Streamlit server did not start in time")

    yield url

    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def app_page(page: Page, streamlit_server: str) -> Page:
    """Navigate to the Streamlit app and wait for it to load."""
    page.goto(streamlit_server, wait_until="networkidle")
    # Wait for Streamlit to finish loading (the app title should appear)
    page.wait_for_selector("text=open-xquant Agent", timeout=15000)
    return page
