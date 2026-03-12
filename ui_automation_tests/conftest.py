import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = 'http://127.0.0.1:5001'
TEST_PORT = 5051
TEST_BASE_URL = f'http://127.0.0.1:{TEST_PORT}'


def _wait_for_server_ready(timeout_seconds=20):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{TEST_BASE_URL}/api/status', timeout=1.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f'UI server did not become ready: {last_error}')
@pytest.fixture(scope='session')
def live_server():
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(
        [
            sys.executable,
            '-c',
            (
                "import sys; "
                f"sys.path.insert(0, r'{REPO_ROOT}'); "
                "from ui.app import app; "
                f"app.run(host='127.0.0.1', port={TEST_PORT}, debug=False)"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    _wait_for_server_ready()

    try:
        yield TEST_BASE_URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture(scope='session')
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture()
def page(browser, live_server):
    context = browser.new_context(viewport={'width': 1440, 'height': 1100})
    page = context.new_page()
    try:
        yield page
    finally:
        context.close()