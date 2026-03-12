import ctypes
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pytest

import panel_app


REPO_ROOT = Path(__file__).resolve().parents[1]


class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long),
    ]


def _panel_url(live_server, quick_lang_enabled):
    return f"{live_server}/panel?{urlencode([('quick_lang_bar', '1' if quick_lang_enabled else '0'), ('quick_lang', 'en'), ('quick_lang', 'zh-CN'), ('quick_lang', 'ja'), ('quick_lang', 'ko')])}"


def _launch_panel_process(live_server, quick_lang_enabled):
    return subprocess.Popen(
        [
            sys.executable,
            'panel_app.py',
            _panel_url(live_server, quick_lang_enabled),
            'reverse-off',
            'floating-off',
            '600',
        ],
        cwd=str(REPO_ROOT),
    )


def _list_panel_handles():
    handles = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_windows(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        text_length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if text_length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(text_length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, len(buffer))
        if buffer.value == 'Yakutan Status Panel':
            handles.append(hwnd)
        return True

    ctypes.windll.user32.EnumWindows(enum_windows, 0)
    return handles


def _connect_window(existing_handles=None, timeout_seconds=15):
    existing_handles = set(existing_handles or [])
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current_handles = _list_panel_handles()
        new_handles = [hwnd for hwnd in current_handles if hwnd not in existing_handles]
        if new_handles:
            return new_handles[0]
        time.sleep(0.25)
    raise RuntimeError('Could not connect to panel window')


def _get_window_height(hwnd):
    rect = RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError('GetWindowRect failed')
    return rect.bottom - rect.top


def _stop_process(process):
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
def test_native_panel_height_changes_with_quick_lang_bar(live_server):
    before_handles = _list_panel_handles()
    shown_process = _launch_panel_process(live_server, True)
    shown_window = _connect_window(before_handles)
    shown_height = _get_window_height(shown_window)
    _stop_process(shown_process)

    before_handles = _list_panel_handles()
    hidden_process = _launch_panel_process(live_server, False)
    hidden_window = _connect_window(before_handles)
    hidden_height = _get_window_height(hidden_window)
    _stop_process(hidden_process)

    actual_delta = shown_height - hidden_height
    expected_deltas = {
        panel_app.QUICK_LANG_BAR_HEIGHT,
        round(panel_app.QUICK_LANG_BAR_HEIGHT * panel_app._get_dpi_scale_factor()),
    }
    assert any(actual_delta == pytest.approx(expected_delta, abs=6) for expected_delta in expected_deltas)