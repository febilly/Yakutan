import sys
import ctypes
import webview
import logging
from urllib.parse import parse_qs, urlparse

DEFAULT_PANEL_WIDTH = 600
WINDOW_TITLE_BAR_HEIGHT = 32
PANEL_CONTENT_HEIGHT_TWO_LINES = 147 - WINDOW_TITLE_BAR_HEIGHT
PANEL_CONTENT_HEIGHT_THREE_LINES = 170 - WINDOW_TITLE_BAR_HEIGHT
QUICK_LANG_BAR_HEIGHT = 27


def _parse_panel_width(raw_value):
    try:
        return max(300, int(raw_value))
    except Exception:
        return DEFAULT_PANEL_WIDTH


def _get_dpi_scale_factor():
    """获取主显示器的 DPI 缩放因子（如 1.0, 1.25, 1.5, 2.25 等）"""
    try:
        return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        return 1.0


def _should_show_quick_lang_bar(url):
    try:
        params = parse_qs(urlparse(url).query)
        raw_value = (params.get('quick_lang_bar') or [None])[0]
        if raw_value is None:
            return True
        return raw_value.strip().lower() not in {'', '0', 'false', 'no', 'off'}
    except Exception:
        return True


class PanelApi:
    def __init__(self):
        self._window = None

    def attach_window(self, window):
        self._window = window

    def close_window(self):
        if self._window is not None:
            self._window.destroy()


def main():
    if len(sys.argv) < 2:
        url = "http://127.0.0.1:5001/panel"
    else:
        url = sys.argv[1]

    show_reverse_translation = len(sys.argv) >= 3 and sys.argv[2] == "reverse-on"
    floating_mode = len(sys.argv) >= 4 and sys.argv[3] == "floating-on"
    panel_width = _parse_panel_width(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_PANEL_WIDTH
    show_quick_lang_bar = _should_show_quick_lang_bar(url)
    content_height = PANEL_CONTENT_HEIGHT_THREE_LINES if show_reverse_translation else PANEL_CONTENT_HEIGHT_TWO_LINES
    if not show_quick_lang_bar:
        content_height = max(60, content_height - QUICK_LANG_BAR_HEIGHT)
    initial_height = content_height if floating_mode else content_height + WINDOW_TITLE_BAR_HEIGHT

    scale = _get_dpi_scale_factor()
    physical_width = int(panel_width * scale)
    physical_height = int(initial_height * scale)
    panel_api = PanelApi()

    # 先隐藏创建，等 resize 到正确物理像素尺寸后再显示，避免闪烁
    window = webview.create_window(
        title='Yakutan Status Panel',
        url=url,
        width=panel_width,
        height=initial_height,
        frameless=floating_mode,
        on_top=floating_mode,
        resizable=True,
        hidden=True,
        js_api=panel_api,
    )
    panel_api.attach_window(window)

    def on_started():
        # pywebview 的 AutoScaleMode.Dpi 缩放行为不一致，
        # 这里用 resize() (直接调 SetWindowPos) 强制设为正确的物理像素尺寸
        window.resize(physical_width, physical_height)
        window.show()

    logging.getLogger('pywebview').setLevel(logging.ERROR)
    webview.start(func=on_started)

if __name__ == '__main__':
    main()
