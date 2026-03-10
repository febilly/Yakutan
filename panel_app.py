import sys
import ctypes
import webview
import logging

PANEL_WIDTH = 800
WINDOW_TITLE_BAR_HEIGHT = 32
PANEL_CONTENT_HEIGHT_TWO_LINES = 117 - WINDOW_TITLE_BAR_HEIGHT
PANEL_CONTENT_HEIGHT_THREE_LINES = 140 - WINDOW_TITLE_BAR_HEIGHT


def _get_dpi_scale_factor():
    """获取主显示器的 DPI 缩放因子（如 1.0, 1.25, 1.5, 2.25 等）"""
    try:
        return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        return 1.0


def main():
    if len(sys.argv) < 2:
        url = "http://127.0.0.1:5001/panel"
    else:
        url = sys.argv[1]

    show_reverse_translation = len(sys.argv) >= 3 and sys.argv[2] == "reverse-on"
    floating_mode = len(sys.argv) >= 4 and sys.argv[3] == "floating-on"
    content_height = PANEL_CONTENT_HEIGHT_THREE_LINES if show_reverse_translation else PANEL_CONTENT_HEIGHT_TWO_LINES
    initial_height = content_height if floating_mode else content_height + WINDOW_TITLE_BAR_HEIGHT

    scale = _get_dpi_scale_factor()
    physical_width = int(PANEL_WIDTH * scale)
    physical_height = int(initial_height * scale)

    # 先隐藏创建，等 resize 到正确物理像素尺寸后再显示，避免闪烁
    window = webview.create_window(
        title='Yakutan Status Panel',
        url=url,
        width=PANEL_WIDTH,
        height=initial_height,
        frameless=floating_mode,
        on_top=floating_mode,
        resizable=True,
        hidden=True,
    )

    def on_started():
        # pywebview 的 AutoScaleMode.Dpi 缩放行为不一致，
        # 这里用 resize() (直接调 SetWindowPos) 强制设为正确的物理像素尺寸
        window.resize(physical_width, physical_height)
        window.show()

    logging.getLogger('pywebview').setLevel(logging.ERROR)
    webview.start(func=on_started)

if __name__ == '__main__':
    main()
