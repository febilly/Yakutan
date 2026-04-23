#!/usr/bin/env python3
"""
VRChat 翻译器 Web UI 启动器
"""
import sys
import os

def _run_panel_mode():
    import panel_app
    panel_args = ['panel_app.py'] + sys.argv[2:]
    panel_app.main(panel_args)


def _run_web_ui_mode():
    # 添加ui目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui'))

    import webbrowser
    from proxy_detector import apply_system_proxy, detect_system_proxy, print_proxy_info

    # 检测并应用系统代理设置
    system_proxies = apply_system_proxy(detect_system_proxy())
    print_proxy_info(system_proxies)

    from ui.app import app

    print("WebUI is now running at http://127.0.0.1:5001")
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        webbrowser.open("http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--panel-app':
        _run_panel_mode()
    else:
        _run_web_ui_mode()
