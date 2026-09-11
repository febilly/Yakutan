#!/usr/bin/env python3
"""
VRChat 翻译器 Web UI 启动器
"""
import sys
import os

UI_PORT = 5001


def _run_panel_mode():
    import panel_app
    panel_args = ['panel_app.py'] + sys.argv[2:]
    panel_app.main(panel_args)


def _run_probe_gpu_mode():
    # 打包版里 probe_gpu_devices 通过本子命令在子进程内枚举 GPU，
    # 输出一行 JSON 后立即退出；绝不能走 WebUI 启动流程。
    import json
    import logging

    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    try:
        from local_inference.gpu_devices import enumerate_gpu_devices
        print(json.dumps(enumerate_gpu_devices(), ensure_ascii=False))
    except Exception as exc:
        print(f"GPU probe failed: {exc}", file=sys.stderr)
        print("[]")


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def _run_web_ui_mode():
    # 添加ui目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui'))

    import webbrowser
    from proxy_detector import refresh_system_proxy_env, print_proxy_info

    # 检测并应用系统代理设置
    system_proxies = refresh_system_proxy_env()
    print_proxy_info(system_proxies)

    from ui.app import app

    print("WebUI is now running at http://127.0.0.1:5001")
    # 端口已被占用说明已有实例在服务（或被其它代码路径重新拉起），
    # 此时不能再开浏览器，避免重复弹窗。
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') and not _port_in_use(UI_PORT):
        webbrowser.open("http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--panel-app':
        _run_panel_mode()
    elif len(sys.argv) >= 2 and sys.argv[1] == '--probe-gpu':
        _run_probe_gpu_mode()
    else:
        _run_web_ui_mode()
