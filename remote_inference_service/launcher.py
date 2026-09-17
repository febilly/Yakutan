#!/usr/bin/env python3
"""Unified launcher and health diagnosis for Yakutan Remote Inference Server on Windows.

Hy-MT2 翻译由服务端进程内引擎直接加载 (local_asr_models/hymt2/*.gguf 存在时)，
无需也不应再启动外部 llama-server —— 否则同一模型会被加载两份、双倍占用显存。
外部 llama-server 仅作为内置引擎不可用时的手动回退 (见 run_hymt2_llama_server.bat)。
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from winutils import REQUIRED_MODULES, check_dependencies, check_port_in_use, get_local_ips

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Color support for Windows console
os.system("")  # Enable ANSI escape sequences on Windows
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "local_asr_models"
SERVER_PY = PROJECT_ROOT / "remote_inference_service" / "server_windows.py"


def check_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    primary_dll = "llama.dll" if sys.platform == "win32" else "libllama.so"
    expected = {
        "SenseVoice (ASR 极速语音识别)": [
            models_dir / "sensevoice-onnx" / "am.mvn",
            models_dir / "sensevoice-onnx" / "embedding.npy",
            models_dir / "sensevoice-onnx" / "sense-voice-encoder-int8.onnx",
            models_dir / "sensevoice-onnx" / "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ],
        "Qwen3-ASR (高精度语音识别)": [
            models_dir / "qwen3-asr" / "qwen3_asr_encoder_frontend.int4.onnx",
            models_dir / "qwen3-asr" / "qwen3_asr_encoder_backend.int4.onnx",
            models_dir / "qwen3-asr" / "qwen3_asr_llm.q4_k.gguf",
            models_dir / "qwen_llama_vulkan_bin" / primary_dll,
        ],
        "Hy-MT2 (实时流式翻译)": list((models_dir / "hymt2").glob("*.gguf")),
    }
    result = {}
    for name, paths in expected.items():
        missing_paths = [p for p in paths if not p.is_file()]
        if name.startswith("Hy-MT2") and not paths:
            missing_paths = [models_dir / "hymt2" / "*.gguf"]
        result[name] = {
            "ready": len(missing_paths) == 0,
            "missing": [p.name for p in missing_paths],
        }
    return result


def install_missing_dependencies() -> bool:
    req_file = PROJECT_ROOT / "requirements-local-inference.txt"
    print(f"\n{CYAN}正在调用 pip 安装本地推理必要依赖...{RESET}")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    code = subprocess.call(cmd)
    return code == 0


def print_banner():
    print(f"""
{CYAN}================================================================{RESET}
{BOLD}        Yakutan 远程推理服务端 一键启动器 (Windows)           {RESET}
{CYAN}================================================================{RESET}
项目路径: {PROJECT_ROOT}
Python:   {sys.executable} (v{sys.version.split()[0]})
""")


def _port_conflict_hosts(host: str) -> tuple[str, ...]:
    """端口冲突探测地址：回环 + （绑定非回环时）默认路由网卡地址。"""
    hosts = ["127.0.0.1"]
    if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::", ""):
        hosts.append(host)
    else:
        for ip in get_local_ips():
            if ip != "127.0.0.1":
                hosts.append(ip)
                break
    return tuple(dict.fromkeys(hosts))


def run():
    parser = argparse.ArgumentParser(description="Yakutan 远程推理服务一键启动")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0 允许局域网访问；仅本机使用请传 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18775, help="WebSocket 端口 (默认 18775)")
    parser.add_argument("--auto-install", action="store_true", help="缺失依赖时自动安装")
    parser.add_argument("--check-only", action="store_true", help="仅执行环境诊断并退出")
    args = parser.parse_args()

    print_banner()

    # 1. 检查端口占用
    if check_port_in_use(args.port, _port_conflict_hosts(args.host)):
        print(f"{RED}[错误] 端口 {args.port} 已被占用！{RESET}")
        print(f"请检查是否已经有一个 Yakutan 服务端实例正在运行（关闭其启动窗口即可结束该实例）。")
        print(f"若端口仍被残留进程占用，可在任务管理器中结束对应的 python.exe 后重试。")
        sys.exit(1)

    # 2. 检查 Python 依赖
    print(f"{BOLD}[1/4] 检查运行依赖...{RESET}")
    installed, missing = check_dependencies()
    if missing:
        print(f"{YELLOW}[提示] 检测到缺少以下本地推理依赖库:{RESET}")
        for m in missing:
            print(f"  - {RED}[MISSING]{RESET} {m}")
        if not args.check_only:
            if args.auto_install:
                ok = install_missing_dependencies()
                if not ok:
                    print(f"{RED}[错误] 依赖安装失败，请手动执行 pip install -r requirements-local-inference.txt{RESET}")
                    sys.exit(1)
            else:
                try:
                    choice = input(f"\n是否现在自动调用 pip 安装补齐缺失依赖？[Y/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "n"
                if choice in ("", "y", "yes"):
                    ok = install_missing_dependencies()
                    if not ok:
                        print(f"{RED}[错误] 依赖安装失败，请检查网络或 pip 配置后重试。{RESET}")
                        sys.exit(1)
                else:
                    print(f"{YELLOW}[跳过安装] 缺失依赖可能导致语音识别模型加载失败。{RESET}")
    else:
        print(f"  {GREEN}[OK]{RESET} 所有核心依赖库均已就绪 ({len(installed)}/{len(REQUIRED_MODULES)})")

    # 3. 检查模型资产
    print(f"\n{BOLD}[2/4] 检查模型资产 (local_asr_models)...{RESET}")
    model_statuses = check_models(MODELS_ROOT)
    for model_name, info in model_statuses.items():
        if info["ready"]:
            print(f"  - {GREEN}✓ 就绪{RESET}: {model_name}")
        else:
            print(f"  - {RED}✕ 缺失{RESET}: {model_name} (缺少文件: {', '.join(info['missing'])})")

    has_asr = model_statuses["SenseVoice (ASR 极速语音识别)"]["ready"] or model_statuses["Qwen3-ASR (高精度语音识别)"]["ready"]
    if not has_asr:
        print(f"\n{RED}[警告] 没有检测到任何可用的 ASR 语音识别模型！{RESET}")
        print(f"请先打开 Yakutan 桌面客户端并在界面下载模型，或将模型拷贝至 {MODELS_ROOT}")

    if model_statuses["Hy-MT2 (实时流式翻译)"]["ready"]:
        print(f"  - {GREEN}✓{RESET} Hy-MT2 将由服务端进程内引擎加载 (无需外部 llama-server，不额外占用显存副本)")
    else:
        print(f"  - {YELLOW}! 提示{RESET}: 未检测到 Hy-MT2 模型；如需本地流式翻译，"
              f"可运行 remote_inference_service/run_hymt2_llama_server.bat 启动外部回退后端")

    if args.check_only:
        print(f"\n{GREEN}诊断完成。{RESET}")
        return

    # 4. 提示客户端连接地址
    print(f"\n{BOLD}[3/4] 网络地址配置{RESET}")
    print(f"服务端监听地址: {args.host}:{args.port}")
    print(f"{CYAN}供其他客户端填写的远程连接地址:{RESET}")
    for ip in get_local_ips():
        print(f"   ► {BOLD}{GREEN}ws://{ip}:{args.port}{RESET}")
    print(f"{YELLOW}[防火墙提醒] 若其他电脑无法连入，请运行「允许局域网访问(防火墙放行).bat」放行 TCP {args.port}。{RESET}")
    print(f"{YELLOW}[安全提醒] 本服务无鉴权，请仅在可信局域网内开放，请勿直接暴露到公网。{RESET}")

    # 5. 进程管理与启动
    subprocesses: list[subprocess.Popen] = []

    def _terminate_subprocesses() -> None:
        for p in subprocesses:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in subprocesses:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    def cleanup_processes(*_):
        print(f"\n{YELLOW}正在安全停止服务端与子进程...{RESET}")
        _terminate_subprocesses()
        print(f"{GREEN}所有服务已停止。{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)

    # 6. 启动主服务 server_windows.py（轮询等待：任何退出路径都会回收子进程并透传退出码）
    print(f"\n{BOLD}[4/4] 启动 WebSocket 主推理服务...{RESET}")
    server_cmd = [
        sys.executable,
        str(SERVER_PY),
        "--host", args.host,
        "--port", str(args.port),
        "--runtime-root", str(PROJECT_ROOT),
        "--models-root", str(MODELS_ROOT),
    ]

    print(f"{GREEN}================================================================{RESET}")
    print(f"{BOLD}服务端已成功进入运行状态！随时按下 Ctrl+C 可安全停止服务。{RESET}")
    print(f"{GREEN}================================================================{RESET}\n")

    main_proc = subprocess.Popen(server_cmd)
    subprocesses.append(main_proc)

    exit_code = 0
    try:
        while True:
            polled = main_proc.poll()
            if polled is not None:
                exit_code = polled
                break
            time.sleep(0.5)
    except SystemExit:
        raise  # 来自 SIGINT 处理器：子进程已在处理器中回收
    except BaseException:
        pass
    finally:
        _terminate_subprocesses()

    if exit_code == 0:
        print(f"{GREEN}服务端已退出。{RESET}")
    else:
        print(f"{RED}[错误] 服务端异常退出 (exit code {exit_code})，请查看上方日志。{RESET}")
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    run()
