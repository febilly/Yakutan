@echo off
chcp 65001 >nul
title Yakutan Hy-MT2 llama-server (Windows fallback backend)

setlocal enabledelayedexpansion

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"
set "PORT=18776"

echo =======================================================
echo          Yakutan Hy-MT2 llama-server 启动脚本
echo =======================================================
echo.
echo [说明] 服务端 (server_windows.py) 已内置进程内 Hy-MT2 引擎，
echo        检测到 local_asr_models\hymt2\*.gguf 时会自动使用，无需本脚本。
echo        本脚本仅作为回退：当内置引擎不可用（如缺少 Vulkan 运行库）时，
echo        由它在 127.0.0.1:18776 提供外部翻译后端。
echo.

:: 按通配符寻找 Hy-MT2 模型 (与 server_windows.py / launcher.py 的 *.gguf 探测保持一致)
set "HYMT2_MODEL="
for %%F in ("%PROJECT_ROOT%\local_asr_models\hymt2\*.gguf") do (
    if not defined HYMT2_MODEL set "HYMT2_MODEL=%%~fF"
)

if not defined HYMT2_MODEL (
    echo [错误] 未找到 Hy-MT2 模型文件:
    echo        %PROJECT_ROOT%\local_asr_models\hymt2\*.gguf
    echo 请先在 Yakutan 客户端中下载 Hy-MT2 模型，或手动放入该目录。
    pause
    exit /b 1
)

:: 寻找 llama-server.exe
set "LLAMA_SERVER="
if exist "%PROJECT_ROOT%\llama-server.exe" (
    set "LLAMA_SERVER=%PROJECT_ROOT%\llama-server.exe"
) else if exist "%PROJECT_ROOT%\local_asr_models\llama-server.exe" (
    set "LLAMA_SERVER=%PROJECT_ROOT%\local_asr_models\llama-server.exe"
) else (
    for %%P in (llama-server.exe) do set "LLAMA_SERVER=%%~$PATH:P"
)

if "%LLAMA_SERVER%"=="" (
    echo [提示] 当前路径及系统 PATH 中未找到 llama-server.exe。
    echo.
    echo 使用外部回退后端需要 llama.cpp 的 llama-server:
    echo 1. 从 llama.cpp 官方 GitHub Releases (https://github.com/ggerganov/llama.cpp/releases)
    echo    下载 Windows 版二进制包（如 llama-bXXXX-bin-win-vulkan-x64.zip 或 cuda 版）。
    echo 2. 将其中的 llama-server.exe 解压到当前项目根目录下。
    echo 3. 重新运行本脚本即可。
    echo.
    pause
    exit /b 1
)

echo 使用 llama-server: %LLAMA_SERVER%
echo 模型文件: %HYMT2_MODEL%
echo 监听端口: %PORT% (仅 127.0.0.1，无需防火墙放行)
echo.

"%LLAMA_SERVER%" -m "%HYMT2_MODEL%" --port %PORT% --host 127.0.0.1 -c 2048 -ngl 99 --flash-attn

if errorlevel 1 (
    echo.
    echo [llama-server 已退出]
    pause
)
