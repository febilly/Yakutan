"""
检测占用本机指定 UDP 端口的进程（用于 OSC 端口冲突提示）。
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Dict, List


def _is_vrchat_process(name: str) -> bool:
    n = (name or "").strip().lower()
    if n.endswith(".exe"):
        n = n[:-4]
    return n == "vrchat"


def _dedupe_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: set[int] = set()
    out: List[Dict[str, object]] = []
    for e in entries:
        pid = int(e["pid"])
        if pid in seen:
            continue
        seen.add(pid)
        out.append(e)
    return out


def _win_powershell_owners(port: int) -> List[Dict[str, object]]:
    ps = (
        f"$eps = @(); "
        f"try {{ $eps = Get-NetUDPEndpoint -LocalPort {int(port)} -ErrorAction Stop }} catch {{ }}; "
        r"foreach ($e in $eps) { "
        r"  $proc = Get-Process -Id $e.OwningProcess -ErrorAction SilentlyContinue; "
        r"  if ($null -ne $proc) { Write-Output ($e.OwningProcess.ToString() + '|' + $proc.ProcessName) } "
        r"}"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=20,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return []
    out: List[Dict[str, object]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        pid_s, _, name = line.partition("|")
        pid_s, name = pid_s.strip(), name.strip()
        if not pid_s.isdigit() or not name:
            continue
        out.append({"pid": int(pid_s), "name": name})
    return out


def _win_netstat_owners(port: int) -> List[Dict[str, object]]:
    token = f":{int(port)}"
    try:
        proc = subprocess.run(
            ["cmd", "/c", "netstat -ano -p udp"],
            capture_output=True,
            text=True,
            timeout=25,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return []
    pids: List[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if "UDP" not in line.upper() or token not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        found = False
        for p in parts:
            if p.endswith(token):
                found = True
                break
        if not found:
            continue
        tail = parts[-1]
        if tail.isdigit():
            pids.append(int(tail))
    pids = list(dict.fromkeys(pids))
    out: List[Dict[str, object]] = []
    for pid in pids:
        name = _win_pid_to_name(pid)
        if name:
            out.append({"pid": pid, "name": name})
    return out


def _win_pid_to_name(pid: int) -> str:
    ps = f"(Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue).ProcessName"
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    return (proc.stdout or "").strip()


def _lsof_udp_owners(port: int) -> List[Dict[str, object]]:
    try:
        proc = subprocess.run(
            ["lsof", "-i", f"UDP:{int(port)}", "-P", "-n"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError):
        return []
    out: List[Dict[str, object]] = []
    for line in (proc.stdout or "").splitlines():
        if not line.strip() or line.startswith("COMMAND"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, pid_s = parts[0], parts[1]
        if not pid_s.isdigit():
            continue
        out.append({"pid": int(pid_s), "name": name})
    return out


def _ss_udp_owners(port: int) -> List[Dict[str, object]]:
    try:
        proc = subprocess.run(
            ["ss", "-H", "-ulnp", f"sport = :{int(port)}"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError):
        return []
    out: List[Dict[str, object]] = []
    for line in (proc.stdout or "").splitlines():
        m = re.search(r'pid=(\d+)', line)
        if not m:
            continue
        pid = int(m.group(1))
        cm = re.search(r'"([^"]+)"', line)
        name = cm.group(1) if cm else "?"
        out.append({"pid": pid, "name": name})
    return out


def _collect_udp_owners(port: int) -> List[Dict[str, object]]:
    if sys.platform == "win32":
        owners = _win_powershell_owners(port)
        if not owners:
            owners = _win_netstat_owners(port)
        return owners
    if sys.platform.startswith("linux"):
        owners = _ss_udp_owners(port)
        if not owners:
            owners = _lsof_udp_owners(port)
        return owners
    if sys.platform == "darwin":
        return _lsof_udp_owners(port)
    return _lsof_udp_owners(port)


def get_non_vrchat_udp_port_occupants(port: int) -> List[Dict[str, object]]:
    """
    返回占用本机 UDP `port` 的进程列表（排除 VRChat），元素为 {'pid', 'name'}。
    检测失败时返回空列表，不阻断服务启动。
    """
    try:
        p = int(port)
    except (TypeError, ValueError):
        return []
    if p < 1 or p > 65535:
        return []
    try:
        raw = _collect_udp_owners(p)
    except Exception:
        return []
    merged = _dedupe_entries(raw)
    return [e for e in merged if not _is_vrchat_process(str(e.get("name", "")))]
