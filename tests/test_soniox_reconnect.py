"""Soniox 断线重连回归测试（审查报告 P1-4）。

覆盖场景：
- 断线后自动重连（指数退避、封顶、失败后持续重试直到成功）
- 重连期间 send_audio_frame 不抛异常（丢弃且有节流日志）
- 重连成功后恢复发送，且会话语义（语言提示/上下文）在建连时重放
- start() 短路消除：连接死亡后 start() 重建连接
- resume() 检测连接死亡后重建连接
- stop() 后不触发重连
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, List, Optional

import pytest

import speech_recognizers.soniox_speech_recognizer as soniox_mod
from speech_recognizers.base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
)

# 退避用例通过 fake sleep 抛异常终止重连线程，线程异常告警属于预期噪声
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)


class RecorderCallback(SpeechRecognitionCallback):
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.errors: List[Exception] = []
        self.results: List[RecognitionEvent] = []
        self._lock = threading.Lock()

    def on_session_started(self) -> None:
        with self._lock:
            self.started += 1

    def on_session_stopped(self) -> None:
        with self._lock:
            self.stopped += 1

    def on_error(self, error: Exception) -> None:
        with self._lock:
            self.errors.append(error)

    def on_result(self, event: RecognitionEvent) -> None:
        with self._lock:
            self.results.append(event)


class FakeWebSocket:
    """可编程的 WebSocket 假实现：脚本条目为消息字符串或待抛出的异常。"""

    def __init__(self, script: Optional[List[Any]] = None) -> None:
        self.script: List[Any] = list(script or [])
        self.sent: List[Any] = []
        self.closed = False
        self.send_error: Optional[Exception] = None
        self._lock = threading.Lock()

    def send(self, data: Any) -> None:
        if self.send_error is not None:
            raise self.send_error
        if self.closed:
            raise soniox_mod.ConnectionClosedError(None, None)
        with self._lock:
            self.sent.append(data)

    def recv(self, timeout: Optional[float] = None) -> str:
        deadline = time.monotonic() + (timeout if timeout is not None else 1.0)
        while True:
            with self._lock:
                if self.script:
                    item = self.script.pop(0)
                    break
            if self.closed:
                raise soniox_mod.ConnectionClosedOK(None, None)
            if time.monotonic() >= deadline:
                raise TimeoutError()
            time.sleep(0.005)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class WsFactory:
    """按 FIFO 顺序返回 FakeWebSocket 或抛出工厂脚本中的异常。"""

    def __init__(self) -> None:
        self.script: List[Any] = []
        self.calls = 0
        self._lock = threading.Lock()

    def queue(self, fake: FakeWebSocket) -> FakeWebSocket:
        self.script.append(fake)
        return fake

    def queue_error(self, error: Exception) -> None:
        self.script.append(error)

    def __call__(self, *args: Any, **kwargs: Any) -> FakeWebSocket:
        with self._lock:
            self.calls += 1
            if self.script:
                item = self.script.pop(0)
            else:
                raise AssertionError("WsFactory exhausted")
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def connected_count(self) -> int:
        return self.calls


def wait_until(condition, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


def _block_background_reconnect(recognizer) -> None:
    """预置 _reconnecting=True，阻断接收线程退出时自动派生后台重连线程。

    用于聚焦 start()/resume() 自身重建路径的用例，避免与后台重连竞态。
    """
    with recognizer._lock:
        recognizer._reconnecting = True


def _unblock_background_reconnect(recognizer) -> None:
    with recognizer._lock:
        recognizer._reconnecting = False


@pytest.fixture()
def fast_reconnect(monkeypatch):
    """把重连退避压缩到毫秒级，保证测试快速且确定性。"""
    monkeypatch.setattr(soniox_mod, "SONIOX_RECONNECT_INITIAL_DELAY", 0.01)
    monkeypatch.setattr(soniox_mod, "SONIOX_RECONNECT_MAX_DELAY", 0.05)
    monkeypatch.setattr(soniox_mod, "refresh_system_proxy_env", lambda: None)
    monkeypatch.setattr(soniox_mod, "SONIOX_DROP_LOG_INTERVAL", 60.0)


def make_recognizer(callback: RecorderCallback) -> soniox_mod.SonioxSpeechRecognizer:
    return soniox_mod.SonioxSpeechRecognizer(
        callback=callback,
        api_key="test-key",
        language_hints=["ja", "en"],
        context={"terms": ["FooTerm"]},
    )


def test_disconnect_triggers_auto_reconnect_and_replays_session(fast_reconnect, monkeypatch):
    """断线后 _recv_worker 自动触发重连，且重连建连时重放语言/上下文配置。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake1)
    assert factory.connected_count == 1

    # 服务端断开连接 -> 接收线程退出 -> 自动重连
    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)

    # 第二次建连重放了会话语义（语言提示 + 上下文）
    config_msg = json.loads(fake2.sent[0])
    assert config_msg["api_key"] == "test-key"
    assert config_msg["language_hints"] == ["ja", "en"]
    assert "FooTerm" in config_msg["context"]["terms"]

    # 回调语义：断开时 stopped，重建后重新 started
    assert callback.started >= 2
    assert callback.stopped >= 1

    recognizer.stop()
    assert recognizer._connected is False
    assert recognizer._ws is None


def test_reconnect_retries_until_success(fast_reconnect, monkeypatch):
    """重连失败（退避）后继续重试，直到成功建立连接。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    factory.queue_error(RuntimeError("network down 1"))
    factory.queue_error(RuntimeError("network down 2"))
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    fake1.script.append(RuntimeError("connection reset"))
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    assert factory.connected_count == 4  # 初连 + 2 次失败 + 最终成功

    recognizer.stop()


def test_send_during_reconnect_does_not_raise_and_resumes_after(fast_reconnect, monkeypatch, capsys):
    """重连等待期间 send_audio_frame 静默不抛（带节流日志）；重连成功后恢复发送。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    # 进入断线窗口（_connected=False 且 _ws 已清空）
    assert wait_until(lambda: not recognizer._connected and recognizer._ws is None)

    # 断线期间反复发送：不抛异常；丢弃日志有节流（60s 窗口内最多一条）
    for _ in range(20):
        recognizer.send_audio_frame(b"\x00\x01")
        time.sleep(0.002)
        if recognizer._connected:
            break
    out = capsys.readouterr().out
    assert out.count("dropping audio frame") == 1

    # 重连成功后恢复发送
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    recognizer.send_audio_frame(b"hello-after-reconnect")
    assert wait_until(lambda: b"hello-after-reconnect" in fake2.sent)

    recognizer.stop()


def test_start_short_circuit_eliminated(fast_reconnect, monkeypatch):
    """连接死亡后 start() 不再被残留 _ws 短路，而是重建连接。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    _block_background_reconnect(recognizer)
    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    assert wait_until(lambda: not recognizer._connected and recognizer._ws is None)
    # 旧实现里这里 _ws 仍指向死连接，start() 直接短路返回
    assert factory.connected_count == 1
    _unblock_background_reconnect(recognizer)

    recognizer.start()
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    assert factory.connected_count == 2

    recognizer.send_audio_frame(b"after-start-rebuild")
    assert wait_until(lambda: b"after-start-rebuild" in fake2.sent)

    # 连接存活时重复 start() 保持幂等，不新建连接
    recognizer.start()
    assert factory.connected_count == 2

    recognizer.stop()


def test_start_rebuilds_even_with_stale_ws_reference(fast_reconnect, monkeypatch):
    """即使 _ws 被外部残留为非 None（旧缺陷形态），start() 也依据 _connected 重建。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    _block_background_reconnect(recognizer)
    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    assert wait_until(lambda: not recognizer._connected and recognizer._ws is None)

    # 模拟旧缺陷形态：断线后 _ws 仍残留死连接引用
    with recognizer._lock:
        recognizer._ws = fake1
    _unblock_background_reconnect(recognizer)

    recognizer.start()
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    assert factory.connected_count == 2

    recognizer.stop()


def test_resume_rebuilds_connection_when_disconnected(fast_reconnect, monkeypatch):
    """pause 期间断线后，resume() 检测 _connected=False 并重建连接。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    recognizer.pause()
    # pause 应发送 finalize
    assert any(
        isinstance(item, (bytes, str)) and json.loads(item).get("type") == "finalize"
        for item in fake1.sent
    )

    _block_background_reconnect(recognizer)
    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    assert wait_until(lambda: not recognizer._connected and recognizer._ws is None)
    _unblock_background_reconnect(recognizer)

    recognizer.resume()
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    recognizer.send_audio_frame(b"after-resume-rebuild")
    assert wait_until(lambda: b"after-resume-rebuild" in fake2.sent)

    recognizer.stop()


def test_resume_falls_back_to_background_reconnect_on_failure(fast_reconnect, monkeypatch):
    """resume() 重建失败时转入后台重连，最终恢复。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    factory.queue_error(RuntimeError("resume boom"))
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    recognizer.pause()
    _block_background_reconnect(recognizer)
    fake1.script.append(soniox_mod.ConnectionClosedError(None, None))
    assert wait_until(lambda: not recognizer._connected and recognizer._ws is None)
    _unblock_background_reconnect(recognizer)

    recognizer.resume()  # 第一次 _connect 失败，转入后台重连
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)

    recognizer.stop()


def test_stop_prevents_reconnect(fast_reconnect, monkeypatch):
    """stop() 后（_should_run=False）不再触发自动重连。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)
    recognizer.stop()

    recognizer._maybe_start_reconnect_thread()
    time.sleep(0.1)
    assert factory.connected_count == 1
    assert recognizer._reconnecting is False


def test_backoff_is_exponential_and_capped(monkeypatch):
    """重连退避指数增长且封顶于 SONIOX_RECONNECT_MAX_DELAY。"""
    monkeypatch.setattr(soniox_mod, "SONIOX_RECONNECT_INITIAL_DELAY", 0.01)
    monkeypatch.setattr(soniox_mod, "SONIOX_RECONNECT_MAX_DELAY", 0.04)
    monkeypatch.setattr(soniox_mod, "refresh_system_proxy_env", lambda: None)

    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    for _ in range(10):
        factory.queue_error(RuntimeError("still down"))
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    delays: List[float] = []

    class _StopLoop(Exception):
        pass

    def fake_sleep(seconds: float) -> None:
        delays.append(seconds)
        if len(delays) >= 5:
            raise _StopLoop()

    monkeypatch.setattr(time, "sleep", fake_sleep)

    with recognizer._lock:
        recognizer._should_run = True
        recognizer._connected = False
        recognizer._reconnecting = False
    recognizer._maybe_start_reconnect_thread()

    # time.sleep 已被替换，这里用忙等而非 wait_until
    deadline = time.monotonic() + 3.0
    while recognizer._reconnecting and time.monotonic() < deadline:
        pass
    assert not recognizer._reconnecting
    assert delays == [0.01, 0.02, 0.04, 0.04, 0.04]

    with recognizer._lock:
        recognizer._should_run = False


def test_send_failure_on_live_connection_closes_and_reconnects(fast_reconnect, monkeypatch):
    """发送音频时连接死亡：置 _connected=False 并唤醒接收线程触发重连。"""
    callback = RecorderCallback()
    recognizer = make_recognizer(callback)
    factory = WsFactory()
    fake1 = factory.queue(FakeWebSocket())
    fake2 = factory.queue(FakeWebSocket())
    monkeypatch.setattr(soniox_mod, "ws_connect", factory)

    recognizer.start()
    assert wait_until(lambda: recognizer._ws is fake1)

    fake1.send_error = RuntimeError("send pipe broken")
    recognizer.send_audio_frame(b"\x00\x00")
    assert wait_until(lambda: not recognizer._connected)
    assert wait_until(lambda: recognizer._connected and recognizer._ws is fake2)
    recognizer.send_audio_frame(b"post-recovery-frame")
    assert wait_until(lambda: b"post-recovery-frame" in fake2.sent)

    recognizer.stop()
