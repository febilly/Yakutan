
import asyncio
import logging
from typing import Optional

from shared.vrchat_bridge import (
    MessageType,
    YakutanMessage,
    ForeignSpeech,
    Heartbeat,
    serialize_message,
    deserialize_message,
    discover_peer,
    connect_bridge_client,
    get_discovery_path,
)
from app_state import get_smart_selector

logger = logging.getLogger(__name__)


class IPCClient:
    def __init__(self, translator=None):
        import config
        self._translator = translator
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._mode = "standalone"
        self._lock = asyncio.Lock()
        self._read_task: Optional[asyncio.Task] = None
        self._wait_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._host = config.IPC_HOST
        self._discovery_file = config.IPC_DISCOVERY_FILE
        self._poll_interval = config.IPC_POLL_INTERVAL
        self._connect_timeout = config.IPC_CONNECT_TIMEOUT
        self._delegate_osc_enabled = False

    def is_connected(self) -> bool:
        return self._mode == "delegate"

    def is_delegate_osc_enabled(self) -> bool:
        return self._delegate_osc_enabled and self.is_connected()

    def get_mode(self) -> str:
        return self._mode

    def set_translator(self, translator) -> None:
        self._translator = translator

    async def start(self):
        try:
            await asyncio.wait_for(self._try_connect(), timeout=self._connect_timeout)
        except (asyncio.TimeoutError, ConnectionError):
            logger.info("Subtitle IPC server not found, entering wait mode (polling every %ss)", self._poll_interval)
            await self._enter_wait_mode()
        except Exception as e:
            logger.warning("[IPC] Initial connection failed: %s", e)
            await self._enter_wait_mode()

    async def _try_connect(self):
        port = await discover_peer(self._discovery_file)
        if port is None:
            raise ConnectionError("No peer discovered")

        try:
            reader, writer = await connect_bridge_client(self._host, port)
        except Exception:
            logger.warning("[IPC] Peer discovered on port %s but connection failed", port)
            raise

        async with self._lock:
            self._reader = reader
            self._writer = writer
            self._mode = "delegate"

        logger.info("Connected to subtitle IPC server on port %s, delegating OSC sending", port)
        self._read_task = asyncio.create_task(self._read_loop())

    async def _enter_wait_mode(self):
        async with self._lock:
            if self._mode == "waiting":
                return
            self._mode = "waiting"
            self._reader = None
            self._writer = None

        if self._wait_task is None or self._wait_task.done():
            self._wait_task = asyncio.create_task(self._wait_loop())

    async def _wait_loop(self):
        while True:
            try:
                await self._try_connect()
                return
            except ConnectionError:
                pass
            except Exception as e:
                if "Peer discovered" in str(e) or "connection failed" in str(e):
                    logger.warning("[IPC] %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _read_loop(self):
        try:
            while True:
                if self._reader is None:
                    break
                line = await self._reader.readline()
                if not line:
                    break

                data = deserialize_message(line.decode("utf-8").strip())
                if data is None:
                    continue

                msg_type = data.get("type")
                if msg_type == MessageType.FOREIGN_SPEECH.value:
                    source_text = data.get("source_text", "")
                    detected_language = data.get("detected_language")
                    if source_text and self._translator is not None:
                        try:
                            self._translator.add_external_speech(source_text)
                            logger.debug("[IPC] Injected foreign speech: %s", source_text)
                        except Exception as e:
                            logger.error("[IPC] Failed to add external speech: %s", e)
                    if detected_language:
                        try:
                            get_smart_selector().record_language(detected_language)
                            logger.debug("[IPC] Recorded foreign language: %s", detected_language)
                        except Exception as e:
                            logger.error("[IPC] Failed to record foreign language: %s", e)
                elif msg_type == MessageType.HEARTBEAT.value:
                    pass
                elif msg_type == MessageType.OSC_STATE.value:
                    enabled = bool(data.get("enabled", False))
                    self._delegate_osc_enabled = enabled
                    logger.info("[IPC] OSC delegation state from server: enabled=%s", enabled)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[IPC] Read loop error: %s", e)
        finally:
            await self._on_disconnect()

    async def _on_disconnect(self):
        await self._close_connection()
        try:
            get_smart_selector().clear_history()
            logger.info("[IPC] Cleared smart target language history on disconnect")
        except Exception:
            pass
        if self._mode != "standalone":
            async with self._lock:
                self._mode = "waiting"
                self._delegate_osc_enabled = False
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        while True:
            try:
                await self._try_connect()
                return
            except ConnectionError:
                pass
            except Exception as e:
                if "Peer discovered" in str(e) or "connection failed" in str(e):
                    logger.warning("[IPC] %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _close_connection(self):
        writer = None
        async with self._lock:
            writer = self._writer
            self._writer = None
            self._reader = None

        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def send_message(self, text: str, ongoing: bool):
        if not self.is_connected():
            return

        try:
            msg = YakutanMessage(text=text, ongoing=ongoing)
            line = serialize_message(msg)
            async with self._lock:
                if self._writer is not None:
                    self._writer.write(line.encode("utf-8"))
                    await self._writer.drain()
        except Exception as e:
            logger.error("[IPC] Failed to send message: %s", e)
            await self._on_disconnect()

    async def set_typing(self, typing: bool):
        if not self.is_connected():
            return
        await self.send_message("", typing)

    async def stop(self):
        async with self._lock:
            self._mode = "standalone"
            self._translator = None

        for task in (self._read_task, self._wait_task, self._reconnect_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self._close_connection()
        try:
            get_smart_selector().clear_history()
        except Exception:
            pass
