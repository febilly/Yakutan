import asyncio
import json
import os
import sys
import time
import logging
import tempfile
import atexit
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, Union

DEFAULT_PORT_RANGE = range(17353, 17364)
DISCOVERY_FILE_NAME = "vrchat_bridge_discovery.json"

class MessageType(str, Enum):
    YAKUTAN_MESSAGE = "YAKUTAN_MESSAGE"
    FOREIGN_SPEECH = "FOREIGN_SPEECH"
    HEARTBEAT = "HEARTBEAT"

@dataclass
class YakutanMessage:
    text: str
    ongoing: bool = False
    type: MessageType = MessageType.YAKUTAN_MESSAGE

@dataclass
class ForeignSpeech:
    source_text: str
    detected_language: Optional[str] = None
    type: MessageType = MessageType.FOREIGN_SPEECH

@dataclass
class Heartbeat:
    type: MessageType = MessageType.HEARTBEAT

def serialize_message(message: Union[YakutanMessage, ForeignSpeech, Heartbeat]) -> str:
    data = asdict(message)
    if isinstance(data['type'], MessageType):
        data['type'] = data['type'].value
    return json.dumps(data) + "\n"

def deserialize_message(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def get_discovery_path() -> str:
    if sys.platform == "win32":
        temp_dir = os.environ.get("TEMP", os.environ.get("TMP", tempfile.gettempdir()))
    else:
        temp_dir = "/tmp"
    return os.path.join(temp_dir, DISCOVERY_FILE_NAME)

def _write_discovery_file(path: str, host: str, port: int):
    try:
        data = {
            "host": host,
            "port": port,
            "pid": os.getpid(),
            "timestamp": time.time()
        }
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Failed to write discovery file: {e}")

def _cleanup_discovery_file(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if data.get("pid") == os.getpid():
                    os.remove(path)
        except Exception:
            pass

async def _dummy_client_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Placeholder client handler - actual handling is done by the caller."""
    pass

async def start_bridge_server(host: str, port_range: range = DEFAULT_PORT_RANGE):
    for port in port_range:
        try:
            server = await asyncio.start_server(_dummy_client_handler, host, port)
            return server, port
        except OSError:
            continue
    raise OSError(f"Could not bind to any port in range {port_range}")

async def discover_peer(discovery_file: str, timeout: float = 30.0) -> Optional[int]:
    if not os.path.exists(discovery_file):
        return None
    try:
        with open(discovery_file, "r") as f:
            data = json.load(f)
        
        if time.time() - data.get("timestamp", 0) > timeout:
            return None
            
        pid = data.get("pid")
        if pid:
            if sys.platform == "win32":
                import ctypes
                PROCESS_QUERY_INFORMATION = 0x0400
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if handle == 0:
                    return None
                ctypes.windll.kernel32.CloseHandle(handle)
            else:
                try:
                    os.kill(pid, 0)
                except OSError:
                    return None
                    
        return data.get("port")
    except Exception:
        return None

async def start_server_with_discovery(host: str, port_range: range = DEFAULT_PORT_RANGE, discovery_file: Optional[str] = None):
    if discovery_file is None:
        discovery_file = get_discovery_path()
        
    server, port = await start_bridge_server(host, port_range)
    _write_discovery_file(discovery_file, host, port)
    
    atexit.register(_cleanup_discovery_file, discovery_file)
    
    return server, port

async def connect_bridge_client(host: str, port: int) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection(host, port)

async def wait_and_connect(host: str, discovery_file: Optional[str] = None):
    if discovery_file is None:
        discovery_file = get_discovery_path()
        
    backoff = 1.0
    while True:
        port = await discover_peer(discovery_file)
        if port:
            try:
                reader, writer = await connect_bridge_client(host, port)
                return reader, writer
            except Exception:
                pass
        
        await asyncio.sleep(min(backoff, 30.0))
        if port:
            backoff *= 2
        else:
            backoff = 3.0
