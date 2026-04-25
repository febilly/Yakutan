"""
配置文件 - 统一管理所有配置项
"""
import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {'', '0', 'false', 'no', 'off'}


def _get_env_int(name: str, default: int, *, min_v: int = 1, max_v: int = 65535) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        v = int(str(raw).strip(), 10)
        return max(min_v, min(max_v, v))
    except ValueError:
        return default

# ============================================================================
# 语音识别后端配置
# ============================================================================

# 是否使用国际版端点（阿里云 DashScope）
# 国际版用户需要设置为 True
USE_INTERNATIONAL_ENDPOINT = False

# 首选的语音识别后端
PREFERRED_ASR_BACKEND = 'qwen'  # 可选: 'dashscope', 'qwen', 'soniox', 'doubao_file', 'local'
                                # 注意: 'dashscope' (Fun-ASR) 仅支持中国大陆版

# 有效的后端列表
VALID_ASR_BACKENDS = {'dashscope', 'qwen', 'soniox', 'doubao_file', 'local'}

# ============================================================================
# 语音识别模型配置
# ============================================================================

# DashScope 后端使用的模型
DASHSCOPE_ASR_MODEL = 'fun-asr-realtime'

# Qwen 后端使用的模型
QWEN_ASR_MODEL = 'qwen3-asr-flash-realtime-2026-02-10'

# Qwen WebSocket URL
QWEN_ASR_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
QWEN_ASR_URL_INTERNATIONAL = 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime'

# ============================================================================
# Soniox 配置
# ============================================================================

# Soniox WebSocket URL
SONIOX_WEBSOCKET_URL = 'wss://stt-rt.soniox.com/transcribe-websocket'

# Soniox 模型
SONIOX_MODEL = 'stt-rt-v3'

# Soniox 语言提示（用于提高识别准确度）
SONIOX_LANGUAGE_HINTS = ['en', 'zh', 'ja', 'ko']

# 是否启用端点检测（自动断句）
SONIOX_ENABLE_ENDPOINT_DETECTION = True

# ============================================================================
# 豆包录音文件识别（极速版）配置
# ============================================================================

DOUBAO_ASR_FLASH_URL = 'https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash'
DOUBAO_ASR_RESOURCE_ID = 'volc.seedasr.auc'
DOUBAO_ASR_MODEL_NAME = 'bigmodel'
DOUBAO_ASR_TIMEOUT_SECONDS = 60
DOUBAO_ASR_MAX_BUFFER_SECONDS = 60

# ============================================================================
# 本地语音识别配置（默认值沿用 LiveTranslate）
# ============================================================================

# 本地 ASR 引擎
# 可选: 'sensevoice', 'qwen3-asr'（已移除 Fun-ASR-Nano）
# sensevoice：INT8 ONNX，固定 CPU（约 1.5–2.5GB 内存；发布版可内置模型）
# qwen3-asr：ONNX 编码默认 CPU；可选 DirectML（LOCAL_QWEN_ENCODER_USE_DML）。GGUF 解码可走 Vulkan，约需显存视配置而定
LOCAL_ASR_ENGINE = 'sensevoice'
_VALID_LOCAL_ASR_ENGINES = frozenset({'sensevoice', 'qwen3-asr'})
if LOCAL_ASR_ENGINE not in _VALID_LOCAL_ASR_ENGINES:
    LOCAL_ASR_ENGINE = 'sensevoice'

# 本地 VAD 配置（默认值沿用 LiveTranslate VADProcessor.__init__）
LOCAL_VAD_MODE = 'silero'  # 可选: 'silero', 'energy', 'disabled'
LOCAL_VAD_THRESHOLD = 0.50
LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
# 单段口语送入 VAD 的最长时长（秒）；超过后对本段仅送入静音块直至 VAD 静音或闭麦结束本段（不按时长强制切句）
LOCAL_VAD_MAX_SPEECH_DURATION = 30.0
LOCAL_VAD_SILENCE_DURATION = 0.8
# 起声时拼接的预缓冲音频时长（秒），用于避免漏掉第一个字
LOCAL_VAD_PRE_SPEECH_DURATION = 0.2

# 本地增量识别（中间结果）
LOCAL_INCREMENTAL_ASR = True
LOCAL_INTERIM_INTERVAL = 2.0

# Qwen3-ASR：GGUF 解码器 KV 上下文长度（token）；增大占显存/内存。
LOCAL_QWEN_ASR_N_CTX = 2048
# 传入 LLM system 区的背景/滚动文本：按模型分词后最多保留的 token 数（取尾部）。
LOCAL_QWEN_CONTEXT_MAX_TOKENS = 1024
# 是否在每条识别后打印 Qwen3-ASR 各阶段耗时（ONNX 编码 / LLM prefill / 生成），使用 INFO 级别。环境变量 LOCAL_QWEN_LOG_PIPELINE_TIMING=0 可关闭。
# 需在 config.LOG_LEVEL 为 INFO/DEBUG 时才能在终端看到（默认 ERROR 时不会输出）。
LOCAL_QWEN_LOG_PIPELINE_TIMING = _get_env_bool('LOCAL_QWEN_LOG_PIPELINE_TIMING', True)
# ONNX 音频编码（前后端）是否使用 DirectML；False 时仅用 CPUExecutionProvider（Mel 本就为 CPU）。
LOCAL_QWEN_ENCODER_USE_DML = _get_env_bool('LOCAL_QWEN_ENCODER_USE_DML', False)

# ============================================================================
# 音频参数配置
# ============================================================================

SAMPLE_RATE = 16000  # 采样率 (Hz)
CHANNELS = 1  # 单声道
DTYPE = 'int16'  # 数据类型
BITS = 16  # 每个采样的位数
FORMAT_PCM = 'pcm'  # 音频数据格式
BLOCK_SIZE = 1600  # 每个缓冲区的帧数

# 是否将重采样后的音频保存到本地 WAV（调试用）
SAVE_POST_RESAMPLE_AUDIO = _get_env_bool('SAVE_POST_RESAMPLE_AUDIO', False)

# 是否将重采样前的原始采集音频保存到本地 WAV（调试用）
SAVE_PRE_RESAMPLE_AUDIO = _get_env_bool('SAVE_PRE_RESAMPLE_AUDIO', False)

# 调试音频输出目录（相对路径时相对于项目根目录）
DEBUG_AUDIO_OUTPUT_DIR = os.getenv('DEBUG_AUDIO_OUTPUT_DIR', 'debug_audio').strip() or 'debug_audio'

# ============================================================================
# 翻译语言配置
# ============================================================================

SOURCE_LANGUAGE = 'auto'  # 翻译源语言（'auto' 为自动检测，或指定如 'en', 'ja' 等）
TARGET_LANGUAGE = 'ja'  # 翻译目标语言（'zh-CN'=简体中文, 'en'=英文, 'ja'=日文 等）
SECONDARY_TARGET_LANGUAGE = None  # 第二输出语言（可选，启用后将并行输出两种译文）
FALLBACK_LANGUAGE = 'en'  # 备用翻译语言（当源语言和目标语言相同时使用）
                           # 设置为 None（非字符串）则禁用备用语言功能

# 智能目标语言（根据最近别人说的话自动推断翻译目标语言）
SMART_TARGET_LANGUAGE_ENABLED = False
SMART_TARGET_LANGUAGE_STRATEGY = "most_common"  # 可选: most_common, latest, weighted
SMART_TARGET_LANGUAGE_WINDOW_SIZE = 10
SMART_TARGET_LANGUAGE_COUNT = 1
SMART_TARGET_LANGUAGE_EXCLUDE_SELF_LANGUAGE = True
SMART_TARGET_LANGUAGE_MANUAL_SECONDARY = None
SMART_TARGET_LANGUAGE_FALLBACK = "en"

# 后端本次启动的时刻（毫秒），整个进程生命周期内固定不变
BACKEND_BOOT_MS = int(time.time() * 1000)

# 配置最后一次被成功应用的时刻（毫秒），每次 POST /api/config 或 /api/target-language 成功后刷新
CONFIG_APPLIED_AT_MS = BACKEND_BOOT_MS


def bump_config_applied_at_ms() -> int:
    global CONFIG_APPLIED_AT_MS
    CONFIG_APPLIED_AT_MS = int(time.time() * 1000)
    return CONFIG_APPLIED_AT_MS

# ============================================================================
# 翻译 API 配置
# ============================================================================

# 翻译 API 类型
# 可选: 'google_web', 'google_dictionary', 'deepl', 'openrouter',
#      'openrouter_streaming', 'openrouter_streaming_deepl_hybrid', 'qwen_mt'
# 注意:
# - openrouter / openrouter_streaming 表示基于 OpenAI 兼容接口的 LLM 翻译
# - openrouter_streaming 是 LLM 翻译的流式模式，支持翻译部分结果
# - openrouter_streaming_deepl_hybrid 在静音触发终译时，按流式更新次数阈值决定
#   使用 DeepL（更新次数较少）或 LLM（更新次数较多）进行最终翻译
TRANSLATION_API_TYPE = 'qwen_mt'

# LLM（OpenAI 兼容接口）配置
LLM_BASE_URL = (
    os.getenv('LLM_BASE_URL', '').strip()
    or os.getenv('OPENAI_BASE_URL', '').strip()
    or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1').strip()
)
LLM_MODEL = (
    os.getenv('LLM_MODEL', '').strip()
    or os.getenv('OPENAI_MODEL', '').strip()
    or os.getenv('OPENROUTER_TRANSLATION_MODEL', 'qwen/qwen3-235b-a22b-2507').strip()
)
LLM_TRANSLATION_TEMPERATURE = 0.2
LLM_TRANSLATION_TIMEOUT = 30
LLM_TRANSLATION_MAX_RETRIES = 3

# LLM 翻译正式程度
# 可选: 'low', 'medium', 'high'
# 默认保持接近当前偏口语、轻礼貌的风格
LLM_TRANSLATION_FORMALITY = (
    os.getenv('LLM_TRANSLATION_FORMALITY', 'medium').strip().lower() or 'medium'
)

# LLM 句子风格
# 可选: 'standard', 'light'
LLM_TRANSLATION_STYLE = (
    os.getenv('LLM_TRANSLATION_STYLE', 'light').strip().lower() or 'light'
)

# OpenAI 兼容翻译接口的 extra_body 控制
# 留空表示不发送 extra_body，由用户在网页中按需填写 JSON 对象
OPENAI_COMPAT_EXTRA_BODY_JSON = ''

# LLM 并行双发（两次相同请求，取先返回结果）：off 关闭；final_only 仅终译
# （流式时对中间断句不双发）；all 对每个请求都双发。会增加 token 用量
LLM_PARALLEL_FASTEST_MODE = 'off'

# ============================================================================
# 翻译功能配置
# ============================================================================

# 是否启用翻译功能
ENABLE_TRANSLATION = True  # True: 识别后翻译文本
                           # False: 直接发送识别结果，不翻译

# 是否启用流式翻译（翻译部分结果）
# 当 TRANSLATION_API_TYPE 为 'openrouter_streaming' 或
# 'openrouter_streaming_deepl_hybrid' 时自动启用
TRANSLATE_PARTIAL_RESULTS = False

# 触发流式中间翻译所需的最小文本长度（字符数）
# 仅影响中间翻译触发，不影响最终整句翻译
MIN_PARTIAL_TRANSLATION_CHARS = 2

# 混合模式阈值：静音触发终译时，若本句已发送的流式翻译请求次数 <= 此值，优先用 DeepL 终译
# 否则沿用 LLM 终译，降低译文大幅跳变的概率
STREAMING_FINAL_DEEPL_MAX_UPDATES = 1

# 是否为日语译文添加假名标注（仅目标语言为日语时生效）
ENABLE_JA_FURIGANA = False

# 是否为中文添加拼音标注（带声调）
ENABLE_ZH_PINYIN = False

# 是否去除文本句尾句号（仅移除末尾单个 。 / . / ．）
REMOVE_TRAILING_PERIOD = False

# 文本花体风格（fancify-text）
# 可选:
# 'none', 'sansSerif', 'bold', 'italic', 'boldItalic', 'monospaced',
# 'boldSerif', 'italicSerif', 'boldItalicSerif', 'doubleStruck', 'script',
# 'fraktur', 'boldFraktur', 'blue', 'smallCaps', 'curly', 'cool', 'magic'
TEXT_FANCY_STYLE = 'none'

# 发往 VRChat / OSC 的显示文本最大长度。
# 项目内所有与聊天框文本上限相关的裁剪逻辑都应统一使用这个值。
OSC_TEXT_MAX_LENGTH = 144


def is_osc_compat_mode_enabled() -> bool:
    return bool(globals().get('OSC_COMPAT_MODE', False))


def get_effective_osc_text_max_length() -> Optional[int]:
    """兼容模式下取消长度限制；其它模式沿用统一上限。"""
    if is_osc_compat_mode_enabled():
        return None
    try:
        value = int(globals().get('OSC_TEXT_MAX_LENGTH', 144))
    except (TypeError, ValueError):
        value = 144
    return max(1, value)

# 是否启用反向翻译功能
ENABLE_REVERSE_TRANSLATION = True  # True: 翻译后再反向翻译回源语言
                                    # False: 不进行反向翻译

# 是否显示原文及语言标识
# True: 保持当前行为（显示如 [en→ja] 译文 (原文)）
# False: 只显示译文本身（不显示语言标识与原文）
SHOW_ORIGINAL_AND_LANG_TAG = True

# 翻译上下文前缀
CONTEXT_PREFIX = "This is an audio transcription of a conversation within the online multiplayer social game VRChat:"

# 翻译上下文大小（保留多少条历史记录）
TRANSLATION_CONTEXT_SIZE = 6

# 是否启用上下文感知翻译
TRANSLATION_CONTEXT_AWARE = True

# ============================================================================
# 麦克风控制配置
# ============================================================================

# 选择的麦克风输入设备（PyAudio device index）
# None 表示使用系统默认输入设备
MIC_DEVICE_INDEX = None

# 是否考虑游戏内麦克风的开关情况
ENABLE_MIC_CONTROL = True  # True: 根据 VRChat 麦克风状态控制识别的启动/停止
                           # False: 程序启动时立即开始识别,忽略麦克风开关消息

# 收到静音消息后延迟停止识别的秒数
MUTE_DELAY_SECONDS = 0.2  # 设置为 0 则立即停止

# ============================================================================
# 热词配置
# ============================================================================

# 是否启用热词功能
ENABLE_HOT_WORDS = True

# 热词文件路径
HOT_WORDS_DIR = 'hot_words'
HOT_WORDS_PRIVATE_DIR = 'hot_words_private'

# ============================================================================
# VAD 配置（仅 Qwen 后端）
# ============================================================================

# 是否启用服务器端VAD（语音活动检测）
ENABLE_VAD = True  # True: 启用VAD，服务器自动检测语音结束并断句
                   # False: 禁用VAD，需要手动调用commit()来触发断句
                   # 注意：VAD和手动commit不能同时使用
                   # - 启用VAD时，pause()会发送静音音频触发断句，而不是调用commit()
                   # - 禁用VAD时，pause()会调用commit()手动断句

# VAD阈值（0.0-1.0），值越小越敏感
VAD_THRESHOLD = 0.2

# VAD静音持续时间（毫秒），检测到此时长的静音后触发断句
VAD_SILENCE_DURATION_MS = 1200

# ============================================================================
# WebSocket 保活配置（仅 Qwen 后端）
# ============================================================================

# WebSocket心跳间隔（秒），防止长时间闲置导致连接超时
KEEPALIVE_INTERVAL = 30  # 设置为0则禁用心跳功能
                         # 建议值：30-60秒，根据服务器超时设置调整

# ============================================================================
# 显示配置
# ============================================================================

# 小面板默认宽度（像素）
PANEL_WIDTH = max(300, int(os.getenv('PANEL_WIDTH', '600') or 600))

# 是否显示识别中的部分结果（ongoing）
SHOW_PARTIAL_RESULTS = False  # True: 显示部分识别结果到聊天框（可能覆盖掉之前的翻译结果）
                               # False: 只显示完整识别结果

# ============================================================================
# 语言检测器配置
# ============================================================================

# 语言检测器类型
# 可选: 'cjke' (中日韩英), 'enzh' (中英), 'fasttext' (通用)
LANGUAGE_DETECTOR_TYPE = 'cjke'

# ============================================================================
# 日志配置
# ============================================================================

# 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'ERROR'

# ============================================================================
# OSC 配置
# ============================================================================

# OSC 服务器配置
OSC_SERVER_IP = '127.0.0.1'
OSC_SERVER_PORT = 9000

# OSC 客户端配置
OSC_CLIENT_IP = '127.0.0.1'
OSC_CLIENT_PORT = 9001

# 发往 VRChat 的 OSC（如聊天框）使用的目标 UDP 端口，默认与游戏一致为 9000
OSC_SEND_TARGET_PORT = _get_env_int('OSC_SEND_TARGET_PORT', 9000)

# 兼容模式：不使用 OSCQuery，而是在固定端口监听兼容 OSC 的游戏事件。
OSC_COMPAT_MODE = _get_env_bool('OSC_COMPAT_MODE', False)
OSC_COMPAT_LISTEN_PORT = _get_env_int('OSC_COMPAT_LISTEN_PORT', 9001)

# 是否绕过「VRChat OSC 所用 UDP 端口」占用检测（可由网页高级设置或环境变量覆盖）
BYPASS_OSC_UDP_PORT_CHECK = _get_env_bool('BYPASS_OSC_UDP_PORT_CHECK', False)

# 出错时是否仍将错误消息发送到 OSC（小面板始终显示错误，不受此项影响）
OSC_SEND_ERROR_MESSAGES = _get_env_bool('OSC_SEND_ERROR_MESSAGES', False)

# ============================================================================
# 线程池配置
# ============================================================================

# 线程池最大工作线程数
MAX_WORKERS = 8

# ============================================================================
# 反向翻译配置
# ============================================================================

# 是否启用反向翻译验证
ENABLE_BACKWARDS_TRANSLATION = True

# 反向翻译目标语言
BACKWARDS_TRANSLATION_TARGET = 'en'

# ============================================================================
# 模型名称常量（用于热词表创建等）
# ============================================================================

# DashScope 热词表目标模型
DASHSCOPE_HOTWORD_MODEL = 'fun-asr-realtime'

# ============================================================================
# IPC 配置 (Yakutan <-> realtime-subtitle)
# ============================================================================

# 是否启用 IPC 功能
IPC_ENABLED = _get_env_bool('IPC_ENABLED', True)

# IPC 服务器地址
IPC_HOST = os.getenv('IPC_HOST', '127.0.0.1').strip()

# IPC 端口范围
IPC_PORT_RANGE = range(17353, 17364)

import tempfile
import sys
from shared.vrchat_bridge import get_discovery_path
IPC_DISCOVERY_FILE = os.getenv(
    'IPC_DISCOVERY_FILE',
    get_discovery_path()
).strip()

# IPC 发现超时时间（秒）
IPC_DISCOVERY_TIMEOUT = 30.0

# IPC 连接超时时间（秒）
IPC_CONNECT_TIMEOUT = 2.0

# IPC 最大重连延迟（秒）
IPC_RECONNECT_MAX_DELAY = 30.0

# IPC 轮询间隔（秒，当服务器未启动时）
IPC_POLL_INTERVAL = 3.0
