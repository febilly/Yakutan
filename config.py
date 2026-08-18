"""
配置文件 - 统一管理所有配置项
"""
import os
import time
from typing import Optional
from shared.vrchat_text_limits import (
    VRCHAT_OSC_TEXT_MAX_LENGTH,
    normalize_osc_text_max_length,
)

def _read_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {'', '0', 'false', 'no', 'off'}


def _read_env_int(name: str, default: int, *, min_v: int = 1, max_v: int = 65535) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        v = int(str(raw).strip(), 10)
        return max(min_v, min(max_v, v))
    except (TypeError, ValueError):
        return default


def _read_first_env(*names: str, default: str = '') -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default

# ============================================================================
# 语音识别后端配置
# ============================================================================

# 是否使用国际版端点（阿里云 DashScope）
# 国际版用户需要设置为 True
USE_INTERNATIONAL_ENDPOINT = False

# 首选的语音识别后端
PREFERRED_ASR_BACKEND = 'qwen'  # 可选: 'dashscope', 'qwen', 'qwen_audio3', 'soniox', 'doubao_file', 'local'
                                # 注意: 'dashscope' (Fun-ASR) 仅支持中国大陆版

# 有效的后端列表
VALID_ASR_BACKENDS = {'dashscope', 'qwen', 'qwen_audio3', 'soniox', 'doubao_file', 'local'}

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

# Qwen-Audio-3.0 后端使用的模型（与 Fun-ASR 共用 Recognition run-task 协议）
QWEN_AUDIO3_ASR_MODEL = 'qwen-audio-3.0-asr-flash-streaming'

# Recognition（run-task）协议的 WebSocket URL；国际版需要切到 dashscope-intl
DASHSCOPE_RECOGNITION_URL = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'
DASHSCOPE_RECOGNITION_URL_INTERNATIONAL = 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference'

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
# qwen3-asr：GGUF 解码跟随"运行位置"（GPU→走 Vulkan；CPU→CPU）；ONNX 音频编码固定在 CPU。
# 约需显存视配置而定
LOCAL_ASR_ENGINE = 'sensevoice'
_VALID_LOCAL_ASR_ENGINES = frozenset({'sensevoice', 'qwen3-asr'})
if LOCAL_ASR_ENGINE not in _VALID_LOCAL_ASR_ENGINES:
    LOCAL_ASR_ENGINE = 'sensevoice'

# ============================================================================
# 统一 VAD 配置（同时控制在线 API 与本地 ASR）
# ============================================================================
# 一个总开关 + 一组参数，既用于在线 API 后端的发送门控（静音时不向 ASR
# 发送音频以省流），也用于本地 ASR 的分段断句。
VAD_ENABLED = True
LOCAL_VAD_MODE = 'silero'  # 可选: 'silero', 'energy'；在线门控固定使用 Silero
LOCAL_VAD_THRESHOLD = 0.50
LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
# 单段口语送入 VAD 的最长时长（秒）；超过后对本段仅送入静音块直至 VAD 静音或闭麦结束本段（不按时长强制切句）
LOCAL_VAD_MAX_SPEECH_DURATION = 30.0
LOCAL_VAD_SILENCE_DURATION = 0.8
# 起声时拼接的预缓冲音频时长（秒），用于避免漏掉第一个字
LOCAL_VAD_PRE_SPEECH_DURATION = 0.2

# 本地识别的运行位置：'auto'（自动挑一张 GPU，独显优先）、'cpu' 或 'vulkan:N'。
# 只对 Qwen3-ASR 的 GGUF 解码器生效——SenseVoice 是 INT8 ONNX，固定跑 CPU。
LOCAL_ASR_DEVICE = 'auto'

# 本地增量识别（中间结果）
# 触发方式基于 VAD：说话中检测到短停顿（如逗号/分句位置）立即做一次全量本地识别，
# 并配有限流（最小间隔）与保底（最长间隔）机制；不再使用固定间隔轮询。
LOCAL_INCREMENTAL_ASR = True
# 短停顿触发：说话中出现该时长（毫秒）的连续静音时，视为一个分句位置，
# 立即对当前累积音频做一次完整本地识别并产出中间结果。
LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS = 10
# 限流：两次增量更新之间的最小间隔（秒），短停顿触发至少间隔该时长一次。
LOCAL_INCREMENTAL_MIN_UPDATE_INTERVAL = 3.0
# 保底：连续该时长（秒）没有任何增量更新时，强制刷新一次中间结果。
LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL = 4.0

# Qwen3-ASR：GGUF 解码器 KV 上下文长度（token）；增大占显存/内存。
LOCAL_QWEN_ASR_N_CTX = 2048
# 传入 LLM system 区的背景/滚动文本：按模型分词后最多保留的 token 数（取尾部）。
LOCAL_QWEN_CONTEXT_MAX_TOKENS = 1024
# 是否在每条识别后打印 Qwen3-ASR 各阶段耗时（ONNX 编码 / LLM prefill / 生成），使用 INFO 级别。旧版 CLI 可在 .env 中用 LOCAL_QWEN_LOG_PIPELINE_TIMING=0 关闭。
# 需在 config.LOG_LEVEL 为 INFO/DEBUG 时才能在终端看到（默认 ERROR 时不会输出）。
LOCAL_QWEN_LOG_PIPELINE_TIMING = True

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
SAVE_POST_RESAMPLE_AUDIO = False

# 是否将重采样前的原始采集音频保存到本地 WAV（调试用）
SAVE_PRE_RESAMPLE_AUDIO = False

# 调试音频输出目录（相对路径时相对于项目根目录）
DEBUG_AUDIO_OUTPUT_DIR = 'debug_audio'

# 采集侧 VAD 调试日志。仅旧版 CLI 会通过 ``apply_cli_env`` 从 .env 覆盖。
ENABLE_VAD_GATING_VERBOSE = False

# ============================================================================
# 翻译语言配置
# ============================================================================

SOURCE_LANGUAGE = 'auto'  # 翻译源语言（'auto' 为自动检测，或指定如 'en', 'ja' 等）
TARGET_LANGUAGE = 'ja'  # 翻译目标语言（'zh-CN'=简体中文, 'en'=英文, 'ja'=日文 等）
SECONDARY_TARGET_LANGUAGE = None  # 第二输出语言（可选，启用后将并行输出两种译文）
FALLBACK_LANGUAGE = 'en'  # 备用翻译语言（当源语言和目标语言相同时使用）
                           # 设置为 None（非字符串）则禁用备用语言功能

# 智能目标语言（根据最近别人说的话自动推断翻译目标语言）
SMART_TARGET_PRIMARY_ENABLED = False
SMART_TARGET_SECONDARY_ENABLED = False
SMART_TARGET_LANGUAGE_STRATEGY = "most_common"  # 可选: most_common, latest, weighted
SMART_TARGET_LANGUAGE_WINDOW_SIZE = 5
SMART_TARGET_LANGUAGE_EXCLUDE_SELF_LANGUAGE = True
SMART_TARGET_LANGUAGE_FALLBACK = "en"
SMART_TARGET_LANGUAGE_MIN_SAMPLES = 3

# 废弃/不推荐直接使用的变量（仅为向后兼容保留，后续逻辑应迁移到 PRIMARY_ENABLED / SECONDARY_ENABLED）
SMART_TARGET_LANGUAGE_ENABLED = SMART_TARGET_PRIMARY_ENABLED or SMART_TARGET_SECONDARY_ENABLED
SMART_TARGET_LANGUAGE_COUNT = 2 if SMART_TARGET_SECONDARY_ENABLED else 1
SMART_TARGET_LANGUAGE_MANUAL_SECONDARY = None

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
#      'openrouter_streaming', 'openrouter_streaming_deepl_hybrid', 'qwen_mt',
#      'hymt2'（Hy-MT2 WebSocket 流式修订翻译，需自行填写 HYMT2_WEBSOCKET_URL）
# 注意:
# - openrouter / openrouter_streaming 表示基于 OpenAI 兼容接口的 LLM 翻译
# - openrouter_streaming 是 LLM 翻译的流式模式，支持翻译部分结果
# - openrouter_streaming_deepl_hybrid 在静音触发终译时，按流式更新次数阈值决定
#   使用 DeepL（更新次数较少）或 LLM（更新次数较多）进行最终翻译
DEFAULT_TRANSLATION_API_TYPE = 'openrouter_streaming'
DEFAULT_LLM_TEMPLATE = 'deepseek-v4-flash'
DEFAULT_LLM_BASE_URL = 'https://api.deepseek.com'
DEFAULT_LLM_MODEL = 'deepseek-v4-flash'
DEFAULT_LLM_EXTRA_BODY_JSON = '{"thinking": {"type": "disabled"}}'
DEFAULT_LLM_TRANSLATION_FORMALITY = 'medium'
DEFAULT_LLM_TRANSLATION_STYLE = 'standard'

TRANSLATION_API_TYPE = DEFAULT_TRANSLATION_API_TYPE

# LLM（OpenAI 兼容接口）配置
LLM_BASE_URL = DEFAULT_LLM_BASE_URL
LLM_MODEL = DEFAULT_LLM_MODEL
LLM_TEMPLATE = DEFAULT_LLM_TEMPLATE
LLM_TRANSLATION_TEMPERATURE = 0.2
LLM_TRANSLATION_TIMEOUT = 30
LLM_TRANSLATION_MAX_RETRIES = 3

# LLM 翻译正式程度
# 可选: 'low', 'medium', 'high'
# 默认保持接近当前偏口语、轻礼貌的风格
LLM_TRANSLATION_FORMALITY = DEFAULT_LLM_TRANSLATION_FORMALITY

# LLM 句子风格
# 可选: 'standard', 'light'
LLM_TRANSLATION_STYLE = DEFAULT_LLM_TRANSLATION_STYLE

# OpenAI 兼容翻译接口的 extra_body 控制
# 留空表示不发送 extra_body，由用户在网页中按需填写 JSON 对象
OPENAI_COMPAT_EXTRA_BODY_JSON = DEFAULT_LLM_EXTRA_BODY_JSON

# LLM 并行双发（两次相同请求，取先返回结果）：off 关闭；final_only 仅终译
# （流式时对中间断句不双发）；all 对每个请求都双发。会增加 token 用量
LLM_PARALLEL_FASTEST_MODE = 'off'

# ============================================================================
# Hy-MT2 流式修订翻译配置（WebSocket 无状态协议，见 Hy-MT2 INTEGRATION.md）
# ============================================================================

# Hy-MT2 接入方式：'api' (外部 WebSocket 服务) 或 'local' (本地 GGUF 模型)
HYMT2_BACKEND = 'api'

# 本地 GGUF 推理的运行位置：'auto'（自动挑一张 GPU，独显优先）、'cpu'（纯 CPU）
# 或 'vulkan:N'（指定第 N 张 GPU）。仅在 HYMT2_BACKEND = 'local' 时有意义。
HYMT2_LOCAL_DEVICE = 'auto'

# Hy-MT2 服务的 WebSocket 地址。该地址**不内置默认值**——服务属于用户自托管/
# 内网部署，必须由用户自行填写（网页「翻译API设置」或 .env 的 HYMT2_WEBSOCKET_URL）。
# 例如：ws://127.0.0.1:18765
HYMT2_WEBSOCKET_URL = ''

# 单次请求/建连超时（秒）
HYMT2_TIMEOUT_SECONDS = 30

# 建连失败重试次数
HYMT2_MAX_RETRIES = 3

def sanitize_local_device(value) -> str:
    """把本地推理设备设置收敛成 'auto' / 'cpu' / 'vulkan:N'。

    实现在 local_asr.gpu_devices；这里做一层薄封装，好让 config 的调用方
    （网页后端等）不必关心 local_asr 是否可用（精简构建里可能被裁掉）。
    """
    try:
        from local_asr.gpu_devices import sanitize_device
    except Exception:
        normalized = str(value or '').strip().lower()
        if normalized == 'cpu':
            return 'cpu'
        if normalized.startswith('vulkan:') and normalized[7:].isdigit():
            return normalized
        return 'auto'
    return sanitize_device(value)

# 运行期凭据。WebUI 只会通过页面请求更新这些进程内字段，不会读取或写入
# os.environ；旧版 CLI 则由 ``apply_cli_env`` 从 .env 显式填充。
DASHSCOPE_API_KEY = ''
DEEPL_API_KEY = ''
LLM_API_KEY = ''
OPENAI_API_KEY = ''
SONIOX_API_KEY = ''
DOUBAO_API_KEY = ''
DOUBAO_APP_ID = ''
DOUBAO_ACCESS_KEY = ''
LLM_APP_URL = ''
LLM_APP_TITLE = ''

# ============================================================================
# 翻译功能配置
# ============================================================================

# 是否启用翻译功能
ENABLE_TRANSLATION = True  # True: 识别后翻译文本
                           # False: 直接发送识别结果，不翻译

# 是否启用流式翻译（翻译部分结果）
# 当 TRANSLATION_API_TYPE 为 'openrouter_streaming'、'openrouter_streaming_deepl_hybrid'
# 或 'hymt2'（且开启流式开关）时启用
TRANSLATE_PARTIAL_RESULTS = True

# 网页「流式翻译模式」开关的**各模型独立偏好**。TRANSLATE_PARTIAL_RESULTS 只记录当前
# 生效模型的状态，切换模型后另一个模型的偏好需要单独记住，否则重启/对账后会被重置。
# 仅供 WebUI 往返保存，识别与翻译流程不读取这两个值。
LLM_STREAMING_PREF = True
HYMT2_STREAMING_PREF = True

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

# 是否对阿拉伯文/希伯来文做 OSC 显示重排，便于在不支持 RTL 正确渲染的游戏环境显示
ENABLE_ARABIC_RESHAPER = True

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
OSC_TEXT_MAX_LENGTH = VRCHAT_OSC_TEXT_MAX_LENGTH


def is_osc_compat_mode_enabled() -> bool:
    return bool(globals().get('OSC_COMPAT_MODE', False))


def get_effective_osc_text_max_length() -> Optional[int]:
    """兼容模式下取消长度限制；其它模式沿用统一上限。"""
    if is_osc_compat_mode_enabled():
        return None
    return normalize_osc_text_max_length(
        globals().get('OSC_TEXT_MAX_LENGTH', VRCHAT_OSC_TEXT_MAX_LENGTH)
    )

# 是否启用反向翻译功能
ENABLE_REVERSE_TRANSLATION = False  # True: 翻译后再反向翻译回源语言
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

TERMINOLOGY_ENABLED = True

# ============================================================================
# 麦克风控制配置
# ============================================================================

# 选择的麦克风输入设备（PyAudio device index）
# None 表示使用系统默认输入设备
MIC_DEVICE_INDEX = None

# 是否考虑游戏内麦克风的开关情况
ENABLE_MIC_CONTROL = False  # True: 根据 VRChat 麦克风状态控制识别的启动/停止
                           # False: 程序启动时立即开始识别,忽略麦克风开关消息

# 收到静音消息后延迟停止识别的秒数
MUTE_DELAY_SECONDS = 0.2  # 设置为 0 则立即停止

# 快速开关麦克风以清空消息框：在收到静音消息后的该时间窗口内再次收到静音消息，
# 则向 OSC 发送空字符串以清空 VRChat 聊天框
ENABLE_DOUBLE_MUTE_CLEAR = True
DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
# 触发清空后，丢弃此时间窗口内迟到返回的识别/翻译结果，避免把已撤回的内容重新发出
# （识别重新开始/恢复会立即解除丢弃，此处仅作为非麦克风控制模式下的安全上限）
DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0

# ============================================================================
# 热词配置
# ============================================================================

# 是否启用热词功能
ENABLE_HOT_WORDS = True

# 热词文件路径
HOT_WORDS_DIR = 'hot_words'
HOT_WORDS_PRIVATE_DIR = 'hot_words_private'

# ============================================================================
# 服务端 VAD 配置（仅 Qwen 后端）
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
VAD_SILENCE_DURATION_MS = 800

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
PANEL_WIDTH = 600

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
OSC_SEND_TARGET_PORT = 9000

# 兼容模式：不使用 OSCQuery，而是在固定端口监听兼容 OSC 的游戏事件。
OSC_COMPAT_MODE = False
OSC_COMPAT_LISTEN_PORT = 9001

# 是否绕过「VRChat OSC 所用 UDP 端口」占用检测（可由网页高级设置覆盖；
# 旧版 CLI 也可在 .env 中设置）
BYPASS_OSC_UDP_PORT_CHECK = False

# 出错时是否仍将错误消息发送到 OSC（小面板始终显示错误，不受此项影响）
OSC_SEND_ERROR_MESSAGES = False

# OSCQuery 运行配置。WebUI 使用这里的默认值，CLI 可由 .env 覆盖。
OSC_QUERY_ENABLED = True
OSCQUERY_APP_NAME = 'DeafaultAppName'

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
IPC_ENABLED = True

# IPC 服务器地址
IPC_HOST = '127.0.0.1'

# IPC 端口范围
IPC_PORT_RANGE = range(17353, 17364)

from shared.vrchat_bridge import get_discovery_path
IPC_DISCOVERY_FILE = get_discovery_path()

# IPC 发现超时时间（秒）
IPC_DISCOVERY_TIMEOUT = 30.0

# IPC 连接超时时间（秒）
IPC_CONNECT_TIMEOUT = 2.0

# IPC 最大重连延迟（秒）
IPC_RECONNECT_MAX_DELAY = 30.0

# IPC 轮询间隔（秒，当服务器未启动时）
IPC_POLL_INTERVAL = 3.0


def apply_cli_env() -> None:
    """Apply legacy CLI settings from the already-loaded process environment.

    ``main.py`` is the only application entry point that calls this function.
    WebUI imports ``config`` directly and therefore always starts from the
    built-in defaults until the browser submits its saved settings.
    """
    global VAD_ENABLED
    global LOCAL_QWEN_LOG_PIPELINE_TIMING
    global LOCAL_ASR_DEVICE
    global SAVE_POST_RESAMPLE_AUDIO, SAVE_PRE_RESAMPLE_AUDIO
    global DEBUG_AUDIO_OUTPUT_DIR, ENABLE_VAD_GATING_VERBOSE
    global TRANSLATION_API_TYPE, LLM_BASE_URL, LLM_MODEL, LLM_TEMPLATE
    global LLM_TRANSLATION_FORMALITY, LLM_TRANSLATION_STYLE
    global OPENAI_COMPAT_EXTRA_BODY_JSON, TRANSLATE_PARTIAL_RESULTS
    global HYMT2_BACKEND, HYMT2_LOCAL_DEVICE
    global HYMT2_WEBSOCKET_URL, HYMT2_TIMEOUT_SECONDS, HYMT2_MAX_RETRIES
    global DASHSCOPE_API_KEY, DEEPL_API_KEY, LLM_API_KEY, OPENAI_API_KEY
    global SONIOX_API_KEY, DOUBAO_API_KEY, DOUBAO_APP_ID, DOUBAO_ACCESS_KEY
    global LLM_APP_URL, LLM_APP_TITLE
    global PANEL_WIDTH
    global OSC_SEND_TARGET_PORT, OSC_COMPAT_MODE, OSC_COMPAT_LISTEN_PORT
    global BYPASS_OSC_UDP_PORT_CHECK, OSC_SEND_ERROR_MESSAGES
    global OSC_QUERY_ENABLED, OSCQUERY_APP_NAME
    global IPC_ENABLED, IPC_HOST, IPC_DISCOVERY_FILE

    VAD_ENABLED = _read_env_bool('VAD_ENABLED', VAD_ENABLED)
    LOCAL_QWEN_LOG_PIPELINE_TIMING = _read_env_bool(
        'LOCAL_QWEN_LOG_PIPELINE_TIMING', LOCAL_QWEN_LOG_PIPELINE_TIMING
    )
    LOCAL_ASR_DEVICE = sanitize_local_device(
        _read_first_env('LOCAL_ASR_DEVICE', default=LOCAL_ASR_DEVICE)
    )
    SAVE_POST_RESAMPLE_AUDIO = _read_env_bool(
        'SAVE_POST_RESAMPLE_AUDIO', SAVE_POST_RESAMPLE_AUDIO
    )
    SAVE_PRE_RESAMPLE_AUDIO = _read_env_bool(
        'SAVE_PRE_RESAMPLE_AUDIO', SAVE_PRE_RESAMPLE_AUDIO
    )
    DEBUG_AUDIO_OUTPUT_DIR = _read_first_env(
        'DEBUG_AUDIO_OUTPUT_DIR', default=DEBUG_AUDIO_OUTPUT_DIR
    )
    ENABLE_VAD_GATING_VERBOSE = _read_env_bool(
        'ENABLE_VAD_GATING_VERBOSE',
        _read_env_bool('ENABLE_LOCAL_VAD_GATING_VERBOSE', ENABLE_VAD_GATING_VERBOSE),
    )

    TRANSLATION_API_TYPE = _read_first_env(
        'TRANSLATION_API_TYPE', default=TRANSLATION_API_TYPE
    )
    LLM_BASE_URL = _read_first_env(
        'LLM_BASE_URL',
        'OPENAI_BASE_URL',
        'OPENROUTER_BASE_URL',
        default=LLM_BASE_URL,
    )
    LLM_MODEL = _read_first_env(
        'LLM_MODEL',
        'OPENAI_MODEL',
        'OPENROUTER_TRANSLATION_MODEL',
        default=LLM_MODEL,
    )
    LLM_TEMPLATE = _read_first_env('LLM_TEMPLATE', default=LLM_TEMPLATE)
    LLM_TRANSLATION_FORMALITY = _read_first_env(
        'LLM_TRANSLATION_FORMALITY', default=LLM_TRANSLATION_FORMALITY
    ).lower()
    LLM_TRANSLATION_STYLE = _read_first_env(
        'LLM_TRANSLATION_STYLE', default=LLM_TRANSLATION_STYLE
    ).lower()
    OPENAI_COMPAT_EXTRA_BODY_JSON = os.getenv(
        'OPENAI_COMPAT_EXTRA_BODY_JSON', OPENAI_COMPAT_EXTRA_BODY_JSON
    ).strip()
    TRANSLATE_PARTIAL_RESULTS = _read_env_bool(
        'TRANSLATE_PARTIAL_RESULTS',
        TRANSLATION_API_TYPE in (
            'openrouter_streaming',
            'openrouter_streaming_deepl_hybrid',
            'hymt2',
        ),
    )

    HYMT2_BACKEND = _read_first_env(
        'HYMT2_BACKEND', default=HYMT2_BACKEND
    )
    HYMT2_LOCAL_DEVICE = sanitize_local_device(
        _read_first_env('HYMT2_LOCAL_DEVICE', default=HYMT2_LOCAL_DEVICE)
    )
    HYMT2_WEBSOCKET_URL = _read_first_env(
        'HYMT2_WEBSOCKET_URL', default=HYMT2_WEBSOCKET_URL
    )
    HYMT2_TIMEOUT_SECONDS = max(
        1, _read_env_int('HYMT2_TIMEOUT_SECONDS', HYMT2_TIMEOUT_SECONDS, max_v=600)
    )
    HYMT2_MAX_RETRIES = max(
        0, _read_env_int('HYMT2_MAX_RETRIES', HYMT2_MAX_RETRIES, max_v=20)
    )

    DASHSCOPE_API_KEY = _read_first_env('DASHSCOPE_API_KEY')
    DEEPL_API_KEY = _read_first_env('DEEPL_API_KEY')
    LLM_API_KEY = _read_first_env('LLM_API_KEY', 'OPENROUTER_API_KEY')
    OPENAI_API_KEY = _read_first_env('OPENAI_API_KEY')
    SONIOX_API_KEY = _read_first_env('SONIOX_API_KEY')
    DOUBAO_API_KEY = _read_first_env('DOUBAO_API_KEY')
    DOUBAO_APP_ID = _read_first_env('DOUBAO_APP_ID')
    DOUBAO_ACCESS_KEY = _read_first_env('DOUBAO_ACCESS_KEY')
    LLM_APP_URL = _read_first_env('LLM_APP_URL', 'OPENROUTER_APP_URL')
    LLM_APP_TITLE = _read_first_env('LLM_APP_TITLE', 'OPENROUTER_APP_TITLE')

    PANEL_WIDTH = max(300, _read_env_int('PANEL_WIDTH', PANEL_WIDTH, max_v=10000))
    OSC_SEND_TARGET_PORT = _read_env_int(
        'OSC_SEND_TARGET_PORT', OSC_SEND_TARGET_PORT
    )
    OSC_COMPAT_MODE = _read_env_bool('OSC_COMPAT_MODE', OSC_COMPAT_MODE)
    OSC_COMPAT_LISTEN_PORT = _read_env_int(
        'OSC_COMPAT_LISTEN_PORT', OSC_COMPAT_LISTEN_PORT
    )
    BYPASS_OSC_UDP_PORT_CHECK = _read_env_bool(
        'BYPASS_OSC_UDP_PORT_CHECK', BYPASS_OSC_UDP_PORT_CHECK
    )
    OSC_SEND_ERROR_MESSAGES = _read_env_bool(
        'OSC_SEND_ERROR_MESSAGES', OSC_SEND_ERROR_MESSAGES
    )
    OSC_QUERY_ENABLED = _read_env_bool('OSC_QUERY_ENABLED', OSC_QUERY_ENABLED)
    OSCQUERY_APP_NAME = _read_first_env(
        'OSCQUERY_APP_NAME', default=OSCQUERY_APP_NAME
    )

    IPC_ENABLED = _read_env_bool('IPC_ENABLED', IPC_ENABLED)
    IPC_HOST = _read_first_env('IPC_HOST', default=IPC_HOST)
    IPC_DISCOVERY_FILE = _read_first_env(
        'IPC_DISCOVERY_FILE', default=IPC_DISCOVERY_FILE
    )


def get_default_ui_config() -> dict:
    """Return a fresh copy of the browser-managed configuration defaults."""
    return {
        'asr': {
            'preferred_backend': 'qwen',
            'keepalive_interval': 30,
            'enable_hot_words': True,
            'use_international_endpoint': False,
        },
        'vad': {
            'enabled': True,
            'mode': 'silero',
            'threshold': 0.50,
            'min_speech_duration': 1.0,
            'max_speech_duration': 30.0,
            'silence_duration': 0.8,
            'pre_speech_duration': 0.2,
        },
        'translation': {
            'enable_translation': True,
            'source_language': 'auto',
            'target_language': 'ja',
            'secondary_target_language': None,
            'fallback_language': 'en',
            'api_type': DEFAULT_TRANSLATION_API_TYPE,
            'translate_partial_results': True,
            'llm_template': DEFAULT_LLM_TEMPLATE,
            'llm_base_url': DEFAULT_LLM_BASE_URL,
            'llm_model': DEFAULT_LLM_MODEL,
            'llm_translation_formality': DEFAULT_LLM_TRANSLATION_FORMALITY,
            'llm_translation_style': DEFAULT_LLM_TRANSLATION_STYLE,
            'openai_compat_extra_body_json': DEFAULT_LLM_EXTRA_BODY_JSON,
            'llm_parallel_fastest_mode': 'off',
            'hymt2_backend': 'api',
            'hymt2_websocket_url': '',
            'show_partial_results': False,
            'enable_furigana': False,
            'enable_pinyin': False,
            'enable_arabic_reshaper': True,
            'remove_trailing_period': False,
            'text_fancy_style': 'none',
            'enable_reverse_translation': False,
            'show_original_and_lang_tag': True,
        },
        'mic_control': {
            'enable_mic_control': False,
            'mute_delay_seconds': 0.2,
            'mic_device_index': None,
            'enable_double_mute_clear': True,
        },
        'language_detector': {
            'type': 'cjke',
        },
        'smart_target_language': {
            'primary_enabled': False,
            'secondary_enabled': False,
            'strategy': 'most_common',
            'window_size': 5,
            'exclude_self_language': True,
            'min_samples': 3,
        },
        'panel': {
            'width': 600,
        },
        'osc': {
            'send_target_port': 9000,
            'compat_mode': False,
            'compat_listen_port': 9001,
            'bypass_udp_port_check': False,
            'send_error_messages': False,
        },
    }
