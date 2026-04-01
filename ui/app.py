"""
Web UI for VRChat Translator
提供配置管理和服务控制的Web界面
"""
import asyncio
import json
import threading
import logging
from typing import Optional
from urllib.parse import urlencode
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os

# 添加父目录到路径以导入config和main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
try:
    from local_asr import (
        LOCAL_ASR_DISPLAY_NAMES,
        LOCAL_ASR_ENGINES,
        get_local_asr_features,
        is_local_asr_build_enabled,
        is_local_asr_ui_enabled,
    )
    from local_asr.model_manager import download_asr as download_local_asr_model
    from local_asr.model_manager import download_silero, get_engine_status, is_silero_cached
except ImportError:  # pragma: no cover
    LOCAL_ASR_ENGINES = ()
    LOCAL_ASR_DISPLAY_NAMES = {}

    def get_local_asr_features():
        return {
            'local_asr_build_enabled': False,
            'local_asr_ui_enabled': False,
            'engines': {},
        }

    def is_local_asr_build_enabled():
        return False

    def is_local_asr_ui_enabled():
        return False

    def download_silero():
        raise RuntimeError('Local ASR unavailable')

    def is_silero_cached():
        return False

    def download_local_asr_model(*args, **kwargs):
        raise RuntimeError('Local ASR unavailable')

    def get_engine_status(*args, **kwargs):
        raise RuntimeError('Local ASR unavailable')
from resource_path import get_resource_path

# 配置Flask使用正确的模板和静态文件路径
template_folder = get_resource_path('ui/templates')
static_folder = get_resource_path('ui/static')

app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)
CORS(app)

# 禁用Flask的请求日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 全局状态
service_status = {
    'running': False,
    'recognition_active': False,
    'backend': config.PREFERRED_ASR_BACKEND
}

service_thread: Optional[threading.Thread] = None
service_loop: Optional[asyncio.AbstractEventLoop] = None
stop_event: Optional[asyncio.Event] = None
local_asr_download_state = {
    'running': False,
    'engine': None,
    'status': '',
    'error': None,
}
local_asr_download_lock = threading.Lock()


def set_or_clear_env_var(name: str, value: Optional[str]) -> None:
    """设置环境变量；空值时删除，确保运行中服务读取到最新密钥。"""
    normalized = (value or '').strip()
    if normalized:
        os.environ[name] = normalized
    else:
        os.environ.pop(name, None)


def _get_feature_flags() -> dict:
    return get_local_asr_features()


def _local_asr_config_dict() -> dict:
    return {
        'engine': getattr(config, 'LOCAL_ASR_ENGINE', 'sensevoice'),
        'vad_mode': getattr(config, 'LOCAL_VAD_MODE', 'silero'),
        'vad_threshold': getattr(config, 'LOCAL_VAD_THRESHOLD', 0.50),
        'min_speech_duration': getattr(config, 'LOCAL_VAD_MIN_SPEECH_DURATION', 1.0),
        'max_speech_duration': getattr(config, 'LOCAL_VAD_MAX_SPEECH_DURATION', 30.0),
        'silence_duration': getattr(config, 'LOCAL_VAD_SILENCE_DURATION', 0.8),
        'incremental_asr': getattr(config, 'LOCAL_INCREMENTAL_ASR', True),
        'interim_interval': getattr(config, 'LOCAL_INTERIM_INTERVAL', 2.0),
    }


def _sanitize_preferred_backend(value: Optional[str]) -> str:
    backend = (value or config.PREFERRED_ASR_BACKEND or 'qwen').strip() or 'qwen'
    if backend == 'local' and not is_local_asr_ui_enabled():
        return 'qwen'
    return backend


def _update_local_asr_download_state(**changes) -> None:
    with local_asr_download_lock:
        local_asr_download_state.update(changes)


def _snapshot_local_asr_download_state() -> dict:
    with local_asr_download_lock:
        return dict(local_asr_download_state)


def _download_local_asr_worker(engine: str) -> None:
    _update_local_asr_download_state(
        running=True,
        engine=engine,
        status=f'准备下载 {LOCAL_ASR_DISPLAY_NAMES.get(engine, engine)}',
        error=None,
    )
    try:
        if not is_silero_cached():
            _update_local_asr_download_state(status='下载 Silero VAD...')
            download_silero()
        _update_local_asr_download_state(
            status=f'下载 {LOCAL_ASR_DISPLAY_NAMES.get(engine, engine)} 模型与运行时...',
        )
        download_local_asr_model(engine)
        _update_local_asr_download_state(
            running=False,
            status='下载完成',
            error=None,
        )
    except Exception as e:
        _update_local_asr_download_state(
            running=False,
            status='下载失败',
            error=str(e),
        )


def get_config_dict():
    """获取当前配置"""
    return {
        'features': _get_feature_flags(),
        # 语音识别配置
        'asr': {
            'preferred_backend': _sanitize_preferred_backend(config.PREFERRED_ASR_BACKEND),
            'enable_vad': config.ENABLE_VAD,
            'vad_threshold': config.VAD_THRESHOLD,
            'vad_silence_duration_ms': config.VAD_SILENCE_DURATION_MS,
            'keepalive_interval': config.KEEPALIVE_INTERVAL,
            'enable_hot_words': config.ENABLE_HOT_WORDS,
            'use_international_endpoint': config.USE_INTERNATIONAL_ENDPOINT,
        },
        # 翻译配置
        'translation': {
            'enable_translation': config.ENABLE_TRANSLATION,
            'source_language': config.SOURCE_LANGUAGE,
            'target_language': config.TARGET_LANGUAGE,
            'secondary_target_language': getattr(config, 'SECONDARY_TARGET_LANGUAGE', None),
            'fallback_language': config.FALLBACK_LANGUAGE,
            'api_type': config.TRANSLATION_API_TYPE,
            'llm_base_url': getattr(config, 'LLM_BASE_URL', ''),
            'llm_model': getattr(config, 'LLM_MODEL', ''),
            'openai_compat_extra_body_json': getattr(config, 'OPENAI_COMPAT_EXTRA_BODY_JSON', ''),
            'llm_parallel_fastest_mode': getattr(
                config, 'LLM_PARALLEL_FASTEST_MODE', 'off'
            ),
            'enable_llm_parallel_fastest': (
                getattr(config, 'LLM_PARALLEL_FASTEST_MODE', 'off')
                not in ('off', None, '')
            ),
            'show_partial_results': config.SHOW_PARTIAL_RESULTS,
            'enable_furigana': getattr(config, 'ENABLE_JA_FURIGANA', False),
            'enable_pinyin': getattr(config, 'ENABLE_ZH_PINYIN', False),
            'enable_reverse_translation': config.ENABLE_REVERSE_TRANSLATION,
            'show_original_and_lang_tag': getattr(config, 'SHOW_ORIGINAL_AND_LANG_TAG', True),
        },
        # 麦克风控制配置
        'mic_control': {
            'enable_mic_control': config.ENABLE_MIC_CONTROL,
            'mute_delay_seconds': config.MUTE_DELAY_SECONDS,
            'mic_device_index': getattr(config, 'MIC_DEVICE_INDEX', None),
        },
        # 语言检测器配置
        'language_detector': {
            'type': config.LANGUAGE_DETECTOR_TYPE,
        },
        'panel': {
            'width': getattr(config, 'PANEL_WIDTH', 600),
        },
        # OSC配置
        'osc': {
            'server_ip': config.OSC_SERVER_IP,
            'server_port': config.OSC_SERVER_PORT,
            'client_ip': config.OSC_CLIENT_IP,
            'client_port': config.OSC_CLIENT_PORT,
        },
        'local_asr': _local_asr_config_dict() if is_local_asr_ui_enabled() else None,
        'config_applied_at_ms': int(getattr(config, 'CONFIG_APPLIED_AT_MS', 0) or 0),
        'backend_boot_ms': int(getattr(config, 'BACKEND_BOOT_MS', 0) or 0),
    }


def update_config(config_data):
    """更新配置"""
    try:
        api_keys = config_data.get('api_keys') or {}
        if 'llm' in api_keys:
            set_or_clear_env_var('LLM_API_KEY', api_keys['llm'])

        # 更新ASR配置
        if 'asr' in config_data:
            asr = config_data['asr']
            if 'preferred_backend' in asr:
                config.PREFERRED_ASR_BACKEND = _sanitize_preferred_backend(asr['preferred_backend'])
            if 'enable_vad' in asr:
                config.ENABLE_VAD = asr['enable_vad']
            if 'vad_threshold' in asr:
                config.VAD_THRESHOLD = float(asr['vad_threshold'])
            if 'vad_silence_duration_ms' in asr:
                config.VAD_SILENCE_DURATION_MS = int(asr['vad_silence_duration_ms'])
            if 'keepalive_interval' in asr:
                config.KEEPALIVE_INTERVAL = int(asr['keepalive_interval'])
            if 'enable_hot_words' in asr:
                config.ENABLE_HOT_WORDS = asr['enable_hot_words']
            if 'use_international_endpoint' in asr:
                config.USE_INTERNATIONAL_ENDPOINT = asr['use_international_endpoint']
        
        # 更新翻译配置
        if 'translation' in config_data:
            trans = config_data['translation']
            if 'enable_translation' in trans:
                config.ENABLE_TRANSLATION = trans['enable_translation']
            if 'source_language' in trans:
                config.SOURCE_LANGUAGE = trans['source_language']
            if 'target_language' in trans:
                config.TARGET_LANGUAGE = trans['target_language']
            if 'secondary_target_language' in trans:
                config.SECONDARY_TARGET_LANGUAGE = trans['secondary_target_language'] if trans['secondary_target_language'] else None
            if 'fallback_language' in trans:
                config.FALLBACK_LANGUAGE = trans['fallback_language'] if trans['fallback_language'] else None
            if 'api_type' in trans:
                config.TRANSLATION_API_TYPE = trans['api_type']
                # 前端的"流式翻译模式"开关会将 api_type 设为 'openrouter_streaming'
                # 此时启用部分结果翻译（实时翻译未完成的句子）
                config.TRANSLATE_PARTIAL_RESULTS = trans['api_type'] in (
                    'openrouter_streaming',
                    'openrouter_streaming_deepl_hybrid',
                )
            if 'llm_base_url' in trans:
                config.LLM_BASE_URL = (trans['llm_base_url'] or '').strip()
            if 'llm_model' in trans:
                config.LLM_MODEL = (trans['llm_model'] or '').strip()
            if 'openai_compat_extra_body_json' in trans:
                raw_extra_body = (trans['openai_compat_extra_body_json'] or '').strip()
                if raw_extra_body:
                    parsed_extra_body = json.loads(raw_extra_body)
                    if not isinstance(parsed_extra_body, dict):
                        return False, 'msg.invalidExtraBodyJson', 'OpenAI 兼容 extra_body 必须是 JSON 对象'
                config.OPENAI_COMPAT_EXTRA_BODY_JSON = raw_extra_body
            if 'llm_parallel_fastest_mode' in trans:
                mode = trans['llm_parallel_fastest_mode']
                if mode not in ('off', 'final_only', 'all'):
                    return (
                        False,
                        'msg.invalidParallelFastestMode',
                        '并行双发模式必须是 off、final_only 或 all',
                    )
                config.LLM_PARALLEL_FASTEST_MODE = mode
            elif 'enable_llm_parallel_fastest' in trans:
                config.LLM_PARALLEL_FASTEST_MODE = (
                    'final_only' if trans['enable_llm_parallel_fastest'] else 'off'
                )
            if 'show_partial_results' in trans:
                config.SHOW_PARTIAL_RESULTS = trans['show_partial_results']
            if 'enable_furigana' in trans:
                config.ENABLE_JA_FURIGANA = trans['enable_furigana']
            if 'enable_pinyin' in trans:
                config.ENABLE_ZH_PINYIN = trans['enable_pinyin']
            if 'enable_reverse_translation' in trans:
                config.ENABLE_REVERSE_TRANSLATION = trans['enable_reverse_translation']
            if 'show_original_and_lang_tag' in trans:
                config.SHOW_ORIGINAL_AND_LANG_TAG = bool(trans['show_original_and_lang_tag'])
        
        # 更新麦克风控制配置
        if 'mic_control' in config_data:
            mic = config_data['mic_control']
            if 'enable_mic_control' in mic:
                config.ENABLE_MIC_CONTROL = mic['enable_mic_control']
            if 'mute_delay_seconds' in mic:
                config.MUTE_DELAY_SECONDS = float(mic['mute_delay_seconds'])
            if 'mic_device_index' in mic:
                value = mic['mic_device_index']
                if value is None or value == '':
                    config.MIC_DEVICE_INDEX = None
                else:
                    config.MIC_DEVICE_INDEX = int(value)
        
        # 更新语言检测器配置
        if 'language_detector' in config_data:
            ld = config_data['language_detector']
            if 'type' in ld:
                config.LANGUAGE_DETECTOR_TYPE = ld['type']

        if 'panel' in config_data:
            panel = config_data['panel']
            if 'width' in panel:
                config.PANEL_WIDTH = max(300, int(panel['width']))

        if is_local_asr_ui_enabled() and 'local_asr' in config_data and config_data['local_asr']:
            local_asr = config_data['local_asr']
            if 'engine' in local_asr:
                _eng = str(local_asr['engine'] or 'sensevoice')
                if _eng not in LOCAL_ASR_ENGINES:
                    _eng = 'sensevoice'
                config.LOCAL_ASR_ENGINE = _eng
            if 'vad_mode' in local_asr:
                config.LOCAL_VAD_MODE = str(local_asr['vad_mode'] or 'silero')
            if 'vad_threshold' in local_asr:
                config.LOCAL_VAD_THRESHOLD = float(local_asr['vad_threshold'])
            if 'min_speech_duration' in local_asr:
                config.LOCAL_VAD_MIN_SPEECH_DURATION = float(local_asr['min_speech_duration'])
            if 'max_speech_duration' in local_asr:
                config.LOCAL_VAD_MAX_SPEECH_DURATION = float(local_asr['max_speech_duration'])
            if 'silence_duration' in local_asr:
                config.LOCAL_VAD_SILENCE_DURATION = float(local_asr['silence_duration'])
            if 'incremental_asr' in local_asr:
                config.LOCAL_INCREMENTAL_ASR = bool(local_asr['incremental_asr'])
            if 'interim_interval' in local_asr:
                config.LOCAL_INTERIM_INTERVAL = float(local_asr['interim_interval'])
        
        config.bump_config_applied_at_ms()
        return True, 'msg.configUpdated', '配置已更新'
    except json.JSONDecodeError:
        return False, 'msg.invalidExtraBodyJson', 'OpenAI 兼容 extra_body 不是合法的 JSON 对象'
    except Exception as e:
        print(f'Error updating config: {e}')
        return False, 'msg.configUpdateFailed', '配置更新失败'


def run_service_async():
    """在独立线程中运行异步服务"""
    global service_loop, stop_event, service_status
    
    # 创建新的事件循环
    service_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(service_loop)
    
    # 导入main模块并运行
    try:
        import main
        # 在 main() 函数内部会创建新的 stop_event，所以这里不需要清除
        service_loop.run_until_complete(main.main(keep_oscquery_alive=True))
        # 运行完成后获取 main 模块中的 stop_event 引用
        stop_event = main.stop_event
    except Exception as e:
        print(f'Service error: {e}')
    finally:
        service_loop.close()
        service_loop = None
        stop_event = None
        service_status['running'] = False
        service_status['recognition_active'] = False


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    return jsonify(get_config_dict())


@app.route('/api/features', methods=['GET'])
def get_features():
    """获取按构建/依赖裁剪后的前端功能开关。"""
    return jsonify(_get_feature_flags())


@app.route('/api/local-asr/status', methods=['GET'])
def get_local_asr_status():
    """获取本地 ASR 下载/可用状态。"""
    if not is_local_asr_build_enabled():
        return jsonify({'success': False, 'message': 'Local ASR is disabled in this build'}), 404

    engines = {}
    for engine in LOCAL_ASR_ENGINES:
        try:
            engines[engine] = get_engine_status(engine)
        except Exception as e:
            engines[engine] = {
                'engine': engine,
                'display_name': LOCAL_ASR_DISPLAY_NAMES.get(engine, engine),
                'ready': False,
                'error': str(e),
            }
    return jsonify({
        'success': True,
        'ui_enabled': is_local_asr_ui_enabled(),
        'engines': engines,
        'download': _snapshot_local_asr_download_state(),
    })


@app.route('/api/local-asr/download', methods=['POST'])
def download_local_asr():
    """后台下载本地 ASR 模型与运行时。"""
    if not is_local_asr_build_enabled():
        return jsonify({'success': False, 'message': 'Local ASR is disabled in this build'}), 404

    data = request.json or {}
    engine = str(data.get('engine') or getattr(config, 'LOCAL_ASR_ENGINE', 'sensevoice'))
    if engine not in LOCAL_ASR_ENGINES:
        return jsonify({'success': False, 'message': f'Unsupported engine: {engine}'}), 400

    snapshot = _snapshot_local_asr_download_state()
    if snapshot.get('running'):
        return jsonify({'success': False, 'message': 'Another local ASR download is already running'}), 409

    worker = threading.Thread(
        target=_download_local_asr_worker,
        args=(engine,),
        daemon=True,
    )
    worker.start()
    return jsonify({'success': True, 'message': 'download started', 'engine': engine})


@app.route('/api/local-asr/download-progress', methods=['GET'])
def get_local_asr_download_progress():
    """查询本地 ASR 下载状态。"""
    if not is_local_asr_build_enabled():
        return jsonify({'success': False, 'message': 'Local ASR is disabled in this build'}), 404
    return jsonify({'success': True, **_snapshot_local_asr_download_state()})


@app.route('/api/env', methods=['GET'])
def get_env_status():
    """获取环境变量状态（不返回敏感信息）"""
    llm_api_key_set = bool(
        os.getenv('LLM_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or os.getenv('OPENROUTER_API_KEY')
    )
    return jsonify({
        'llm': {
            'api_key_set': llm_api_key_set,
        },
    })


@app.route('/api/config', methods=['POST'])
def update_config_api():
    """更新配置"""
    try:
        config_data = request.json
        success, message_id, message = update_config(config_data)
        if success:
            # 如果服务正在运行，通知主服务线程在下一次可行时重载翻译器实例（无需重启识别线程）
            try:
                if service_status.get('running') and service_loop is not None:
                    import main as main_module
                    # 在主服务的事件循环线程安全地调度重初始化操作
                    service_loop.call_soon_threadsafe(getattr(main_module, 'reinitialize_translator_compat', lambda: None))
            except Exception as e:
                print(f'Error notifying service to reload translator: {e}')

            return jsonify({
                'success': True,
                'message_id': message_id,
                'message': message,
                'config_applied_at_ms': int(getattr(config, 'CONFIG_APPLIED_AT_MS', 0) or 0),
            })
        else:
            status_code = 400 if message_id == 'msg.invalidExtraBodyJson' else 500
            return jsonify({'success': False, 'message_id': message_id, 'message': message}), status_code
    except Exception as e:
        print(f'Error updating config: {e}')
        return jsonify({'success': False, 'message_id': 'msg.configUpdateFailed', 'message': '配置更新失败'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    status = dict(service_status)
    status['target_language'] = config.TARGET_LANGUAGE
    status['config_applied_at_ms'] = int(getattr(config, 'CONFIG_APPLIED_AT_MS', 0) or 0)
    status['backend_boot_ms'] = int(getattr(config, 'BACKEND_BOOT_MS', 0) or 0)
    status['local_asr_ui_enabled'] = is_local_asr_ui_enabled()
    return jsonify(status)

@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    """获取最新的一条字幕状态"""
    try:
        import sys
        if 'main' in sys.modules:
            import main as m
            if hasattr(m, 'subtitles_state'):
                state = m.subtitles_state.copy()
                state['running'] = service_status['running']
                state['show_reverse_translation'] = bool(getattr(config, 'ENABLE_REVERSE_TRANSLATION', False))
                state['target_language'] = config.TARGET_LANGUAGE
                state['config_applied_at_ms'] = int(getattr(config, 'CONFIG_APPLIED_AT_MS', 0) or 0)
                state['backend_boot_ms'] = int(getattr(config, 'BACKEND_BOOT_MS', 0) or 0)
                return jsonify(state)
    except Exception:
        pass
    
    return jsonify({
        "original": "",
        "translated": "",
        "reverse_translated": "",
        "ongoing": False,
        "show_reverse_translation": bool(getattr(config, 'ENABLE_REVERSE_TRANSLATION', False)),
        "running": service_status['running'],
        "target_language": config.TARGET_LANGUAGE,
        "config_applied_at_ms": int(getattr(config, 'CONFIG_APPLIED_AT_MS', 0) or 0),
        "backend_boot_ms": int(getattr(config, 'BACKEND_BOOT_MS', 0) or 0),
    })

@app.route('/api/target-language', methods=['POST'])
def set_target_language():
    """快速切换目标翻译语言（小面板快捷按钮）"""
    try:
        data = request.json or {}
        lang = (data.get('target_language') or '').strip()
        if not lang:
            return jsonify({'success': False, 'message': 'target_language is required'}), 400

        config.TARGET_LANGUAGE = lang
        applied_ms = config.bump_config_applied_at_ms()

        # 如果服务正在运行，热加载翻译器
        try:
            if service_status.get('running') and service_loop is not None:
                import main as main_module
                service_loop.call_soon_threadsafe(getattr(main_module, 'reinitialize_translator_compat', lambda: None))
        except Exception as e:
            print(f'Error notifying service to reload translator: {e}')

        return jsonify({
            'success': True,
            'target_language': lang,
            'config_applied_at_ms': applied_ms,
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/panel', methods=['GET'])
def panel():
    """渲染迷你面板"""
    return render_template('panel.html')

@app.route('/api/open-panel', methods=['POST'])
def open_panel():
    """打开迷你面板窗口"""
    try:
        # 接收大面板传来的 API Keys 并写入环境变量
        data = request.json or {}
        api_keys = data.get('api_keys', {})
        floating_mode = bool(data.get('floating_mode', False))
        quick_language_settings = data.get('quick_language_settings') or {}
        if api_keys.get('dashscope'):
            os.environ['DASHSCOPE_API_KEY'] = api_keys['dashscope']
        if api_keys.get('deepl'):
            os.environ['DEEPL_API_KEY'] = api_keys['deepl']
        if api_keys.get('llm'):
            os.environ['LLM_API_KEY'] = api_keys['llm']
        if api_keys.get('soniox'):
            os.environ['SONIOX_API_KEY'] = api_keys['soniox']
        if api_keys.get('doubao'):
            os.environ['DOUBAO_API_KEY'] = api_keys['doubao']

        quick_lang_defaults = ['en', 'zh-CN', 'ja', 'ko']
        raw_quick_langs = quick_language_settings.get('languages')
        panel_quick_langs = []
        for index, fallback in enumerate(quick_lang_defaults):
            value = fallback
            if isinstance(raw_quick_langs, list) and index < len(raw_quick_langs):
                candidate = str(raw_quick_langs[index]).strip()
                if candidate:
                    value = candidate
            panel_quick_langs.append(value)

        raw_quick_lang_enabled = quick_language_settings.get('enabled', True)
        if isinstance(raw_quick_lang_enabled, str):
            quick_lang_enabled = raw_quick_lang_enabled.strip().lower() not in {'', '0', 'false', 'no', 'off'}
        else:
            quick_lang_enabled = bool(raw_quick_lang_enabled)

        import subprocess
        python_exe = sys.executable
        # script is at project root
        panel_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "panel_app.py")
        panel_query = [('quick_lang_bar', '1' if quick_lang_enabled else '0')]
        panel_query.extend(('quick_lang', lang) for lang in panel_quick_langs)
        panel_url = f"http://127.0.0.1:5001/panel?{urlencode(panel_query)}"
        initial_mode = "reverse-on" if getattr(config, 'ENABLE_REVERSE_TRANSLATION', False) else "reverse-off"
        floating_mode_arg = "floating-on" if floating_mode else "floating-off"
        panel_width_arg = str(max(300, int(getattr(config, 'PANEL_WIDTH', 600))))
        panel_args = [panel_url, initial_mode, floating_mode_arg, panel_width_arg]
        if getattr(sys, 'frozen', False):
            launch_cmd = [python_exe, '--panel-app', *panel_args]
        else:
            launch_cmd = [python_exe, panel_script, *panel_args]

        subprocess.Popen(launch_cmd)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/audio/input-devices', methods=['GET'])
def list_input_devices():
    """列出当前系统可用的麦克风输入设备（PyAudio）。"""
    try:
        try:
            import pyaudio
        except Exception as e:  # pragma: no cover
            return jsonify({'devices': [], 'default_index': None, 'default_name': None, 'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None), 'error': str(e)})

        pa = pyaudio.PyAudio()
        devices = []
        default_index = None
        default_name = None
        preferred_host_api_index = None
        try:
            try:
                default_info = pa.get_default_input_device_info()
                default_index = default_info.get('index')
                default_name = str(default_info.get('name') or '').strip() or None
            except Exception:
                default_index = None
                default_name = None

            # 选择一个 Host API，避免同一设备被不同 Host API 重复枚举
            # 优先 WASAPI（更贴近系统设备管理器的“启用/禁用”状态），否则用默认 Host API
            try:
                host_api_count = pa.get_host_api_count()
                for host_api_idx in range(host_api_count):
                    try:
                        host_api_info = pa.get_host_api_info_by_index(host_api_idx)
                    except Exception:
                        continue
                    name = str(host_api_info.get('name') or '')
                    if 'wasapi' in name.lower():
                        preferred_host_api_index = host_api_idx
                        break
            except Exception:
                preferred_host_api_index = None

            if preferred_host_api_index is None:
                try:
                    preferred_host_api_index = pa.get_default_host_api_info().get('index')
                except Exception:
                    preferred_host_api_index = None

            count = pa.get_device_count()
            seen_names = set()
            for idx in range(count):
                try:
                    info = pa.get_device_info_by_index(idx)
                except Exception:
                    continue

                if preferred_host_api_index is not None and info.get('hostApi') != preferred_host_api_index:
                    continue

                max_in = int(info.get('maxInputChannels', 0) or 0)
                if max_in <= 0:
                    continue

                name = str(info.get('name') or f'Device {idx}')
                name_key = " ".join(name.strip().lower().split())
                if name_key in seen_names:
                    continue
                seen_names.add(name_key)

                devices.append({'index': idx, 'name': name, 'max_input_channels': max_in})
        finally:
            try:
                pa.terminate()
            except Exception:
                pass

        return jsonify({
            'devices': devices,
            'default_index': default_index,
            'default_name': default_name,
            'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None),
        })
    except Exception as e:
        return jsonify({'devices': [], 'default_index': None, 'default_name': None, 'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None), 'error': str(e)}), 500


@app.route('/api/service/start', methods=['POST'])
def start_service():
    """启动服务"""
    global service_thread, service_status
    
    if service_status['running']:
        return jsonify({'success': False, 'message_id': 'msg.serviceAlreadyRunning', 'message': '服务已在运行中'})
    
    try:
        # 从请求中获取 API Keys
        data = request.json or {}
        api_keys = data.get('api_keys', {})
        
        # 设置 API Keys 到环境变量
        if 'dashscope' in api_keys and api_keys['dashscope']:
            os.environ['DASHSCOPE_API_KEY'] = api_keys['dashscope']
        
        if 'deepl' in api_keys and api_keys['deepl']:
            os.environ['DEEPL_API_KEY'] = api_keys['deepl']
        
        if 'llm' in api_keys and api_keys['llm']:
            os.environ['LLM_API_KEY'] = api_keys['llm']
        
        if 'soniox' in api_keys and api_keys['soniox']:
            os.environ['SONIOX_API_KEY'] = api_keys['soniox']

        if 'doubao' in api_keys and api_keys['doubao']:
            os.environ['DOUBAO_API_KEY'] = api_keys['doubao']
        
        service_thread = threading.Thread(target=run_service_async, daemon=True)
        service_thread.start()
        service_status['running'] = True
        service_status['backend'] = _sanitize_preferred_backend(config.PREFERRED_ASR_BACKEND)
        return jsonify({'success': True, 'message_id': 'msg.serviceStarted', 'message': '服务已启动'})
    except Exception as e:
        print(f'Error starting service: {e}')
        return jsonify({'success': False, 'message_id': 'msg.startFailed', 'message': '启动失败'}), 500


@app.route('/api/service/stop', methods=['POST'])
def stop_service():
    """停止服务"""
    global service_thread, service_status, service_loop, stop_event
    
    if not service_status['running']:
        return jsonify({'success': False, 'message_id': 'msg.serviceNotRunning', 'message': '服务未运行'})
    
    try:
        # 从 main 模块获取最新的 stop_event
        import main
        current_stop_event = main.stop_event
        
        if current_stop_event and service_loop:
            service_loop.call_soon_threadsafe(current_stop_event.set)
        
        # 等待线程结束（最多10秒）
        if service_thread:
            service_thread.join(timeout=10)
        
        service_status['running'] = False
        service_status['recognition_active'] = False
        stop_event = None
        return jsonify({'success': True, 'message_id': 'msg.serviceStopped', 'message': '服务已停止'})
    except Exception as e:
        print(f'Error stopping service: {e}')
        return jsonify({'success': False, 'message_id': 'msg.stopFailed', 'message': '停止失败'}), 500


@app.route('/api/service/restart', methods=['POST'])
def restart_service():
    """重启服务"""
    global service_thread, service_status, service_loop, stop_event
    
    if not service_status['running']:
        return jsonify({'success': False, 'message_id': 'msg.noRestartNeeded', 'message': '服务未运行，无需重启'})
    
    try:
        # 从 main 模块获取最新的 stop_event
        import main
        current_stop_event = main.stop_event
        
        # 先停止
        if current_stop_event and service_loop:
            service_loop.call_soon_threadsafe(current_stop_event.set)
        
        if service_thread:
            service_thread.join(timeout=10)
        
        service_status['running'] = False
        service_status['recognition_active'] = False
        stop_event = None
        
        # 再启动
        service_thread = threading.Thread(target=run_service_async, daemon=True)
        service_thread.start()
        service_status['running'] = True
        
        return jsonify({'success': True, 'message_id': 'msg.serviceRestarted', 'message': '服务已重启'})
    except Exception as e:
        print(f'Error restarting service: {e}')
        return jsonify({'success': False, 'message_id': 'msg.restartFailed', 'message': '重启失败'}), 500


@app.route('/api/config/defaults', methods=['GET'])
def get_defaults():
    """获取默认配置"""
    return jsonify({
        'features': _get_feature_flags(),
        'asr': {
            'preferred_backend': 'qwen',  # 可选: 'qwen', 'qwen_international', 'dashscope'
            'enable_vad': True,
            'vad_threshold': 0.2,
            'vad_silence_duration_ms': 800,
            'keepalive_interval': 30,
            'enable_hot_words': True,
        },
        'translation': {
            'enable_translation': True,
            'source_language': 'auto',
            'target_language': 'ja',
            'secondary_target_language': None,
            'fallback_language': 'en',
            'api_type': 'deepl',
            'llm_parallel_fastest_mode': 'off',
            'show_partial_results': False,
            'enable_furigana': False,
            'enable_reverse_translation': True,
        },
        'mic_control': {
            'enable_mic_control': True,
            'mute_delay_seconds': 0.2,
        },
        'language_detector': {
            'type': 'cjke',
        },
        'panel': {
            'width': 600,
        },
        'local_asr': _local_asr_config_dict() if is_local_asr_ui_enabled() else None,
    })


@app.route('/api/check-api-key', methods=['POST'])
def check_api_key():
    """检查API Key是否有效"""
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'valid': False, 'message_id': 'msg.enterDashscopeKey', 'message': '请输入 DashScope API Key'})
        
        # 检查API Key格式
        if not api_key.startswith('sk-'):
            return jsonify({'valid': False, 'message_id': 'msg.invalidKeyFormat', 'message': 'API Key 格式无效（应以 sk- 开头）'})
        
        # 检查API Key是否是占位符
        if api_key == '<your-dashscope-api-key>':
            return jsonify({'valid': False, 'message_id': 'msg.replacePlaceholder', 'message': '请替换占位符为真实的 API Key'})
        
        # 临时设置API Key到环境变量
        os.environ['DASHSCOPE_API_KEY'] = api_key
        
        return jsonify({'valid': True, 'message_id': 'msg.keyFormatValid', 'message': 'API Key 格式有效'})
    except Exception as e:
        print(f'Error checking API key: {e}')
        return jsonify({'valid': False, 'message_id': 'msg.checkFailed', 'message': '检查失败'}), 500


if __name__ == '__main__':
    print('='*60)
    print('VRChat 翻译器 Web UI')
    print('访问 http://localhost:5001 打开控制面板')
    print('按 Ctrl+C 退出')
    print('='*60)
    app.run(host='0.0.0.0', port=5001, debug=False)
