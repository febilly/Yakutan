"""
Web UI for VRChat Translator
提供配置管理和服务控制的Web界面
"""
import asyncio
import json
import threading
import logging
from typing import Optional
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os

# 添加父目录到路径以导入config和main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
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


def get_config_dict():
    """获取当前配置"""
    return {
        # 语音识别配置
        'asr': {
            'preferred_backend': config.PREFERRED_ASR_BACKEND,
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
            'fallback_language': config.FALLBACK_LANGUAGE,
            'api_type': config.TRANSLATION_API_TYPE,
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
        # OSC配置
        'osc': {
            'server_ip': config.OSC_SERVER_IP,
            'server_port': config.OSC_SERVER_PORT,
            'client_ip': config.OSC_CLIENT_IP,
            'client_port': config.OSC_CLIENT_PORT,
        },
    }


def update_config(config_data):
    """更新配置"""
    try:
        # 更新ASR配置
        if 'asr' in config_data:
            asr = config_data['asr']
            if 'preferred_backend' in asr:
                config.PREFERRED_ASR_BACKEND = asr['preferred_backend']
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
            if 'fallback_language' in trans:
                config.FALLBACK_LANGUAGE = trans['fallback_language'] if trans['fallback_language'] else None
            if 'api_type' in trans:
                config.TRANSLATION_API_TYPE = trans['api_type']
                # 前端的"流式翻译模式"开关会将 api_type 设为 'openrouter_streaming'
                # 此时启用部分结果翻译（实时翻译未完成的句子）
                config.TRANSLATE_PARTIAL_RESULTS = (trans['api_type'] == 'openrouter_streaming')
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
        
        return True
    except Exception as e:
        print(f'Error updating config: {e}')
        return False


def run_service_async():
    """在独立线程中运行异步服务"""
    global service_loop, stop_event
    
    # 创建新的事件循环
    service_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(service_loop)
    
    # 导入main模块并运行
    try:
        import main
        # 在 main() 函数内部会创建新的 stop_event，所以这里不需要清除
        service_loop.run_until_complete(main.main())
        # 运行完成后获取 main 模块中的 stop_event 引用
        stop_event = main.stop_event
    except Exception as e:
        print(f'Service error: {e}')
    finally:
        service_loop.close()
        service_loop = None
        stop_event = None


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    return jsonify(get_config_dict())


@app.route('/api/env', methods=['GET'])
def get_env_status():
    """获取环境变量状态（不返回敏感信息）"""
    openai_api_key_set = bool(os.getenv('OPENAI_API_KEY'))
    openrouter_api_key_set = bool(os.getenv('OPENROUTER_API_KEY'))
    return jsonify({
        'openrouter': {
            'api_key_set': openai_api_key_set or openrouter_api_key_set,
        },
        'openai': {
            'api_key_set': openai_api_key_set,
        },
    })


@app.route('/api/config', methods=['POST'])
def update_config_api():
    """更新配置"""
    try:
        config_data = request.json
        if update_config(config_data):
            # 如果服务正在运行，通知主服务线程在下一次可行时重载翻译器实例（无需重启识别线程）
            try:
                if service_status.get('running') and service_loop is not None:
                    import main as main_module
                    # 在主服务的事件循环线程安全地调度重初始化操作
                    service_loop.call_soon_threadsafe(getattr(main_module, 'reinitialize_translator', lambda: None))
            except Exception as e:
                print(f'Error notifying service to reload translator: {e}')

            return jsonify({'success': True, 'message_id': 'msg.configUpdated', 'message': '配置已更新'})
        else:
            return jsonify({'success': False, 'message_id': 'msg.configUpdateFailed', 'message': '配置更新失败'}), 500
    except Exception as e:
        print(f'Error updating config: {e}')
        return jsonify({'success': False, 'message_id': 'msg.configUpdateFailed', 'message': '配置更新失败'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    return jsonify(service_status)


@app.route('/api/audio/input-devices', methods=['GET'])
def list_input_devices():
    """列出当前系统可用的麦克风输入设备（PyAudio）。"""
    try:
        try:
            import pyaudio
        except Exception as e:  # pragma: no cover
            return jsonify({'devices': [], 'default_index': None, 'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None), 'error': str(e)})

        pa = pyaudio.PyAudio()
        devices = []
        default_index = None
        preferred_host_api_index = None
        try:
            try:
                default_info = pa.get_default_input_device_info()
                default_index = default_info.get('index')
            except Exception:
                default_index = None

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
            'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None),
        })
    except Exception as e:
        return jsonify({'devices': [], 'default_index': None, 'selected_index': getattr(config, 'MIC_DEVICE_INDEX', None), 'error': str(e)}), 500


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
        
        if 'openrouter' in api_keys and api_keys['openrouter']:
            os.environ['OPENROUTER_API_KEY'] = api_keys['openrouter']
        
        if 'soniox' in api_keys and api_keys['soniox']:
            os.environ['SONIOX_API_KEY'] = api_keys['soniox']
        
        service_thread = threading.Thread(target=run_service_async, daemon=True)
        service_thread.start()
        service_status['running'] = True
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
            'fallback_language': 'en',
            'api_type': 'deepl',
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
