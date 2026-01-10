// API基础URL
const API_BASE = '/api';

// 自动保存定时器
let autoSaveTimer = null;

// 配置键名
const CONFIG_STORAGE_KEY = 'vrchat_translator_config';

// 待显示的警告消息（用于自动切换翻译API）
let pendingWarningMessage = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function () {
    // 先初始化 i18n 系统
    if (window.i18n) {
        window.i18n.initI18n();
    }

    const targetLangInput = document.getElementById('target-language');
    if (targetLangInput) {
        targetLangInput.addEventListener('input', updateFuriganaVisibility);
    }

    loadConfigFromLocalStorage();
    loadAPIKeys();
    refreshMicDevices(true);
    setupMicDeviceAutoRefresh();
    updateStatus();
    // 每2秒更新一次状态
    setInterval(updateStatus, 2000);

    // 显示配置保存提示
    showConfigStorageInfo();
});

// 刷新后端输入设备列表（PyAudio）
async function refreshMicDevices(preserveSelection = true) {
    const micSelect = document.getElementById('mic-device');
    if (!micSelect) return;

    const previousValue = preserveSelection ? micSelect.value : '';

    try {
        const response = await fetch(`${API_BASE}/audio/input-devices`);
        const data = await response.json();
        const devices = Array.isArray(data.devices) ? data.devices : [];

        // 重建选项
        micSelect.innerHTML = '';
        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.textContent = '系统默认';
        micSelect.appendChild(defaultOpt);

        const indices = new Set();
        devices.forEach((d) => {
            const idx = d.index;
            if (idx === undefined || idx === null) return;
            const idxStr = String(idx);
            indices.add(idxStr);

            const opt = document.createElement('option');
            opt.value = idxStr;
            const name = d.name ? String(d.name) : `Device ${idxStr}`;
            opt.textContent = `${name} (#${idxStr})`;
            micSelect.appendChild(opt);
        });

        // 优先保留当前选择；否则使用后端当前配置；最后回退默认
        const serverSelected = (data.selected_index === undefined || data.selected_index === null) ? '' : String(data.selected_index);
        if (previousValue && indices.has(previousValue)) {
            micSelect.value = previousValue;
        } else if (serverSelected && indices.has(serverSelected)) {
            micSelect.value = serverSelected;
        } else {
            micSelect.value = '';
        }
    } catch (e) {
        // 静默失败：不影响其它功能
        console.warn('获取麦克风列表失败:', e);
    }
}

function setupMicDeviceAutoRefresh() {
    // 尽量监听浏览器设备变化事件（不保证每个环境都触发）
    try {
        if (navigator.mediaDevices) {
            if (typeof navigator.mediaDevices.addEventListener === 'function') {
                navigator.mediaDevices.addEventListener('devicechange', () => refreshMicDevices(true));
            } else if ('ondevicechange' in navigator.mediaDevices) {
                navigator.mediaDevices.ondevicechange = () => refreshMicDevices(true);
            }
        }
    } catch (_) {
        // ignore
    }

    // 后端设备列表变化更可靠：做一个低频轮询刷新
    setInterval(() => refreshMicDevices(true), 5000);
}

// 显示配置保存信息
function showConfigStorageInfo() {
    const lastSaved = localStorage.getItem(CONFIG_STORAGE_KEY + '_timestamp');

    if (lastSaved) {
        const date = new Date(parseInt(lastSaved));
        console.log(`配置已加载 (${date.toLocaleString('zh-CN')})`);
    }
}

// 从localStorage加载API Keys
function loadAPIKeys() {
    const dashscopeKey = localStorage.getItem('dashscope_api_key');
    const deeplKey = localStorage.getItem('deepl_api_key');
    const openrouterKey = localStorage.getItem('openrouter_api_key');
    const useInternational = localStorage.getItem('use_international_endpoint') === 'true';

    if (dashscopeKey) document.getElementById('dashscope-api-key').value = dashscopeKey;
    if (deeplKey) document.getElementById('deepl-api-key').value = deeplKey;
    if (openrouterKey) document.getElementById('openrouter-api-key').value = openrouterKey;

    const sonioxKey = localStorage.getItem('soniox_api_key');
    if (sonioxKey) document.getElementById('soniox-api-key').value = sonioxKey;

    document.getElementById('use-international-endpoint').checked = useInternational;

    // 应用国际版设置对 ASR 选项的影响
    updateAsrOptionsForInternational(useInternational);

    // 如果已有DashScope API Key，自动折叠API Keys区域
    if (dashscopeKey) {
        const apiKeysSection = document.getElementById('api-keys');
        const apiKeysIcon = document.getElementById('api-keys-icon');
        if (apiKeysSection && !apiKeysSection.classList.contains('collapsed')) {
            apiKeysSection.classList.add('collapsed');
            apiKeysIcon.classList.add('collapsed');
            apiKeysIcon.textContent = '▶';
        }
    }

    // 添加API Key change事件监听
    document.getElementById('dashscope-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('deepl-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('openrouter-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('soniox-api-key').addEventListener('input', saveAPIKey);
}

// 处理国际版端点开关变化
function handleInternationalEndpointChange(event) {
    const useInternational = event.target.checked;

    // 保存到 localStorage
    localStorage.setItem('use_international_endpoint', useInternational.toString());

    // 更新 ASR 选项
    updateAsrOptionsForInternational(useInternational);

    // 触发配置保存
    onSettingChange();
}

// 根据国际版设置更新 ASR 选项
function updateAsrOptionsForInternational(useInternational) {
    const asrBackendSelect = document.getElementById('asr-backend');
    const dashscopeOption = asrBackendSelect.querySelector('option[value="dashscope"]');
    const t = window.i18n ? window.i18n.t : (key) => key;

    if (useInternational) {
        // 国际版：禁用 Fun-ASR 选项
        if (dashscopeOption) {
            dashscopeOption.disabled = true;
            dashscopeOption.textContent = t('asr.dashscopeDisabled');
        }

        // 如果当前选中的是 dashscope，自动切换到 qwen
        if (asrBackendSelect.value === 'dashscope') {
            asrBackendSelect.value = 'qwen';
            onSettingChange();
        }
    } else {
        // 中国大陆版：启用 Fun-ASR 选项
        if (dashscopeOption) {
            dashscopeOption.disabled = false;
            dashscopeOption.textContent = t('asr.dashscope');
        }
    }
}

// 保存API Key到localStorage
function saveAPIKey(event) {
    const id = event.target.id;
    const value = event.target.value;
    const keyName = id.replace(/-/g, '_'); // 使用正则表达式替换所有 '-' 为 '_'

    if (value) {
        localStorage.setItem(keyName, value);
    } else {
        localStorage.removeItem(keyName);
    }

    // 触发配置自动保存
    onSettingChange();
}

// 应用语言选择（从下拉列表到文本框）
function applyLanguageSelection(inputId) {
    const selectId = inputId + '-select';
    const inputElement = document.getElementById(inputId);
    const selectElement = document.getElementById(selectId);

    // 获取选择的值（可能是空字符串）
    const selectedValue = selectElement.value;

    // 如果下拉框有选项被选中（包括空值"禁用"）
    if (selectElement.selectedIndex > 0) {  // 跳过 "-- 快速选择 --" 选项
        inputElement.value = selectedValue;
        // 重置下拉列表到提示状态
        selectElement.value = '';
        // 触发配置更新
        updateFuriganaVisibility();
        onSettingChange();
    }
}

// 从 localStorage 加载配置
function loadConfigFromLocalStorage() {
    try {
        const savedConfig = localStorage.getItem(CONFIG_STORAGE_KEY);

        if (savedConfig) {
            const config = JSON.parse(savedConfig);

            // 填充表单
            if (config.translation) {
                document.getElementById('enable-translation').checked = config.translation.enable_translation ?? true;
                document.getElementById('target-language').value = config.translation.target_language || 'ja';
                document.getElementById('fallback-language').value = config.translation.fallback_language || 'en';
                // 处理 openrouter_streaming 的特殊情况
                const apiType = config.translation.api_type || 'qwen_mt';
                if (apiType === 'openrouter_streaming') {
                    document.getElementById('translation-api-type').value = 'openrouter';
                    document.getElementById('openrouter-streaming-mode').checked = true;
                } else {
                    document.getElementById('translation-api-type').value = apiType;
                    document.getElementById('openrouter-streaming-mode').checked = false;
                }
                document.getElementById('source-language').value = config.translation.source_language || 'auto';
                document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
                document.getElementById('enable-furigana').checked = config.translation.enable_furigana ?? false;
                document.getElementById('enable-pinyin').checked = config.translation.enable_pinyin ?? false;
                document.getElementById('enable-reverse-translation').checked = config.translation.enable_reverse_translation ?? true;

                const showTag = document.getElementById('show-original-and-lang-tag');
                if (showTag) {
                    showTag.checked = config.translation.show_original_and_lang_tag ?? true;
                }
            }

            if (config.mic_control) {
                document.getElementById('enable-mic-control').checked = config.mic_control.enable_mic_control ?? true;
                document.getElementById('mute-delay').value = config.mic_control.mute_delay_seconds || 0.2;

                const micSelect = document.getElementById('mic-device');
                if (micSelect) {
                    const idx = config.mic_control.mic_device_index;
                    micSelect.value = (idx === undefined || idx === null) ? '' : String(idx);
                }
            }

            if (config.asr) {
                document.getElementById('asr-backend').value = config.asr.preferred_backend || 'qwen';
                document.getElementById('enable-hot-words').checked = config.asr.enable_hot_words ?? true;
                document.getElementById('enable-vad').checked = config.asr.enable_vad ?? true;
                document.getElementById('vad-threshold').value = config.asr.vad_threshold || 0.2;
                document.getElementById('vad-silence-duration').value = config.asr.vad_silence_duration_ms || 800;
                document.getElementById('keepalive-interval').value = config.asr.keepalive_interval || 30;

                // 加载国际版设置
                const useInternational = config.asr.use_international_endpoint ?? false;
                document.getElementById('use-international-endpoint').checked = useInternational;
                localStorage.setItem('use_international_endpoint', useInternational.toString());
                updateAsrOptionsForInternational(useInternational);
            }

            if (config.language_detector) {
                document.getElementById('language-detector').value = config.language_detector.type || 'cjke';
            }

            console.log('✓ 已从浏览器加载配置');
        } else {
            // 如果没有保存的配置，使用前端默认值
            loadDefaultConfig();
        }

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();
        updateFuriganaVisibility();

    } catch (error) {
        console.error('加载本地配置失败:', error);
        // 出错时使用前端默认值
        loadDefaultConfig();
        toggleTranslationOptions();
        updateFuriganaVisibility();
    }
}

// 加载前端默认配置（不依赖服务器）
function loadDefaultConfig() {
    // 翻译配置
    document.getElementById('enable-translation').checked = true;
    document.getElementById('target-language').value = 'ja';
    document.getElementById('fallback-language').value = 'en';
    document.getElementById('translation-api-type').value = 'qwen_mt';
    document.getElementById('source-language').value = 'auto';
    document.getElementById('show-partial-results').checked = false;
    document.getElementById('enable-furigana').checked = false;
    document.getElementById('enable-pinyin').checked = false;
    document.getElementById('enable-reverse-translation').checked = true;
    document.getElementById('openrouter-streaming-mode').checked = false;

    const showTag = document.getElementById('show-original-and-lang-tag');
    if (showTag) showTag.checked = true;

    // 麦克风控制
    document.getElementById('enable-mic-control').checked = true;
    document.getElementById('mute-delay').value = 0.2;

    // 麦克风设备
    const micSelect = document.getElementById('mic-device');
    if (micSelect) micSelect.value = '';

    // ASR 配置
    document.getElementById('asr-backend').value = 'qwen';  // 可选: 'qwen', 'dashscope'
    document.getElementById('enable-hot-words').checked = true;
    document.getElementById('enable-vad').checked = true;
    document.getElementById('vad-threshold').value = 0.2;
    document.getElementById('vad-silence-duration').value = 800;
    document.getElementById('keepalive-interval').value = 30;
    document.getElementById('use-international-endpoint').checked = false;

    // 语言检测器
    document.getElementById('language-detector').value = 'cjke';

    console.log('✓ 已加载前端默认配置');
    updateFuriganaVisibility();
}

// 从服务器加载配置（仅在本地无配置时使用）
async function loadConfigFromServer() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();

        // 填充表单
        document.getElementById('enable-translation').checked = config.translation.enable_translation;
        document.getElementById('target-language').value = config.translation.target_language;
        document.getElementById('fallback-language').value = config.translation.fallback_language || '';
        // 处理 openrouter_streaming 的特殊情况
        const serverApiType = config.translation.api_type;
        if (serverApiType === 'openrouter_streaming') {
            document.getElementById('translation-api-type').value = 'openrouter';
            document.getElementById('openrouter-streaming-mode').checked = true;
        } else {
            document.getElementById('translation-api-type').value = serverApiType;
            document.getElementById('openrouter-streaming-mode').checked = false;
        }
        document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
        document.getElementById('enable-furigana').checked = config.translation.enable_furigana ?? false;
        document.getElementById('enable-pinyin').checked = config.translation.enable_pinyin ?? false;

        document.getElementById('enable-mic-control').checked = config.mic_control.enable_mic_control;
        document.getElementById('mute-delay').value = config.mic_control.mute_delay_seconds;

        document.getElementById('asr-backend').value = config.asr.preferred_backend;
        document.getElementById('enable-hot-words').checked = config.asr.enable_hot_words;
        document.getElementById('enable-vad').checked = config.asr.enable_vad;
        document.getElementById('vad-threshold').value = config.asr.vad_threshold;
        document.getElementById('vad-silence-duration').value = config.asr.vad_silence_duration_ms;
        document.getElementById('keepalive-interval').value = config.asr.keepalive_interval;

        document.getElementById('language-detector').value = config.language_detector.type;
        document.getElementById('source-language').value = config.translation.source_language;

        // 保存到本地
        saveConfigToLocalStorage();

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();
        updateFuriganaVisibility();

        console.log('已从服务器加载配置');

    } catch (error) {
        console.error('加载服务器配置失败:', error);
    }
}

// 保存配置到 localStorage
function saveConfigToLocalStorage() {
    try {
        // 确定实际的 API 类型（如果是 OpenRouter 且启用了流式模式，使用 openrouter_streaming）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                source_language: document.getElementById('source-language').value,
                show_partial_results: document.getElementById('show-partial-results').checked,
                enable_furigana: document.getElementById('enable-furigana').checked,
                enable_pinyin: document.getElementById('enable-pinyin').checked,
                enable_reverse_translation: document.getElementById('enable-reverse-translation').checked,
            },
            mic_control: {
                enable_mic_control: document.getElementById('enable-mic-control').checked,
                mute_delay_seconds: parseFloat(document.getElementById('mute-delay').value),
            },
            asr: {
                preferred_backend: document.getElementById('asr-backend').value,
                enable_hot_words: document.getElementById('enable-hot-words').checked,
                enable_vad: document.getElementById('enable-vad').checked,
                vad_threshold: parseFloat(document.getElementById('vad-threshold').value),
                vad_silence_duration_ms: parseInt(document.getElementById('vad-silence-duration').value),
                keepalive_interval: parseInt(document.getElementById('keepalive-interval').value),
                use_international_endpoint: document.getElementById('use-international-endpoint').checked,
            },
            language_detector: {
                type: document.getElementById('language-detector').value,
            }
        };

        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
        localStorage.setItem(CONFIG_STORAGE_KEY + '_timestamp', Date.now().toString());

    } catch (error) {
        console.error('保存配置失败:', error);
    }
}

// 加载配置（兼容旧代码）
async function loadConfig() {
    loadConfigFromLocalStorage();
}

// 切换翻译选项的显示/隐藏
function toggleTranslationOptions() {
    const enableTranslation = document.getElementById('enable-translation').checked;
    const translationOptions = document.getElementById('translation-options');

    if (enableTranslation) {
        translationOptions.classList.remove('hidden');
    } else {
        translationOptions.classList.add('hidden');
    }
}

// 假名和拼音选项始终显示，无需根据语言切换
function updateFuriganaVisibility() {
    // 假名和拼音选项始终可见，不再根据目标语言隐藏
}

// 处理翻译API变更
let previousTranslationApi = null;

function handleTranslationApiChange(event) {
    const newApi = event.target.value;
    const warningElement = document.getElementById('translation-api-warning');
    const streamingModeGroup = document.getElementById('openrouter-streaming-mode-group');
    const t = window.i18n ? window.i18n.t : (key) => key;

    // 检查是否需要API Key
    let requiresKey = false;
    let keyName = '';
    let keyInputId = '';
    let apiDisplayName = '';

    // 显示/隐藏 OpenRouter 流式翻译模式选项
    if (newApi === 'openrouter') {
        streamingModeGroup.style.display = 'block';
    } else {
        streamingModeGroup.style.display = 'none';
        // 如果切换到不支持流式的API，自动关闭流式翻译开关
        document.getElementById('openrouter-streaming-mode').checked = false;
    }

    if (newApi === 'deepl') {
        requiresKey = true;
        keyName = 'deepl_api_key';
        keyInputId = 'deepl-api-key';
        apiDisplayName = 'DeepL';
    } else if (newApi === 'openrouter') {
        requiresKey = true;
        keyName = 'openrouter_api_key';
        keyInputId = 'openrouter-api-key';
        apiDisplayName = 'OpenRouter';
    }

    // 如果需要API Key但未提供
    if (requiresKey) {
        const apiKey = localStorage.getItem(keyName) || document.getElementById(keyInputId).value.trim();

        if (!apiKey) {
            // 恢复到之前的选项
            if (previousTranslationApi) {
                event.target.value = previousTranslationApi;
            } else {
                // 如果没有之前的值，默认切换到Google Dictionary
                event.target.value = 'google_dictionary';
            }

            // 显示警告消息
            warningElement.textContent = '⚠️ ' + t('msg.apiKeyRequired', { api: apiDisplayName });
            warningElement.style.display = 'block';

            // 5秒后自动隐藏
            setTimeout(() => {
                warningElement.style.display = 'none';
            }, 5000);

            // 展开API Keys配置区域
            const apiKeysSection = document.getElementById('api-keys');
            if (apiKeysSection.classList.contains('collapsed')) {
                toggleCollapsible('api-keys');
            }

            // 高亮对应的API Key输入框
            const keyInput = document.getElementById(keyInputId);
            if (keyInput) {
                keyInput.classList.add('error-highlight');
                setTimeout(() => {
                    keyInput.classList.remove('error-highlight');
                }, 3000);
            }

            return; // 不触发保存
        }
    }

    // 清除警告消息
    warningElement.style.display = 'none';

    // 记录当前选择作为下次的"之前选项"
    previousTranslationApi = newApi;

    // 触发配置保存
    onSettingChange();
}

// 初始化时记录当前的翻译API
document.addEventListener('DOMContentLoaded', function () {
    setTimeout(() => {
        const apiSelect = document.getElementById('translation-api-type');
        previousTranslationApi = apiSelect.value;

        // 初始化 OpenRouter 流式模式选项的显示状态
        const streamingModeGroup = document.getElementById('openrouter-streaming-mode-group');
        if (apiSelect.value === 'openrouter') {
            streamingModeGroup.style.display = 'block';
        } else {
            streamingModeGroup.style.display = 'none';
        }
    }, 100);
});

// 当设置改变时自动保存（延迟保存，避免频繁请求）
function onSettingChange(changedElement = null) {
    // 清除之前的定时器
    if (autoSaveTimer) {
        clearTimeout(autoSaveTimer);
    }

    // 200ms后自动保存（更快响应，静默保存）
    autoSaveTimer = setTimeout(async () => {
        // 先保存到本地浏览器
        saveConfigToLocalStorage();

        // 再保存到服务器
        await saveConfig(true); // true表示是自动保存，不重启服务

        // 如果服务正在运行，仅在必要时（如 ASR 后端等）重启；否则只保存并通知后端热加载配置
        const statusResponse = await fetch(`${API_BASE}/status`);
        const status = await statusResponse.json();
        if (status.running) {
            try {
            // 优先使用传入的变更元素；否则退回到当前活动元素
            const el = changedElement || document.activeElement;

                // 检查元素是否有 data-restart-required 属性
                const needRestart = el && el.getAttribute('data-restart-required') === 'true';

                if (needRestart) {
                    // 仅在确实修改了需要重启的项时重启服务
                    await restartService();
                } else {
                    // 不需要重启：后端（同进程）会收到配置变更并在下一次翻译时热加载
                    console.log('配置已保存；无需重启服务，已通知后端热加载。');
                }
            } catch (e) {
                console.error('处理自动重启逻辑失败:', e);
            }
        }
    }, 200);
}

// 保存配置
async function saveConfig(autoSave = false) {
    const t = window.i18n ? window.i18n.t : (key) => key;

    try {
        // 确定实际的 API 类型（如果是 OpenRouter 且启用了流式模式，使用 openrouter_streaming）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                source_language: document.getElementById('source-language').value,
                show_partial_results: document.getElementById('show-partial-results').checked,
                enable_furigana: document.getElementById('enable-furigana').checked,
                enable_pinyin: document.getElementById('enable-pinyin').checked,
                enable_reverse_translation: document.getElementById('enable-reverse-translation').checked,
                show_original_and_lang_tag: document.getElementById('show-original-and-lang-tag')
                    ? document.getElementById('show-original-and-lang-tag').checked
                    : true,
            },
            mic_control: {
                enable_mic_control: document.getElementById('enable-mic-control').checked,
                mute_delay_seconds: parseFloat(document.getElementById('mute-delay').value),
                mic_device_index: (() => {
                    const v = document.getElementById('mic-device') ? document.getElementById('mic-device').value : '';
                    return v === '' ? null : parseInt(v);
                })(),
            },
            asr: {
                preferred_backend: document.getElementById('asr-backend').value,
                enable_hot_words: document.getElementById('enable-hot-words').checked,
                enable_vad: document.getElementById('enable-vad').checked,
                vad_threshold: parseFloat(document.getElementById('vad-threshold').value),
                vad_silence_duration_ms: parseInt(document.getElementById('vad-silence-duration').value),
                keepalive_interval: parseInt(document.getElementById('keepalive-interval').value),
            },
            language_detector: {
                type: document.getElementById('language-detector').value,
            }
        };

        const response = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config),
        });

        const result = await response.json();

        if (result.success) {
            if (!autoSave) {
                showMessage(t('msg.configSaved'), 'success');
            }
        } else {
            const localizedMsg = localizeBackendMessage(result.message_id, result.message);
            showMessage(t('msg.saveConfigFailed') + ': ' + localizedMsg, 'error');
        }
    } catch (error) {
        console.error('保存配置失败:', error);
        if (!autoSave) {
            showMessage(t('msg.saveConfigFailed'), 'error');
        }
    }
}

// 更新服务状态
async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();

        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');

        const t = window.i18n ? window.i18n.t : (key) => key;

        if (status.running) {
            statusText.textContent = t('status.running');
            statusDot.classList.add('running');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            statusText.textContent = t('status.notRunning');
            statusDot.classList.remove('running');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    } catch (error) {
        console.error('更新状态失败:', error);
    }
}

// 启动服务
async function startService() {
    const startBtn = document.getElementById('start-btn');
    const t = window.i18n ? window.i18n.t : (key) => key;

    startBtn.disabled = true;
    startBtn.textContent = t('btn.starting');
    pendingWarningMessage = null;

    try {
        // 先检查 DashScope API Key
        const dashscopeKey = document.getElementById('dashscope-api-key').value.trim();

        if (!dashscopeKey) {
            showMessage('❌ ' + t('msg.dashscopeRequired'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');

            // 展开API Keys配置区域，提示用户输入
            const apiKeysSection = document.getElementById('api-keys');
            if (apiKeysSection.classList.contains('collapsed')) {
                toggleCollapsible('api-keys');
            }

            // 高亮并震动 API Key 输入框
            highlightAPIKeyInput('dashscope-api-key');

            return;
        }

        // 验证 DashScope API Key格式
        const checkDashscopeResponse = await fetch(`${API_BASE}/check-api-key`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_key: dashscopeKey }),
        });

        const checkDashscopeResult = await checkDashscopeResponse.json();

        if (!checkDashscopeResult.valid) {
            // 后端返回消息ID，需要本地化
            const localizedMsg = localizeBackendMessage(checkDashscopeResult.message_id, checkDashscopeResult.message);
            showMessage('❌ ' + t('msg.dashscopeValidationFailed') + localizedMsg, 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');

            // 高亮并震动 API Key 输入框
            highlightAPIKeyInput('dashscope-api-key');

            return;
        }

        // 检查翻译API所需的 Key 是否齐全
        const enableTranslation = document.getElementById('enable-translation').checked;
        const translationApiSelect = document.getElementById('translation-api-type');
        let translationApiType = translationApiSelect.value;
        const deeplKey = document.getElementById('deepl-api-key').value.trim();
        const openrouterKey = document.getElementById('openrouter-api-key').value.trim();

        if (enableTranslation) {
            const requiresDeeplKey = translationApiType === 'deepl' && !deeplKey;
            const requiresOpenrouterKey = translationApiType === 'openrouter' && !openrouterKey;

            if (requiresDeeplKey || requiresOpenrouterKey) {
                translationApiType = 'google_dictionary';
                translationApiSelect.value = translationApiType;
                translationApiSelect.dispatchEvent(new Event('change'));
                saveConfigToLocalStorage();
                pendingWarningMessage = t('msg.autoSwitchToGoogle');
                showMessage('⚠️ ' + pendingWarningMessage, 'warning');
            }
        }

        // 所有检查通过，先同步配置到服务器
        try {
            await saveConfig(true); // autoSave = true，不显示成功消息
            console.log('✓ 配置已同步到服务器');
        } catch (error) {
            console.error('同步配置失败:', error);
            showMessage('❌ ' + t('msg.syncConfigFailed'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');
            return;
        }

        // 准备 API Keys
        const sonioxKey = document.getElementById('soniox-api-key').value.trim();
        const apiKeys = {
            dashscope: dashscopeKey,
            deepl: deeplKey,
            openrouter: openrouterKey,
            soniox: sonioxKey
        };

        // 检查 Soniox 后端是否需要 API Key
        const asrBackend = document.getElementById('asr-backend').value;
        if (asrBackend === 'soniox' && !sonioxKey) {
            showMessage('❌ ' + t('msg.sonioxKeyRequired'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');
            highlightAPIKeyInput('soniox-api-key');
            return;
        }

        // 启动服务
        const response = await fetch(`${API_BASE}/service/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_keys: apiKeys }),
        });

        const result = await response.json();

        if (result.success) {
            if (pendingWarningMessage) {
                showMessage('⚠️ ' + pendingWarningMessage + ' ' + t('msg.serviceStartSuccess'), 'warning');
                pendingWarningMessage = null;
            } else {
                showMessage('✅ ' + t('msg.serviceStartSuccess'), 'success');
            }
            setTimeout(updateStatus, 500);
        } else {
            const localizedMsg = localizeBackendMessage(result.message_id, result.message);
            showMessage('❌ ' + t('msg.serviceStartFailed') + localizedMsg, 'error');
            startBtn.disabled = false;
        }
    } catch (error) {
        console.error('启动服务失败:', error);
        showMessage('❌ ' + t('msg.startServiceFailed'), 'error');
        startBtn.disabled = false;
    } finally {
        pendingWarningMessage = null;
        startBtn.textContent = t('btn.startService');
    }
}

// 高亮 API Key 输入框
function highlightAPIKeyInput(inputId) {
    const apiKeyInput = document.getElementById(inputId);
    if (!apiKeyInput) return;

    apiKeyInput.classList.add('error-highlight');
    apiKeyInput.focus();

    // 3秒后移除高亮
    setTimeout(() => {
        apiKeyInput.classList.remove('error-highlight');
    }, 3000);
}

// 停止服务
async function stopService() {
    const stopBtn = document.getElementById('stop-btn');
    const t = window.i18n ? window.i18n.t : (key) => key;

    stopBtn.disabled = true;
    stopBtn.textContent = t('btn.stopping');

    try {
        const response = await fetch(`${API_BASE}/service/stop`, {
            method: 'POST',
        });

        const result = await response.json();

        if (result.success) {
            showMessage(t('msg.serviceStopSuccess'), 'success');
            setTimeout(updateStatus, 500);
        } else {
            const localizedMsg = localizeBackendMessage(result.message_id, result.message);
            showMessage(t('msg.serviceStopFailed') + localizedMsg, 'error');
            stopBtn.disabled = false;
        }
    } catch (error) {
        console.error('停止服务失败:', error);
        showMessage(t('msg.stopServiceFailed'), 'error');
        stopBtn.disabled = false;
    } finally {
        stopBtn.textContent = t('btn.stopService');
    }
}

// 重启服务
async function restartService() {
    try {
        const response = await fetch(`${API_BASE}/service/restart`, {
            method: 'POST',
        });

        const result = await response.json();

        if (result.success) {
            console.log('服务已重启');
            setTimeout(updateStatus, 500);
        }
    } catch (error) {
        console.error('重启服务失败:', error);
    }
}

// 恢复默认设置
async function resetToDefaults() {
    const t = window.i18n ? window.i18n.t : (key) => key;

    if (!confirm(t('msg.confirmReset'))) {
        return;
    }

    try {
        // 使用前端默认配置
        loadDefaultConfig();

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();

        // 先保存到本地浏览器
        saveConfigToLocalStorage();

        // 再保存到服务器
        await saveConfig();
        showMessage('✅ ' + t('msg.defaultsRestored'), 'success');

        // 如果服务正在运行，重启
        const statusResponse = await fetch(`${API_BASE}/status`);
        const status = await statusResponse.json();
        if (status.running) {
            await restartService();
        }
    } catch (error) {
        console.error('恢复默认设置失败:', error);
        showMessage(t('msg.restoreDefaultsFailed'), 'error');
    }
}

// 显示消息
function showMessage(text, type) {
    const messageEl = document.getElementById('message');
    messageEl.textContent = text;
    messageEl.className = 'message ' + type;

    // 5秒后自动隐藏
    setTimeout(() => {
        messageEl.className = 'message';
    }, 5000);
}

// 显示本地化消息（使用消息ID）
function showLocalizedMessage(messageId, type, params = {}) {
    const text = window.i18n ? window.i18n.t(messageId, params) : messageId;
    showMessage(text, type);
}

// 本地化后端消息
// 后端返回 message_id（消息ID）和 message（默认消息）
// 前端根据 message_id 获取本地化文本，如果没有则使用默认消息
function localizeBackendMessage(messageId, defaultMessage) {
    if (!messageId) {
        return defaultMessage || '';
    }
    const t = window.i18n ? window.i18n.t : (key) => key;
    const localized = t(messageId);
    // 如果翻译结果和 key 相同，说明没有找到翻译，使用默认消息
    return localized !== messageId ? localized : (defaultMessage || messageId);
}

// 折叠/展开面板
function toggleCollapsible(id) {
    const content = document.getElementById(id);
    const icon = document.getElementById(id + '-icon');

    content.classList.toggle('collapsed');
    icon.classList.toggle('collapsed');

    // 更新图标
    if (content.classList.contains('collapsed')) {
        icon.textContent = '▶';
    } else {
        icon.textContent = '▼';
    }
}
