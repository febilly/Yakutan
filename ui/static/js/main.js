// API基础URL
const API_BASE = '/api';

// 自动保存定时器
let autoSaveTimer = null;

// 配置键名
const CONFIG_STORAGE_KEY = 'vrchat_translator_config';

// 待显示的警告消息（用于自动切换翻译API）
let pendingWarningMessage = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    loadConfigFromLocalStorage();
    loadAPIKeys();
    updateStatus();
    // 每2秒更新一次状态
    setInterval(updateStatus, 2000);
    
    // 显示配置保存提示
    showConfigStorageInfo();
});

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
    
    if (dashscopeKey) document.getElementById('dashscope-api-key').value = dashscopeKey;
    if (deeplKey) document.getElementById('deepl-api-key').value = deeplKey;
    if (openrouterKey) document.getElementById('openrouter-api-key').value = openrouterKey;
    
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
                document.getElementById('translation-api-type').value = config.translation.api_type || 'deepl';
                document.getElementById('source-language').value = config.translation.source_language || 'auto';
                document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
                document.getElementById('enable-reverse-translation').checked = config.translation.enable_reverse_translation ?? true;
                document.getElementById('translate-partial-results').checked = config.translation.translate_partial_results ?? false;
            }
            
            if (config.mic_control) {
                document.getElementById('enable-mic-control').checked = config.mic_control.enable_mic_control ?? true;
                document.getElementById('mute-delay').value = config.mic_control.mute_delay_seconds || 0.2;
            }
            
            if (config.asr) {
                document.getElementById('asr-backend').value = config.asr.preferred_backend || 'qwen';
                document.getElementById('enable-hot-words').checked = config.asr.enable_hot_words ?? true;
                document.getElementById('enable-vad').checked = config.asr.enable_vad ?? true;
                document.getElementById('vad-threshold').value = config.asr.vad_threshold || 0.2;
                document.getElementById('vad-silence-duration').value = config.asr.vad_silence_duration_ms || 800;
                document.getElementById('keepalive-interval').value = config.asr.keepalive_interval || 30;
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
        
    } catch (error) {
        console.error('加载本地配置失败:', error);
        // 出错时使用前端默认值
        loadDefaultConfig();
        toggleTranslationOptions();
    }
}

// 加载前端默认配置（不依赖服务器）
function loadDefaultConfig() {
    // 翻译配置
    document.getElementById('enable-translation').checked = true;
    document.getElementById('target-language').value = 'ja';
    document.getElementById('fallback-language').value = 'en';
    document.getElementById('translation-api-type').value = 'deepl';
    document.getElementById('source-language').value = 'auto';
    document.getElementById('show-partial-results').checked = false;
    document.getElementById('enable-reverse-translation').checked = true;
    document.getElementById('translate-partial-results').checked = false;
    
    // 麦克风控制
    document.getElementById('enable-mic-control').checked = true;
    document.getElementById('mute-delay').value = 0.2;
    
    // ASR 配置
    document.getElementById('asr-backend').value = 'qwen';
    document.getElementById('enable-hot-words').checked = true;
    document.getElementById('enable-vad').checked = true;
    document.getElementById('vad-threshold').value = 0.2;
    document.getElementById('vad-silence-duration').value = 800;
    document.getElementById('keepalive-interval').value = 30;
    
    // 语言检测器
    document.getElementById('language-detector').value = 'cjke';
    
    console.log('✓ 已加载前端默认配置');
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
        document.getElementById('translation-api-type').value = config.translation.api_type;
        document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
        document.getElementById('translate-partial-results').checked = config.translation.translate_partial_results ?? false;
        
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
        
        console.log('已从服务器加载配置');
        
    } catch (error) {
        console.error('加载服务器配置失败:', error);
    }
}

// 保存配置到 localStorage
function saveConfigToLocalStorage() {
    try {
        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: document.getElementById('translation-api-type').value,
                source_language: document.getElementById('source-language').value,
                show_partial_results: document.getElementById('show-partial-results').checked,
                enable_reverse_translation: document.getElementById('enable-reverse-translation').checked,
                translate_partial_results: document.getElementById('translate-partial-results').checked,
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

// 处理翻译API变更
let previousTranslationApi = null;

function handleTranslationApiChange(event) {
    const newApi = event.target.value;
    const warningElement = document.getElementById('translation-api-warning');
    const partialResultsGroup = document.getElementById('translate-partial-results-group');
    
    // 检查是否需要API Key
    let requiresKey = false;
    let keyName = '';
    let keyInputId = '';
    let apiDisplayName = '';
    
    // 显示/隐藏流式翻译选项
    if (newApi === 'openrouter_streaming') {
        partialResultsGroup.style.display = 'block';
    } else {
        partialResultsGroup.style.display = 'none';
        // 如果切换到不支持流式的API，自动关闭流式翻译开关
        document.getElementById('translate-partial-results').checked = false;
    }

    if (newApi === 'deepl') {
        requiresKey = true;
        keyName = 'deepl_api_key';
        keyInputId = 'deepl-api-key';
        apiDisplayName = 'DeepL';
    } else if (newApi === 'openrouter' || newApi === 'openrouter_streaming') {
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
            warningElement.textContent = `⚠️ 使用 ${apiDisplayName} 需要配置 API Key，请先在"API Keys 配置"中填写`;
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
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        const apiSelect = document.getElementById('translation-api-type');
        previousTranslationApi = apiSelect.value;
        
        // 初始化流式翻译选项的显示状态
        const partialResultsGroup = document.getElementById('translate-partial-results-group');
        if (apiSelect.value === 'openrouter_streaming') {
            partialResultsGroup.style.display = 'block';
        } else {
            partialResultsGroup.style.display = 'none';
        }
    }, 100);
});

// 当设置改变时自动保存（延迟保存，避免频繁请求）
function onSettingChange() {
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
                // 通过当前活动元素判断此次修改属于哪个配置项
                const el = document.activeElement;
                
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
    try {
        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: document.getElementById('translation-api-type').value,
                source_language: document.getElementById('source-language').value,
                show_partial_results: document.getElementById('show-partial-results').checked,
                translate_partial_results: document.getElementById('translate-partial-results').checked,
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
                showMessage('配置保存成功！', 'success');
            }
        } else {
            showMessage('配置保存失败: ' + result.message, 'error');
        }
    } catch (error) {
        console.error('保存配置失败:', error);
        if (!autoSave) {
            showMessage('保存配置失败', 'error');
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
        
        if (status.running) {
            statusText.textContent = '服务运行中';
            statusDot.classList.add('running');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            statusText.textContent = '服务未运行';
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
    startBtn.disabled = true;
    startBtn.textContent = '启动中...';
    pendingWarningMessage = null;
    
    try {
        // 先检查 DashScope API Key
        const dashscopeKey = document.getElementById('dashscope-api-key').value.trim();
        
        if (!dashscopeKey) {
            showMessage('❌ 错误：必须配置阿里云 DashScope API Key 才能启动服务！', 'error');
            startBtn.disabled = false;
            startBtn.textContent = '启动服务';
            
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
            showMessage('❌ DashScope API Key 验证失败: ' + checkDashscopeResult.message, 'error');
            startBtn.disabled = false;
            startBtn.textContent = '启动服务';
            
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
                pendingWarningMessage = '未检测到所选翻译接口的 API Key，已自动切换为 Google Dictionary。';
                showMessage(`⚠️ ${pendingWarningMessage}`, 'warning');
            }
        }
        
        // 所有检查通过，先同步配置到服务器
        try {
            await saveConfig(true); // autoSave = true，不显示成功消息
            console.log('✓ 配置已同步到服务器');
        } catch (error) {
            console.error('同步配置失败:', error);
            showMessage('❌ 同步配置失败，无法启动服务', 'error');
            startBtn.disabled = false;
            startBtn.textContent = '启动服务';
            return;
        }
        
        // 准备 API Keys
        const apiKeys = {
            dashscope: dashscopeKey,
            deepl: deeplKey,
            openrouter: openrouterKey
        };
        
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
                showMessage(`⚠️ ${pendingWarningMessage}服务已启动成功。`, 'warning');
                pendingWarningMessage = null;
            } else {
                showMessage('✅ 服务启动成功', 'success');
            }
            setTimeout(updateStatus, 500);
        } else {
            showMessage('❌ 服务启动失败: ' + result.message, 'error');
            startBtn.disabled = false;
        }
    } catch (error) {
        console.error('启动服务失败:', error);
        showMessage('❌ 启动服务失败', 'error');
        startBtn.disabled = false;
    } finally {
        pendingWarningMessage = null;
        startBtn.textContent = '启动服务';
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
    stopBtn.disabled = true;
    stopBtn.textContent = '停止中...';
    
    try {
        const response = await fetch(`${API_BASE}/service/stop`, {
            method: 'POST',
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMessage('服务停止成功', 'success');
            setTimeout(updateStatus, 500);
        } else {
            showMessage('服务停止失败: ' + result.message, 'error');
            stopBtn.disabled = false;
        }
    } catch (error) {
        console.error('停止服务失败:', error);
        showMessage('停止服务失败', 'error');
        stopBtn.disabled = false;
    } finally {
        stopBtn.textContent = '停止服务';
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
    if (!confirm('确定要恢复默认设置吗？（API Keys将被保留）')) {
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
        showMessage('✅ 已恢复默认设置', 'success');
        
        // 如果服务正在运行，重启
        const statusResponse = await fetch(`${API_BASE}/status`);
        const status = await statusResponse.json();
        if (status.running) {
            await restartService();
        }
    } catch (error) {
        console.error('恢复默认设置失败:', error);
        showMessage('恢复默认设置失败', 'error');
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
