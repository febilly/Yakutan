/**
 * 国际化 (i18n) 模块
 * 支持界面语言切换，设计为可扩展结构以便后续添加更多语言
 */

// 支持的语言列表
const SUPPORTED_LANGUAGES = {
    'zh-CN': '简体中文',
    'en': 'English'
};

// 默认语言
const DEFAULT_LANGUAGE = 'zh-CN';

// 本地存储键名
const LANGUAGE_STORAGE_KEY = 'ui_language';

// 当前语言
let currentLanguage = DEFAULT_LANGUAGE;

// 翻译文本
const translations = {
    'zh-CN': {
        // 页面标题和头部
        'page.title': 'Yakutan 控制面板',
        'header.title': '🎤 Yakutan 控制面板',
        'status.notRunning': '服务未运行',
        'status.running': '服务运行中',

        // 服务控制
        'section.serviceControl': '服务控制',
        'btn.startService': '启动服务',
        'btn.stopService': '停止服务',
        'btn.resetDefaults': '恢复默认设置',
        'hint.autoSave': '所有配置将自动保存在浏览器本地',
        'btn.starting': '启动中...',
        'btn.stopping': '停止中...',

        // 基本设置
        'section.basicSettings': '基本设置',
        'label.enableTranslation': '启用翻译',
        'label.showPartialResults': '输出中间结果',
        'hint.partialResults': '不推荐在开启翻译时使用',
        'label.targetLanguage': '目标语言',
        'hint.targetLanguage': '可直接输入语言代码，或从下拉列表快速选择',
        'label.fallbackLanguage': '备用语言（当源语言与目标语言相同时使用）',
        'hint.fallbackLanguage': '可直接输入语言代码，留空则禁用备用语言',
        'label.enableFurigana': '日语译文添加假名',
        'hint.enableFurigana': '仅在目标语言为日语时可用',
        'select.quickSelect': '-- 快速选择 --',
        'select.disabled': '禁用',

        // 语言选项
        'lang.zhCN': '简体中文 (zh-CN)',
        'lang.zhTW': '繁体中文 (zh-TW)',
        'lang.en': '英语 (en)',
        'lang.enGB': '英语（英式） (en-GB)',
        'lang.ja': '日语 (ja)',
        'lang.ko': '韩语 (ko)',
        'lang.es': '西班牙语 (es)',
        'lang.fr': '法语 (fr)',
        'lang.de': '德语 (de)',
        'lang.ru': '俄语 (ru)',
        'lang.ar': '阿拉伯语 (ar)',
        'lang.pt': '葡萄牙语 (pt)',
        'lang.it': '意大利语 (it)',

        // 翻译API设置
        'section.translationApi': '翻译API设置',
        'label.translationApi': '翻译API',
        'api.qwenMt': 'Qwen-MT（阿里云，使用 DashScope Key）',
        'api.deepl': 'DeepL（高质量）',
        'api.googleDict': 'Google Dictionary（免费，更快，请注意网络连通性）',
        'api.googleWeb': 'Google Web（免费，备用，请注意网络连通性）',
        'api.openrouter': 'OpenRouter（LLM）',
        'label.streamingMode': '流式翻译模式',
        'hint.streamingMode': '启用后支持翻译部分结果（实时翻译未完成的句子）',
        'label.reverseTranslation': '启用反向翻译',
        'hint.reverseTranslation': '总是使用 Google Dictionary API，请注意网络连通性',

        // API Keys配置
        'section.apiKeys': 'API Keys 配置',
        'label.dashscopeKey': '阿里云 DashScope API Key',
        'label.required': '*必需',
        'label.international': '国际版',
        'hint.dashscopeKey': 'Qwen 和 FunASR 语音识别均需要此 Key。',
        'link.getChinaKey': '获取中国大陆版API Key',
        'link.getIntlKey': '获取国际版API Key',
        'label.deeplKey': 'DeepL API Key (可选，用于翻译)',
        'link.getApiKey': '获取API Key →',
        'label.openrouterKey': 'OpenRouter API Key (可选，用于LLM翻译)',

        // 语音识别设置
        'section.asrSettings': '语音识别设置',
        'label.asrBackend': '识别后端',
        'asr.qwen': 'Qwen3 ASR（推荐）',
        'asr.dashscope': 'Fun-ASR（仅中国大陆版可用）',
        'asr.dashscopeDisabled': 'Fun-ASR（国际版不可用）',
        'label.pauseOnMute': '游戏静音时暂停转录',
        'hint.pauseOnMute': '第一次解除静音后开始转录',
        'label.enableHotWords': '启用热词',
        'hint.enableHotWords': '提高特定词汇的识别准确度',
        'label.muteDelay': '静音延迟（秒）',
        'hint.muteDelay': '静音后延迟停止识别的时间，防止漏掉最后一个字',

        // 高级设置
        'section.advancedSettings': '高级设置',
        'subsection.vad': 'VAD（语音活动检测）设置 - 仅Qwen后端',
        'label.enableVad': '启用VAD',
        'hint.enableVad': '自动检测语音结束并断句',
        'label.vadThreshold': 'VAD阈值（0.0-1.0）',
        'hint.vadThreshold': '值越小越敏感，越容易触发断句',
        'label.vadSilenceDuration': 'VAD静音持续时间（毫秒）',
        'hint.vadSilenceDuration': '检测到此时长的静音后触发断句',
        'subsection.websocket': 'WebSocket保活设置 - 仅Qwen后端',
        'label.keepaliveInterval': '心跳间隔（秒）',
        'hint.keepaliveInterval': '防止长时间闲置导致连接超时，设置为0禁用',
        'subsection.langDetector': '语言检测器设置',
        'label.detectorType': '检测器类型',
        'detector.cjke': '中日韩英检测器（推荐）',
        'detector.enzh': '中英检测器',
        'detector.fasttext': '通用检测器（支持更多语言）',
        'subsection.sourceLang': '源语言设置',
        'label.sourceLanguage': '源语言',
        'sourceLang.auto': '自动检测',
        'sourceLang.zh': '中文',
        'sourceLang.en': '英语',
        'sourceLang.ja': '日语',
        'sourceLang.ko': '韩语',
        'hint.sourceLanguage': '建议保持"自动检测"',

        // 页脚
        'footer.text': 'Yakutan',

        // 消息 - 来自后端的消息ID
        'msg.configUpdated': '配置已更新',
        'msg.configUpdateFailed': '配置更新失败',
        'msg.serviceAlreadyRunning': '服务已在运行中',
        'msg.serviceStarted': '服务已启动',
        'msg.startFailed': '启动失败',
        'msg.serviceNotRunning': '服务未运行',
        'msg.serviceStopped': '服务已停止',
        'msg.stopFailed': '停止失败',
        'msg.noRestartNeeded': '服务未运行，无需重启',
        'msg.serviceRestarted': '服务已重启',
        'msg.restartFailed': '重启失败',
        'msg.enterDashscopeKey': '请输入 DashScope API Key',
        'msg.invalidKeyFormat': 'API Key 格式无效（应以 sk- 开头）',
        'msg.replacePlaceholder': '请替换占位符为真实的 API Key',
        'msg.keyFormatValid': 'API Key 格式有效',
        'msg.checkFailed': '检查失败',

        // 前端消息
        'msg.configSaved': '配置保存成功！',
        'msg.saveConfigFailed': '保存配置失败',
        'msg.dashscopeRequired': '错误：必须配置阿里云 DashScope API Key 才能启动服务！',
        'msg.dashscopeValidationFailed': 'DashScope API Key 验证失败: ',
        'msg.syncConfigFailed': '同步配置失败，无法启动服务',
        'msg.serviceStartSuccess': '服务启动成功',
        'msg.serviceStartFailed': '服务启动失败: ',
        'msg.startServiceFailed': '启动服务失败',
        'msg.serviceStopSuccess': '服务停止成功',
        'msg.serviceStopFailed': '服务停止失败: ',
        'msg.stopServiceFailed': '停止服务失败',
        'msg.defaultsRestored': '已恢复默认设置',
        'msg.restoreDefaultsFailed': '恢复默认设置失败',
        'msg.confirmReset': '确定要恢复默认设置吗？（API Keys将被保留）',
        'msg.apiKeyRequired': '使用 {api} 需要配置 API Key，请先在"API Keys 配置"中填写',
        'msg.autoSwitchToGoogle': '未检测到所选翻译接口的 API Key，已自动切换为 Google Dictionary。',

        // 语言选择器
        'label.uiLanguage': '界面语言'
    },

    'en': {
        // Page title and header
        'page.title': 'Yakutan Control Panel',
        'header.title': '🎤 Yakutan Control Panel',
        'status.notRunning': 'Service Not Running',
        'status.running': 'Service Running',

        // Service control
        'section.serviceControl': 'Service Control',
        'btn.startService': 'Start Service',
        'btn.stopService': 'Stop Service',
        'btn.resetDefaults': 'Reset to Defaults',
        'hint.autoSave': 'All settings are automatically saved in the browser',
        'btn.starting': 'Starting...',
        'btn.stopping': 'Stopping...',

        // Basic settings
        'section.basicSettings': 'Basic Settings',
        'label.enableTranslation': 'Enable Translation',
        'label.showPartialResults': 'Show Partial Results',
        'hint.partialResults': 'Not recommended when translation is enabled',
        'label.targetLanguage': 'Target Language',
        'hint.targetLanguage': 'Enter language code directly or select from the dropdown',
        'label.fallbackLanguage': 'Fallback Language (used when source equals target)',
        'hint.fallbackLanguage': 'Enter language code directly, leave empty to disable',
        'label.enableFurigana': 'Add furigana to Japanese output',
        'hint.enableFurigana': 'Available only when target language is Japanese',
        'select.quickSelect': '-- Quick Select --',
        'select.disabled': 'Disabled',

        // Language options
        'lang.zhCN': 'Simplified Chinese (zh-CN)',
        'lang.zhTW': 'Traditional Chinese (zh-TW)',
        'lang.en': 'English (en)',
        'lang.enGB': 'British English (en-GB)',
        'lang.ja': 'Japanese (ja)',
        'lang.ko': 'Korean (ko)',
        'lang.es': 'Spanish (es)',
        'lang.fr': 'French (fr)',
        'lang.de': 'German (de)',
        'lang.ru': 'Russian (ru)',
        'lang.ar': 'Arabic (ar)',
        'lang.pt': 'Portuguese (pt)',
        'lang.it': 'Italian (it)',

        // Translation API settings
        'section.translationApi': 'Translation API Settings',
        'label.translationApi': 'Translation API',
        'api.qwenMt': 'Qwen-MT (Alibaba Cloud, uses DashScope Key)',
        'api.deepl': 'DeepL (High Quality)',
        'api.googleDict': 'Google Dictionary (Free, Faster, check network connectivity)',
        'api.googleWeb': 'Google Web (Free, Backup, check network connectivity)',
        'api.openrouter': 'OpenRouter (LLM)',
        'label.streamingMode': 'Streaming Translation Mode',
        'hint.streamingMode': 'Enable to translate partial results in real-time',
        'label.reverseTranslation': 'Enable Reverse Translation',
        'hint.reverseTranslation': 'Always uses Google Dictionary API, check network connectivity',

        // API Keys configuration
        'section.apiKeys': 'API Keys Configuration',
        'label.dashscopeKey': 'Alibaba Cloud DashScope API Key',
        'label.required': '*Required',
        'label.international': 'International',
        'hint.dashscopeKey': 'Required for both Qwen and FunASR speech recognition.',
        'link.getChinaKey': 'Get China Mainland API Key',
        'link.getIntlKey': 'Get International API Key',
        'label.deeplKey': 'DeepL API Key (optional, for translation)',
        'link.getApiKey': 'Get API Key →',
        'label.openrouterKey': 'OpenRouter API Key (optional, for LLM translation)',

        // Speech recognition settings
        'section.asrSettings': 'Speech Recognition Settings',
        'label.asrBackend': 'Recognition Backend',
        'asr.qwen': 'Qwen3 ASR (Recommended)',
        'asr.dashscope': 'Fun-ASR (China Mainland only)',
        'asr.dashscopeDisabled': 'Fun-ASR (Not available for International)',
        'label.pauseOnMute': 'Pause transcription when muted in game',
        'hint.pauseOnMute': 'Starts transcription after first unmute',
        'label.enableHotWords': 'Enable Hot Words',
        'hint.enableHotWords': 'Improves recognition accuracy for specific words',
        'label.muteDelay': 'Mute Delay (seconds)',
        'hint.muteDelay': 'Delay before stopping recognition after mute, prevents missing last word',

        // Advanced settings
        'section.advancedSettings': 'Advanced Settings',
        'subsection.vad': 'VAD (Voice Activity Detection) Settings - Qwen backend only',
        'label.enableVad': 'Enable VAD',
        'hint.enableVad': 'Automatically detect end of speech and segment',
        'label.vadThreshold': 'VAD Threshold (0.0-1.0)',
        'hint.vadThreshold': 'Lower values are more sensitive, easier to trigger segmentation',
        'label.vadSilenceDuration': 'VAD Silence Duration (ms)',
        'hint.vadSilenceDuration': 'Triggers segmentation after this duration of silence',
        'subsection.websocket': 'WebSocket Keep-alive Settings - Qwen backend only',
        'label.keepaliveInterval': 'Heartbeat Interval (seconds)',
        'hint.keepaliveInterval': 'Prevents connection timeout during idle, set to 0 to disable',
        'subsection.langDetector': 'Language Detector Settings',
        'label.detectorType': 'Detector Type',
        'detector.cjke': 'CJK-English Detector (Recommended)',
        'detector.enzh': 'English-Chinese Detector',
        'detector.fasttext': 'Universal Detector (supports more languages)',
        'subsection.sourceLang': 'Source Language Settings',
        'label.sourceLanguage': 'Source Language',
        'sourceLang.auto': 'Auto Detect',
        'sourceLang.zh': 'Chinese',
        'sourceLang.en': 'English',
        'sourceLang.ja': 'Japanese',
        'sourceLang.ko': 'Korean',
        'hint.sourceLanguage': 'Recommended to keep "Auto Detect"',

        // Footer
        'footer.text': 'Yakutan',

        // Messages - Backend message IDs
        'msg.configUpdated': 'Configuration updated',
        'msg.configUpdateFailed': 'Configuration update failed',
        'msg.serviceAlreadyRunning': 'Service is already running',
        'msg.serviceStarted': 'Service started',
        'msg.startFailed': 'Start failed',
        'msg.serviceNotRunning': 'Service is not running',
        'msg.serviceStopped': 'Service stopped',
        'msg.stopFailed': 'Stop failed',
        'msg.noRestartNeeded': 'Service is not running, no restart needed',
        'msg.serviceRestarted': 'Service restarted',
        'msg.restartFailed': 'Restart failed',
        'msg.enterDashscopeKey': 'Please enter DashScope API Key',
        'msg.invalidKeyFormat': 'Invalid API Key format (should start with sk-)',
        'msg.replacePlaceholder': 'Please replace the placeholder with a real API Key',
        'msg.keyFormatValid': 'API Key format is valid',
        'msg.checkFailed': 'Check failed',

        // Frontend messages
        'msg.configSaved': 'Configuration saved successfully!',
        'msg.saveConfigFailed': 'Failed to save configuration',
        'msg.dashscopeRequired': 'Error: Alibaba Cloud DashScope API Key is required to start the service!',
        'msg.dashscopeValidationFailed': 'DashScope API Key validation failed: ',
        'msg.syncConfigFailed': 'Failed to sync configuration, cannot start service',
        'msg.serviceStartSuccess': 'Service started successfully',
        'msg.serviceStartFailed': 'Service start failed: ',
        'msg.startServiceFailed': 'Failed to start service',
        'msg.serviceStopSuccess': 'Service stopped successfully',
        'msg.serviceStopFailed': 'Service stop failed: ',
        'msg.stopServiceFailed': 'Failed to stop service',
        'msg.defaultsRestored': 'Defaults restored',
        'msg.restoreDefaultsFailed': 'Failed to restore defaults',
        'msg.confirmReset': 'Are you sure you want to reset to defaults? (API Keys will be preserved)',
        'msg.apiKeyRequired': 'API Key is required for {api}, please fill it in "API Keys Configuration" first',
        'msg.autoSwitchToGoogle': 'API Key for selected translation API not found, automatically switched to Google Dictionary.',

        // Language selector
        'label.uiLanguage': 'UI Language'
    }
};

/**
 * 获取当前语言
 * @returns {string} 当前语言代码
 */
function getCurrentLanguage() {
    return currentLanguage;
}

/**
 * 设置当前语言
 * @param {string} lang - 语言代码
 */
function setLanguage(lang) {
    if (SUPPORTED_LANGUAGES[lang]) {
        currentLanguage = lang;
        localStorage.setItem(LANGUAGE_STORAGE_KEY, lang);
        applyTranslations();
        // 更新页面标题
        document.title = t('page.title');
    }
}

/**
 * 从本地存储加载语言设置
 */
function loadLanguageFromStorage() {
    const savedLang = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (savedLang && SUPPORTED_LANGUAGES[savedLang]) {
        currentLanguage = savedLang;
    } else {
        // 尝试检测浏览器语言
        const browserLang = navigator.language || navigator.userLanguage;
        if (browserLang) {
            // 先尝试精确匹配
            if (SUPPORTED_LANGUAGES[browserLang]) {
                currentLanguage = browserLang;
            } else {
                // 尝试匹配语言前缀
                const langPrefix = browserLang.split('-')[0];
                if (langPrefix === 'zh') {
                    currentLanguage = 'zh-CN';
                } else if (langPrefix === 'en') {
                    currentLanguage = 'en';
                }
            }
        }
    }
}

/**
 * 翻译文本
 * @param {string} key - 翻译键
 * @param {Object} params - 可选的替换参数
 * @returns {string} 翻译后的文本
 */
function t(key, params = {}) {
    const langData = translations[currentLanguage] || translations[DEFAULT_LANGUAGE];
    let text = langData[key] || translations[DEFAULT_LANGUAGE][key] || key;
    
    // 替换参数 {param}
    for (const [paramKey, paramValue] of Object.entries(params)) {
        text = text.replace(new RegExp(`\\{${paramKey}\\}`, 'g'), paramValue);
    }
    
    return text;
}

/**
 * 根据后端消息ID获取本地化消息
 * @param {string} messageId - 后端消息ID
 * @param {Object} params - 可选的替换参数
 * @returns {string} 本地化后的消息
 */
function localizeMessage(messageId, params = {}) {
    return t(messageId, params);
}

/**
 * 应用翻译到页面元素
 */
function applyTranslations() {
    // 更新所有带有 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (key) {
            el.textContent = t(key);
        }
    });

    // 更新所有带有 data-i18n-placeholder 属性的元素
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (key) {
            el.placeholder = t(key);
        }
    });

    // 更新所有带有 data-i18n-title 属性的元素
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        if (key) {
            el.title = t(key);
        }
    });

    // 更新语言选择器的当前值
    const langSelector = document.getElementById('language-selector');
    if (langSelector) {
        langSelector.value = currentLanguage;
    }
}

/**
 * 初始化语言选择器
 */
function initLanguageSelector() {
    const selector = document.getElementById('language-selector');
    if (!selector) return;

    // 清空并填充选项
    selector.innerHTML = '';
    for (const [code, name] of Object.entries(SUPPORTED_LANGUAGES)) {
        const option = document.createElement('option');
        option.value = code;
        option.textContent = name;
        selector.appendChild(option);
    }

    // 设置当前值
    selector.value = currentLanguage;

    // 添加变更事件
    selector.addEventListener('change', (e) => {
        setLanguage(e.target.value);
    });
}

/**
 * 初始化i18n模块
 */
function initI18n() {
    loadLanguageFromStorage();
    initLanguageSelector();
    applyTranslations();
    document.title = t('page.title');
}

// 导出函数供其他模块使用
window.i18n = {
    t,
    localizeMessage,
    getCurrentLanguage,
    setLanguage,
    applyTranslations,
    initI18n,
    SUPPORTED_LANGUAGES
};
