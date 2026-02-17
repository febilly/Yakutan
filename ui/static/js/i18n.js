/**
 * 国际化 (i18n) 模块
 * 支持界面语言切换，设计为可扩展结构以便后续添加更多语言
 */

// 支持的语言列表
const SUPPORTED_LANGUAGES = {
    'zh-CN': '简体中文',
    'en': 'English',
    'ja': '日本語',
    'ko': '한국어'
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
        'label.enableFurigana': '日语添加假名',
        'hint.enableFurigana': '为日语文本的汉字标注假名读音',
        'label.enablePinyin': '中文添加拼音',
        'hint.enablePinyin': '为中文标注拼音（带声调）',
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
            'api.openrouterStreamingDeeplHybrid': 'OpenRouter 流式 + DeepL 静音终译（混合）',
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
        'hint.openrouterEnvLocked': '已从环境变量读取，输入框已锁定',
        'placeholder.openrouterEnvConfigured': '已在环境变量配置',

        // 语音识别设置
        'section.asrSettings': '语音识别设置',
        'label.asrBackend': '识别后端',
        'asr.qwen': 'Qwen3 ASR（推荐）',
        'asr.dashscope': 'Fun-ASR（仅中国大陆版可用）',
        'asr.dashscopeDisabled': 'Fun-ASR（国际版不可用）',
        'asr.soniox': 'Soniox（多语言，需要API Key）',
        'label.sonioxKey': 'Soniox API Key (可选，用于多语言识别)',
        'hint.sonioxKey': '支持60+语言的语音识别。',
        'label.pauseOnMute': '游戏静音时暂停转录',
        'hint.pauseOnMute': '第一次解除静音后开始转录',
        'label.enableHotWords': '启用热词',
        'hint.enableHotWords': '提高特定词汇的识别准确度',
        'label.muteDelay': '静音延迟（秒）',
        'hint.muteDelay': '静音后延迟停止识别的时间，防止漏掉最后一个字',

        // 高级设置
        'section.advancedSettings': '高级设置',
        'subsection.display': '显示设置',
        'label.showOriginalAndLangTag': '显示原文及语言标识',
        'hint.showOriginalAndLangTag': '关闭后只显示译文',
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
        'label.micDevice': '麦克风',
        'label.device': '设备',
        'option.systemDefault': '系统默认',

        // 页脚
        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

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
        'msg.sonioxKeyRequired': 'Soniox 后端需要配置 API Key',

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
        'label.enableFurigana': 'Add furigana to Japanese text',
        'hint.enableFurigana': 'Add hiragana readings to Japanese kanji',
        'label.enablePinyin': 'Add pinyin to Chinese text',
        'hint.enablePinyin': 'Add pinyin with tones to Chinese characters',
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
            'api.openrouterStreamingDeeplHybrid': 'OpenRouter Streaming + DeepL Mute Finalization (Hybrid)',
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
        'hint.openrouterEnvLocked': 'Loaded from environment variable; input is locked',
        'placeholder.openrouterEnvConfigured': 'Configured via environment variable',

        // Speech recognition settings
        'section.asrSettings': 'Speech Recognition Settings',
        'label.asrBackend': 'Recognition Backend',
        'asr.qwen': 'Qwen3 ASR (Recommended)',
        'asr.dashscope': 'Fun-ASR (China Mainland only)',
        'asr.dashscopeDisabled': 'Fun-ASR (Not available for International)',
        'asr.soniox': 'Soniox (Multilingual, requires API Key)',
        'label.sonioxKey': 'Soniox API Key (optional, for multilingual recognition)',
        'hint.sonioxKey': 'Supports 60+ languages for speech recognition.',
        'label.pauseOnMute': 'Pause transcription when muted in game',
        'hint.pauseOnMute': 'Starts transcription after first unmute',
        'label.enableHotWords': 'Enable Hot Words',
        'hint.enableHotWords': 'Improves recognition accuracy for specific words',
        'label.muteDelay': 'Mute Delay (seconds)',
        'hint.muteDelay': 'Delay before stopping recognition after mute, prevents missing last word',

        // Advanced settings
        'section.advancedSettings': 'Advanced Settings',
        'subsection.display': 'Display',
        'label.showOriginalAndLangTag': 'Show original text and language tag',
        'hint.showOriginalAndLangTag': 'When off, show translation only',
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
        'label.micDevice': 'Microphone',
        'label.device': 'Device',
        'option.systemDefault': 'System Default',

        // Footer
        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

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
        'msg.sonioxKeyRequired': 'Soniox backend requires API Key',

        // Language selector
        'label.uiLanguage': 'UI Language'
    },

    'ja': {
        'page.title': 'Yakutan コントロールパネル',
        'header.title': '🎤 Yakutan コントロールパネル',
        'status.notRunning': 'サービス停止中',
        'status.running': 'サービス稼働中',

        'section.serviceControl': 'サービス制御',
        'btn.startService': 'サービス開始',
        'btn.stopService': 'サービス停止',
        'btn.resetDefaults': 'デフォルトに戻す',
        'hint.autoSave': 'すべての設定はブラウザに自動保存されます',
        'btn.starting': '開始中...',
        'btn.stopping': '停止中...',

        'section.basicSettings': '基本設定',
        'label.enableTranslation': '翻訳を有効化',
        'label.showPartialResults': '途中結果を表示',
        'hint.partialResults': '翻訳有効時は非推奨です',
        'label.targetLanguage': '翻訳先言語',
        'hint.targetLanguage': '言語コードを直接入力するか、ドロップダウンから選択してください',
        'label.fallbackLanguage': 'フォールバック言語（原文と言語が同じ場合に使用）',
        'hint.fallbackLanguage': '言語コードを直接入力。空欄で無効化されます',
        'label.enableFurigana': '日本語にふりがなを追加',
        'hint.enableFurigana': '日本語テキストの漢字に読み仮名を付与します',
        'label.enablePinyin': '中国語にピンインを追加',
        'hint.enablePinyin': '中国語に声調付きピンインを付与します',
        'select.quickSelect': '-- クイック選択 --',
        'select.disabled': '無効',

        'lang.zhCN': '簡体字中国語 (zh-CN)',
        'lang.zhTW': '繁体字中国語 (zh-TW)',
        'lang.en': '英語 (en)',
        'lang.enGB': '英語（英国） (en-GB)',
        'lang.ja': '日本語 (ja)',
        'lang.ko': '韓国語 (ko)',
        'lang.es': 'スペイン語 (es)',
        'lang.fr': 'フランス語 (fr)',
        'lang.de': 'ドイツ語 (de)',
        'lang.ru': 'ロシア語 (ru)',
        'lang.ar': 'アラビア語 (ar)',
        'lang.pt': 'ポルトガル語 (pt)',
        'lang.it': 'イタリア語 (it)',

        'section.translationApi': '翻訳 API 設定',
        'label.translationApi': '翻訳 API',
        'api.qwenMt': 'Qwen-MT（Alibaba Cloud、DashScope Key を使用）',
        'api.deepl': 'DeepL（高品質）',
        'api.googleDict': 'Google Dictionary（無料・高速、ネットワーク接続に注意）',
        'api.googleWeb': 'Google Web（無料・予備、ネットワーク接続に注意）',
        'api.openrouter': 'OpenRouter（LLM）',
        'api.openrouterStreamingDeeplHybrid': 'OpenRouter ストリーミング + DeepL ミュート終訳（ハイブリッド）',
        'label.streamingMode': 'ストリーミング翻訳モード',
        'hint.streamingMode': '有効化すると途中結果をリアルタイム翻訳できます',
        'label.reverseTranslation': '逆翻訳を有効化',
        'hint.reverseTranslation': '常に Google Dictionary API を使用します。ネットワーク接続に注意してください',

        'section.apiKeys': 'API Keys 設定',
        'label.dashscopeKey': 'Alibaba Cloud DashScope API Key',
        'label.required': '*必須',
        'label.international': '国際版',
        'hint.dashscopeKey': 'Qwen と FunASR の音声認識の両方で必要です。',
        'link.getChinaKey': '中国本土版 API Key を取得',
        'link.getIntlKey': '国際版 API Key を取得',
        'label.deeplKey': 'DeepL API Key（任意、翻訳用）',
        'link.getApiKey': 'API Key を取得 →',
        'label.openrouterKey': 'OpenRouter API Key（任意、LLM 翻訳用）',
        'hint.openrouterEnvLocked': '環境変数から読み込み済みのため、入力欄はロックされています',
        'placeholder.openrouterEnvConfigured': '環境変数で設定済み',

        'section.asrSettings': '音声認識設定',
        'label.asrBackend': '認識バックエンド',
        'asr.qwen': 'Qwen3 ASR（推奨）',
        'asr.dashscope': 'Fun-ASR（中国本土版のみ）',
        'asr.dashscopeDisabled': 'Fun-ASR（国際版では利用不可）',
        'asr.soniox': 'Soniox（多言語、API Key が必要）',
        'label.sonioxKey': 'Soniox API Key（任意、多言語認識用）',
        'hint.sonioxKey': '60 以上の言語の音声認識に対応しています。',
        'label.pauseOnMute': 'ゲームでミュート中は文字起こしを停止',
        'hint.pauseOnMute': '最初にミュート解除した後に文字起こしを開始します',
        'label.enableHotWords': 'ホットワードを有効化',
        'hint.enableHotWords': '特定語彙の認識精度を向上させます',
        'label.muteDelay': 'ミュート遅延（秒）',
        'hint.muteDelay': 'ミュート後に認識停止まで待機し、最後の語句の欠落を防ぎます',

        'section.advancedSettings': '詳細設定',
        'subsection.display': '表示設定',
        'label.showOriginalAndLangTag': '原文と言語タグを表示',
        'hint.showOriginalAndLangTag': 'オフ時は翻訳文のみ表示します',
        'subsection.vad': 'VAD（音声区間検出）設定 - Qwen バックエンドのみ',
        'label.enableVad': 'VAD を有効化',
        'hint.enableVad': '発話終了を自動検出して区切ります',
        'label.vadThreshold': 'VAD しきい値（0.0-1.0）',
        'hint.vadThreshold': '小さいほど敏感になり、区切りが発生しやすくなります',
        'label.vadSilenceDuration': 'VAD 無音継続時間（ms）',
        'hint.vadSilenceDuration': 'この長さの無音を検出すると区切りを実行します',
        'subsection.websocket': 'WebSocket キープアライブ設定 - Qwen バックエンドのみ',
        'label.keepaliveInterval': 'ハートビート間隔（秒）',
        'hint.keepaliveInterval': 'アイドル時の接続タイムアウトを防ぎます。0 で無効化',
        'subsection.langDetector': '言語検出器設定',
        'label.detectorType': '検出器タイプ',
        'detector.cjke': 'CJK-英語検出器（推奨）',
        'detector.enzh': '英中検出器',
        'detector.fasttext': '汎用検出器（より多くの言語に対応）',
        'subsection.sourceLang': '原文言語設定',
        'label.sourceLanguage': '原文言語',
        'sourceLang.auto': '自動検出',
        'sourceLang.zh': '中国語',
        'sourceLang.en': '英語',
        'sourceLang.ja': '日本語',
        'sourceLang.ko': '韓国語',
        'hint.sourceLanguage': '通常は「自動検出」のままを推奨します',
        'label.micDevice': 'マイク',
        'label.device': 'デバイス',
        'option.systemDefault': 'システム既定',

        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        'msg.configUpdated': '設定を更新しました',
        'msg.configUpdateFailed': '設定の更新に失敗しました',
        'msg.serviceAlreadyRunning': 'サービスは既に稼働中です',
        'msg.serviceStarted': 'サービスを開始しました',
        'msg.startFailed': '開始に失敗しました',
        'msg.serviceNotRunning': 'サービスは停止中です',
        'msg.serviceStopped': 'サービスを停止しました',
        'msg.stopFailed': '停止に失敗しました',
        'msg.noRestartNeeded': 'サービスは停止中のため、再起動は不要です',
        'msg.serviceRestarted': 'サービスを再起動しました',
        'msg.restartFailed': '再起動に失敗しました',
        'msg.enterDashscopeKey': 'DashScope API Key を入力してください',
        'msg.invalidKeyFormat': 'API Key 形式が無効です（sk- で始まる必要があります）',
        'msg.replacePlaceholder': 'プレースホルダーを実際の API Key に置き換えてください',
        'msg.keyFormatValid': 'API Key 形式は有効です',
        'msg.checkFailed': '確認に失敗しました',

        'msg.configSaved': '設定を保存しました！',
        'msg.saveConfigFailed': '設定の保存に失敗しました',
        'msg.dashscopeRequired': 'エラー：サービス開始には Alibaba Cloud DashScope API Key が必要です！',
        'msg.dashscopeValidationFailed': 'DashScope API Key の検証に失敗しました: ',
        'msg.syncConfigFailed': '設定の同期に失敗したため、サービスを開始できません',
        'msg.serviceStartSuccess': 'サービスを開始しました',
        'msg.serviceStartFailed': 'サービス開始に失敗しました: ',
        'msg.startServiceFailed': 'サービス開始に失敗しました',
        'msg.serviceStopSuccess': 'サービスを停止しました',
        'msg.serviceStopFailed': 'サービス停止に失敗しました: ',
        'msg.stopServiceFailed': 'サービス停止に失敗しました',
        'msg.defaultsRestored': 'デフォルトを復元しました',
        'msg.restoreDefaultsFailed': 'デフォルト復元に失敗しました',
        'msg.confirmReset': 'デフォルトに戻しますか？（API Keys は保持されます）',
        'msg.apiKeyRequired': '{api} を使用するには API Key が必要です。先に「API Keys 設定」で入力してください',
        'msg.autoSwitchToGoogle': '選択した翻訳 API の API Key が見つからないため、Google Dictionary に自動切替しました。',
        'msg.sonioxKeyRequired': 'Soniox バックエンドには API Key が必要です',

        'label.uiLanguage': '表示言語'
    },

    'ko': {
        'page.title': 'Yakutan 제어판',
        'header.title': '🎤 Yakutan 제어판',
        'status.notRunning': '서비스 중지됨',
        'status.running': '서비스 실행 중',

        'section.serviceControl': '서비스 제어',
        'btn.startService': '서비스 시작',
        'btn.stopService': '서비스 중지',
        'btn.resetDefaults': '기본값 복원',
        'hint.autoSave': '모든 설정은 브라우저에 자동 저장됩니다',
        'btn.starting': '시작 중...',
        'btn.stopping': '중지 중...',

        'section.basicSettings': '기본 설정',
        'label.enableTranslation': '번역 사용',
        'label.showPartialResults': '중간 결과 표시',
        'hint.partialResults': '번역 사용 시 권장되지 않습니다',
        'label.targetLanguage': '대상 언어',
        'hint.targetLanguage': '언어 코드를 직접 입력하거나 드롭다운에서 선택하세요',
        'label.fallbackLanguage': '대체 언어(원문과 대상 언어가 같을 때 사용)',
        'hint.fallbackLanguage': '언어 코드를 직접 입력하고, 비워 두면 비활성화됩니다',
        'label.enableFurigana': '일본어 후리가나 추가',
        'hint.enableFurigana': '일본어 한자에 읽는 법(히라가나)을 추가합니다',
        'label.enablePinyin': '중국어 병음 추가',
        'hint.enablePinyin': '중국어에 성조 포함 병음을 추가합니다',
        'select.quickSelect': '-- 빠른 선택 --',
        'select.disabled': '사용 안 함',

        'lang.zhCN': '중국어 간체 (zh-CN)',
        'lang.zhTW': '중국어 번체 (zh-TW)',
        'lang.en': '영어 (en)',
        'lang.enGB': '영국식 영어 (en-GB)',
        'lang.ja': '일본어 (ja)',
        'lang.ko': '한국어 (ko)',
        'lang.es': '스페인어 (es)',
        'lang.fr': '프랑스어 (fr)',
        'lang.de': '독일어 (de)',
        'lang.ru': '러시아어 (ru)',
        'lang.ar': '아랍어 (ar)',
        'lang.pt': '포르투갈어 (pt)',
        'lang.it': '이탈리아어 (it)',

        'section.translationApi': '번역 API 설정',
        'label.translationApi': '번역 API',
        'api.qwenMt': 'Qwen-MT (Alibaba Cloud, DashScope Key 사용)',
        'api.deepl': 'DeepL (고품질)',
        'api.googleDict': 'Google Dictionary (무료, 빠름, 네트워크 연결 확인)',
        'api.googleWeb': 'Google Web (무료, 백업용, 네트워크 연결 확인)',
        'api.openrouter': 'OpenRouter (LLM)',
        'api.openrouterStreamingDeeplHybrid': 'OpenRouter 스트리밍 + DeepL 음소거 최종 번역 (하이브리드)',
        'label.streamingMode': '스트리밍 번역 모드',
        'hint.streamingMode': '활성화하면 중간 결과를 실시간으로 번역합니다',
        'label.reverseTranslation': '역방향 번역 사용',
        'hint.reverseTranslation': '항상 Google Dictionary API를 사용합니다. 네트워크 연결을 확인하세요',

        'section.apiKeys': 'API Keys 설정',
        'label.dashscopeKey': 'Alibaba Cloud DashScope API Key',
        'label.required': '*필수',
        'label.international': '국제판',
        'hint.dashscopeKey': 'Qwen 및 FunASR 음성 인식 모두에 필요합니다.',
        'link.getChinaKey': '중국 본토용 API Key 받기',
        'link.getIntlKey': '국제판 API Key 받기',
        'label.deeplKey': 'DeepL API Key (선택, 번역용)',
        'link.getApiKey': 'API Key 받기 →',
        'label.openrouterKey': 'OpenRouter API Key (선택, LLM 번역용)',
        'hint.openrouterEnvLocked': '환경 변수에서 로드되어 입력란이 잠겨 있습니다',
        'placeholder.openrouterEnvConfigured': '환경 변수로 설정됨',

        'section.asrSettings': '음성 인식 설정',
        'label.asrBackend': '인식 백엔드',
        'asr.qwen': 'Qwen3 ASR (권장)',
        'asr.dashscope': 'Fun-ASR (중국 본토 전용)',
        'asr.dashscopeDisabled': 'Fun-ASR (국제판에서는 사용 불가)',
        'asr.soniox': 'Soniox (다국어, API Key 필요)',
        'label.sonioxKey': 'Soniox API Key (선택, 다국어 인식용)',
        'hint.sonioxKey': '60개 이상의 언어 음성 인식을 지원합니다.',
        'label.pauseOnMute': '게임 음소거 시 전사 일시중지',
        'hint.pauseOnMute': '처음 음소거 해제 후 전사가 시작됩니다',
        'label.enableHotWords': '핫워드 사용',
        'hint.enableHotWords': '특정 단어의 인식 정확도를 높입니다',
        'label.muteDelay': '음소거 지연(초)',
        'hint.muteDelay': '음소거 후 인식 중지까지 지연하여 마지막 단어 누락을 방지합니다',

        'section.advancedSettings': '고급 설정',
        'subsection.display': '표시 설정',
        'label.showOriginalAndLangTag': '원문 및 언어 태그 표시',
        'hint.showOriginalAndLangTag': '끄면 번역문만 표시합니다',
        'subsection.vad': 'VAD(음성 활동 감지) 설정 - Qwen 백엔드 전용',
        'label.enableVad': 'VAD 사용',
        'hint.enableVad': '발화 종료를 자동 감지하여 문장을 분할합니다',
        'label.vadThreshold': 'VAD 임계값 (0.0-1.0)',
        'hint.vadThreshold': '값이 낮을수록 민감도가 높아 분할이 쉽게 발생합니다',
        'label.vadSilenceDuration': 'VAD 무음 지속 시간(ms)',
        'hint.vadSilenceDuration': '이 시간만큼 무음을 감지하면 분할을 수행합니다',
        'subsection.websocket': 'WebSocket Keep-alive 설정 - Qwen 백엔드 전용',
        'label.keepaliveInterval': '하트비트 간격(초)',
        'hint.keepaliveInterval': '유휴 상태에서 연결 타임아웃을 방지합니다. 0으로 비활성화',
        'subsection.langDetector': '언어 감지기 설정',
        'label.detectorType': '감지기 유형',
        'detector.cjke': 'CJK-영어 감지기 (권장)',
        'detector.enzh': '영중 감지기',
        'detector.fasttext': '범용 감지기(더 많은 언어 지원)',
        'subsection.sourceLang': '원문 언어 설정',
        'label.sourceLanguage': '원문 언어',
        'sourceLang.auto': '자동 감지',
        'sourceLang.zh': '중국어',
        'sourceLang.en': '영어',
        'sourceLang.ja': '일본어',
        'sourceLang.ko': '한국어',
        'hint.sourceLanguage': '"자동 감지" 유지를 권장합니다',
        'label.micDevice': '마이크',
        'label.device': '장치',
        'option.systemDefault': '시스템 기본값',

        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        'msg.configUpdated': '설정이 업데이트되었습니다',
        'msg.configUpdateFailed': '설정 업데이트에 실패했습니다',
        'msg.serviceAlreadyRunning': '서비스가 이미 실행 중입니다',
        'msg.serviceStarted': '서비스가 시작되었습니다',
        'msg.startFailed': '시작에 실패했습니다',
        'msg.serviceNotRunning': '서비스가 실행 중이 아닙니다',
        'msg.serviceStopped': '서비스가 중지되었습니다',
        'msg.stopFailed': '중지에 실패했습니다',
        'msg.noRestartNeeded': '서비스가 실행 중이 아니므로 재시작이 필요하지 않습니다',
        'msg.serviceRestarted': '서비스가 재시작되었습니다',
        'msg.restartFailed': '재시작에 실패했습니다',
        'msg.enterDashscopeKey': 'DashScope API Key를 입력하세요',
        'msg.invalidKeyFormat': 'API Key 형식이 올바르지 않습니다 (sk-로 시작해야 함)',
        'msg.replacePlaceholder': '플레이스홀더를 실제 API Key로 바꿔 주세요',
        'msg.keyFormatValid': 'API Key 형식이 유효합니다',
        'msg.checkFailed': '확인에 실패했습니다',

        'msg.configSaved': '설정이 저장되었습니다!',
        'msg.saveConfigFailed': '설정 저장에 실패했습니다',
        'msg.dashscopeRequired': '오류: 서비스를 시작하려면 Alibaba Cloud DashScope API Key가 필요합니다!',
        'msg.dashscopeValidationFailed': 'DashScope API Key 검증 실패: ',
        'msg.syncConfigFailed': '설정 동기화에 실패하여 서비스를 시작할 수 없습니다',
        'msg.serviceStartSuccess': '서비스가 시작되었습니다',
        'msg.serviceStartFailed': '서비스 시작 실패: ',
        'msg.startServiceFailed': '서비스 시작에 실패했습니다',
        'msg.serviceStopSuccess': '서비스가 중지되었습니다',
        'msg.serviceStopFailed': '서비스 중지 실패: ',
        'msg.stopServiceFailed': '서비스 중지에 실패했습니다',
        'msg.defaultsRestored': '기본값이 복원되었습니다',
        'msg.restoreDefaultsFailed': '기본값 복원에 실패했습니다',
        'msg.confirmReset': '정말 기본값으로 복원하시겠습니까? (API Keys는 유지됩니다)',
        'msg.apiKeyRequired': '{api}를 사용하려면 API Key가 필요합니다. 먼저 "API Keys 설정"에서 입력하세요',
        'msg.autoSwitchToGoogle': '선택한 번역 API의 API Key를 찾을 수 없어 Google Dictionary로 자동 전환했습니다.',
        'msg.sonioxKeyRequired': 'Soniox 백엔드는 API Key가 필요합니다',

        'label.uiLanguage': 'UI 언어'
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
        document.dispatchEvent(new CustomEvent('i18n:languageChanged', {
            detail: { language: currentLanguage }
        }));
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
                } else if (langPrefix === 'ja') {
                    currentLanguage = 'ja';
                } else if (langPrefix === 'ko') {
                    currentLanguage = 'ko';
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
    document.documentElement.lang = currentLanguage;

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
