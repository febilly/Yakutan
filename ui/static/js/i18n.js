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

// 浏览器 UI 语言无法映射到已支持的中英日韩界面时回退为英语
const UI_LANGUAGE_FALLBACK = 'en';

// 本地存储键名
const LANGUAGE_STORAGE_KEY = 'ui_language';
const LANGUAGE_USER_SELECTED_KEY = 'ui_language_user_selected';

// 当前语言
let currentLanguage = DEFAULT_LANGUAGE;

// 翻译文本
const translations = {
    'zh-CN': {
        // 页面标题和头部
        'page.title': 'Yakutan 控制面板',
        'header.title': '🎤 Yakutan 控制面板',
        'status.notRunning': '服务未运行',
        'status.starting': '服务启动中',
        'status.stopping': '服务停止中',
        'status.running': '服务运行中',
        'status.ipcConnected': '已连接',
        'status.ipcConnectedDelegate': '已连接 Realtime Subtitle',

        // 服务控制
        'section.serviceControl': '服务控制',
        'btn.startService': '启动服务',
        'btn.stopService': '停止服务',
        'btn.resetDefaults': '恢复默认设置',
        'hint.autoSave': '所有配置将自动保存在浏览器本地',
        'btn.starting': '启动中...',
        'btn.stopping': '停止中...',
        'btn.openPanel': '小面板',
        'btn.clearLanguageInput': '清除输入',
        'label.floatingPanelMode': '悬浮窗模式',

        // 基本设置
        'section.basicSettings': '基本设置',
        'label.enableTranslation': '启用翻译',
        'label.showPartialResults': '输出中间结果',
        'hint.partialResults': '不推荐在开启翻译时使用',
        'label.targetLanguage': '目标语言',
        'hint.targetLanguage': '点击右侧箭头选择语言，也可自行输入语言代码',
        'label.secondaryTargetLanguage': '第二输出语言',
        'hint.secondaryTargetLanguage': '可选；启用后会并行输出两行译文',
        'label.fallbackLanguage': '备用语言',
        'hint.fallbackLanguage': '留空则禁用；当源语言与目标语言相同时改用这里',
        'label.enableFurigana': '日语添加假名',
        'hint.enableFurigana': '为日语文本的汉字标注假名读音',
        'label.enablePinyin': '中文添加拼音',
        'hint.enablePinyin': '为中文标注拼音（带声调）',
        'label.enableArabicReshaper': '阿拉伯文重排',
        'hint.enableArabicReshaper': '让阿拉伯文在 VRChat 中按正确字形和方向显示',
        'label.textFancyStyle': '文本风格',
        'hint.textFancyStyle': '使用 fancify-text 为文本增加 Unicode 风格',
        'option.textFancyStyle.none': '无效果',
        'option.textFancyStyle.smallCaps': 'smallCaps - sᴍᴀʟʟCᴀᴘs',
        'option.textFancyStyle.curly': 'curly - ƈųཞɭყ',
        'option.textFancyStyle.magic': 'magic - ɱαɠιƈ',
        'select.quickSelect': '-- 快速选择 --',
        'select.none': '无',
        'select.disabled': '禁用',

        // 语言选项
        'lang.zhCN': '简体中文 (zh-CN)',
        'lang.zhTW': '繁体中文 (zh-TW)',
        'lang.asrZh': '中文 (zh)',
        'lang.en': '英语 (en)',
        'lang.enGB': '英语（英式） (en-GB)',
        'lang.ja': '日语 (ja)',
        'lang.ko': '韩语 (ko)',
        'lang.ar': '阿拉伯语 (ar)',
        'lang.de': '德语 (de)',
        'lang.es': '西班牙语 (es)',
        'lang.fr': '法语 (fr)',
        'lang.id': '印尼语 (id)',
        'lang.it': '意大利语 (it)',
        'lang.pt': '葡萄牙语 (pt)',
        'lang.ru': '俄语 (ru)',
        'lang.th': '泰语 (th)',
        'lang.tl': '他加禄语（菲律宾） (tl)',
        'lang.tr': '土耳其语 (tr)',

        // 翻译API设置
        'section.translationApi': '翻译API设置',
        'label.translationApi': '翻译API',
        'api.qwenMt': 'Qwen-MT（阿里云，使用 DashScope Key）',
        'api.deepl': 'DeepL（高质量）',
        'api.googleDict': 'Google Dictionary（免费，更快，请注意网络连通性）',
        'api.googleWeb': 'Google Web（免费，备用，请注意网络连通性）',
        'api.openrouter': 'LLM（自定义兼容接口，可选流式翻译）',
            'api.openrouterStreamingDeeplHybrid': 'LLM 流式 + DeepL 终译（混合）',
        'label.streamingMode': '流式翻译模式',
        'hint.streamingMode': '启用后会边说边翻译，不用等整句话说完才出结果。请注意这会大幅增加 token 用量。',
        'feature.llmStreamingPromo': '新：推荐尝试LLM流式翻译',
        'feature.switchToLlmStreaming': '点此切换到LLM流式翻译',
        'hint.sensitiveWordsRisk': '请注意敏感词问题，必要时请切换到 DeepL 或 Google 翻译',
        'section.llmSettings': 'LLM 设置',
        'label.llmTemplate': '模板',
        'btn.llmTemplateDashscopeQwenFlash': '阿里 Qwen3.5-Flash',
        'btn.llmTemplateDashscopeQwenPlus': '阿里 Qwen3.5-Plus',
        'btn.llmTemplateDeepSeekV4Flash': 'DeepSeek v4 Flash',
        'btn.llmTemplateOpenRouter': 'OpenRouter',
        'btn.llmTemplateLongCat': 'LongCat（免费）',
        'btn.llmTemplateMercury2': 'Mercury 2（免费）',
        'btn.llmTemplateCustom1': '自定义 1',
        'btn.llmTemplateCustom2': '自定义 2',
        'btn.llmTemplateCustom3': '自定义 3',
        'hint.llmTemplate': '模板会直接填写下方配置；阿里模板会把当前 DashScope Key 复制到 LLM Key。自定义 1~3 会自动保存你填写的地址、模型名、Key、extra_body 等信息。',
        'hint.llmTemplateKeySource': '{provider} API Key 获取地址：',
        'label.llmBaseUrl': 'LLM 地址',
        'hint.llmBaseUrl': '填写兼容接口的根地址，例如 https://openrouter.ai/api/v1 或 https://dashscope.aliyuncs.com/compatible-mode/v1，不要带 /chat/completions。',
        'label.llmModel': 'LLM 模型名',
        'label.llmTranslationFormality': '翻译正式程度',
        'option.llmTranslationFormality.low': '低（朋友之间聊天）',
        'option.llmTranslationFormality.medium': '中（与人初次见面）',
        'option.llmTranslationFormality.high': '高（非常礼貌）',
        'hint.llmTranslationFormality': '仅影响 LLM 译文的语气和礼貌程度。低更随意，中更自然礼貌，高更郑重正式。',
        'label.llmTranslationStyle': '句子风格',
        'option.llmTranslationStyle.standard': '标准',
        'option.llmTranslationStyle.light': '轻快',
        'hint.llmTranslationStyle': '在不改变原意的前提下调整句子的气质。标准最稳，轻快更有聊天感。',
        'label.llmKey': 'LLM API Key',
        'hint.llmKey': '仅在选择 LLM 翻译时使用。',
        'label.openaiCompatExtraBodyJson': '自定义 extra_body（可选）',
        'hint.openaiCompatExtraBodyJson': '留空则不发送 extra_body。',
        'label.llmParallelFastestMode': '并行双发取最快返回',
        'option.llmParallelFastest.off': '禁用',
        'option.llmParallelFastest.finalOnly': '仅对终译启用双发',
        'option.llmParallelFastest.all': '对所有请求都启用双发',
        'hint.llmParallelFastestMode': '禁用则不双发。「仅终译」在流式翻译时不对中间断句双发；「全部」对每个请求都双发。会增加 token 用量。',
        'label.reverseTranslation': '启用反向翻译',
        'hint.reverseTranslation': '将译文翻译回原文所用的语言，并在小面板中显示。总是使用 Google Dictionary API，请注意网络连通性',

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
        'label.openrouterKey': 'LLM API Key（可选）',
        'hint.openrouterEnvLocked': '已从环境变量读取',
        'placeholder.openrouterEnvConfigured': '已在环境变量配置',
        'label.doubaoKey': '豆包录音文件 API Key（可选）',
        'hint.doubaoKey': '用于豆包录音文件识别后端。',

        // 语音识别设置
        'section.asrSettings': '语音识别设置',
        'section.localAsrSettings': '本地音频识别',
        'label.asrBackend': '识别后端',
        'asr.qwen': 'Qwen3 ASR（推荐）',
        'asr.dashscope': 'Fun-ASR（仅中国大陆版可用）',
        'asr.dashscopeDisabled': 'Fun-ASR（国际版不可用）',
        'asr.doubaoFile': '豆包录音文件识别（关麦后返回）',
        'asr.soniox': 'Soniox（对多语混讲支持较好）',
        'asr.local': '本地识别（SenseVoice / Qwen3）',
        'localAsr.engine.sensevoice': 'SenseVoice Small（快速、CPU）',
        'localAsr.engine.qwen3': 'Qwen3-ASR（准确、GPU）',
        'localAsr.engine.sensevoiceHint': 'INT8 ONNX，固定 CPU；',
        'localAsr.engine.qwen3Hint': '在 GPU 上运行（Vulkan/DirectML），约需 1.6GB 显存。',
        'label.sonioxKey': 'Soniox API Key（可选）',
        'hint.sonioxKey': '支持60+语言的语音识别。',
        'label.pauseOnMute': '游戏静音时暂停转录',
        'hint.pauseOnMute': '第一次解除静音后开始转录',
        'label.enableHotWords': '启用热词',
        'hint.enableHotWords': '提高特定词汇的识别准确度',
        'label.muteDelay': '静音延迟（秒）',
        'hint.muteDelay': '静音后延迟停止识别的时间，防止漏掉最后一个字',

        // 高级设置
        'section.textPostProcessing': '文本后处理',
        'section.advancedSettings': '高级设置',
        'subsection.display': '显示设置',
        'subsection.micControl': '麦克风控制',
        'label.panelWidth': '小面板宽度（像素）',
        'hint.panelWidth': '修改后对新打开的小面板生效',
        'label.removeTrailingPeriod': '去除句尾句号',
        'hint.removeTrailingPeriod': '移除文本末尾的 。 和 .',
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
        'subsection.oscNetwork': 'OSC',
        'label.oscCompatMode': '兼容模式',
        'hint.oscCompatMode': '用于其他兼容OSC的游戏，如Resonite。在用于VRChat时请勿打开此选项。',
        'label.oscCompatListenPort': '监听端口',
        'hint.oscCompatListenPort': '兼容模式开启时使用固定监听端口接收 OSC，默认 9001。',
        'label.bypassOscUdpPortCheck': '绕过 VRChat OSC 端口占用检查',
        'hint.bypassOscUdpPortCheck':
            '开启后启动/重启服务时不再检查 UDP 端口占用（可能影响游戏接收 OSC，仅在知情时使用）',
        'label.oscSendTargetPort': 'OSC 发送目标端口',
        'hint.oscSendTargetPort':
            'VRChat 接收 OSC 的 UDP 端口，默认 9000。请务必谨慎修改：端口错误会导致聊天框等消息无法送达；仅在与游戏或网络环境实际监听端口一致时调整。',
        'label.oscSendErrorMessages': '报错时将错误消息发送到 OSC',
        'hint.oscSendErrorMessages': '默认关闭。关闭后错误仍会显示在小窗口，但不会发送到游戏 OSC。',
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
        'hint.sourceLanguage': '建议保持"自动检测"，也可自行输入语言代码',
        'label.micDevice': '麦克风',

        // 智能目标语言
        'section.smartTargetLanguage': '智能目标语言',
        'badge.activated': '已激活',
        'label.smartTargetPrimaryEnabled': '主目标语言自动推断',
        'label.smartTargetSecondaryEnabled': '第二目标语言自动推断',
        'label.excludeSelfLanguage': '排除自身语言',
        'hint.excludeSelfLanguage': '自动推断时不选择语音识别检测到的源语言',
        'label.smartTargetStrategy': '推断策略',
        'option.strategy.mostCommon': '频率最高 (Most Common)',
        'option.strategy.latest': '最新语言 (Latest)',
        'option.strategy.weighted': '权重衰减 (Weighted)',
        'label.smartTargetWindowSize': '采样窗口大小',
        'label.smartTargetMinSamples': '最小采样数',
        'label.recentLanguages': '最近检测到的语言',
        'hint.noRecentLanguages': '暂无数据',

        'label.dependencies': '依赖',
        'label.device': '设备',
        'option.systemDefault': '系统默认',
        'option.systemDefaultWithDevice': '系统默认（{name}）',
        'option.cpu': 'CPU',
        'option.cuda': 'CUDA',
        'option.localVadSilero': 'Silero',
        'option.localVadEnergy': '能量检测',
        'option.localVadDisabled': '禁用',
        'label.localAsrEngine': '本地识别引擎',
        'label.localAsrDevice': '计算设备',
        'label.localAsrLanguage': '语言提示',
        'label.localVadMode': '本地 VAD 模式',
        'label.localVadThreshold': '本地 VAD 阈值',
        'label.localMinSpeechDuration': '最短语音时长（秒）',
        'label.localMaxSpeechDuration': '单段最长语音采集（秒）',
        'label.localSilenceDuration': '静音持续时间（秒）',
        'label.localPreSpeechDuration': '起声预缓冲（秒）',
        'label.localIncrementalAsr': '启用增量识别',
        'label.localInterimInterval': '中间结果间隔（秒）',
        'label.localAsrStatus': '模型状态',
        'hint.localAsrNotChecked': '尚未检查本地模型状态',
        'hint.localAsrStatus': '首次使用前请先下载对应引擎模型和运行时。',
        'hint.localAsrNeedsDownload': '当前引擎尚未准备好，请先下载模型和运行时。',
        'hint.localAsrUsesGlobalSourceLang': '识别语言与上方「语音识别」中的源语言一致；留空或 auto 为自动检测。',
        'status.downloading': '下载中',
        'status.localAsrReady': '{engine} 已准备就绪',
        'btn.downloadLocalAsr': '下载本地识别模型',

        // 页脚
        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        // 消息 - 来自后端的消息ID
        'msg.configUpdated': '配置已更新',
        'msg.configUpdateFailed': '配置更新失败',
        'msg.serviceAlreadyRunning': '服务已在运行中',
        'msg.serviceBusy': '服务正在切换状态，请稍后再试',
        'msg.serviceStarted': '服务已启动',
        'msg.udpPortConflictWarning': '检测到以下程序占用了本机 UDP {port} 端口：{programs}。若非 VRChat，可能导致游戏无法接收 OSC/字幕消息。',
        'msg.udpPortOccupied': 'UDP 端口被占用，无法启动或重启服务。',
        'msg.udpPortBlockedCancel': '已取消操作。请先释放该 UDP 端口或关闭占用进程后再试。',
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
        'msg.localAsrDownloadStarted': '已开始下载本地识别模型',
        'msg.localAsrDownloadFailed': '下载本地识别模型失败',
        'msg.localAsrDisabledInBuild': '当前标准版未启用本地识别功能',
        'msg.localAsrNotReady': '本地识别模型尚未准备好，请先下载',
        'msg.localAsrNeedPythonDeps': '主模型已在本地，但缺少运行依赖: {deps}。请执行 pip install -r requirements-local-asr.txt 后重启应用。',
        'msg.localAsrNeedSilero': 'Silero VAD（ONNX）未就绪。请在下方点击「下载本地识别模型」。',

        // 前端消息
        'msg.configSaved': '配置保存成功！',
        'msg.saveConfigFailed': '保存配置失败',
        'msg.invalidExtraBodyJson': 'extra_body 必须是合法的 JSON 对象',
        'msg.invalidLlmTranslationFormality': 'LLM 翻译正式程度无效',
        'msg.invalidLlmTranslationStyle': 'LLM 句子风格无效',
        'msg.invalidParallelFastestMode': '并行双发模式无效',
        'msg.dashscopeRequired': '错误：必须配置阿里云 DashScope API Key 才能启动服务！',
        'msg.dashscopeValidationFailed': 'DashScope API Key 验证失败: ',
        'msg.syncConfigFailed': '同步配置失败，无法启动服务',
        'msg.serviceStartSuccess': '服务启动成功',
        'msg.serviceStarting': '服务正在启动',
        'msg.acceleratorProcessModeWarning': '检测到疑似有加速器正在运行，如您在使用加速器，请确保加速器当前模式不是“进程模式”。',
        'msg.serviceStartFailed': '服务启动失败: ',
        'msg.startServiceFailed': '启动服务失败',
        'msg.serviceStopSuccess': '服务停止成功',
        'msg.serviceStopping': '服务正在停止',
        'msg.serviceStopFailed': '服务停止失败: ',
        'msg.stopServiceFailed': '停止服务失败',
        'msg.panelOpened': '小面板已打开',
        'msg.panelFailed': '无法打开小面板: ',
        'msg.defaultsRestored': '已恢复默认设置',
        'msg.restoreDefaultsFailed': '恢复默认设置失败',
        'msg.confirmReset': '确定要恢复默认设置吗？（API Keys将被保留）',
        'msg.apiKeyRequired': '使用 {api} 需要配置 API Key，请先在"API Keys 配置"中填写',
        'msg.autoSwitchToGoogle': '未检测到所选翻译接口的 API Key，已自动切换为 Google Dictionary。',
        'msg.llmFieldRequired': '{field} 未填写，请先补全 LLM 设置。',
        'msg.llmTemplateDashscopeCopied': '已将当前 DashScope Key 复制到 LLM Key。',
        'msg.llmTemplateDashscopeKeyMissing': '未检测到 DashScope Key；已套用模板其它字段，请手动填写 LLM Key。',
        'msg.sonioxKeyRequired': 'Soniox 后端需要配置 API Key',
        'msg.doubaoKeyRequired': '豆包录音文件后端需要配置 API Key',
        'msg.doubaoKeyFormat': '豆包 API Key 无效',

        // 语言选择器
        'label.uiLanguage': '界面语言',

        // 快捷切换按钮
        'section.quickLangButtons': '小面板设置',
        'label.enableQuickLangBar': '显示快捷切换按钮',
        'hint.quickLangButtons': '设置小面板底部的4个快捷语言切换按钮',
        'label.quickLangSlot': '按钮',
    },

    'en': {
        // Page title and header
        'page.title': 'Yakutan Control Panel',
        'header.title': '🎤 Yakutan Control Panel',
        'status.notRunning': 'Service Not Running',
        'status.starting': 'Service Starting',
        'status.stopping': 'Service Stopping',
        'status.running': 'Service Running',
        'status.ipcConnected': 'Connected',
        'status.ipcConnectedDelegate': 'Connected Realtime Subtitle',

        // Service control
        'section.serviceControl': 'Service Control',
        'btn.startService': 'Start Service',
        'btn.stopService': 'Stop Service',
        'btn.resetDefaults': 'Reset to Defaults',
        'hint.autoSave': 'All settings are automatically saved in the browser',
        'btn.starting': 'Starting...',
        'btn.stopping': 'Stopping...',
        'btn.openPanel': 'Mini Panel',
        'btn.clearLanguageInput': 'Clear input',
        'label.floatingPanelMode': 'Floating Panel Mode',

        // Basic settings
        'section.basicSettings': 'Basic Settings',
        'label.enableTranslation': 'Enable Translation',
        'label.showPartialResults': 'Show Partial Results',
        'hint.partialResults': 'Not recommended when translation is enabled',
        'label.targetLanguage': 'Target Language',
        'hint.targetLanguage': 'Use the arrow on the right to choose a language, or enter the language code manually',
        'label.secondaryTargetLanguage': 'Second Output',
        'hint.secondaryTargetLanguage': 'Optional. When enabled, two translated lines are sent in parallel',
        'label.fallbackLanguage': 'Fallback',
        'hint.fallbackLanguage': 'Leave empty to disable it; used when the source already matches the target',
        'label.enableFurigana': 'Add furigana to Japanese text',
        'hint.enableFurigana': 'Add hiragana readings to Japanese kanji',
        'label.enablePinyin': 'Add pinyin to Chinese text',
        'hint.enablePinyin': 'Add pinyin with tones to Chinese characters',
        'label.enableArabicReshaper': 'Reshape Arabic text',
        'hint.enableArabicReshaper': 'Display Arabic with correct forms and direction in VRChat',
        'label.textFancyStyle': 'Text Style',
        'hint.textFancyStyle': 'Use fancify-text to apply a Unicode text style',
        'option.textFancyStyle.none': 'No Effect',
        'option.textFancyStyle.smallCaps': 'smallCaps - sᴍᴀʟʟCᴀᴘs',
        'option.textFancyStyle.curly': 'curly - ƈųཞɭყ',
        'option.textFancyStyle.magic': 'magic - ɱαɠιƈ',
        'select.quickSelect': '-- Quick Select --',
        'select.none': 'None',
        'select.disabled': 'Disabled',

        // Language options
        'lang.zhCN': 'Simplified Chinese (zh-CN)',
        'lang.zhTW': 'Traditional Chinese (zh-TW)',
        'lang.asrZh': 'Chinese (zh)',
        'lang.en': 'English (en)',
        'lang.enGB': 'British English (en-GB)',
        'lang.ja': 'Japanese (ja)',
        'lang.ko': 'Korean (ko)',
        'lang.ar': 'Arabic (ar)',
        'lang.de': 'German (de)',
        'lang.es': 'Spanish (es)',
        'lang.fr': 'French (fr)',
        'lang.id': 'Indonesian (id)',
        'lang.it': 'Italian (it)',
        'lang.pt': 'Portuguese (pt)',
        'lang.ru': 'Russian (ru)',
        'lang.th': 'Thai (th)',
        'lang.tl': 'Filipino/Tagalog (tl)',
        'lang.tr': 'Turkish (tr)',

        // Translation API settings
        'section.translationApi': 'Translation API Settings',
        'label.translationApi': 'Translation API',
        'api.qwenMt': 'Qwen-MT (Alibaba Cloud, uses DashScope Key)',
        'api.deepl': 'DeepL (High Quality)',
        'api.googleDict': 'Google Dictionary (Free, Faster, check network connectivity)',
        'api.googleWeb': 'Google Web (Free, Backup, check network connectivity)',
        'api.openrouter': 'LLM (Custom Compatible API, Optional Streaming)',
            'api.openrouterStreamingDeeplHybrid': 'LLM Streaming + DeepL Final Translation (Hybrid)',
        'label.streamingMode': 'Streaming Translation Mode',
        'hint.streamingMode': 'When enabled, translation appears while you are still speaking instead of waiting for the whole sentence to finish. Note: this significantly increases token usage.',
        'feature.llmStreamingPromo': 'New: Try LLM streaming translation (recommended)',
        'feature.switchToLlmStreaming': 'Switch to LLM streaming translation',
        'hint.sensitiveWordsRisk': 'Please watch for sensitive-word filtering issues. Switch to DeepL or Google Translate when necessary.',
        'section.llmSettings': 'LLM Settings',
        'label.llmTemplate': 'Templates',
        'btn.llmTemplateDashscopeQwenFlash': 'Alibaba Qwen3.5-Flash',
        'btn.llmTemplateDashscopeQwenPlus': 'Alibaba Qwen3.5-Plus',
        'btn.llmTemplateDeepSeekV4Flash': 'DeepSeek v4 Flash',
        'btn.llmTemplateOpenRouter': 'OpenRouter',
        'btn.llmTemplateLongCat': 'LongCat (Free)',
        'btn.llmTemplateMercury2': 'Mercury 2 (Free)',
        'btn.llmTemplateCustom1': 'Custom 1',
        'btn.llmTemplateCustom2': 'Custom 2',
        'btn.llmTemplateCustom3': 'Custom 3',
        'hint.llmTemplate': 'Templates fill the fields below directly. The Alibaba template copies the current DashScope key into the LLM key field. Custom 1-3 automatically save the base URL, model, key, extra_body, and related inputs you enter.',
        'hint.llmTemplateKeySource': 'Get the API key for {provider} here: ',
        'label.llmBaseUrl': 'LLM Base URL',
        'hint.llmBaseUrl': 'Enter the compatible API base URL, for example https://openrouter.ai/api/v1 or https://dashscope.aliyuncs.com/compatible-mode/v1. Do not include /chat/completions.',
        'label.llmModel': 'LLM Model',
        'label.llmTranslationFormality': 'Translation Formality',
        'option.llmTranslationFormality.low': 'Low (Friends Chatting)',
        'option.llmTranslationFormality.medium': 'Medium (First Meeting)',
        'option.llmTranslationFormality.high': 'High (Very Polite)',
        'hint.llmTranslationFormality': 'Controls the tone and politeness of LLM translations only. Low is casual, medium is naturally polite, and high is clearly formal.',
        'label.llmTranslationStyle': 'Sentence Style',
        'option.llmTranslationStyle.standard': 'Standard',
        'option.llmTranslationStyle.light': 'Lively',
        'hint.llmTranslationStyle': 'Adjusts the vibe of the sentence without changing the meaning. Standard is the safest, and lively feels more chatty.',
        'label.llmKey': 'LLM API Key',
        'hint.llmKey': 'Used only when LLM translation is selected.',
        'label.openaiCompatExtraBodyJson': 'Custom extra_body (Optional)',
        'hint.openaiCompatExtraBodyJson': 'Leave empty to avoid sending extra_body.',
        'label.llmParallelFastestMode': 'Send Two Requests And Use The Faster One',
        'option.llmParallelFastest.off': 'Disabled',
        'option.llmParallelFastest.finalOnly': 'Final translation only',
        'option.llmParallelFastest.all': 'All requests',
        'hint.llmParallelFastestMode': 'Off: no dual-send. Final only skips dual-send for partial updates while streaming. All: dual-send for every request. Increases token usage.',
        'label.reverseTranslation': 'Enable Reverse Translation',
        'hint.reverseTranslation': 'Translates the output back toward the source language and shows it in the mini panel. Always uses Google Dictionary API. Please ensure network connectivity.',

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
        'label.openrouterKey': 'LLM API Key (optional)',
        'hint.openrouterEnvLocked': 'Loaded from environment variable',
        'placeholder.openrouterEnvConfigured': 'Configured via environment variable',
        'label.doubaoKey': 'Doubao File API Key (optional)',
        'hint.doubaoKey': 'Used by Doubao file transcription backend.',

        // Speech recognition settings
        'section.asrSettings': 'Speech Recognition Settings',
        'section.localAsrSettings': 'Local Speech Recognition',
        'label.asrBackend': 'Recognition Backend',
        'asr.qwen': 'Qwen3 ASR (Recommended)',
        'asr.dashscope': 'Fun-ASR (China Mainland only)',
        'asr.dashscopeDisabled': 'Fun-ASR (Not available for International)',
        'asr.doubaoFile': 'Doubao File ASR (returns after mute)',
        'asr.soniox': 'Soniox (strong mixed-language support)',
        'asr.local': 'Local ASR (SenseVoice / Qwen3)',
        'localAsr.engine.sensevoice': 'SenseVoice Small (Fast, CPU)',
        'localAsr.engine.qwen3': 'Qwen3-ASR (Accurate, GPU)',
        'localAsr.engine.sensevoiceHint': 'INT8 ONNX on CPU.',
        'localAsr.engine.qwen3Hint': 'Runs on GPU (Vulkan/DirectML); about 1.6GB VRAM.',
        'label.sonioxKey': 'Soniox API Key (optional)',
        'hint.sonioxKey': 'Supports 60+ languages for speech recognition.',
        'label.pauseOnMute': 'Pause transcription when muted in game',
        'hint.pauseOnMute': 'Starts transcription after first unmute',
        'label.enableHotWords': 'Enable Hot Words',
        'hint.enableHotWords': 'Improves recognition accuracy for specific words',
        'label.muteDelay': 'Mute Delay (seconds)',
        'hint.muteDelay': 'Delay before stopping recognition after mute, prevents missing last word',

        // Advanced settings
        'section.textPostProcessing': 'Text Post-Processing',
        'section.advancedSettings': 'Advanced Settings',
        'subsection.display': 'Display',
        'subsection.micControl': 'Microphone Control',
        'label.panelWidth': 'Mini Panel Width (px)',
        'hint.panelWidth': 'Changes apply to newly opened mini panels',
        'label.removeTrailingPeriod': 'Remove trailing period',
        'hint.removeTrailingPeriod': 'Remove a final 。 or . at the end of text',
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
        'subsection.oscNetwork': 'OSC',
        'label.oscCompatMode': 'Compatibility mode',
        'hint.oscCompatMode': 'For other OSC-compatible games such as Resonite. Do not enable this option when using VRChat.',
        'label.oscCompatListenPort': 'Listen port',
        'hint.oscCompatListenPort': 'When compatibility mode is on, use this fixed port to listen for OSC events. Default: 9001.',
        'label.bypassOscUdpPortCheck': 'Bypass VRChat OSC UDP port check',
        'hint.bypassOscUdpPortCheck':
            'When on, start/restart will not block if the OSC UDP port appears in use (may break in-game OSC; use only if you understand the risk).',
        'label.oscSendTargetPort': 'OSC send target UDP port',
        'hint.oscSendTargetPort':
            'UDP port where VRChat listens for OSC (default 9000). Change with care: a wrong port prevents chatbox and similar messages from arriving; only set if it matches your game or network setup.',
        'label.oscSendErrorMessages': 'Send error messages to OSC',
        'hint.oscSendErrorMessages': 'Off by default. Errors still appear in the mini panel, but will not be sent to in-game OSC.',
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
        'hint.sourceLanguage': 'Recommended to keep "Auto Detect"; you can also type a language code.',
        'label.micDevice': 'Microphone',

        // Smart Target Language
        'section.smartTargetLanguage': 'Smart Target Language',
        'badge.activated': 'Activated',
        'label.smartTargetPrimaryEnabled': 'Auto-detect primary target language',
        'label.smartTargetSecondaryEnabled': 'Auto-detect secondary target language',
        'label.excludeSelfLanguage': 'Exclude self language',
        'hint.excludeSelfLanguage': 'When auto-detecting, do not select the source language detected by speech recognition',
        'label.smartTargetStrategy': 'Inference strategy',
        'option.strategy.mostCommon': 'Most Common',
        'option.strategy.latest': 'Latest',
        'option.strategy.weighted': 'Weighted Decay',
        'label.smartTargetWindowSize': 'Sampling window size',
        'label.smartTargetMinSamples': 'Minimum samples',
        'label.recentLanguages': 'Recently detected languages',
        'hint.noRecentLanguages': 'No data yet',

        'label.dependencies': 'Dependencies',
        'label.device': 'Device',
        'option.systemDefault': 'System Default',
        'option.systemDefaultWithDevice': 'System default ({name})',
        'option.cpu': 'CPU',
        'option.cuda': 'CUDA',
        'option.localVadSilero': 'Silero',
        'option.localVadEnergy': 'Energy',
        'option.localVadDisabled': 'Disabled',
        'label.localAsrEngine': 'Local ASR Engine',
        'label.localAsrDevice': 'Compute Device',
        'label.localAsrLanguage': 'Language Hint',
        'label.localVadMode': 'Local VAD Mode',
        'label.localVadThreshold': 'Local VAD Threshold',
        'label.localMinSpeechDuration': 'Min Speech Duration (s)',
        'label.localMaxSpeechDuration': 'Max speech per utterance (s)',
        'label.localSilenceDuration': 'Silence Duration (s)',
        'label.localPreSpeechDuration': 'Pre-speech Buffer (s)',
        'label.localIncrementalAsr': 'Enable Incremental ASR',
        'label.localInterimInterval': 'Interim Interval (s)',
        'label.localAsrStatus': 'Model Status',
        'hint.localAsrNotChecked': 'Local ASR model status has not been checked yet',
        'hint.localAsrStatus': 'Download the selected engine model and runtime before first use.',
        'hint.localAsrNeedsDownload': 'The selected engine is not ready yet. Please download the model and runtime first.',
        'hint.localAsrUsesGlobalSourceLang': 'Recognition language follows the global “Source language” under Speech recognition; leave empty or use auto for auto-detect.',
        'status.downloading': 'Downloading',
        'status.localAsrReady': '{engine} is ready',
        'btn.downloadLocalAsr': 'Download Local ASR Model',

        // Footer
        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        // Messages - Backend message IDs
        'msg.configUpdated': 'Configuration updated',
        'msg.configUpdateFailed': 'Configuration update failed',
        'msg.serviceAlreadyRunning': 'Service is already running',
        'msg.serviceBusy': 'Service is changing state, please wait',
        'msg.serviceStarted': 'Service started',
        'msg.udpPortConflictWarning': 'The following program(s) are using local UDP port {port}: {programs}. If this is not VRChat, the game may not receive OSC/subtitle messages.',
        'msg.udpPortOccupied': 'UDP port is in use; cannot start or restart the service.',
        'msg.udpPortBlockedCancel': 'Action cancelled. Free the UDP port or close the conflicting process and try again.',
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
        'msg.localAsrDownloadStarted': 'Started downloading the local ASR model',
        'msg.localAsrDownloadFailed': 'Failed to download the local ASR model',
        'msg.localAsrDisabledInBuild': 'Local ASR is disabled in this standard build',
        'msg.localAsrNotReady': 'The local ASR model is not ready yet. Please download it first',
        'msg.localAsrNeedPythonDeps': 'Model files are present but Python dependencies are missing: {deps}. Run: pip install -r requirements-local-asr.txt then restart.',
        'msg.localAsrNeedSilero': 'Silero VAD (ONNX) is missing. Click download under Local ASR settings.',

        // Frontend messages
        'msg.configSaved': 'Configuration saved successfully!',
        'msg.saveConfigFailed': 'Failed to save configuration',
        'msg.invalidExtraBodyJson': 'extra_body must be a valid JSON object',
        'msg.invalidLlmTranslationFormality': 'Invalid LLM translation formality',
        'msg.invalidLlmTranslationStyle': 'Invalid LLM sentence style',
        'msg.invalidParallelFastestMode': 'Invalid parallel dual-send mode',
        'msg.dashscopeRequired': 'Error: Alibaba Cloud DashScope API Key is required to start the service!',
        'msg.dashscopeValidationFailed': 'DashScope API Key validation failed: ',
        'msg.syncConfigFailed': 'Failed to sync configuration, cannot start service',
        'msg.serviceStartSuccess': 'Service started successfully',
        'msg.serviceStarting': 'Service is starting',
        'msg.acceleratorProcessModeWarning': 'A possible accelerator was detected. If you are using one, make sure it is not in "Process Mode".',
        'msg.serviceStartFailed': 'Service start failed: ',
        'msg.startServiceFailed': 'Failed to start service',
        'msg.serviceStopSuccess': 'Service stopped successfully',
        'msg.serviceStopping': 'Service is stopping',
        'msg.serviceStopFailed': 'Service stop failed: ',
        'msg.stopServiceFailed': 'Failed to stop service',
        'msg.panelOpened': 'Mini Panel opened',
        'msg.panelFailed': 'Failed to open Mini Panel: ',
        'msg.defaultsRestored': 'Defaults restored',
        'msg.restoreDefaultsFailed': 'Failed to restore defaults',
        'msg.confirmReset': 'Are you sure you want to reset to defaults? (API Keys will be preserved)',
        'msg.apiKeyRequired': 'API Key is required for {api}, please fill it in "API Keys Configuration" first',
        'msg.autoSwitchToGoogle': 'API Key for selected translation API not found, automatically switched to Google Dictionary.',
        'msg.llmFieldRequired': '{field} is required. Please complete the LLM settings first.',
        'msg.llmTemplateDashscopeCopied': 'Copied the current DashScope key into the LLM key field.',
        'msg.llmTemplateDashscopeKeyMissing': 'DashScope key not found. Other template fields were filled; please enter the LLM key manually.',
        'msg.sonioxKeyRequired': 'Soniox backend requires API Key',
        'msg.doubaoKeyRequired': 'Doubao file backend requires API Key',
        'msg.doubaoKeyFormat': 'Invalid Doubao API Key',

        // Language selector
        'label.uiLanguage': 'UI Language',

        // Quick language buttons
        'section.quickLangButtons': 'Mini Panel Settings',
        'label.enableQuickLangBar': 'Show quick switch buttons',
        'hint.quickLangButtons': 'Configure the 4 quick language switch buttons at the bottom of the mini panel',
        'label.quickLangSlot': 'Button',
    },

    'ja': {
        'page.title': 'Yakutan コントロールパネル',
        'header.title': '🎤 Yakutan コントロールパネル',
        'status.notRunning': 'サービス停止中',
        'status.starting': 'サービス起動中',
        'status.stopping': 'サービス停止処理中',
        'status.running': 'サービス稼働中',
        'status.ipcConnected': '接続済み',
        'status.ipcConnectedDelegate': 'Realtime Subtitle に接続済み',

        'section.serviceControl': 'サービス制御',
        'btn.startService': 'サービス開始',
        'btn.stopService': 'サービス停止',
        'btn.resetDefaults': 'デフォルトに戻す',
        'hint.autoSave': 'すべての設定はブラウザに自動保存されます',
        'btn.starting': '開始中...',
        'btn.stopping': '停止中...',
        'btn.openPanel': 'ミニパネル',
        'btn.clearLanguageInput': '入力をクリア',
        'label.floatingPanelMode': 'フローティングウィンドウ',

        'section.basicSettings': '基本設定',
        'label.enableTranslation': '翻訳を有効化',
        'label.showPartialResults': '途中結果を表示',
        'hint.partialResults': '翻訳有効時は非推奨です',
        'label.targetLanguage': '翻訳先言語',
        'hint.targetLanguage': '右側の矢印から言語を選ぶか、言語コードを直接入力してください',
        'label.secondaryTargetLanguage': '第2出力言語',
        'hint.secondaryTargetLanguage': '任意です。有効化すると2行の訳文を並行出力します',
        'label.fallbackLanguage': 'フォールバック言語',
        'hint.fallbackLanguage': '空欄で無効化します。原文と言語が同じ場合はここを使います',
        'label.enableFurigana': '日本語にふりがなを追加',
        'hint.enableFurigana': '日本語テキストの漢字に読み仮名を付与します',
        'label.enablePinyin': '中国語にピンインを追加',
        'hint.enablePinyin': '中国語に声調付きピンインを付与します',
        'label.enableArabicReshaper': 'アラビア語の表示を整形',
        'hint.enableArabicReshaper': 'VRChat でアラビア語を正しい字形と方向で表示します',
        'label.textFancyStyle': 'テキストスタイル',
        'hint.textFancyStyle': 'fancify-text で Unicode スタイルを適用します',
        'option.textFancyStyle.none': '効果なし',
        'option.textFancyStyle.smallCaps': 'smallCaps - sᴍᴀʟʟCᴀᴘs',
        'option.textFancyStyle.curly': 'curly - ƈųཞɭყ',
        'option.textFancyStyle.magic': 'magic - ɱαɠιƈ',
        'select.quickSelect': '-- クイック選択 --',
        'select.none': 'なし',
        'select.disabled': '無効',

        'lang.zhCN': '簡体字中国語 (zh-CN)',
        'lang.zhTW': '繁体字中国語 (zh-TW)',
        'lang.asrZh': '中国語 (zh)',
        'lang.en': '英語 (en)',
        'lang.enGB': '英語（英国） (en-GB)',
        'lang.ja': '日本語 (ja)',
        'lang.ko': '韓国語 (ko)',
        'lang.ar': 'アラビア語 (ar)',
        'lang.de': 'ドイツ語 (de)',
        'lang.es': 'スペイン語 (es)',
        'lang.fr': 'フランス語 (fr)',
        'lang.id': 'インドネシア語 (id)',
        'lang.it': 'イタリア語 (it)',
        'lang.pt': 'ポルトガル語 (pt)',
        'lang.ru': 'ロシア語 (ru)',
        'lang.th': 'タイ語 (th)',
        'lang.tl': 'フィリピン語/タガログ語 (tl)',
        'lang.tr': 'トルコ語 (tr)',

        'section.translationApi': '翻訳 API 設定',
        'label.translationApi': '翻訳 API',
        'api.qwenMt': 'Qwen-MT（Alibaba Cloud、DashScope Key を使用）',
        'api.deepl': 'DeepL（高品質）',
        'api.googleDict': 'Google Dictionary（無料・高速、ネットワーク接続に注意）',
        'api.googleWeb': 'Google Web（無料・予備、ネットワーク接続に注意）',
        'api.openrouter': 'LLM（カスタム互換 API、ストリーミング任意）',
        'api.openrouterStreamingDeeplHybrid': 'LLM ストリーミング + DeepL 終訳（ハイブリッド）',
        'label.streamingMode': 'ストリーミング翻訳モード',
        'hint.streamingMode': '有効にすると、話し終わるまで待たずに、話している途中から翻訳が出ます。token 使用量が大幅に増える点に注意してください。',
        'feature.llmStreamingPromo': '新：LLM ストリーミング翻訳のお試しをおすすめします',
        'feature.switchToLlmStreaming': 'LLM ストリーミング翻訳に切り替える',
        'hint.sensitiveWordsRisk': 'センシティブワードの影響に注意し、必要に応じて DeepL または Google 翻訳へ切り替えてください',
        'section.llmSettings': 'LLM 設定',
        'label.llmTemplate': 'テンプレート',
        'btn.llmTemplateDashscopeQwenFlash': 'Alibaba Qwen3.5-Flash',
        'btn.llmTemplateDashscopeQwenPlus': 'Alibaba Qwen3.5-Plus',
        'btn.llmTemplateDeepSeekV4Flash': 'DeepSeek v4 Flash',
        'btn.llmTemplateOpenRouter': 'OpenRouter',
        'btn.llmTemplateLongCat': 'LongCat（無料）',
        'btn.llmTemplateMercury2': 'Mercury 2（無料）',
        'btn.llmTemplateCustom1': 'カスタム 1',
        'btn.llmTemplateCustom2': 'カスタム 2',
        'btn.llmTemplateCustom3': 'カスタム 3',
        'hint.llmTemplate': 'テンプレートは下の項目を直接入力します。Alibaba テンプレートは現在の DashScope Key を LLM Key にコピーします。カスタム 1〜3 は入力した URL、モデル名、Key、extra_body などを自動保存します。',
        'hint.llmTemplateKeySource': '{provider} の API Key 取得先: ',
        'label.llmBaseUrl': 'LLM アドレス',
        'hint.llmBaseUrl': '互換 API のベース URL を入力してください。例: https://openrouter.ai/api/v1 または https://dashscope.aliyuncs.com/compatible-mode/v1。/chat/completions は付けないでください。',
        'label.llmModel': 'LLM モデル名',
        'label.llmTranslationFormality': '翻訳の丁寧さ',
        'option.llmTranslationFormality.low': '低（友達同士の会話）',
        'option.llmTranslationFormality.medium': '中（初対面の相手）',
        'option.llmTranslationFormality.high': '高（とても丁寧）',
        'hint.llmTranslationFormality': 'LLM 訳文の語調と丁寧さだけを調整します。低はくだけた会話調、中は自然な丁寧語、高は改まった丁寧語です。',
        'label.llmTranslationStyle': '文のスタイル',
        'option.llmTranslationStyle.standard': '標準',
        'option.llmTranslationStyle.light': '軽快',
        'hint.llmTranslationStyle': '意味は変えずに文の雰囲気を調整します。標準は最も無難で、軽快は会話感が強くなります。',
        'label.llmKey': 'LLM API Key',
        'hint.llmKey': 'LLM 翻訳を選択したときのみ使用します。',
        'label.openaiCompatExtraBodyJson': 'カスタム extra_body（任意）',
        'hint.openaiCompatExtraBodyJson': '空欄なら extra_body を送信しません。',
        'label.llmParallelFastestMode': '並列 2 発で最速応答を採用',
        'option.llmParallelFastest.off': '無効',
        'option.llmParallelFastest.finalOnly': '最終訳のみ双発',
        'option.llmParallelFastest.all': '全リクエストで双発',
        'hint.llmParallelFastestMode': '無効は双発しません。「最終のみ」はストリーミング中の途中訳では双発しません。「全部」は毎回双発します。token 使用量が増えます。',
        'label.reverseTranslation': '逆翻訳を有効化',
        'hint.reverseTranslation': '訳文を原文側の言語へ戻す訳を小パネルに表示します。常に Google Dictionary API を使用します。ネットワーク接続にご注意ください',

        'section.apiKeys': 'API Keys 設定',
        'label.dashscopeKey': 'Alibaba Cloud DashScope API Key',
        'label.required': '*必須',
        'label.international': '国際版',
        'hint.dashscopeKey': 'Qwen と FunASR の音声認識の両方で必要です。',
        'link.getChinaKey': '中国本土版 API Key を取得',
        'link.getIntlKey': '国際版 API Key を取得',
        'label.deeplKey': 'DeepL API Key（任意、翻訳用）',
        'link.getApiKey': 'API Key を取得 →',
        'label.openrouterKey': 'LLM API Key（任意）',
        'hint.openrouterEnvLocked': '環境変数から読み込み済み',
        'placeholder.openrouterEnvConfigured': '環境変数で設定済み',
        'label.doubaoKey': 'Doubao 録音ファイル API Key（任意）',
        'hint.doubaoKey': 'Doubao 録音ファイル認識バックエンド用。',

        'section.asrSettings': '音声認識設定',
        'section.localAsrSettings': 'ローカル音声認識',
        'label.asrBackend': '認識バックエンド',
        'asr.qwen': 'Qwen3 ASR（推奨）',
        'asr.dashscope': 'Fun-ASR（中国本土版のみ）',
        'asr.dashscopeDisabled': 'Fun-ASR（国際版では利用不可）',
        'asr.doubaoFile': 'Doubao 録音ファイル認識（ミュート後に返却）',
        'asr.soniox': 'Soniox（多言語混話の扱いに強い）',
        'asr.local': 'ローカル ASR（SenseVoice / Qwen3）',
        'localAsr.engine.sensevoice': 'SenseVoice Small（高速、CPU）',
        'localAsr.engine.qwen3': 'Qwen3-ASR（高精度、GPU）',
        'localAsr.engine.sensevoiceHint': 'INT8 ONNX（CPU）。',
        'localAsr.engine.qwen3Hint': 'GPU（Vulkan/DirectML）で動作。VRAM 約 1.6GB。',
        'label.sonioxKey': 'Soniox API Key（任意）',
        'hint.sonioxKey': '60 以上の言語の音声認識に対応しています。',
        'label.pauseOnMute': 'ゲームでミュート中は文字起こしを停止',
        'hint.pauseOnMute': '最初にミュート解除した後に文字起こしを開始します',
        'label.enableHotWords': 'ホットワードを有効化',
        'hint.enableHotWords': '特定語彙の認識精度を向上させます',
        'label.muteDelay': 'ミュート遅延（秒）',
        'hint.muteDelay': 'ミュート後に認識停止まで待機し、最後の語句の欠落を防ぎます',

        'section.textPostProcessing': 'テキスト後処理',
        'section.advancedSettings': '詳細設定',
        'subsection.display': '表示設定',
        'subsection.micControl': 'マイク制御',
        'label.panelWidth': 'ミニパネルの幅（px）',
        'hint.panelWidth': '変更は新しく開くミニパネルに反映されます',
        'label.removeTrailingPeriod': '文末の句点を削除',
        'hint.removeTrailingPeriod': 'テキスト末尾の 。 と . を削除します',
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
        'subsection.oscNetwork': 'OSC',
        'label.oscCompatMode': '互換モード',
        'hint.oscCompatMode': 'Resonite など、OSC 互換ゲーム向けです。VRChat で使用する場合はこのオプションを有効にしないでください。',
        'label.oscCompatListenPort': '待受ポート',
        'hint.oscCompatListenPort': '互換モード時はこの固定ポートで OSC を受信します。既定値は 9001 です。',
        'label.bypassOscUdpPortCheck': 'VRChat OSC の UDP ポート占有チェックをバイパス',
        'hint.bypassOscUdpPortCheck':
            'オンにすると起動・再起動時に UDP ポート占有でブロックしません（ゲーム側 OSC に影響する可能性があります。理解した上で利用してください）',
        'label.oscSendTargetPort': 'OSC 送信先 UDP ポート',
        'hint.oscSendTargetPort':
            'VRChat が OSC を受信する UDP ポート（既定 9000）。変更は慎重に：誤ったポートではチャットボックス等が届きません。ゲームや環境の実際の待受ポートと一致するときのみ変更してください。',
        'label.oscSendErrorMessages': 'エラー時も OSC に送信',
        'hint.oscSendErrorMessages': '既定ではオフです。オフでもエラーはミニパネルに表示されますが、ゲームの OSC には送信しません。',
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
        'hint.sourceLanguage': '通常は「自動検出」のままを推奨します。言語コードを直接入力することもできます。',
        'label.micDevice': 'マイク',

        // スマートターゲット言語
        'section.smartTargetLanguage': 'スマートターゲット言語',
        'badge.activated': '有効',
        'label.smartTargetPrimaryEnabled': '主要ターゲット言語を自動推定',
        'label.smartTargetSecondaryEnabled': '第二ターゲット言語を自動推定',
        'label.excludeSelfLanguage': '自身の言語を除外',
        'hint.excludeSelfLanguage': '自動推定時に音声認識で検出されたソース言語を選択しない',
        'label.smartTargetStrategy': '推定戦略',
        'option.strategy.mostCommon': '頻度最高',
        'option.strategy.latest': '最新言語',
        'option.strategy.weighted': '重み減衰',
        'label.smartTargetWindowSize': 'サンプリングウィンドウサイズ',
        'label.smartTargetMinSamples': '最小サンプル数',
        'label.recentLanguages': '最近検出された言語',
        'hint.noRecentLanguages': 'データなし',

        'label.dependencies': '依存関係',
        'label.device': 'デバイス',
        'option.systemDefault': 'システム既定',
        'option.systemDefaultWithDevice': 'システム既定（{name}）',
        'option.cpu': 'CPU',
        'option.cuda': 'CUDA',
        'option.localVadSilero': 'Silero',
        'option.localVadEnergy': 'エネルギー検出',
        'option.localVadDisabled': '無効',
        'label.localAsrEngine': 'ローカル認識エンジン',
        'label.localAsrDevice': '計算デバイス',
        'label.localAsrLanguage': '言語ヒント',
        'label.localVadMode': 'ローカル VAD モード',
        'label.localVadThreshold': 'ローカル VAD しきい値',
        'label.localMinSpeechDuration': '最短発話長（秒）',
        'label.localMaxSpeechDuration': '1発話あたり最長（秒）',
        'label.localSilenceDuration': '無音継続時間（秒）',
        'label.localPreSpeechDuration': '発話前バッファ（秒）',
        'label.localIncrementalAsr': '増量認識を有効化',
        'label.localInterimInterval': '中間結果の間隔（秒）',
        'label.localAsrStatus': 'モデル状態',
        'hint.localAsrNotChecked': 'ローカル ASR モデル状態はまだ確認されていません',
        'hint.localAsrStatus': '初回利用前に、選択したエンジンのモデルとランタイムをダウンロードしてください。',
        'hint.localAsrNeedsDownload': '選択中のエンジンはまだ準備できていません。先にモデルとランタイムをダウンロードしてください。',
        'hint.localAsrUsesGlobalSourceLang': '認識言語は上の「音声認識」内のソース言語に従います。空欄または auto で自動検出です。',
        'status.downloading': 'ダウンロード中',
        'status.localAsrReady': '{engine} の準備ができました',
        'btn.downloadLocalAsr': 'ローカル認識モデルをダウンロード',

        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        'msg.configUpdated': '設定を更新しました',
        'msg.configUpdateFailed': '設定の更新に失敗しました',
        'msg.serviceAlreadyRunning': 'サービスは既に稼働中です',
        'msg.serviceBusy': 'サービスは状態切り替え中です。しばらく待ってください',
        'msg.serviceStarted': 'サービスを開始しました',
        'msg.udpPortConflictWarning': '次のプログラムがローカル UDP ポート {port} を使用中です: {programs}。VRChat 以外の場合、ゲームが OSC/字幕を受信できない可能性があります。',
        'msg.udpPortOccupied': 'UDP ポートが使用中のため、サービスを開始・再起動できません。',
        'msg.udpPortBlockedCancel': '操作をキャンセルしました。該当 UDP ポートを解放するか、使用中のプロセスを終了してから再度お試しください。',
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
        'msg.localAsrDownloadStarted': 'ローカル ASR モデルのダウンロードを開始しました',
        'msg.localAsrDownloadFailed': 'ローカル ASR モデルのダウンロードに失敗しました',
        'msg.localAsrDisabledInBuild': 'この標準版ビルドではローカル ASR は無効です',
        'msg.localAsrNotReady': 'ローカル ASR モデルの準備ができていません。先にダウンロードしてください',
        'msg.localAsrNeedPythonDeps': 'モデルはありますが Python 依存パッケージが不足: {deps}。pip install -r requirements-local-asr.txt を実行後、再起動。',
        'msg.localAsrNeedSilero': 'Silero VAD（ONNX）がありません。ローカル ASR 設定からダウンロードしてください。',

        'msg.configSaved': '設定を保存しました！',
        'msg.saveConfigFailed': '設定の保存に失敗しました',
        'msg.invalidExtraBodyJson': 'extra_body は有効な JSON オブジェクトである必要があります',
        'msg.invalidLlmTranslationFormality': 'LLM 翻訳の丁寧さ設定が無効です',
        'msg.invalidLlmTranslationStyle': 'LLM 文スタイル設定が無効です',
        'msg.invalidParallelFastestMode': '並列双発モードが無効です',
        'msg.dashscopeRequired': 'エラー：サービス開始には Alibaba Cloud DashScope API Key が必要です！',
        'msg.dashscopeValidationFailed': 'DashScope API Key の検証に失敗しました: ',
        'msg.syncConfigFailed': '設定の同期に失敗したため、サービスを開始できません',
        'msg.serviceStartSuccess': 'サービスを開始しました',
        'msg.serviceStarting': 'サービスを起動しています',
        'msg.acceleratorProcessModeWarning': '加速器らしきものが動作中です。使用中の場合は、現在のモードが「プロセスモード」ではないことを確認してください。',
        'msg.serviceStartFailed': 'サービス開始に失敗しました: ',
        'msg.startServiceFailed': 'サービス開始に失敗しました',
        'msg.serviceStopSuccess': 'サービスを停止しました',
        'msg.serviceStopping': 'サービスを停止しています',
        'msg.serviceStopFailed': 'サービス停止に失敗しました: ',
        'msg.stopServiceFailed': 'サービス停止に失敗しました',
        'msg.panelOpened': 'ミニパネルを開きました',
        'msg.panelFailed': 'ミニパネルを開けません: ',
        'msg.defaultsRestored': 'デフォルトを復元しました',
        'msg.restoreDefaultsFailed': 'デフォルト復元に失敗しました',
        'msg.confirmReset': 'デフォルトに戻しますか？（API Keys は保持されます）',
        'msg.apiKeyRequired': '{api} を使用するには API Key が必要です。先に「API Keys 設定」で入力してください',
        'msg.autoSwitchToGoogle': '選択した翻訳 API の API Key が見つからないため、Google Dictionary に自動切替しました。',
        'msg.llmFieldRequired': '{field} が未入力です。先に LLM 設定を補完してください。',
        'msg.llmTemplateDashscopeCopied': '現在の DashScope Key を LLM Key にコピーしました。',
        'msg.llmTemplateDashscopeKeyMissing': 'DashScope Key が見つかりません。他のテンプレート項目のみ反映したので、LLM Key は手動で入力してください。',
        'msg.sonioxKeyRequired': 'Soniox バックエンドには API Key が必要です',
        'msg.doubaoKeyRequired': 'Doubao 録音ファイルバックエンドには API Key が必要です',
        'msg.doubaoKeyFormat': 'Doubao API Key が無効です',

        'label.uiLanguage': '表示言語',

        // クイック言語ボタン
        'section.quickLangButtons': 'ミニパネル設定',
        'label.enableQuickLangBar': 'クイック切替ボタンを表示',
        'hint.quickLangButtons': 'ミニパネル下部の 4 つの言語クイック切替ボタンを設定します',
        'label.quickLangSlot': 'ボタン',
    },

    'ko': {
        'page.title': 'Yakutan 제어판',
        'header.title': '🎤 Yakutan 제어판',
        'status.notRunning': '서비스 중지됨',
        'status.starting': '서비스 시작 중',
        'status.stopping': '서비스 중지 중',
        'status.running': '서비스 실행 중',
        'status.ipcConnected': '연결됨',
        'status.ipcConnectedDelegate': 'Realtime Subtitle 연결됨',

        'section.serviceControl': '서비스 제어',
        'btn.startService': '서비스 시작',
        'btn.stopService': '서비스 중지',
        'btn.resetDefaults': '기본값 복원',
        'hint.autoSave': '모든 설정은 브라우저에 자동 저장됩니다',
        'btn.starting': '시작 중...',
        'btn.stopping': '중지 중...',
        'btn.openPanel': '미니 패널',
        'btn.clearLanguageInput': '입력 지우기',
        'label.floatingPanelMode': '플로팅 창 모드',

        'section.basicSettings': '기본 설정',
        'label.enableTranslation': '번역 사용',
        'label.showPartialResults': '중간 결과 표시',
        'hint.partialResults': '번역 사용 시 권장되지 않습니다',
        'label.targetLanguage': '대상 언어',
        'hint.targetLanguage': '오른쪽 화살표를 눌러 언어를 선택하거나 언어 코드를 직접 입력하세요',
        'label.secondaryTargetLanguage': '두 번째 출력 언어',
        'hint.secondaryTargetLanguage': '선택 사항입니다. 활성화하면 두 줄의 번역이 병렬로 출력됩니다',
        'label.fallbackLanguage': '대체 언어',
        'hint.fallbackLanguage': '비워 두면 비활성화되며, 원문 언어가 대상 언어와 같을 때 사용됩니다',
        'label.enableFurigana': '일본어 후리가나 추가',
        'hint.enableFurigana': '일본어 한자에 읽는 법(히라가나)을 추가합니다',
        'label.enablePinyin': '중국어 병음 추가',
        'hint.enablePinyin': '중국어에 성조 포함 병음을 추가합니다',
        'label.enableArabicReshaper': '아랍어 표시 재정렬',
        'hint.enableArabicReshaper': 'VRChat에서 아랍어가 올바른 글자 모양과 방향으로 표시되게 합니다',
        'label.textFancyStyle': '텍스트 스타일',
        'hint.textFancyStyle': 'fancify-text로 유니코드 텍스트 스타일을 적용합니다',
        'option.textFancyStyle.none': '효과 없음',
        'option.textFancyStyle.smallCaps': 'smallCaps - sᴍᴀʟʟCᴀᴘs',
        'option.textFancyStyle.curly': 'curly - ƈųཞɭყ',
        'option.textFancyStyle.magic': 'magic - ɱαɠιƈ',
        'select.quickSelect': '-- 빠른 선택 --',
        'select.none': '없음',
        'select.disabled': '사용 안 함',

        'lang.zhCN': '중국어 간체 (zh-CN)',
        'lang.zhTW': '중국어 번체 (zh-TW)',
        'lang.asrZh': '중국어 (zh)',
        'lang.en': '영어 (en)',
        'lang.enGB': '영국식 영어 (en-GB)',
        'lang.ja': '일본어 (ja)',
        'lang.ko': '한국어 (ko)',
        'lang.ar': '아랍어 (ar)',
        'lang.de': '독일어 (de)',
        'lang.es': '스페인어 (es)',
        'lang.fr': '프랑스어 (fr)',
        'lang.id': '인도네시아어 (id)',
        'lang.it': '이탈리아어 (it)',
        'lang.pt': '포르투갈어 (pt)',
        'lang.ru': '러시아어 (ru)',
        'lang.th': '태국어 (th)',
        'lang.tl': '필리핀어/타갈로그어 (tl)',
        'lang.tr': '터키어 (tr)',

        'section.translationApi': '번역 API 설정',
        'label.translationApi': '번역 API',
        'api.qwenMt': 'Qwen-MT (Alibaba Cloud, DashScope Key 사용)',
        'api.deepl': 'DeepL (고품질)',
        'api.googleDict': 'Google Dictionary (무료, 빠름, 네트워크 연결 확인)',
        'api.googleWeb': 'Google Web (무료, 백업용, 네트워크 연결 확인)',
        'api.openrouter': 'LLM (사용자 지정 호환 API, 스트리밍 선택 가능)',
        'api.openrouterStreamingDeeplHybrid': 'LLM 스트리밍 + DeepL 최종 번역 (하이브리드)',
        'label.streamingMode': '스트리밍 번역 모드',
        'hint.streamingMode': '활성화하면 말을 다 마칠 때까지 기다리지 않고, 말하는 도중부터 번역 결과가 나옵니다. 토큰 사용량이 크게 증가할 수 있습니다.',
        'feature.llmStreamingPromo': '신규: LLM 스트리밍 번역 사용을 추천합니다',
        'feature.switchToLlmStreaming': 'LLM 스트리밍 번역으로 전환',
        'hint.sensitiveWordsRisk': '민감어 필터링 문제에 주의하고, 필요하면 DeepL 또는 Google 번역으로 전환하세요',
        'section.llmSettings': 'LLM 설정',
        'label.llmTemplate': '템플릿',
        'btn.llmTemplateDashscopeQwenFlash': 'Alibaba Qwen3.5-Flash',
        'btn.llmTemplateDashscopeQwenPlus': 'Alibaba Qwen3.5-Plus',
        'btn.llmTemplateDeepSeekV4Flash': 'DeepSeek v4 Flash',
        'btn.llmTemplateOpenRouter': 'OpenRouter',
        'btn.llmTemplateLongCat': 'LongCat(무료)',
        'btn.llmTemplateMercury2': 'Mercury 2(무료)',
        'btn.llmTemplateCustom1': '사용자 정의 1',
        'btn.llmTemplateCustom2': '사용자 정의 2',
        'btn.llmTemplateCustom3': '사용자 정의 3',
        'hint.llmTemplate': '템플릿은 아래 항목을 바로 채웁니다. Alibaba 템플릿은 현재 DashScope Key를 LLM Key로 복사합니다. 사용자 정의 1~3은 입력한 주소, 모델명, Key, extra_body 등을 자동 저장합니다.',
        'hint.llmTemplateKeySource': '{provider} API Key 발급 주소: ',
        'label.llmBaseUrl': 'LLM 주소',
        'hint.llmBaseUrl': '호환 API의 기본 주소를 입력하세요. 예: https://openrouter.ai/api/v1 또는 https://dashscope.aliyuncs.com/compatible-mode/v1. /chat/completions 는 포함하지 마세요.',
        'label.llmModel': 'LLM 모델명',
        'label.llmTranslationFormality': '번역 격식 수준',
        'option.llmTranslationFormality.low': '낮음 (친구끼리 대화)',
        'option.llmTranslationFormality.medium': '중간 (처음 만난 사람)',
        'option.llmTranslationFormality.high': '높음 (매우 공손함)',
        'hint.llmTranslationFormality': 'LLM 번역의 말투와 공손함만 조절합니다. 낮음은 편한 대화체, 중간은 자연스럽게 공손한 말투, 높음은 더 격식 있는 표현입니다.',
        'label.llmTranslationStyle': '문장 스타일',
        'option.llmTranslationStyle.standard': '표준',
        'option.llmTranslationStyle.light': '경쾌',
        'hint.llmTranslationStyle': '뜻은 바꾸지 않고 문장의 분위기만 조절합니다. 표준은 가장 안정적이고, 경쾌는 더 대화체에 가깝습니다.',
        'label.llmKey': 'LLM API Key',
        'hint.llmKey': 'LLM 번역을 선택했을 때만 사용합니다.',
        'label.openaiCompatExtraBodyJson': '사용자 정의 extra_body(선택)',
        'hint.openaiCompatExtraBodyJson': '비워 두면 extra_body를 보내지 않습니다.',
        'label.llmParallelFastestMode': '병렬 2회 요청 후 가장 빠른 응답 사용',
        'option.llmParallelFastest.off': '사용 안 함',
        'option.llmParallelFastest.finalOnly': '최종 번역만 이중 발송',
        'option.llmParallelFastest.all': '모든 요청 이중 발송',
        'hint.llmParallelFastestMode': '사용 안 함: 이중 발송 없음. 최종만: 스트리밍 중간 번역은 이중 발송 안 함. 전체: 매 요청 이중 발송. 토큰 사용이 늘어납니다.',
        'label.reverseTranslation': '역방향 번역 사용',
        'hint.reverseTranslation': '번역문을 원문 언어 쪽으로 되돌린 뜻을 작은 패널에 표시합니다. 항상 Google Dictionary API를 사용합니다. 네트워크 연결을 확인하세요',

        'section.apiKeys': 'API Keys 설정',
        'label.dashscopeKey': 'Alibaba Cloud DashScope API Key',
        'label.required': '*필수',
        'label.international': '국제판',
        'hint.dashscopeKey': 'Qwen 및 FunASR 음성 인식 모두에 필요합니다.',
        'link.getChinaKey': '중국 본토용 API Key 받기',
        'link.getIntlKey': '국제판 API Key 받기',
        'label.deeplKey': 'DeepL API Key (선택, 번역용)',
        'link.getApiKey': 'API Key 받기 →',
        'label.openrouterKey': 'LLM API Key (선택)',
        'hint.openrouterEnvLocked': '환경 변수에서 로드됨',
        'placeholder.openrouterEnvConfigured': '환경 변수로 설정됨',
        'label.doubaoKey': 'Doubao 파일 API Key (선택)',
        'hint.doubaoKey': 'Doubao 파일 음성 인식 백엔드용입니다.',

        'section.asrSettings': '음성 인식 설정',
        'section.localAsrSettings': '로컬 음성 인식',
        'label.asrBackend': '인식 백엔드',
        'asr.qwen': 'Qwen3 ASR (권장)',
        'asr.dashscope': 'Fun-ASR (중국 본토 전용)',
        'asr.dashscopeDisabled': 'Fun-ASR (국제판에서는 사용 불가)',
        'asr.doubaoFile': 'Doubao 파일 ASR (음소거 후 반환)',
        'asr.soniox': 'Soniox (다국어 혼용 인식에 유리)',
        'asr.local': '로컬 ASR (SenseVoice / Qwen3)',
        'localAsr.engine.sensevoice': 'SenseVoice Small (빠름, CPU)',
        'localAsr.engine.qwen3': 'Qwen3-ASR (정확, GPU)',
        'localAsr.engine.sensevoiceHint': 'INT8 ONNX(CPU).',
        'localAsr.engine.qwen3Hint': 'GPU(Vulkan/DirectML)에서 실행, VRAM 약 1.6GB.',
        'label.sonioxKey': 'Soniox API Key (선택)',
        'hint.sonioxKey': '60개 이상의 언어 음성 인식을 지원합니다.',
        'label.pauseOnMute': '게임 음소거 시 전사 일시중지',
        'hint.pauseOnMute': '처음 음소거 해제 후 전사가 시작됩니다',
        'label.enableHotWords': '핫워드 사용',
        'hint.enableHotWords': '특정 단어의 인식 정확도를 높입니다',
        'label.muteDelay': '음소거 지연(초)',
        'hint.muteDelay': '음소거 후 인식 중지까지 지연하여 마지막 단어 누락을 방지합니다',

        'section.textPostProcessing': '텍스트 후처리',
        'section.advancedSettings': '고급 설정',
        'subsection.display': '표시 설정',
        'subsection.micControl': '마이크 제어',
        'label.panelWidth': '미니 패널 너비(px)',
        'hint.panelWidth': '변경 사항은 새로 여는 미니 패널에 적용됩니다',
        'label.removeTrailingPeriod': '문장 끝 마침표 제거',
        'hint.removeTrailingPeriod': '텍스트 끝의 。 와 . 를 제거합니다',
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
        'subsection.oscNetwork': 'OSC',
        'label.oscCompatMode': '호환 모드',
        'hint.oscCompatMode': 'Resonite 같은 다른 OSC 호환 게임용입니다. VRChat에서 사용할 때는 이 옵션을 켜지 마세요.',
        'label.oscCompatListenPort': '수신 포트',
        'hint.oscCompatListenPort': '호환 모드에서는 이 고정 포트로 OSC를 수신합니다. 기본값은 9001입니다.',
        'label.bypassOscUdpPortCheck': 'VRChat OSC UDP 포트 점검 건너뛰기',
        'hint.bypassOscUdpPortCheck':
            '켜면 시작/다시 시작 시 UDP 포트 사용 중이어도 막지 않습니다(인게임 OSC에 문제가 생길 수 있습니다. 인지한 경우에만 사용하세요).',
        'label.oscSendTargetPort': 'OSC 전송 대상 UDP 포트',
        'hint.oscSendTargetPort':
            'VRChat이 OSC를 수신하는 UDP 포트(기본 9000). 변경은 신중히: 잘못된 포트면 채팅 상자 등이 전달되지 않습니다. 게임/환경에서 실제로 열어둔 포트와 같을 때만 바꾸세요.',
        'label.oscSendErrorMessages': '오류 메시지도 OSC로 전송',
        'hint.oscSendErrorMessages': '기본값은 꺼짐입니다. 꺼져 있어도 오류는 미니 패널에 계속 표시되지만 게임 OSC로는 보내지 않습니다.',
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
        'hint.sourceLanguage': '"자동 감지" 유지를 권장합니다. 언어 코드를 직접 입력할 수도 있습니다.',
        'label.micDevice': '마이크',

        // 스마트 대상 언어
        'section.smartTargetLanguage': '스마트 대상 언어',
        'badge.activated': '활성화됨',
        'label.smartTargetPrimaryEnabled': '주 대상 언어 자동 추론',
        'label.smartTargetSecondaryEnabled': '두 번째 대상 언어 자동 추론',
        'label.excludeSelfLanguage': '자신의 언어 제외',
        'hint.excludeSelfLanguage': '자동 추론 시 음성 인식이 감지한 소스 언어 선택 안 함',
        'label.smartTargetStrategy': '추론 전략',
        'option.strategy.mostCommon': '빈도 최고',
        'option.strategy.latest': '최신 언어',
        'option.strategy.weighted': '가중치 감쇠',
        'label.smartTargetWindowSize': '샘플링 윈도우 크기',
        'label.smartTargetMinSamples': '최소 샘플 수',
        'label.recentLanguages': '최근 감지된 언어',
        'hint.noRecentLanguages': '데이터 없음',

        'label.dependencies': '의존성',
        'label.device': '장치',
        'option.systemDefault': '시스템 기본값',
        'option.systemDefaultWithDevice': '시스템 기본값({name})',
        'option.cpu': 'CPU',
        'option.cuda': 'CUDA',
        'option.localVadSilero': 'Silero',
        'option.localVadEnergy': '에너지 검출',
        'option.localVadDisabled': '사용 안 함',
        'label.localAsrEngine': '로컬 인식 엔진',
        'label.localAsrDevice': '연산 장치',
        'label.localAsrLanguage': '언어 힌트',
        'label.localVadMode': '로컬 VAD 모드',
        'label.localVadThreshold': '로컬 VAD 임계값',
        'label.localMinSpeechDuration': '최소 발화 길이(초)',
        'label.localMaxSpeechDuration': '발화당 최대(초)',
        'label.localSilenceDuration': '무음 지속 시간(초)',
        'label.localPreSpeechDuration': '발화 전 버퍼(초)',
        'label.localIncrementalAsr': '증분 인식 사용',
        'label.localInterimInterval': '중간 결과 간격(초)',
        'label.localAsrStatus': '모델 상태',
        'hint.localAsrNotChecked': '로컬 ASR 모델 상태를 아직 확인하지 않았습니다',
        'hint.localAsrStatus': '처음 사용하기 전에 선택한 엔진의 모델과 런타임을 먼저 다운로드하세요.',
        'hint.localAsrNeedsDownload': '선택한 엔진이 아직 준비되지 않았습니다. 모델과 런타임을 먼저 다운로드하세요.',
        'hint.localAsrUsesGlobalSourceLang': '인식 언어는 위쪽 「음성 인식」의 원본 언어 설정을 따릅니다. 비우거나 auto면 자동 감지입니다.',
        'status.downloading': '다운로드 중',
        'status.localAsrReady': '{engine} 준비 완료',
        'btn.downloadLocalAsr': '로컬 인식 모델 다운로드',

        'footer.text': 'Yakutan',
        'link.github': 'GitHub',

        'msg.configUpdated': '설정이 업데이트되었습니다',
        'msg.configUpdateFailed': '설정 업데이트에 실패했습니다',
        'msg.serviceAlreadyRunning': '서비스가 이미 실행 중입니다',
        'msg.serviceBusy': '서비스 상태가 변경 중입니다. 잠시 후 다시 시도하세요',
        'msg.serviceStarted': '서비스가 시작되었습니다',
        'msg.udpPortConflictWarning': '다음 프로그램이 로컬 UDP 포트 {port}를 사용 중입니다: {programs}. VRChat이 아니면 게임이 OSC/자막을 받지 못할 수 있습니다.',
        'msg.udpPortOccupied': 'UDP 포트가 사용 중이어 서비스를 시작하거나 다시 시작할 수 없습니다.',
        'msg.udpPortBlockedCancel': '작업이 취소되었습니다. 해당 UDP 포트를 비우거나 점유 프로세스를 종료한 뒤 다시 시도하세요.',
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
        'msg.localAsrDownloadStarted': '로컬 ASR 모델 다운로드를 시작했습니다',
        'msg.localAsrDownloadFailed': '로컬 ASR 모델 다운로드에 실패했습니다',
        'msg.localAsrDisabledInBuild': '이 표준 빌드에서는 로컬 ASR 이 비활성화되어 있습니다',
        'msg.localAsrNotReady': '로컬 ASR 모델이 아직 준비되지 않았습니다. 먼저 다운로드하세요',
        'msg.localAsrNeedPythonDeps': '모델은 있으나 Python 패키지 부족: {deps}. pip install -r requirements-local-asr.txt 후 재시작.',
        'msg.localAsrNeedSilero': 'Silero VAD(ONNX) 없음. 로컬 ASR에서 다운로드.',

        'msg.configSaved': '설정이 저장되었습니다!',
        'msg.saveConfigFailed': '설정 저장에 실패했습니다',
        'msg.invalidExtraBodyJson': 'extra_body는 올바른 JSON 객체여야 합니다',
        'msg.invalidLlmTranslationFormality': 'LLM 번역 격식 수준 값이 올바르지 않습니다',
        'msg.invalidLlmTranslationStyle': 'LLM 문장 스타일 값이 올바르지 않습니다',
        'msg.invalidParallelFastestMode': '병렬 이중 발송 모드가 올바르지 않습니다',
        'msg.dashscopeRequired': '오류: 서비스를 시작하려면 Alibaba Cloud DashScope API Key가 필요합니다!',
        'msg.dashscopeValidationFailed': 'DashScope API Key 검증 실패: ',
        'msg.syncConfigFailed': '설정 동기화에 실패하여 서비스를 시작할 수 없습니다',
        'msg.serviceStartSuccess': '서비스가 시작되었습니다',
        'msg.serviceStarting': '서비스를 시작하는 중입니다',
        'msg.acceleratorProcessModeWarning': '가속기 실행이 감지되었습니다. 사용 중이라면 현재 모드가 "프로세스 모드"가 아닌지 확인하세요.',
        'msg.serviceStartFailed': '서비스 시작 실패: ',
        'msg.startServiceFailed': '서비스 시작에 실패했습니다',
        'msg.serviceStopSuccess': '서비스가 중지되었습니다',
        'msg.serviceStopping': '서비스를 중지하는 중입니다',
        'msg.serviceStopFailed': '서비스 중지 실패: ',
        'msg.stopServiceFailed': '서비스 중지에 실패했습니다',
        'msg.panelOpened': '미니 패널이 열렸습니다',
        'msg.panelFailed': '미니 패널을 열 수 없습니다: ',
        'msg.defaultsRestored': '기본값이 복원되었습니다',
        'msg.restoreDefaultsFailed': '기본값 복원에 실패했습니다',
        'msg.confirmReset': '정말 기본값으로 복원하시겠습니까? (API Keys는 유지됩니다)',
        'msg.apiKeyRequired': '{api}를 사용하려면 API Key가 필요합니다. 먼저 "API Keys 설정"에서 입력하세요',
        'msg.autoSwitchToGoogle': '선택한 번역 API의 API Key를 찾을 수 없어 Google Dictionary로 자동 전환했습니다.',
        'msg.llmFieldRequired': '{field} 항목이 비어 있습니다. 먼저 LLM 설정을 완성하세요.',
        'msg.llmTemplateDashscopeCopied': '현재 DashScope Key를 LLM Key로 복사했습니다.',
        'msg.llmTemplateDashscopeKeyMissing': 'DashScope Key를 찾지 못했습니다. 다른 템플릿 항목만 채웠으니 LLM Key는 직접 입력하세요.',
        'msg.sonioxKeyRequired': 'Soniox 백엔드는 API Key가 필요합니다',
        'msg.doubaoKeyRequired': 'Doubao 파일 백엔드는 API Key가 필요합니다',
        'msg.doubaoKeyFormat': 'Doubao API Key가 올바르지 않습니다',

        'label.uiLanguage': 'UI 언어',

        // 빠른 언어 버튼
        'section.quickLangButtons': '미니 패널 설정',
        'label.enableQuickLangBar': '빠른 전환 버튼 표시',
        'hint.quickLangButtons': '미니 패널 하단의 4개 언어 빠른 전환 버튼을 설정합니다',
        'label.quickLangSlot': '버튼',
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
function setLanguage(lang, options = {}) {
    const { persist = true, userSelected = true } = options;
    if (SUPPORTED_LANGUAGES[lang]) {
        currentLanguage = lang;
        if (persist) {
            localStorage.setItem(LANGUAGE_STORAGE_KEY, lang);
            localStorage.setItem(LANGUAGE_USER_SELECTED_KEY, userSelected ? 'true' : 'false');
        }
        applyTranslations();
        // 更新页面标题
        document.title = t('page.title');
        document.dispatchEvent(new CustomEvent('i18n:languageChanged', {
            detail: { language: currentLanguage }
        }));
    }
}

function detectSystemLanguage() {
    const browserLanguages = Array.isArray(navigator.languages) && navigator.languages.length > 0
        ? navigator.languages
        : [navigator.language || navigator.userLanguage].filter(Boolean);

    for (const browserLang of browserLanguages) {
        if (!browserLang) continue;

        if (SUPPORTED_LANGUAGES[browserLang]) {
            return browserLang;
        }

        const langPrefix = browserLang.split(/[-_]/)[0].toLowerCase();
        if (langPrefix === 'zh') {
            return 'zh-CN';
        }
        if (langPrefix === 'en') {
            return 'en';
        }
        if (langPrefix === 'ja') {
            return 'ja';
        }
        if (langPrefix === 'ko') {
            return 'ko';
        }
    }

    return UI_LANGUAGE_FALLBACK;
}

/**
 * 从本地存储加载语言设置
 */
function loadLanguageFromStorage() {
    const savedLang = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    const isUserSelected = localStorage.getItem(LANGUAGE_USER_SELECTED_KEY) === 'true';
    const systemLang = detectSystemLanguage();

    if (isUserSelected && savedLang && SUPPORTED_LANGUAGES[savedLang]) {
        currentLanguage = savedLang;
    } else if (!isUserSelected && systemLang && SUPPORTED_LANGUAGES[systemLang]) {
        currentLanguage = systemLang;
        localStorage.setItem(LANGUAGE_STORAGE_KEY, systemLang);
        localStorage.setItem(LANGUAGE_USER_SELECTED_KEY, 'false');
    } else if (savedLang && SUPPORTED_LANGUAGES[savedLang]) {
        currentLanguage = savedLang;
    } else {
        currentLanguage = UI_LANGUAGE_FALLBACK;
        localStorage.setItem(LANGUAGE_STORAGE_KEY, UI_LANGUAGE_FALLBACK);
        localStorage.setItem(LANGUAGE_USER_SELECTED_KEY, 'false');
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
        setLanguage(e.target.value, { persist: true, userSelected: true });
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
