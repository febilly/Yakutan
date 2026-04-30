// API基础URL
const API_BASE = '/api';

// 自动保存定时器
let autoSaveTimer = null;

/** PyAudio 解析到的系统默认输入设备 identity，用于检测默认设备是否变化 */
let lastMicDefaultIdentity = null;

// 配置键名
const CONFIG_STORAGE_KEY = 'vrchat_translator_config';

// ===================== 大面板 ↔ 后端 同步基础设施 =====================
//
// 规则：
//   1. 后端在内存里维护 backend_boot_ms（进程启动时固定）和 config_applied_at_ms（每次配置变更刷新）
//   2. 大面板在 localStorage 维护 touch_main_ms（用户在大面板做了任何更改就写 Date.now()）
//      以及 last_seen_boot_ms（上次见到的后端 boot 时刻）
//   3. 小面板不维护时间戳；只推后端、拉后端
//
// 打开大面板时（reconcile）：
//   - boot_ms ≠ last_seen_boot_ms → 后端重启过 → 大面板本地配置推到后端
//   - boot_ms = last_seen_boot_ms → 同一运行期 → config_applied_at_ms vs touch_main_ms 谁大听谁
//
// 运行中轮询（updateStatus）：
//   - config_applied_at_ms > touch_main_ms → 拉服务器的（小面板改了语言等）

const CONFIG_TOUCH_MAIN_MS_KEY = 'vrchat_translator_touch_main_ms';
const LAST_SEEN_BOOT_MS_KEY = 'vrchat_translator_last_seen_boot_ms';

function _readLocalStorageInt(key) {
    const v = parseInt(localStorage.getItem(key) || '0', 10);
    return Number.isFinite(v) ? v : 0;
}

function getMainConfigTouchMs() {
    return _readLocalStorageInt(CONFIG_TOUCH_MAIN_MS_KEY);
}

function setMainConfigTouchMs(ms) {
    const n = Math.floor(Number(ms));
    if (Number.isFinite(n) && n > 0) {
        localStorage.setItem(CONFIG_TOUCH_MAIN_MS_KEY, String(n));
    }
}

function getLastSeenBootMs() {
    return _readLocalStorageInt(LAST_SEEN_BOOT_MS_KEY);
}

function setLastSeenBootMs(ms) {
    const n = Math.floor(Number(ms));
    if (Number.isFinite(n) && n > 0) {
        localStorage.setItem(LAST_SEEN_BOOT_MS_KEY, String(n));
    }
}

function touchMainPanelUserEditedAt() {
    setMainConfigTouchMs(Date.now());
}

// ===================== 同步基础设施结束 =====================

const PANEL_FLOATING_MODE_STORAGE_KEY = 'panel_floating_mode';
const LLM_SELECTED_TEMPLATE_STORAGE_KEY = 'llm_selected_template';
const LLM_TEMPLATE_KEY_STORAGE_PREFIX = 'llm_template_key_';
const LLM_TEMPLATE_BASEURL_STORAGE_PREFIX = 'llm_template_baseurl_';
const LLM_TEMPLATE_MODEL_STORAGE_PREFIX = 'llm_template_model_';
const LLM_TEMPLATE_EXTRABODY_STORAGE_PREFIX = 'llm_template_extrabody_';
const LLM_TEMPLATE_PARALLEL_STORAGE_PREFIX = 'llm_template_parallel_';

// 待显示的警告消息（用于自动切换翻译API）
let pendingWarningMessage = null;

// 环境变量状态（由后端提供）
let envStatus = {
    llm: {
        api_key_set: false,
    },
};

let featureFlags = {
    local_asr_build_enabled: false,
    local_asr_ui_enabled: false,
    engines: {},
};

let localAsrStatus = {
    running: false,
    engine: null,
    status: '',
    error: null,
};

function updateLocalAsrEngineHint() {
    const el = document.getElementById('local-asr-engine-hint');
    const engine = document.getElementById('local-asr-engine')?.value || 'sensevoice';
    if (!el) return;
    if (!isLocalAsrUiEnabled()) {
        el.textContent = '';
        return;
    }
    const t = window.i18n ? window.i18n.t : (key) => key;
    const key =
        engine === 'qwen3-asr' ? 'localAsr.engine.qwen3Hint' : 'localAsr.engine.sensevoiceHint';
    el.textContent = t(key);
}

function isLocalAsrUiEnabled() {
    return !!featureFlags.local_asr_ui_enabled;
}

function getLocalAsrConfigFromForm() {
    return {
        engine: document.getElementById('local-asr-engine')?.value || 'sensevoice',
        vad_mode: document.getElementById('local-vad-mode')?.value || 'silero',
        vad_threshold: parseFloat(document.getElementById('local-vad-threshold')?.value || '0.5'),
        min_speech_duration: parseFloat(document.getElementById('local-min-speech-duration')?.value || '1'),
        max_speech_duration: parseFloat(document.getElementById('local-max-speech-duration')?.value || '30'),
        silence_duration: parseFloat(document.getElementById('local-silence-duration')?.value || '0.8'),
        pre_speech_duration: parseFloat(document.getElementById('local-pre-speech-duration')?.value || '0.2'),
        incremental_asr: document.getElementById('local-incremental-asr')?.checked ?? true,
        interim_interval: parseFloat(document.getElementById('local-interim-interval')?.value || '2'),
    };
}

function applyLocalAsrConfig(config) {
    if (!config) return;
    if (document.getElementById('local-asr-engine')) {
        document.getElementById('local-asr-engine').value = config.engine || 'sensevoice';
    }
    if (document.getElementById('local-vad-mode')) {
        document.getElementById('local-vad-mode').value = config.vad_mode || 'silero';
    }
    if (document.getElementById('local-vad-threshold')) {
        document.getElementById('local-vad-threshold').value = config.vad_threshold ?? 0.5;
    }
    if (document.getElementById('local-min-speech-duration')) {
        document.getElementById('local-min-speech-duration').value = config.min_speech_duration ?? 1.0;
    }
    if (document.getElementById('local-max-speech-duration')) {
        document.getElementById('local-max-speech-duration').value = config.max_speech_duration ?? 30.0;
    }
    if (document.getElementById('local-silence-duration')) {
        document.getElementById('local-silence-duration').value = config.silence_duration ?? 0.8;
    }
    if (document.getElementById('local-pre-speech-duration')) {
        document.getElementById('local-pre-speech-duration').value = config.pre_speech_duration ?? 0.2;
    }
    if (document.getElementById('local-incremental-asr')) {
        document.getElementById('local-incremental-asr').checked = config.incremental_asr ?? true;
    }
    if (document.getElementById('local-interim-interval')) {
        document.getElementById('local-interim-interval').value = config.interim_interval ?? 2.0;
    }
    updateLocalAsrEngineHint();
}

function ensureLocalAsrBackendOption() {
    const select = document.getElementById('asr-backend');
    if (!select) return;
    const t = window.i18n ? window.i18n.t : (key) => key;
    let option = select.querySelector('option[value="local"]');
    if (isLocalAsrUiEnabled()) {
        if (!option) {
            option = document.createElement('option');
            option.value = 'local';
            option.setAttribute('data-i18n', 'asr.local');
            select.appendChild(option);
        }
        option.textContent = t('asr.local');
    } else if (option) {
        if (select.value === 'local') {
            select.value = 'qwen';
        }
        option.remove();
    }
}

function sanitizeAsrBackendValue(value) {
    const normalized = value || 'qwen';
    if (normalized === 'local' && !isLocalAsrUiEnabled()) {
        return 'qwen';
    }
    return normalized;
}

function renderLocalAsrStatus(payload) {
    const box = document.getElementById('local-asr-status');
    const button = document.getElementById('local-asr-download-btn');
    if (!box || !button) return;
    const t = window.i18n ? window.i18n.t : (key) => key;
    const engine = document.getElementById('local-asr-engine')?.value || 'sensevoice';
    const engineStatus = payload?.engines?.[engine];
    localAsrStatus = payload?.download || localAsrStatus;

    if (!payload || !engineStatus) {
        box.textContent = t('hint.localAsrNotChecked');
        button.disabled = false;
        return;
    }

    if (localAsrStatus.running) {
        box.textContent = `${t('status.downloading')}: ${localAsrStatus.status || ''}`;
        button.disabled = true;
        return;
    }

    if (engineStatus.ready) {
        box.textContent = t('status.localAsrReady', { engine: engineStatus.display_name || engine });
    } else {
        const issues = [];
        if (Array.isArray(engineStatus.runtime_issues) && engineStatus.runtime_issues.length) {
            issues.push(`${t('label.dependencies')}: ${engineStatus.runtime_issues.join(', ')}`);
        }
        if (engineStatus.error) {
            issues.push(engineStatus.error);
        }
        if (Array.isArray(engineStatus.missing) && engineStatus.missing.length) {
            issues.push(t('hint.localAsrNeedsDownload'));
        }
        box.textContent = issues.length
            ? issues.join(' | ')
            : t('hint.localAsrNeedsDownload');
        button.disabled = false;
    }
}

async function refreshLocalAsrStatus() {
    if (!isLocalAsrUiEnabled()) return;
    try {
        const response = await fetch(`${API_BASE}/local-asr/status`);
        if (!response.ok) return;
        const payload = await response.json();
        renderLocalAsrStatus(payload);
    } catch (error) {
        console.warn('获取本地 ASR 状态失败:', error);
    }
}

async function downloadLocalAsrModels() {
    const t = window.i18n ? window.i18n.t : (key) => key;
    if (!isLocalAsrUiEnabled()) return;
    const localConfig = getLocalAsrConfigFromForm();
    const button = document.getElementById('local-asr-download-btn');
    if (button) {
        button.disabled = true;
    }
    try {
        const response = await fetch(`${API_BASE}/local-asr/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                engine: localConfig.engine,
            }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.message || 'download failed');
        }
        showMessage(t('msg.localAsrDownloadStarted'), 'success');
        await refreshLocalAsrStatus();
    } catch (error) {
        console.error('下载本地 ASR 失败:', error);
        showMessage(t('msg.localAsrDownloadFailed') + ': ' + error.message, 'error');
        if (button) {
            button.disabled = false;
        }
    }
}

function updateLocalAsrUiVisibility() {
    const card = document.getElementById('local-asr-card');
    if (!card) return;
    ensureLocalAsrBackendOption();
    if (isLocalAsrUiEnabled()) {
        card.style.display = 'block';
        updateLocalAsrEngineHint();
        void refreshLocalAsrStatus();
    } else {
        card.style.display = 'none';
    }
}

function onLocalAsrSettingChange(changedElement = null) {
    if (!isLocalAsrUiEnabled()) return;
    if (changedElement && changedElement.id === 'local-asr-engine') {
        updateLocalAsrEngineHint();
        void refreshLocalAsrStatus();
    }
    onSettingChange(changedElement);
}

async function loadServerFeatures() {
    try {
        const response = await fetch(`${API_BASE}/features`);
        if (!response.ok) return;
        featureFlags = await response.json();
        updateLocalAsrUiVisibility();
    } catch (error) {
        console.warn('加载功能开关失败:', error);
    }
}

const LANGUAGE_OPTIONS = [
    { code: 'zh-CN', labelKey: 'lang.zhCN' },
    { code: 'zh-TW', labelKey: 'lang.zhTW' },
    { code: 'en', labelKey: 'lang.en' },
    { code: 'en-GB', labelKey: 'lang.enGB' },
    { code: 'ja', labelKey: 'lang.ja' },
    { code: 'ko', labelKey: 'lang.ko' },
    { code: 'ar', labelKey: 'lang.ar' },
    { code: 'de', labelKey: 'lang.de' },
    { code: 'es', labelKey: 'lang.es' },
    { code: 'fr', labelKey: 'lang.fr' },
    { code: 'id', labelKey: 'lang.id' },
    { code: 'it', labelKey: 'lang.it' },
    { code: 'pt', labelKey: 'lang.pt' },
    { code: 'ru', labelKey: 'lang.ru' },
    { code: 'th', labelKey: 'lang.th' },
    { code: 'tl', labelKey: 'lang.tl' },
    { code: 'tr', labelKey: 'lang.tr' },
];

/** 语音识别源语言下拉：中文/英文不展示地区变体（仅 zh、en） */
const SOURCE_LANGUAGE_COMBO_OPTIONS = [
    { code: 'zh', labelKey: 'lang.asrZh' },
    { code: 'en', labelKey: 'lang.en' },
    { code: 'ja', labelKey: 'lang.ja' },
    { code: 'ko', labelKey: 'lang.ko' },
    { code: 'ar', labelKey: 'lang.ar' },
    { code: 'de', labelKey: 'lang.de' },
    { code: 'es', labelKey: 'lang.es' },
    { code: 'fr', labelKey: 'lang.fr' },
    { code: 'id', labelKey: 'lang.id' },
    { code: 'it', labelKey: 'lang.it' },
    { code: 'pt', labelKey: 'lang.pt' },
    { code: 'ru', labelKey: 'lang.ru' },
    { code: 'th', labelKey: 'lang.th' },
    { code: 'tl', labelKey: 'lang.tl' },
    { code: 'tr', labelKey: 'lang.tr' },
];

function normalizeSourceLanguageInputValue(stored) {
    const s = stored == null ? '' : String(stored).trim();
    const lower = s.toLowerCase();
    if (!s || lower === 'auto' || lower === 'auto-detect') {
        return '';
    }
    if (lower === 'zh' || lower === 'zh-cn' || lower === 'zh-tw' || lower === 'zh-hans' || lower === 'zh-hant') {
        return 'zh';
    }
    if (lower === 'en' || lower === 'en-gb' || lower === 'en-us') {
        return 'en';
    }
    return s;
}

function getSourceLanguageEffective() {
    const el = document.getElementById('source-language');
    if (!el) return 'auto';
    const raw = el.value.trim();
    const lower = raw.toLowerCase();
    if (!raw || lower === 'auto' || lower === 'auto-detect') {
        return 'auto';
    }
    if (lower === 'zh' || lower === 'zh-cn' || lower === 'zh-tw' || lower === 'zh-hans' || lower === 'zh-hant') {
        return 'zh';
    }
    if (lower === 'en' || lower === 'en-gb' || lower === 'en-us') {
        return 'en';
    }
    return raw;
}

function normalizeSourceLanguageInputOnBlur(input) {
    const v = input.value.trim().toLowerCase();
    if (!v || v === 'auto' || v === 'auto-detect') {
        input.value = '';
    } else if (v === 'zh-cn' || v === 'zh-tw' || v === 'zh-hans' || v === 'zh-hant') {
        input.value = 'zh';
    } else if (v === 'en-gb' || v === 'en-us') {
        input.value = 'en';
    }
}

function applySourceLanguageInputFromStored(stored) {
    const el = document.getElementById('source-language');
    if (!el) return;
    el.value = normalizeSourceLanguageInputValue(stored);
}

/** 手动指定的源语言是否为 zh / en / ja / ko（不含 auto） */
function sourceLanguageIsCjkeFamily(effective) {
    if (effective === 'auto' || !effective) {
        return false;
    }
    const norm = String(effective).trim().toLowerCase();
    return norm === 'zh' || norm === 'en' || norm === 'ja' || norm === 'ko';
}

/** 浏览器首选界面语言是否为中英日韩；无法读取时返回 undefined */
function isBrowserPrimaryLanguageCjke() {
    if (typeof navigator === 'undefined') {
        return undefined;
    }
    const list = Array.isArray(navigator.languages) && navigator.languages.length
        ? navigator.languages
        : [navigator.language || navigator.userLanguage || ''].filter(Boolean);
    if (!list.length) {
        return undefined;
    }
    const raw = String(list[0]).trim();
    if (!raw) {
        return undefined;
    }
    const s = raw.toLowerCase().replace(/_/g, '-');
    if (s === 'zh' || s.startsWith('zh-')) return true;
    if (s === 'en' || s.startsWith('en-')) return true;
    if (s === 'ja' || s.startsWith('ja-')) return true;
    if (s === 'ko' || s.startsWith('ko-')) return true;
    return false;
}

function shouldAutoUseGenericLanguageDetector() {
    const primCjke = isBrowserPrimaryLanguageCjke();
    if (primCjke === false) {
        return true;
    }
    const effective = getSourceLanguageEffective();
    if (effective !== 'auto' && !sourceLanguageIsCjkeFamily(effective)) {
        return true;
    }
    return false;
}

/** 与中日韩英检测器匹配：手动选了中英日韩，或自动检测且浏览器首选为中英日韩 */
function shouldAutoUseCjkeLanguageDetector() {
    if (shouldAutoUseGenericLanguageDetector()) {
        return false;
    }
    const effective = getSourceLanguageEffective();
    if (sourceLanguageIsCjkeFamily(effective)) {
        return true;
    }
    if (effective === 'auto' && isBrowserPrimaryLanguageCjke() === true) {
        return true;
    }
    return false;
}

/** 按源语言 / 浏览器语言自动在 fasttext 与 cjke 之间切换；其它类型（如 enzh）在命中规则时也会被覆盖 */
function applyAutoLanguageDetectorIfNeeded() {
    const sel = document.getElementById('language-detector');
    if (!sel) return false;
    if (shouldAutoUseGenericLanguageDetector()) {
        if (sel.value === 'fasttext') return false;
        sel.value = 'fasttext';
        return true;
    }
    if (shouldAutoUseCjkeLanguageDetector()) {
        if (sel.value === 'cjke') return false;
        sel.value = 'cjke';
        return true;
    }
    return false;
}

const LLM_PARALLEL_FASTEST_MODES = ['off', 'final_only', 'all'];
const DEFAULT_LLM_TEMPLATE_NAME = 'custom1';

const LLM_TEMPLATE_CONFIGS = {
    'dashscope-qwen35-flash': {
        baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        model: 'qwen3.5-flash',
        extraBody: '{"enable_thinking": false}',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateDashscopeQwenFlash',
        copyDashscopeKey: true,
    },
    'dashscope-qwen35-plus': {
        baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        model: 'qwen3.5-plus',
        extraBody: '{"enable_thinking": false}',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateDashscopeQwenPlus',
        copyDashscopeKey: true,
    },
    'deepseek-v4-flash': {
        baseUrl: 'https://api.deepseek.com',
        model: 'deepseek-v4-flash',
        extraBody: '{"thinking": {"type": "disabled"}}',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateDeepSeekV4Flash',
    },
    openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateOpenRouter',
    },
    longcat: {
        baseUrl: 'https://api.longcat.chat/openai/v1',
        model: 'LongCat-Flash-Lite',
        extraBody: '',
        parallelFastestMode: 'all',
        providerLabelKey: 'btn.llmTemplateLongCat',
    },
    mercury2: {
        baseUrl: 'https://api.inceptionlabs.ai/v1',
        model: 'mercury-2',
        extraBody: '',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateMercury2',
    },
    custom1: {
        baseUrl: '',
        model: '',
        extraBody: '',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateCustom1',
        isCustom: true,
    },
    custom2: {
        baseUrl: '',
        model: '',
        extraBody: '',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateCustom2',
        isCustom: true,
    },
    custom3: {
        baseUrl: '',
        model: '',
        extraBody: '',
        parallelFastestMode: 'off',
        providerLabelKey: 'btn.llmTemplateCustom3',
        isCustom: true,
    },
};

function resolveLLMParallelFastestModeFromStoredTranslation(trans) {
    if (!trans) return 'off';
    const raw = trans.llm_parallel_fastest_mode;
    if (LLM_PARALLEL_FASTEST_MODES.includes(raw)) {
        return raw;
    }
    if (typeof trans.enable_llm_parallel_fastest === 'boolean') {
        return trans.enable_llm_parallel_fastest ? 'final_only' : 'off';
    }
    const url = (trans.llm_base_url || '').toLowerCase();
    if (url.includes('longcat.chat')) {
        return 'all';
    }
    return 'off';
}

function setLLMParallelFastestModeSelect(value) {
    const el = document.getElementById('llm-parallel-fastest-mode');
    if (!el) return;
    el.value = LLM_PARALLEL_FASTEST_MODES.includes(value) ? value : 'off';
}

function getLLMParallelFastestModeSelect() {
    const el = document.getElementById('llm-parallel-fastest-mode');
    if (!el) return 'off';
    return LLM_PARALLEL_FASTEST_MODES.includes(el.value) ? el.value : 'off';
}

let activeLLMTemplate = null;

function isValidLLMTemplateName(templateName) {
    return !!(templateName && Object.prototype.hasOwnProperty.call(LLM_TEMPLATE_CONFIGS, templateName));
}

function normalizeLLMTemplateName(templateName) {
    return isValidLLMTemplateName(templateName) ? templateName : '';
}

function isCustomLLMTemplate(templateName) {
    return !!(templateName && LLM_TEMPLATE_CONFIGS[templateName]?.isCustom);
}

function inferLLMTemplateFromCurrentFields() {
    const baseUrl = (document.getElementById('llm-base-url')?.value || '').trim();
    const model = (document.getElementById('llm-model')?.value || '').trim();

    for (const [templateName, templateConfig] of Object.entries(LLM_TEMPLATE_CONFIGS)) {
        if (templateConfig.isCustom) {
            const storedBaseUrl = getStoredLLMTemplateBaseUrl(templateName);
            const storedModel = getStoredLLMTemplateModel(templateName);
            if (!(storedBaseUrl || storedModel)) {
                continue;
            }
            if (baseUrl === storedBaseUrl && model === storedModel) {
                return templateName;
            }
            continue;
        }
        if (baseUrl !== templateConfig.baseUrl) {
            continue;
        }
        if (Object.prototype.hasOwnProperty.call(templateConfig, 'model')) {
            if (model === templateConfig.model) {
                return templateName;
            }
            continue;
        }
        if (baseUrl === templateConfig.baseUrl) {
            return templateName;
        }
    }

    return null;
}

function updateLLMTemplateButtonStates() {
    const selectedTemplate = getSelectedLLMTemplateName();
    document.querySelectorAll('[data-llm-template]').forEach((button) => {
        const isSelected = button.getAttribute('data-llm-template') === selectedTemplate;
        button.classList.toggle('active', isSelected);
        button.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
    });
}

function updateLLMTemplateFieldLocks() {
    const baseUrlInput = document.getElementById('llm-base-url');
    if (!baseUrlInput) return;

    const selectedTemplate = getSelectedLLMTemplateName();
    const isEditable = isCustomLLMTemplate(selectedTemplate);
    baseUrlInput.readOnly = !isEditable;
    baseUrlInput.classList.toggle('readonly-field', !isEditable);
}

function setSelectedLLMTemplateName(templateName, persist = true) {
    const normalized = normalizeLLMTemplateName(templateName) || DEFAULT_LLM_TEMPLATE_NAME;
    activeLLMTemplate = normalized;
    if (persist) {
        localStorage.setItem(LLM_SELECTED_TEMPLATE_STORAGE_KEY, normalized);
    }
    updateLLMTemplateButtonStates();
    updateLLMTemplateFieldLocks();
    return normalized;
}

function getSelectedLLMTemplateName() {
    const normalizedActive = normalizeLLMTemplateName(activeLLMTemplate);
    if (normalizedActive) {
        return normalizedActive;
    }
    const storedTemplate = normalizeLLMTemplateName(localStorage.getItem(LLM_SELECTED_TEMPLATE_STORAGE_KEY));
    return storedTemplate || DEFAULT_LLM_TEMPLATE_NAME;
}

function hasAnyLLMConnectionFieldValue() {
    return !!(
        (document.getElementById('llm-base-url')?.value || '').trim()
        || (document.getElementById('llm-model')?.value || '').trim()
        || (document.getElementById('llm-api-key')?.value || '').trim()
        || (document.getElementById('openai-compat-extra-body-json')?.value || '').trim()
        || getLLMParallelFastestModeSelect() !== 'off'
    );
}

function ensureSelectedLLMTemplate(configTranslation = null) {
    const configTemplate = normalizeLLMTemplateName(configTranslation?.llm_template);
    if (configTemplate) {
        return setSelectedLLMTemplateName(configTemplate);
    }

    const storedTemplate = normalizeLLMTemplateName(localStorage.getItem(LLM_SELECTED_TEMPLATE_STORAGE_KEY));
    if (storedTemplate) {
        activeLLMTemplate = storedTemplate;
        updateLLMTemplateButtonStates();
        updateLLMTemplateFieldLocks();
        return storedTemplate;
    }

    const inferredTemplate = inferLLMTemplateFromCurrentFields();
    if (inferredTemplate) {
        setSelectedLLMTemplateName(inferredTemplate);
        persistCurrentLLMTemplateState();
        return inferredTemplate;
    }

    setSelectedLLMTemplateName(DEFAULT_LLM_TEMPLATE_NAME);
    if (hasAnyLLMConnectionFieldValue()) {
        persistCurrentLLMTemplateState();
    }
    return DEFAULT_LLM_TEMPLATE_NAME;
}

function getLLMTemplateKeyStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_KEY_STORAGE_PREFIX}${templateName}` : null;
}

function getLLMTemplateBaseUrlStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_BASEURL_STORAGE_PREFIX}${templateName}` : null;
}

function getLLMTemplateModelStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_MODEL_STORAGE_PREFIX}${templateName}` : null;
}

function getLLMTemplateExtraBodyStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_EXTRABODY_STORAGE_PREFIX}${templateName}` : null;
}

function getLLMTemplateParallelStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_PARALLEL_STORAGE_PREFIX}${templateName}` : null;
}

function getCurrentLLMTemplateName() {
    return getSelectedLLMTemplateName();
}

function getStoredLLMTemplateKey(templateName) {
    const storageKey = getLLMTemplateKeyStorageKey(templateName);
    if (!storageKey) return '';
    return localStorage.getItem(storageKey) || '';
}

function setStoredLLMTemplateKey(templateName, value) {
    const storageKey = getLLMTemplateKeyStorageKey(templateName);
    if (!storageKey) return;

    const normalized = (value || '').trim();
    if (normalized) {
        localStorage.setItem(storageKey, normalized);
    } else {
        localStorage.removeItem(storageKey);
    }
}

function getStoredLLMTemplateBaseUrl(templateName) {
    const storageKey = getLLMTemplateBaseUrlStorageKey(templateName);
    if (!storageKey) return '';
    return localStorage.getItem(storageKey) || '';
}

function setStoredLLMTemplateBaseUrl(templateName, value) {
    const storageKey = getLLMTemplateBaseUrlStorageKey(templateName);
    if (!storageKey) return;

    const normalized = (value || '').trim();
    if (normalized) {
        localStorage.setItem(storageKey, normalized);
    } else {
        localStorage.removeItem(storageKey);
    }
}

function getStoredLLMTemplateModel(templateName) {
    const storageKey = getLLMTemplateModelStorageKey(templateName);
    if (!storageKey) return '';
    return localStorage.getItem(storageKey) || '';
}

function setStoredLLMTemplateModel(templateName, value) {
    const storageKey = getLLMTemplateModelStorageKey(templateName);
    if (!storageKey) return;

    const normalized = (value || '').trim();
    if (normalized) {
        localStorage.setItem(storageKey, normalized);
    } else {
        localStorage.removeItem(storageKey);
    }
}

/** @returns {string | null} 有记录则为字符串（可为 "" 表示用户明确留空）；从未保存过则为 null */
function getStoredLLMTemplateExtraBody(templateName) {
    const storageKey = getLLMTemplateExtraBodyStorageKey(templateName);
    if (!storageKey) return null;
    return localStorage.getItem(storageKey);
}

function setStoredLLMTemplateExtraBody(templateName, value) {
    const storageKey = getLLMTemplateExtraBodyStorageKey(templateName);
    if (!storageKey) return;
    const normalized = value == null ? '' : String(value).trim();
    localStorage.setItem(storageKey, normalized);
}

function getStoredLLMTemplateParallelMode(templateName) {
    const storageKey = getLLMTemplateParallelStorageKey(templateName);
    if (!storageKey) return '';
    const value = localStorage.getItem(storageKey) || '';
    return LLM_PARALLEL_FASTEST_MODES.includes(value) ? value : '';
}

function setStoredLLMTemplateParallelMode(templateName, value) {
    const storageKey = getLLMTemplateParallelStorageKey(templateName);
    if (!storageKey) return;

    const normalized = LLM_PARALLEL_FASTEST_MODES.includes(value) ? value : 'off';
    localStorage.setItem(storageKey, normalized);
}

function persistCurrentLLMTemplateBaseUrl() {
    const templateName = getCurrentLLMTemplateName();
    const baseUrlInput = document.getElementById('llm-base-url');
    if (!templateName || !baseUrlInput || !isCustomLLMTemplate(templateName)) return;

    setStoredLLMTemplateBaseUrl(templateName, baseUrlInput.value);
}

function persistCurrentLLMTemplateKey() {
    const templateName = getCurrentLLMTemplateName();
    const keyInput = document.getElementById('llm-api-key');
    if (!templateName || !keyInput) return;

    setStoredLLMTemplateKey(templateName, keyInput.value);
}

function persistCurrentLLMTemplateModel() {
    const templateName = getCurrentLLMTemplateName();
    const modelInput = document.getElementById('llm-model');
    if (!templateName || !modelInput) return;

    if (templateName === 'openrouter' || isCustomLLMTemplate(templateName)) {
        setStoredLLMTemplateModel(templateName, modelInput.value);
    }
}

function persistCurrentLLMTemplateExtraBody() {
    const templateName = getCurrentLLMTemplateName();
    const extraInput = document.getElementById('openai-compat-extra-body-json');
    if (!templateName || !extraInput) return;

    setStoredLLMTemplateExtraBody(templateName, extraInput.value);
}

function persistCurrentLLMTemplateParallelMode() {
    const templateName = getCurrentLLMTemplateName();
    if (!templateName || !isCustomLLMTemplate(templateName)) return;

    setStoredLLMTemplateParallelMode(templateName, getLLMParallelFastestModeSelect());
}

function persistCurrentLLMTemplateState() {
    persistCurrentLLMTemplateBaseUrl();
    persistCurrentLLMTemplateKey();
    persistCurrentLLMTemplateModel();
    persistCurrentLLMTemplateExtraBody();
    persistCurrentLLMTemplateParallelMode();
}

function snapshotLLMTemplateStorage() {
    const snapshot = {};
    snapshot[LLM_SELECTED_TEMPLATE_STORAGE_KEY] = getSelectedLLMTemplateName();

    Object.keys(LLM_TEMPLATE_CONFIGS).forEach((templateName) => {
        snapshot[getLLMTemplateKeyStorageKey(templateName)] = getStoredLLMTemplateKey(templateName);

        const baseUrlStorageKey = getLLMTemplateBaseUrlStorageKey(templateName);
        snapshot[baseUrlStorageKey] = getStoredLLMTemplateBaseUrl(templateName);

        const modelStorageKey = getLLMTemplateModelStorageKey(templateName);
        snapshot[modelStorageKey] = getStoredLLMTemplateModel(templateName);

        const extraKey = getLLMTemplateExtraBodyStorageKey(templateName);
        if (extraKey) {
            const rawExtra = localStorage.getItem(extraKey);
            if (rawExtra !== null) {
                snapshot[extraKey] = rawExtra;
            }
        }

        const parallelStorageKey = getLLMTemplateParallelStorageKey(templateName);
        if (parallelStorageKey) {
            const rawParallelMode = localStorage.getItem(parallelStorageKey);
            if (rawParallelMode !== null) {
                snapshot[parallelStorageKey] = rawParallelMode;
            }
        }
    });

    return snapshot;
}

function restoreLLMTemplateStorage(snapshot) {
    if (!snapshot) return;

    Object.entries(snapshot).forEach(([storageKey, value]) => {
        if (storageKey === LLM_SELECTED_TEMPLATE_STORAGE_KEY) {
            if (normalizeLLMTemplateName(value)) {
                localStorage.setItem(storageKey, value);
                activeLLMTemplate = value;
            } else {
                localStorage.removeItem(storageKey);
                activeLLMTemplate = null;
            }
            updateLLMTemplateButtonStates();
            updateLLMTemplateFieldLocks();
            return;
        }
        if (
            storageKey.startsWith(LLM_TEMPLATE_EXTRABODY_STORAGE_PREFIX)
            || storageKey.startsWith(LLM_TEMPLATE_PARALLEL_STORAGE_PREFIX)
        ) {
            if (value === '') {
                localStorage.setItem(storageKey, '');
            } else if (value) {
                localStorage.setItem(storageKey, String(value));
            } else {
                localStorage.removeItem(storageKey);
            }
            return;
        }
        if (value) {
            localStorage.setItem(storageKey, value);
        } else {
            localStorage.removeItem(storageKey);
        }
    });
}

/** 若当前 Base URL 命中某一 LLM 预设且该预设下曾保存过 extra_body，则覆盖输入框（优先于配置里的全局值） */
function applyStoredExtraBodyForActiveLLMTemplate() {
    const templateName = getCurrentLLMTemplateName();
    const extraBodyInput = document.getElementById('openai-compat-extra-body-json');
    if (!templateName || !extraBodyInput) return;

    const storedExtra = getStoredLLMTemplateExtraBody(templateName);
    if (storedExtra !== null) {
        extraBodyInput.value = storedExtra;
    }
}

function resolveLLMTemplateKeySource(templateName) {
    const t = window.i18n ? window.i18n.t : (key) => key;
    const useInternationalEndpoint = document.getElementById('use-international-endpoint')?.checked ?? false;
    const templateConfig = LLM_TEMPLATE_CONFIGS[templateName];
    if (!templateConfig) return null;
    if (templateConfig.isCustom) return null;

    let url = '';
    if (templateName === 'dashscope-qwen35-flash' || templateName === 'dashscope-qwen35-plus') {
        url = useInternationalEndpoint
            ? 'https://modelstudio.console.aliyun.com/ap-southeast-1?tab=doc#/api-key'
            : 'https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key';
    } else if (templateName === 'openrouter') {
        url = 'https://openrouter.ai/settings/keys';
    } else if (templateName === 'deepseek-v4-flash') {
        url = 'https://platform.deepseek.com/api_keys';
    } else if (templateName === 'longcat') {
        url = 'https://longcat.chat/platform/api_keys';
    } else if (templateName === 'mercury2') {
        url = 'https://platform.inceptionlabs.ai/dashboard/api-keys';
    }

    if (!url) return null;

    return {
        providerName: t(templateConfig.providerLabelKey),
        url,
    };
}

function updateLLMTemplateKeySourceHint(templateName = activeLLMTemplate) {
    const container = document.getElementById('llm-template-key-source-hint');
    const label = document.getElementById('llm-template-key-source-label');
    const link = document.getElementById('llm-template-key-source-link');
    const t = window.i18n ? window.i18n.t : (key) => key;

    if (!container || !label || !link) return;

    const source = resolveLLMTemplateKeySource(templateName);
    if (!source) {
        container.style.display = 'none';
        label.textContent = '';
        link.textContent = '';
        link.removeAttribute('href');
        activeLLMTemplate = templateName || null;
        return;
    }

    activeLLMTemplate = templateName;
    label.textContent = t('hint.llmTemplateKeySource', { provider: source.providerName });
    link.href = source.url;
    link.textContent = source.url;
    link.title = source.url;
    container.style.display = 'block';
}

function syncLLMTemplateKeySourceHintFromInputs() {
    updateLLMTemplateKeySourceHint(getCurrentLLMTemplateName());
    updateLLMTemplateButtonStates();
    updateLLMTemplateFieldLocks();
}

function shouldShowLLMSettings(apiType) {
    return apiType === 'openrouter' || apiType === 'openrouter_streaming_deepl_hybrid';
}

const VALID_LLM_TRANSLATION_FORMALITY = ['low', 'medium', 'high'];
const VALID_LLM_TRANSLATION_STYLE = ['standard', 'light'];
const DEFAULT_LLM_TRANSLATION_FORMALITY = 'medium';
const DEFAULT_LLM_TRANSLATION_STYLE = 'light';

function sanitizeLLMTranslationFormality(value) {
    const normalized = (value || DEFAULT_LLM_TRANSLATION_FORMALITY).trim().toLowerCase();
    return VALID_LLM_TRANSLATION_FORMALITY.includes(normalized)
        ? normalized
        : DEFAULT_LLM_TRANSLATION_FORMALITY;
}

function sanitizeLLMTranslationStyle(value) {
    const normalized = (value || DEFAULT_LLM_TRANSLATION_STYLE).trim().toLowerCase();
    return VALID_LLM_TRANSLATION_STYLE.includes(normalized)
        ? normalized
        : DEFAULT_LLM_TRANSLATION_STYLE;
}

/** LLM 地址、模型、Key 是否已就绪（与启动校验一致：Key 可为环境/后端已设置而输入框为空）。 */
function isLLMConnectionFieldsComplete() {
    const base = (document.getElementById('llm-base-url')?.value || '').trim();
    const model = (document.getElementById('llm-model')?.value || '').trim();
    const llmKey = (document.getElementById('llm-api-key')?.value || '').trim();
    const hasKey = envStatus.llm.api_key_set || !!llmKey;
    return !!(base && model && hasKey);
}

/** 同步「流式翻译模式」分组：仅普通 LLM 显示开关；混合模式固定流式（由 api_type），不显示开关且不改动开关状态，避免影响切回 LLM 时的偏好。 */
function updateOpenRouterStreamingUi() {
    const apiSelect = document.getElementById('translation-api-type');
    const streamingModeGroup = document.getElementById('openrouter-streaming-mode-group');
    const streamingMode = document.getElementById('openrouter-streaming-mode');
    if (!apiSelect || !streamingModeGroup || !streamingMode) return;

    const v = apiSelect.value;
    if (v === 'openrouter_streaming_deepl_hybrid') {
        streamingModeGroup.style.display = 'none';
        streamingMode.disabled = false;
        return;
    }
    if (v === 'openrouter') {
        streamingModeGroup.style.display = 'block';
        streamingMode.disabled = false;
        return;
    }
    streamingModeGroup.style.display = 'none';
    streamingMode.checked = false;
    streamingMode.disabled = false;
}

function updateLLMSettingsVisibility(apiType = null, expandPanel = false) {
    const actualApiType = apiType || (document.getElementById('translation-api-type')
        ? document.getElementById('translation-api-type').value
        : 'qwen_mt');
    const wrapper = document.getElementById('llm-settings-wrapper');
    const formalityGroup = document.getElementById('llm-translation-formality-group');
    const styleGroup = document.getElementById('llm-translation-style-group');
    const toneGrid = document.getElementById('llm-tone-grid');
    if (!wrapper || !formalityGroup || !styleGroup || !toneGrid) return;

    const shouldShow = shouldShowLLMSettings(actualApiType);
    wrapper.style.display = shouldShow ? 'block' : 'none';
    toneGrid.style.display = shouldShow ? 'grid' : 'none';
    formalityGroup.style.display = shouldShow ? 'block' : 'none';
    styleGroup.style.display = shouldShow ? 'block' : 'none';

    if (shouldShow && expandPanel) {
        ensureCollapsibleExpanded('llm-settings');
    }
}

function updateSensitiveWordsHint(apiType = null) {
    const actualApiType = apiType || (document.getElementById('translation-api-type')
        ? document.getElementById('translation-api-type').value
        : 'qwen_mt');
    const hint = document.getElementById('qwen-mt-sensitive-words-hint');
    if (!hint) return;

    hint.style.display = actualApiType === 'qwen_mt' ? 'block' : 'none';
}

/** Qwen ASR / Fun-ASR / Qwen-MT 依赖 DashScope Key，其余场景不校验 */
function currentConfigRequiresDashscopeKey() {
    const asrBackend = document.getElementById('asr-backend')?.value;
    const enableTranslation = document.getElementById('enable-translation')?.checked ?? false;
    const translationApiType = document.getElementById('translation-api-type')?.value ?? '';
    const byAsr = asrBackend === 'qwen' || asrBackend === 'dashscope';
    const byTranslation = enableTranslation && translationApiType === 'qwen_mt';
    return byAsr || byTranslation;
}

function updateDashscopeKeyFieldState() {
    const badge = document.getElementById('dashscope-key-required-badge');
    const input = document.getElementById('dashscope-api-key');
    if (!input) return;
    const need = currentConfigRequiresDashscopeKey();
    if (badge) {
        if (need) {
            badge.hidden = false;
            badge.setAttribute('data-i18n', 'label.required');
            badge.textContent = window.i18n ? window.i18n.t('label.required') : '*必需';
        } else {
            badge.hidden = true;
            badge.textContent = '';
            badge.removeAttribute('data-i18n');
        }
    }
    input.required = !!need;
}

function ensureCollapsibleExpanded(id) {
    const content = document.getElementById(id);
    if (content && content.classList.contains('collapsed')) {
        toggleCollapsible(id);
    }
}

function highlightInput(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;
    input.classList.add('error-highlight');
    input.focus();
    setTimeout(() => {
        input.classList.remove('error-highlight');
    }, 3000);
}

function expandLLMSettingsPanel(inputIds = []) {
    const wrapper = document.getElementById('llm-settings-wrapper');
    ensureCollapsibleExpanded('translation-api');
    if (wrapper) {
        wrapper.style.display = 'block';
        wrapper.classList.add('error-highlight');
        setTimeout(() => wrapper.classList.remove('error-highlight'), 3000);
    }
    ensureCollapsibleExpanded('llm-settings');
    inputIds.forEach(highlightInput);
}

function persistSecretInputValue(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;
    const keyName = inputId.replace(/-/g, '_');
    const value = input.value;
    if (value) {
        localStorage.setItem(keyName, value);
    } else {
        localStorage.removeItem(keyName);
    }
}

function updateSecretVisibilityToggleState(toggleButton, targetInput) {
    if (!toggleButton || !targetInput) return;

    const isVisible = targetInput.type === 'text';
    toggleButton.classList.toggle('visible', isVisible);
    toggleButton.setAttribute('aria-pressed', isVisible ? 'true' : 'false');
    toggleButton.setAttribute('aria-label', isVisible ? 'Hide key' : 'Show key');
    toggleButton.title = isVisible ? 'Hide key' : 'Show key';
}

function setupSecretVisibilityToggles() {
    document.querySelectorAll('.secret-visibility-toggle[data-target-input]').forEach((toggleButton) => {
        const targetInputId = toggleButton.getAttribute('data-target-input');
        const targetInput = targetInputId ? document.getElementById(targetInputId) : null;
        if (!targetInput) return;

        updateSecretVisibilityToggleState(toggleButton, targetInput);

        if (toggleButton.dataset.bound === 'true') {
            return;
        }

        toggleButton.addEventListener('click', () => {
            targetInput.type = targetInput.type === 'password' ? 'text' : 'password';
            updateSecretVisibilityToggleState(toggleButton, targetInput);
        });

        toggleButton.dataset.bound = 'true';
    });
}

function loadPanelFloatingModeSetting() {
    const toggle = document.getElementById('panel-floating-mode');
    if (!toggle) return;

    toggle.checked = localStorage.getItem(PANEL_FLOATING_MODE_STORAGE_KEY) === 'true';
}

function savePanelFloatingModeSetting() {
    const toggle = document.getElementById('panel-floating-mode');
    if (!toggle) return;

    localStorage.setItem(PANEL_FLOATING_MODE_STORAGE_KEY, toggle.checked.toString());
}

function onPanelFloatingModeChange() {
    savePanelFloatingModeSetting();
}

function resetPanelFloatingModeSetting() {
    const toggle = document.getElementById('panel-floating-mode');
    if (toggle) {
        toggle.checked = false;
    }

    localStorage.removeItem(PANEL_FLOATING_MODE_STORAGE_KEY);
}

function getNormalizedPanelWidth() {
    const input = document.getElementById('panel-width');
    if (!input) return 600;

    const rawValue = parseInt(input.value, 10);
    const normalizedValue = Math.min(2000, Math.max(300, Number.isFinite(rawValue) ? rawValue : 600));
    input.value = normalizedValue;
    return normalizedValue;
}

function getOscSendTargetPortFromForm() {
    const input = document.getElementById('osc-send-target-port');
    if (!input) return 9000;
    const raw = parseInt(input.value, 10);
    const p = Math.max(1, Math.min(65535, Number.isFinite(raw) ? raw : 9000));
    input.value = p;
    return p;
}

function isOscCompatModeEnabled() {
    return document.getElementById('osc-compat-mode')?.checked === true;
}

function getOscCompatListenPortFromForm() {
    const input = document.getElementById('osc-compat-listen-port');
    if (!input) return 9001;
    const raw = parseInt(input.value, 10);
    const p = Math.max(1, Math.min(65535, Number.isFinite(raw) ? raw : 9001));
    input.value = p;
    return p;
}

function updateOscCompatModeUi() {
    const compatEnabled = isOscCompatModeEnabled();
    const compatPortGroup = document.getElementById('osc-compat-listen-port-group');
    if (compatPortGroup) {
        compatPortGroup.style.display = compatEnabled ? 'block' : 'none';
    }

    const compatPortInput = document.getElementById('osc-compat-listen-port');
    if (compatPortInput) {
        compatPortInput.disabled = !compatEnabled;
        getOscCompatListenPortFromForm();
    }

    const bypassOscEl = document.getElementById('bypass-osc-udp-port-check');
    const bypassOscGroup = document.getElementById('bypass-osc-udp-port-check-group');
    if (bypassOscEl) {
        bypassOscEl.disabled = compatEnabled;
    }
    if (bypassOscGroup) {
        bypassOscGroup.classList.toggle('disabled-option', compatEnabled);
    }
}

function shouldSkipOscUdpPortCheck() {
    return isOscCompatModeEnabled()
        || document.getElementById('bypass-osc-udp-port-check')?.checked === true;
}

function applyLLMTemplate(templateName) {
    const previousTemplateName = getCurrentLLMTemplateName();
    const baseUrlInput = document.getElementById('llm-base-url');
    const modelInput = document.getElementById('llm-model');
    const keyInput = document.getElementById('llm-api-key');
    const extraBodyInput = document.getElementById('openai-compat-extra-body-json');
    const parallelFastestSelect = document.getElementById('llm-parallel-fastest-mode');
    const dashscopeKeyInput = document.getElementById('dashscope-api-key');
    const t = window.i18n ? window.i18n.t : (key) => key;
    const templateConfig = LLM_TEMPLATE_CONFIGS[templateName];

    if (!baseUrlInput || !modelInput || !keyInput || !extraBodyInput || !parallelFastestSelect || !templateConfig) {
        return;
    }

    if (previousTemplateName) {
        persistCurrentLLMTemplateState();
    }

    setSelectedLLMTemplateName(templateName);

    if (templateConfig.isCustom) {
        baseUrlInput.value = getStoredLLMTemplateBaseUrl(templateName);
        modelInput.value = getStoredLLMTemplateModel(templateName);
    } else {
        baseUrlInput.value = templateConfig.baseUrl;
    }
    if (!templateConfig.isCustom && Object.prototype.hasOwnProperty.call(templateConfig, 'model')) {
        modelInput.value = templateConfig.model;
    } else if (templateName === 'openrouter') {
        modelInput.value = getStoredLLMTemplateModel(templateName);
    }
    const storedExtra = getStoredLLMTemplateExtraBody(templateName);
    if (storedExtra !== null) {
        extraBodyInput.value = storedExtra;
    } else if (Object.prototype.hasOwnProperty.call(templateConfig, 'extraBody')) {
        extraBodyInput.value = templateConfig.extraBody;
    } else {
        extraBodyInput.value = '';
    }
    if (templateConfig.isCustom) {
        parallelFastestSelect.value = getStoredLLMTemplateParallelMode(templateName) || 'off';
    } else {
        parallelFastestSelect.value = templateConfig.parallelFastestMode || 'off';
    }

    const storedKey = getStoredLLMTemplateKey(templateName);
    if (storedKey) {
        keyInput.value = storedKey;
        persistSecretInputValue('llm-api-key');
        envStatus.llm.api_key_set = true;
    } else if (templateConfig.isCustom) {
        keyInput.value = '';
        persistSecretInputValue('llm-api-key');
        envStatus.llm.api_key_set = false;
    } else if (templateConfig.copyDashscopeKey) {
        const dashscopeKey = dashscopeKeyInput ? dashscopeKeyInput.value.trim() : '';
        if (dashscopeKey) {
            keyInput.value = dashscopeKey;
            setStoredLLMTemplateKey(templateName, dashscopeKey);
            persistSecretInputValue('llm-api-key');
            envStatus.llm.api_key_set = true;
            showMessage('✅ ' + t('msg.llmTemplateDashscopeCopied'), 'success');
        } else {
            keyInput.value = '';
            persistSecretInputValue('llm-api-key');
            envStatus.llm.api_key_set = false;
            showMessage('⚠️ ' + t('msg.llmTemplateDashscopeKeyMissing'), 'warning');
        }
    } else {
        keyInput.value = '';
        persistSecretInputValue('llm-api-key');
        envStatus.llm.api_key_set = false;
    }

    updateLLMTemplateKeySourceHint(templateName);
    onSettingChange(baseUrlInput);
}

function closeLanguageMenus(exceptCombo = null) {
    document.querySelectorAll('.language-combo.open').forEach((combo) => {
        if (combo !== exceptCombo) {
            combo.classList.remove('open');
        }
    });
}

function refreshLanguageComboClearLabels() {
    const t = window.i18n ? window.i18n.t : (key) => key;
    const label = t('btn.clearLanguageInput');
    document.querySelectorAll('.language-combo-clear').forEach((btn) => {
        btn.setAttribute('aria-label', label);
        btn.title = label;
    });
}

function syncLanguageComboClearVisibility(combo) {
    if (!combo) return;
    const input = combo.querySelector('.language-combo-input');
    const clearBtn = combo.querySelector('.language-combo-clear');
    if (!input || !clearBtn) return;
    clearBtn.hidden = input.value.trim() === '';
}

function syncAllLanguageComboClearButtons() {
    document.querySelectorAll('.language-combo').forEach(syncLanguageComboClearVisibility);
}

function renderLanguageComboMenu(combo) {
    if (!combo) return;

    const input = combo.querySelector('.language-combo-input');
    const menu = combo.querySelector('.language-combo-menu');
    const t = window.i18n ? window.i18n.t : (key) => key;

    if (!input || !menu) return;

    const currentValue = input.value.trim().toLowerCase();
    menu.innerHTML = '';

    if (input.id === 'secondary-target-language' || input.id === 'fallback-language') {
        const noneButton = document.createElement('button');
        noneButton.type = 'button';
        noneButton.className = 'language-combo-option';
        if (!currentValue) {
            noneButton.classList.add('active');
        }

        const noneCode = document.createElement('span');
        noneCode.className = 'language-combo-option-plain';
        noneCode.textContent = t('select.none');
        noneButton.appendChild(noneCode);

        noneButton.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });

        noneButton.addEventListener('click', () => {
            input.value = '';
            syncLanguageComboClearVisibility(combo);
            renderLanguageComboMenu(combo);
            closeLanguageMenus();
            onSettingChange(input);
            input.focus();
        });

        menu.appendChild(noneButton);
    }

    if (input.id === 'source-language') {
        const autoButton = document.createElement('button');
        autoButton.type = 'button';
        autoButton.className = 'language-combo-option';
        const isAuto = !currentValue || currentValue === 'auto' || currentValue === 'auto-detect';
        if (isAuto) {
            autoButton.classList.add('active');
        }

        const autoCode = document.createElement('span');
        autoCode.className = 'language-combo-option-code';
        autoCode.textContent = 'auto';

        const autoLabel = document.createElement('span');
        autoLabel.className = 'language-combo-option-label';
        autoLabel.textContent = t('sourceLang.auto');

        autoButton.appendChild(autoCode);
        autoButton.appendChild(autoLabel);

        autoButton.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });

        autoButton.addEventListener('click', () => {
            input.value = '';
            syncLanguageComboClearVisibility(combo);
            renderLanguageComboMenu(combo);
            closeLanguageMenus();
            onSettingChange(input);
            input.focus();
        });

        menu.appendChild(autoButton);
    }

    const comboLanguageOptions = input.id === 'source-language' ? SOURCE_LANGUAGE_COMBO_OPTIONS : LANGUAGE_OPTIONS;

    comboLanguageOptions.forEach((option) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'language-combo-option';
        let isActive = option.code.toLowerCase() === currentValue;
        if (input.id === 'source-language') {
            const oc = option.code.toLowerCase();
            if (oc === 'zh') {
                isActive = ['zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant'].includes(currentValue);
            } else if (oc === 'en') {
                isActive = currentValue === 'en' || currentValue === 'en-gb' || currentValue === 'en-us';
            }
        }
        if (isActive) {
            button.classList.add('active');
        }

        const code = document.createElement('span');
        code.className = 'language-combo-option-code';
        code.textContent = option.code;

        const label = document.createElement('span');
        label.className = 'language-combo-option-label';
        label.textContent = t(option.labelKey);

        button.appendChild(code);
        button.appendChild(label);

        button.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });

        button.addEventListener('click', () => {
            input.value = option.code;
            syncLanguageComboClearVisibility(combo);
            renderLanguageComboMenu(combo);
            closeLanguageMenus();
            if (input.id === 'target-language') {
                updateFuriganaVisibility();
            }
            if (input.id.startsWith('quick-lang-')) {
                onQuickLangChange();
            } else {
                onSettingChange(input);
            }
            input.focus();
        });

        menu.appendChild(button);
    });
}

function renderLanguageComboMenus() {
    document.querySelectorAll('.language-combo').forEach((combo) => {
        renderLanguageComboMenu(combo);
    });
}

function setupLanguageComboboxes() {
    document.querySelectorAll('.language-combo').forEach((combo) => {
        if (combo.dataset.initialized === 'true') {
            return;
        }

        const input = combo.querySelector('.language-combo-input');
        const toggle = combo.querySelector('.language-combo-toggle');

        if (!input || !toggle) {
            return;
        }

        const openMenu = () => {
            closeLanguageMenus(combo);
            renderLanguageComboMenu(combo);
            combo.classList.add('open');
        };

        toggle.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (combo.classList.contains('open')) {
                combo.classList.remove('open');
            } else {
                openMenu();
            }
            input.focus();
        });

        input.addEventListener('input', () => {
            syncLanguageComboClearVisibility(combo);
            if (combo.classList.contains('open')) {
                renderLanguageComboMenu(combo);
            }
        });

        input.addEventListener('keydown', (event) => {
            if (event.key === 'ArrowDown') {
                event.preventDefault();
                openMenu();
            } else if (event.key === 'Escape') {
                combo.classList.remove('open');
            }
        });

        const clearBtn = combo.querySelector('.language-combo-clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                input.value = '';
                syncLanguageComboClearVisibility(combo);
                closeLanguageMenus(combo);
                renderLanguageComboMenu(combo);
                onSettingChange(input);
                input.focus();
            });
        }

        if (input.id === 'source-language') {
            input.addEventListener('blur', () => {
                normalizeSourceLanguageInputOnBlur(input);
                syncLanguageComboClearVisibility(combo);
                onSettingChange(input);
            });
        }

        syncLanguageComboClearVisibility(combo);

        combo.dataset.initialized = 'true';
    });

    refreshLanguageComboClearLabels();
    renderLanguageComboMenus();
}

const QUICK_LANG_STORAGE_KEY = 'panel_quick_languages';
const QUICK_LANG_BAR_ENABLED_KEY = 'panel_quick_lang_bar_enabled';
const QUICK_LANG_DEFAULTS = ['en', 'zh-CN', 'ja', 'ko'];

function loadQuickLanguageSettings() {
    try {
        const toggle = document.getElementById('enable-quick-lang-bar');
        if (toggle) {
            const storedEnabled = localStorage.getItem(QUICK_LANG_BAR_ENABLED_KEY);
            toggle.checked = storedEnabled === null ? true : storedEnabled === 'true';
        }

        const stored = localStorage.getItem(QUICK_LANG_STORAGE_KEY);
        let langs = QUICK_LANG_DEFAULTS;
        if (stored) {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed) && parsed.length === 4) {
                langs = parsed;
            }
        }
        for (let i = 0; i < 4; i++) {
            const input = document.getElementById(`quick-lang-${i + 1}`);
            if (input) {
                input.value = langs[i] || QUICK_LANG_DEFAULTS[i];
            }
        }
    } catch (e) {
        console.warn('Failed to load quick language settings:', e);
    }
}

function saveQuickLanguageSettings() {
    const langs = [];
    for (let i = 0; i < 4; i++) {
        const input = document.getElementById(`quick-lang-${i + 1}`);
        langs.push(input ? input.value.trim() || QUICK_LANG_DEFAULTS[i] : QUICK_LANG_DEFAULTS[i]);
    }
    localStorage.setItem(QUICK_LANG_STORAGE_KEY, JSON.stringify(langs));
}

function resetQuickLanguageSettings() {
    for (let i = 0; i < 4; i++) {
        const input = document.getElementById(`quick-lang-${i + 1}`);
        if (input) {
            input.value = QUICK_LANG_DEFAULTS[i];
        }
    }

    const toggle = document.getElementById('enable-quick-lang-bar');
    if (toggle) {
        toggle.checked = true;
    }

    localStorage.setItem(QUICK_LANG_STORAGE_KEY, JSON.stringify(QUICK_LANG_DEFAULTS));
    localStorage.removeItem(QUICK_LANG_BAR_ENABLED_KEY);
}

function onQuickLangChange() {
    saveQuickLanguageSettings();
}

function onEnableQuickLangBarChange() {
    const toggle = document.getElementById('enable-quick-lang-bar');
    if (!toggle) return;
    localStorage.setItem(QUICK_LANG_BAR_ENABLED_KEY, toggle.checked.toString());
}

function getQuickLanguageSettingsForPanel() {
    const languages = [];
    for (let i = 0; i < 4; i++) {
        const input = document.getElementById(`quick-lang-${i + 1}`);
        languages.push(input ? input.value.trim() || QUICK_LANG_DEFAULTS[i] : QUICK_LANG_DEFAULTS[i]);
    }

    const toggle = document.getElementById('enable-quick-lang-bar');
    return {
        enabled: toggle ? toggle.checked : true,
        languages,
    };
}
// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', async function () {
    // 先初始化 i18n 系统
    if (window.i18n) {
        window.i18n.initI18n();
    }

    setupLanguageComboboxes();
    await loadServerFeatures();

    document.addEventListener('click', function (event) {
        if (!event.target.closest('.language-combo')) {
            closeLanguageMenus();
        }
    });

    const targetLangInput = document.getElementById('target-language');
    if (targetLangInput) {
        targetLangInput.addEventListener('input', updateFuriganaVisibility);
    }

    const hadSavedConfig = !!localStorage.getItem(CONFIG_STORAGE_KEY);
    loadConfigFromLocalStorage();
    if (hadSavedConfig) {
        await reconcileMainConfigWithServerAtStartup();
    } else {
        await loadConfigFromServer();
    }

    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.disabled = false;
    }

    loadPanelFloatingModeSetting();
    loadQuickLanguageSettings();
    loadAPIKeys();
    initializeCollapsibleStates();
    applyStoredExtraBodyForActiveLLMTemplate();
    setupSecretVisibilityToggles();
    applyAsrBackendLocks();
    updateOscCompatModeUi();
    loadEnvStatus();
    refreshMicDevices(true);
    setupMicDeviceAutoRefresh();
    updateDashscopeKeyFieldState();
    updateStatus();
    updateIpcStatus();
    // 每2秒更新一次状态
    setInterval(updateStatus, 2000);
    setInterval(updateIpcStatus, 2500);
    setInterval(() => {
        if (isLocalAsrUiEnabled()) {
            void refreshLocalAsrStatus();
        }
    }, 3000);

    // 显示配置保存提示
    showConfigStorageInfo();

    const switchLlmStreamingBtn = document.getElementById('switch-to-llm-streaming-btn');
    if (switchLlmStreamingBtn) {
        switchLlmStreamingBtn.addEventListener('click', switchToLLMStreamingTranslation);
    }
});

document.addEventListener('i18n:languageChanged', function () {
    const useInternational = document.getElementById('use-international-endpoint')?.checked ?? false;
    updateAsrOptionsForInternational(useInternational);
    updateLocalAsrEngineHint();
    updateLocalAsrUiVisibility();
    renderLanguageComboMenus();
    refreshLanguageComboClearLabels();
    refreshMicDevices(true);
    updateDashscopeKeyFieldState();
    updateStatus();
    updateLLMTemplateKeySourceHint();
});

// 从服务器加载环境变量状态
async function loadEnvStatus() {
    try {
        const response = await fetch(`${API_BASE}/env`);
        if (!response.ok) return;
        const data = await response.json();
        if (data && data.llm) {
            envStatus.llm.api_key_set = !!data.llm.api_key_set;
        }
    } catch (e) {
        // 静默失败：不影响其它功能
        console.warn('获取环境变量状态失败:', e);
    }
}

/**
 * Windows/PortAudio 下 get_default_input_device_info 的设备名常被截断（约 32 字符），
 * 而枚举列表里的名称是完整的；且默认设备 index 可能因 Host API 过滤与列表不一致。
 * 用列表项解析「系统默认」应展示的完整名称。
 */
function resolveDefaultMicDisplayName(devices, defaultIndex, defaultNameApi) {
    const api = (defaultNameApi || '').trim();
    const fromIndex = devices.find((ad) => ad.index === defaultIndex);
    if (fromIndex && fromIndex.name) {
        return String(fromIndex.name).trim();
    }
    if (!api) return '';
    const exact = devices.find((d) => d.name && String(d.name).trim() === api);
    if (exact && exact.name) {
        return String(exact.name).trim();
    }
    let best = '';
    for (const d of devices) {
        const n = d.name ? String(d.name).trim() : '';
        if (!n || !n.startsWith(api)) continue;
        if (n.length > best.length) best = n;
    }
    return best || api;
}

// 系统默认麦克风变化且当前选用「系统默认」时，保存并重启服务以重新打开采集流
async function maybeRestartForNewSystemDefaultMic() {
    try {
        const statusRes = await fetch(`${API_BASE}/status`);
        const status = await statusRes.json();
        if (getServiceLifecycle(status) !== 'running') return;
        const ok = await saveConfig(true);
        if (!ok) return;
        await restartService();
    } catch (e) {
        console.warn('默认麦克风变更后重启服务失败:', e);
    }
}

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
        const t = window.i18n ? window.i18n.t : (key) => key;
        const defaultIndex = data.default_index;
        const defaultNameApi = data.default_name ? String(data.default_name).trim() : '';
        const displayDefaultName = resolveDefaultMicDisplayName(devices, defaultIndex, defaultNameApi);
        defaultOpt.textContent = displayDefaultName
            ? (window.i18n
                ? window.i18n.t('option.systemDefaultWithDevice', { name: displayDefaultName })
                : `系统默认（${displayDefaultName}）`)
            : (window.i18n ? window.i18n.t('option.systemDefault') : '系统默认');
        micSelect.appendChild(defaultOpt);

        const indices = new Set();
        devices.forEach((d) => {
            const idx = d.index;
            if (idx === undefined || idx === null) return;
            const idxStr = String(idx);
            indices.add(idxStr);

            const opt = document.createElement('option');
            opt.value = idxStr;
            const name = d.name ? String(d.name) : `${t('label.device')} ${idxStr}`;
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

        const identity = `${defaultIndex ?? 'none'}|${displayDefaultName}`;
        const prevIdentity = lastMicDefaultIdentity;
        lastMicDefaultIdentity = identity;
        if (prevIdentity !== null && identity !== prevIdentity && micSelect.value === '') {
            void maybeRestartForNewSystemDefaultMic();
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
    const llmKey = localStorage.getItem('llm_api_key');
    const doubaoKey = localStorage.getItem('doubao_api_key');
    const useInternational = localStorage.getItem('use_international_endpoint') === 'true';

    if (dashscopeKey) document.getElementById('dashscope-api-key').value = dashscopeKey;
    if (deeplKey) document.getElementById('deepl-api-key').value = deeplKey;
    if (doubaoKey) document.getElementById('doubao-api-key').value = doubaoKey;

    const sonioxKey = localStorage.getItem('soniox_api_key');
    if (sonioxKey) document.getElementById('soniox-api-key').value = sonioxKey;

    const activeTemplate = ensureSelectedLLMTemplate();
    const storedTemplateKey = getStoredLLMTemplateKey(activeTemplate);
    if (storedTemplateKey) {
        document.getElementById('llm-api-key').value = storedTemplateKey;
    } else if (llmKey) {
        document.getElementById('llm-api-key').value = llmKey;
        persistCurrentLLMTemplateKey();
    }

    if (activeTemplate === 'openrouter') {
        const storedOpenRouterModel = getStoredLLMTemplateModel(activeTemplate);
        if (storedOpenRouterModel) {
            document.getElementById('llm-model').value = storedOpenRouterModel;
        }
    }

    document.getElementById('use-international-endpoint').checked = useInternational;

    // 应用国际版设置对 ASR 选项的影响
    updateAsrOptionsForInternational(useInternational);

    // 如果已有DashScope API Key，自动折叠API Keys区域
    if (dashscopeKey) {
        const apiKeysSection = document.getElementById('api-keys');
        const apiKeysIcon = document.getElementById('api-keys-icon');
        if (apiKeysSection && !apiKeysSection.classList.contains('collapsed')) {
            apiKeysSection.classList.add('collapsed');
            updateCollapsibleIcon(apiKeysIcon, true);
            syncCollapsibleContainerState(apiKeysSection);
        }
    }

    // 添加API Key change事件监听
    document.getElementById('dashscope-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('deepl-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('llm-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('doubao-api-key').addEventListener('input', saveAPIKey);
    document.getElementById('soniox-api-key').addEventListener('input', saveAPIKey);

    ['llm-base-url', 'llm-model', 'openai-compat-extra-body-json'].forEach((inputId) => {
        const input = document.getElementById(inputId);
        if (!input) return;
        input.addEventListener('input', (event) => {
            if (event.target.id === 'llm-base-url') {
                persistCurrentLLMTemplateBaseUrl();
            }
            if (event.target.id === 'llm-model') {
                persistCurrentLLMTemplateModel();
            }
            if (event.target.id === 'openai-compat-extra-body-json') {
                persistCurrentLLMTemplateExtraBody();
            }
            syncLLMTemplateKeySourceHintFromInputs();
            onSettingChange(event.target);
        });
    });

    const llmBaseUrlInput = document.getElementById('llm-base-url');
    if (llmBaseUrlInput && llmBaseUrlInput.dataset.llmExtrabodySyncBound !== 'true') {
        llmBaseUrlInput.dataset.llmExtrabodySyncBound = 'true';
        llmBaseUrlInput.addEventListener('blur', () => {
            applyStoredExtraBodyForActiveLLMTemplate();
        });
    }

    const extraBodyEl = document.getElementById('openai-compat-extra-body-json');
    if (extraBodyEl && extraBodyEl.dataset.llmExtrabodyBlurPersistBound !== 'true') {
        extraBodyEl.dataset.llmExtrabodyBlurPersistBound = 'true';
        extraBodyEl.addEventListener('blur', () => {
            persistCurrentLLMTemplateExtraBody();
        });
    }

    const parallelModeEl = document.getElementById('llm-parallel-fastest-mode');
    if (parallelModeEl && parallelModeEl.dataset.llmParallelPersistBound !== 'true') {
        parallelModeEl.dataset.llmParallelPersistBound = 'true';
        parallelModeEl.addEventListener('change', () => {
            persistCurrentLLMTemplateParallelMode();
            syncLLMTemplateKeySourceHintFromInputs();
            onSettingChange(parallelModeEl);
        });
    }
}

// 处理国际版端点开关变化
function handleInternationalEndpointChange(event) {
    const useInternational = event.target.checked;

    // 保存到 localStorage
    localStorage.setItem('use_international_endpoint', useInternational.toString());

    // 更新 ASR 选项
    updateAsrOptionsForInternational(useInternational);
    updateLLMTemplateKeySourceHint();

    // 触发配置保存
    onSettingChange(event.target);
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

function applyAsrBackendLocks() {
    const asrBackendSelect = document.getElementById('asr-backend');
    const micControlToggle = document.getElementById('enable-mic-control');
    const partialResultsToggle = document.getElementById('show-partial-results');
    const enableTranslationToggle = document.getElementById('enable-translation');
    const translationApiSelect = document.getElementById('translation-api-type');
    const streamingModeToggle = document.getElementById('openrouter-streaming-mode');
    if (!asrBackendSelect || !micControlToggle || !partialResultsToggle || !enableTranslationToggle || !translationApiSelect || !streamingModeToggle) {
        return;
    }

    const isDoubaoFile = asrBackendSelect.value === 'doubao_file';
    const apiTypeVal = translationApiSelect.value;
    const isStreamingTranslation = enableTranslationToggle.checked
        && (
            (apiTypeVal === 'openrouter' && streamingModeToggle.checked)
            || apiTypeVal === 'openrouter_streaming_deepl_hybrid'
        );

    if (isDoubaoFile) {
        micControlToggle.disabled = true;
    } else {
        micControlToggle.disabled = false;
    }

    if (isDoubaoFile || isStreamingTranslation) {
        partialResultsToggle.checked = false;
        partialResultsToggle.disabled = true;
    } else {
        partialResultsToggle.disabled = false;
    }
}

// 保存API Key到localStorage
function saveAPIKey(event) {
    persistSecretInputValue(event.target.id);

    if (event.target.id === 'llm-api-key') {
        persistCurrentLLMTemplateKey();
        envStatus.llm.api_key_set = !!event.target.value.trim();
    }

    // 触发配置自动保存
    onSettingChange(event.target);
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
                document.getElementById('secondary-target-language').value = config.translation.secondary_target_language || '';
                document.getElementById('fallback-language').value = config.translation.fallback_language || 'en';
                // 处理 LLM 流式模式的特殊情况
                const apiType = config.translation.api_type || 'qwen_mt';
                if (apiType === 'openrouter_streaming') {
                    document.getElementById('translation-api-type').value = 'openrouter';
                    document.getElementById('openrouter-streaming-mode').checked = true;
                    document.getElementById('openrouter-streaming-mode').disabled = false;
                } else if (apiType === 'openrouter_streaming_deepl_hybrid') {
                    document.getElementById('translation-api-type').value = 'openrouter_streaming_deepl_hybrid';
                } else {
                    document.getElementById('translation-api-type').value = apiType;
                    document.getElementById('openrouter-streaming-mode').checked = false;
                    document.getElementById('openrouter-streaming-mode').disabled = false;
                }
                document.getElementById('llm-base-url').value = config.translation.llm_base_url || '';
                document.getElementById('llm-model').value = config.translation.llm_model || '';
                document.getElementById('llm-translation-formality').value =
                    sanitizeLLMTranslationFormality(config.translation.llm_translation_formality);
                document.getElementById('llm-translation-style').value =
                    sanitizeLLMTranslationStyle(config.translation.llm_translation_style);
                document.getElementById('openai-compat-extra-body-json').value = config.translation.openai_compat_extra_body_json || '';
                setLLMParallelFastestModeSelect(
                    resolveLLMParallelFastestModeFromStoredTranslation(config.translation)
                );
                ensureSelectedLLMTemplate(config.translation);
                applySourceLanguageInputFromStored(config.translation.source_language ?? 'auto');
                document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
                document.getElementById('enable-furigana').checked = config.translation.enable_furigana ?? false;
                document.getElementById('enable-pinyin').checked = config.translation.enable_pinyin ?? false;
                document.getElementById('remove-trailing-period').checked = config.translation.remove_trailing_period ?? false;
                document.getElementById('text-fancy-style').value = config.translation.text_fancy_style || 'none';
                document.getElementById('enable-reverse-translation').checked = config.translation.enable_reverse_translation ?? false;

                const showTag = document.getElementById('show-original-and-lang-tag');
                if (showTag) {
                    showTag.checked = config.translation.show_original_and_lang_tag ?? true;
                }
            }

            if (config.osc) {
                const oscCompatEl = document.getElementById('osc-compat-mode');
                if (oscCompatEl) {
                    oscCompatEl.checked = config.osc.compat_mode ?? false;
                }
                const oscCompatPortEl = document.getElementById('osc-compat-listen-port');
                if (oscCompatPortEl) {
                    oscCompatPortEl.value = config.osc.compat_listen_port ?? 9001;
                }
                const bypassOscEl = document.getElementById('bypass-osc-udp-port-check');
                if (bypassOscEl) {
                    bypassOscEl.checked = config.osc.bypass_udp_port_check ?? false;
                }
                const oscPortEl = document.getElementById('osc-send-target-port');
                if (oscPortEl) {
                    oscPortEl.value = config.osc.send_target_port ?? 9000;
                }
                const oscSendErrorsEl = document.getElementById('osc-send-error-messages');
                if (oscSendErrorsEl) {
                    oscSendErrorsEl.checked = config.osc.send_error_messages ?? false;
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
                document.getElementById('asr-backend').value = sanitizeAsrBackendValue(config.asr.preferred_backend || 'qwen');
                document.getElementById('enable-hot-words').checked = config.asr.enable_hot_words ?? true;
                document.getElementById('enable-vad').checked = config.asr.enable_vad ?? true;
                document.getElementById('vad-threshold').value = config.asr.vad_threshold || 0.2;
                document.getElementById('vad-silence-duration').value = config.asr.vad_silence_duration_ms || 800;

                // 加载国际版设置
                const useInternational = config.asr.use_international_endpoint ?? false;
                document.getElementById('use-international-endpoint').checked = useInternational;
                localStorage.setItem('use_international_endpoint', useInternational.toString());
                updateAsrOptionsForInternational(useInternational);
            }

            if (isLocalAsrUiEnabled()) {
                applyLocalAsrConfig(config.local_asr || {});
            }

            if (config.language_detector) {
                document.getElementById('language-detector').value = config.language_detector.type || 'cjke';
            }

            applyAutoLanguageDetectorIfNeeded();

            if (config.panel) {
                document.getElementById('panel-width').value = config.panel.width || 600;
                getNormalizedPanelWidth();
            }

            const bypassOscEl = document.getElementById('bypass-osc-udp-port-check');
            if (bypassOscEl && config.osc) {
                bypassOscEl.checked = config.osc.bypass_udp_port_check ?? false;
            }
            const oscCompatEl = document.getElementById('osc-compat-mode');
            if (oscCompatEl && config.osc) {
                oscCompatEl.checked = config.osc.compat_mode ?? false;
            }
            const oscCompatPortEl = document.getElementById('osc-compat-listen-port');
            if (oscCompatPortEl && config.osc) {
                const clp = config.osc.compat_listen_port;
                const p =
                    clp == null || clp === ''
                        ? 9001
                        : Math.max(1, Math.min(65535, parseInt(clp, 10) || 9001));
                oscCompatPortEl.value = String(p);
            }
            const oscPortEl = document.getElementById('osc-send-target-port');
            if (oscPortEl && config.osc) {
                const stp = config.osc.send_target_port;
                const p =
                    stp == null || stp === ''
                        ? 9000
                        : Math.max(1, Math.min(65535, parseInt(stp, 10) || 9000));
                oscPortEl.value = String(p);
            }

            if (config.smart_target_language) {
                const stl = config.smart_target_language;
                if (document.getElementById('smart-target-primary-enabled')) {
                    document.getElementById('smart-target-primary-enabled').checked = stl.primary_enabled ?? true;
                }
                if (document.getElementById('smart-target-secondary-enabled')) {
                    document.getElementById('smart-target-secondary-enabled').checked = stl.secondary_enabled ?? false;
                }
                if (document.getElementById('smart-target-strategy')) {
                    document.getElementById('smart-target-strategy').value = stl.strategy || 'auto';
                }
                if (document.getElementById('smart-target-window-size')) {
                    document.getElementById('smart-target-window-size').value = stl.window_size ?? 5;
                }
                if (document.getElementById('smart-target-exclude-self')) {
                    document.getElementById('smart-target-exclude-self').checked = stl.exclude_self_language ?? true;
                }
                if (document.getElementById('smart-target-min-samples')) {
                    document.getElementById('smart-target-min-samples').value = stl.min_samples ?? 3;
                }
                updateSmartTargetVisibility();
            }

            console.log('✓ 已从浏览器加载配置');
        } else {
            // 如果没有保存的配置，使用前端默认值（由页面初始化处再拉取服务器并 / 或对账）
            loadDefaultConfig();
        }

        // 根据翻译开关显示/隐藏翻译选项
        updateOpenRouterStreamingUi();
        toggleTranslationOptions();
        updateFuriganaVisibility();
        updateLLMSettingsVisibility();
        updateSensitiveWordsHint();
        applyAsrBackendLocks();
        updateOscCompatModeUi();
        ensureSelectedLLMTemplate();
        syncLLMTemplateKeySourceHintFromInputs();
        syncAllLanguageComboClearButtons();
        updateDashscopeKeyFieldState();
        applyStoredExtraBodyForActiveLLMTemplate();
        updateLocalAsrUiVisibility();

    } catch (error) {
        console.error('加载本地配置失败:', error);
        // 出错时使用前端默认值
        loadDefaultConfig();
        toggleTranslationOptions();
        updateFuriganaVisibility();
        updateLLMSettingsVisibility();
        updateSensitiveWordsHint();
        applyAsrBackendLocks();
        updateOscCompatModeUi();
        ensureSelectedLLMTemplate();
        syncLLMTemplateKeySourceHintFromInputs();
        syncAllLanguageComboClearButtons();
        updateDashscopeKeyFieldState();
        applyStoredExtraBodyForActiveLLMTemplate();
        updateLocalAsrUiVisibility();
    }
}

// 加载前端默认配置（不依赖服务器）
function loadDefaultConfig() {
    // 翻译配置
    document.getElementById('enable-translation').checked = true;
    document.getElementById('target-language').value = 'ja';
    document.getElementById('secondary-target-language').value = '';
    document.getElementById('fallback-language').value = 'en';
    document.getElementById('translation-api-type').value = 'qwen_mt';
    applySourceLanguageInputFromStored('auto');
    document.getElementById('show-partial-results').checked = false;
    document.getElementById('enable-furigana').checked = false;
    document.getElementById('enable-pinyin').checked = false;
    document.getElementById('remove-trailing-period').checked = false;
    document.getElementById('text-fancy-style').value = 'none';
    document.getElementById('enable-reverse-translation').checked = false;
    const streamingModeEl = document.getElementById('openrouter-streaming-mode');
    if (streamingModeEl) {
        streamingModeEl.checked = false;
        streamingModeEl.disabled = false;
    }
    setLLMParallelFastestModeSelect('off');
    document.getElementById('llm-base-url').value = '';
    document.getElementById('llm-model').value = '';
    document.getElementById('llm-translation-formality').value = DEFAULT_LLM_TRANSLATION_FORMALITY;
    document.getElementById('llm-translation-style').value = DEFAULT_LLM_TRANSLATION_STYLE;
    document.getElementById('openai-compat-extra-body-json').value = '';

    const showTag = document.getElementById('show-original-and-lang-tag');
    if (showTag) showTag.checked = true;

    // 麦克风控制
    document.getElementById('enable-mic-control').checked = true;
    document.getElementById('mute-delay').value = 0.2;

    // 麦克风设备
    const micSelect = document.getElementById('mic-device');
    if (micSelect) micSelect.value = '';

    // ASR 配置
    document.getElementById('asr-backend').value = 'qwen';
    document.getElementById('enable-hot-words').checked = true;
    document.getElementById('enable-vad').checked = true;
    document.getElementById('vad-threshold').value = 0.2;
    document.getElementById('vad-silence-duration').value = 800;
    document.getElementById('use-international-endpoint').checked = false;

    if (isLocalAsrUiEnabled()) {
        applyLocalAsrConfig({
            engine: 'sensevoice',
            vad_mode: 'silero',
            vad_threshold: 0.50,
            min_speech_duration: 1.0,
            max_speech_duration: 30.0,
            silence_duration: 0.8,
            pre_speech_duration: 0.2,
            incremental_asr: true,
            interim_interval: 2.0,
        });
    }

    // 语言检测器
    document.getElementById('language-detector').value = 'cjke';
    applyAutoLanguageDetectorIfNeeded();

    // 小面板
    document.getElementById('panel-width').value = 600;

    const bypassOscDefault = document.getElementById('bypass-osc-udp-port-check');
    if (bypassOscDefault) {
        bypassOscDefault.checked = false;
    }
    const oscCompatDefault = document.getElementById('osc-compat-mode');
    if (oscCompatDefault) {
        oscCompatDefault.checked = false;
    }
    const oscCompatPortDefault = document.getElementById('osc-compat-listen-port');
    if (oscCompatPortDefault) {
        oscCompatPortDefault.value = '9001';
    }
    const oscSendErrorsDefault = document.getElementById('osc-send-error-messages');
    if (oscSendErrorsDefault) {
        oscSendErrorsDefault.checked = false;
    }
    const oscPortDefault = document.getElementById('osc-send-target-port');
    if (oscPortDefault) {
        oscPortDefault.value = '9000';
    }

    console.log('✓ 已加载前端默认配置');
    ensureSelectedLLMTemplate({ llm_template: DEFAULT_LLM_TEMPLATE_NAME });
    updateOpenRouterStreamingUi();
    updateFuriganaVisibility();
    updateLLMSettingsVisibility();
    updateSensitiveWordsHint();
    applyAsrBackendLocks();
    updateOscCompatModeUi();
    syncLLMTemplateKeySourceHintFromInputs();
    syncAllLanguageComboClearButtons();
    updateDashscopeKeyFieldState();
    updateLocalAsrUiVisibility();
}

function refreshUiAfterServerConfigApply() {
    updateOpenRouterStreamingUi();
    toggleTranslationOptions();
    updateFuriganaVisibility();
    updateLLMSettingsVisibility();
    updateSensitiveWordsHint();
    applyAsrBackendLocks();
    updateOscCompatModeUi();
    syncLLMTemplateKeySourceHintFromInputs();
    syncAllLanguageComboClearButtons();
    updateDashscopeKeyFieldState();
    applyStoredExtraBodyForActiveLLMTemplate();
    updateLocalAsrUiVisibility();
    renderLanguageComboMenus();
}

/** 将 GET /api/config 的 JSON 填回表单（不写入 localStorage、不改 touch） */
function applyServerConfigPayload(config) {
    if (config.features) {
        featureFlags = config.features;
    }
    updateLocalAsrUiVisibility();

    document.getElementById('enable-translation').checked = config.translation.enable_translation;
    document.getElementById('target-language').value = config.translation.target_language;
    document.getElementById('secondary-target-language').value = config.translation.secondary_target_language || '';
    document.getElementById('fallback-language').value = config.translation.fallback_language || '';
    const serverApiType = config.translation.api_type;
    if (serverApiType === 'openrouter_streaming') {
        document.getElementById('translation-api-type').value = 'openrouter';
        document.getElementById('openrouter-streaming-mode').checked = true;
        document.getElementById('openrouter-streaming-mode').disabled = false;
    } else if (serverApiType === 'openrouter_streaming_deepl_hybrid') {
        document.getElementById('translation-api-type').value = 'openrouter_streaming_deepl_hybrid';
    } else {
        document.getElementById('translation-api-type').value = serverApiType;
        document.getElementById('openrouter-streaming-mode').checked = false;
        document.getElementById('openrouter-streaming-mode').disabled = false;
    }
    document.getElementById('llm-base-url').value = config.translation.llm_base_url || '';
    document.getElementById('llm-model').value = config.translation.llm_model || '';
    document.getElementById('llm-translation-formality').value =
        sanitizeLLMTranslationFormality(config.translation.llm_translation_formality);
    document.getElementById('llm-translation-style').value =
        sanitizeLLMTranslationStyle(config.translation.llm_translation_style);
    document.getElementById('openai-compat-extra-body-json').value = config.translation.openai_compat_extra_body_json || '';
    setLLMParallelFastestModeSelect(
        resolveLLMParallelFastestModeFromStoredTranslation(config.translation),
    );
    ensureSelectedLLMTemplate(config.translation);
    document.getElementById('show-partial-results').checked = config.translation.show_partial_results ?? false;
    document.getElementById('enable-furigana').checked = config.translation.enable_furigana ?? false;
    document.getElementById('enable-pinyin').checked = config.translation.enable_pinyin ?? false;
    document.getElementById('remove-trailing-period').checked = config.translation.remove_trailing_period ?? false;
    document.getElementById('text-fancy-style').value = config.translation.text_fancy_style || 'none';
    document.getElementById('enable-reverse-translation').checked = config.translation.enable_reverse_translation ?? false;
    const showTag = document.getElementById('show-original-and-lang-tag');
    if (showTag) {
        showTag.checked = config.translation.show_original_and_lang_tag ?? true;
    }

    document.getElementById('enable-mic-control').checked = config.mic_control.enable_mic_control;
    document.getElementById('mute-delay').value = config.mic_control.mute_delay_seconds;
    const micSelect = document.getElementById('mic-device');
    if (micSelect && config.mic_control) {
        const idx = config.mic_control.mic_device_index;
        micSelect.value = (idx === undefined || idx === null) ? '' : String(idx);
    }

    document.getElementById('asr-backend').value = sanitizeAsrBackendValue(config.asr.preferred_backend);
    document.getElementById('enable-hot-words').checked = config.asr.enable_hot_words;
    document.getElementById('enable-vad').checked = config.asr.enable_vad;
    document.getElementById('vad-threshold').value = config.asr.vad_threshold;
    document.getElementById('vad-silence-duration').value = config.asr.vad_silence_duration_ms;
    const useInternational = config.asr.use_international_endpoint ?? false;
    document.getElementById('use-international-endpoint').checked = useInternational;
    localStorage.setItem('use_international_endpoint', useInternational.toString());
    updateAsrOptionsForInternational(useInternational);

    document.getElementById('language-detector').value = config.language_detector.type;
    applySourceLanguageInputFromStored(config.translation.source_language);
    applyAutoLanguageDetectorIfNeeded();
    document.getElementById('panel-width').value = (config.panel && config.panel.width) || 600;
    getNormalizedPanelWidth();
    const bypassOscEl = document.getElementById('bypass-osc-udp-port-check');
    if (bypassOscEl && config.osc) {
        bypassOscEl.checked = config.osc.bypass_udp_port_check ?? false;
    }
    const oscCompatEl = document.getElementById('osc-compat-mode');
    if (oscCompatEl && config.osc) {
        oscCompatEl.checked = config.osc.compat_mode ?? false;
    }
    const oscCompatPortEl = document.getElementById('osc-compat-listen-port');
    if (oscCompatPortEl && config.osc) {
        const clp = config.osc.compat_listen_port;
        const p =
            clp == null || clp === ''
                ? 9001
                : Math.max(1, Math.min(65535, parseInt(clp, 10) || 9001));
        oscCompatPortEl.value = String(p);
    }
    const oscSendErrorsEl = document.getElementById('osc-send-error-messages');
    if (oscSendErrorsEl && config.osc) {
        oscSendErrorsEl.checked = config.osc.send_error_messages ?? false;
    }
    const oscPortApply = document.getElementById('osc-send-target-port');
    if (oscPortApply && config.osc) {
        const stp = config.osc.send_target_port;
        const p =
            stp == null || stp === ''
                ? 9000
                : Math.max(1, Math.min(65535, parseInt(stp, 10) || 9000));
        oscPortApply.value = String(p);
    }
    if (isLocalAsrUiEnabled()) {
        applyLocalAsrConfig(config.local_asr || {});
    }

    if (config.smart_target_language) {
        const stl = config.smart_target_language;
        if (document.getElementById('smart-target-primary-enabled')) {
            document.getElementById('smart-target-primary-enabled').checked = stl.primary_enabled ?? true;
        }
        if (document.getElementById('smart-target-secondary-enabled')) {
            document.getElementById('smart-target-secondary-enabled').checked = stl.secondary_enabled ?? false;
        }
        if (document.getElementById('smart-target-strategy')) {
            document.getElementById('smart-target-strategy').value = stl.strategy || 'auto';
        }
        if (document.getElementById('smart-target-window-size')) {
            document.getElementById('smart-target-window-size').value = stl.window_size ?? 5;
        }
        if (document.getElementById('smart-target-exclude-self')) {
            document.getElementById('smart-target-exclude-self').checked = stl.exclude_self_language ?? true;
        }
        if (document.getElementById('smart-target-min-samples')) {
            document.getElementById('smart-target-min-samples').value = stl.min_samples ?? 3;
        }
        updateSmartTargetVisibility();
    }
}

/**
 * 打开大面板时与后端对账。
 *
 * 1. 后端 boot_ms ≠ 本地记录 → 后端重启过 → 把大面板本地配置推到后端（用户上次存的就是最终态）
 * 2. 同一运行期 → 比较 config_applied_at_ms 与 touch_main_ms，谁大听谁
 */
async function reconcileMainConfigWithServerAtStartup() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        if (!response.ok) return;
        const serverCfg = await response.json();

        const serverBootMs = Number(serverCfg.backend_boot_ms) || 0;
        const lastSeenBootMs = getLastSeenBootMs();
        const serverAppliedMs = Number(serverCfg.config_applied_at_ms) || 0;
        const clientMs = getMainConfigTouchMs();

        if (serverBootMs !== lastSeenBootMs) {
            // 后端（重新）启动过 → 大面板本地配置推到后端
            setLastSeenBootMs(serverBootMs);
            const ok = await saveConfig(true);
            console.log(ok
                ? '对账：后端新启动，已将本地配置推送到服务器'
                : '对账：后端新启动，本地配置推送失败');
            return;
        }

        // 同一运行期：谁的时间戳更大，听谁的
        if (serverAppliedMs > clientMs) {
            applyServerConfigPayload(serverCfg);
            saveConfigToLocalStorage();
            setMainConfigTouchMs(serverAppliedMs);
            refreshUiAfterServerConfigApply();
            console.log('对账：服务器配置较新，已拉取到本地');
        } else if (clientMs > serverAppliedMs) {
            await saveConfig(true);
            console.log('对账：本地配置较新，已推送到服务器');
        }
        // 相等 → 已同步，无需操作
    } catch (e) {
        console.warn('对账失败:', e);
    }
}

/**
 * 首次打开（localStorage 无配置）时从服务器拉取全量配置。
 */
async function loadConfigFromServer() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const cfg = await response.json();
        applyServerConfigPayload(cfg);
        saveConfigToLocalStorage();

        const appliedMs = Number(cfg.config_applied_at_ms) || 0;
        if (appliedMs > 0) setMainConfigTouchMs(appliedMs);
        const bootMs = Number(cfg.backend_boot_ms) || 0;
        if (bootMs > 0) setLastSeenBootMs(bootMs);

        refreshUiAfterServerConfigApply();
        console.log('已从服务器加载配置（首次打开）');
    } catch (error) {
        console.error('加载服务器配置失败:', error);
    }
}

// 保存配置到 localStorage
function saveConfigToLocalStorage() {
    try {
        applyAsrBackendLocks();

        // 确定实际的 API 类型（LLM + 流式开关 -> openrouter_streaming；混合模式保持独立值）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter_streaming_deepl_hybrid') {
            // 保持 hybrid
        } else if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                secondary_target_language: document.getElementById('secondary-target-language').value || null,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                llm_template: getSelectedLLMTemplateName(),
                llm_base_url: document.getElementById('llm-base-url').value.trim(),
                llm_model: document.getElementById('llm-model').value.trim(),
                llm_translation_formality: sanitizeLLMTranslationFormality(
                    document.getElementById('llm-translation-formality').value
                ),
                llm_translation_style: sanitizeLLMTranslationStyle(
                    document.getElementById('llm-translation-style').value
                ),
                openai_compat_extra_body_json: document.getElementById('openai-compat-extra-body-json').value.trim(),
                llm_parallel_fastest_mode: getLLMParallelFastestModeSelect(),
                source_language: getSourceLanguageEffective(),
                show_partial_results: document.getElementById('show-partial-results').checked,
                enable_furigana: document.getElementById('enable-furigana').checked,
                enable_pinyin: document.getElementById('enable-pinyin').checked,
                remove_trailing_period: document.getElementById('remove-trailing-period').checked,
                text_fancy_style: document.getElementById('text-fancy-style').value || 'none',
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
                preferred_backend: sanitizeAsrBackendValue(document.getElementById('asr-backend').value),
                enable_hot_words: document.getElementById('enable-hot-words').checked,
                enable_vad: document.getElementById('enable-vad').checked,
                vad_threshold: parseFloat(document.getElementById('vad-threshold').value),
                vad_silence_duration_ms: parseInt(document.getElementById('vad-silence-duration').value),
                use_international_endpoint: document.getElementById('use-international-endpoint').checked,
            },
            language_detector: {
                type: document.getElementById('language-detector').value,
            },
            panel: {
                width: getNormalizedPanelWidth(),
            },
            osc: {
                send_target_port: getOscSendTargetPortFromForm(),
                compat_mode: isOscCompatModeEnabled(),
                compat_listen_port: getOscCompatListenPortFromForm(),
                bypass_udp_port_check:
                    document.getElementById('bypass-osc-udp-port-check')?.checked === true,
                send_error_messages:
                    document.getElementById('osc-send-error-messages')?.checked === true,
            },
            smart_target_language: {
                primary_enabled: document.getElementById('smart-target-primary-enabled')?.checked ?? true,
                secondary_enabled: document.getElementById('smart-target-secondary-enabled')?.checked ?? false,
                strategy: document.getElementById('smart-target-strategy')?.value || 'auto',
                window_size: parseInt(document.getElementById('smart-target-window-size')?.value || '5'),
                exclude_self_language: document.getElementById('smart-target-exclude-self')?.checked ?? true,
                min_samples: parseInt(document.getElementById('smart-target-min-samples')?.value || '3'),
            },
            local_asr: isLocalAsrUiEnabled() ? getLocalAsrConfigFromForm() : null,
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

/** 一键：启用翻译、LLM API、流式翻译模式，并走与下拉框变更相同的更新与保存逻辑。 */
function switchToLLMStreamingTranslation() {
    const enableTrans = document.getElementById('enable-translation');
    if (enableTrans && !enableTrans.checked) {
        enableTrans.checked = true;
        toggleTranslationOptions();
    }

    const select = document.getElementById('translation-api-type');
    const streaming = document.getElementById('openrouter-streaming-mode');
    if (!select || !streaming) return;

    select.value = 'openrouter';
    streaming.checked = true;
    select.dispatchEvent(new Event('change', { bubbles: true }));
}

function handleTranslationApiChange(event) {
    const newApi = event.target.value;
    const warningElement = document.getElementById('translation-api-warning');

    updateOpenRouterStreamingUi();
    const expandLlmPanel = (newApi === 'openrouter' || newApi === 'openrouter_streaming_deepl_hybrid')
        && !isLLMConnectionFieldsComplete();
    updateLLMSettingsVisibility(newApi, expandLlmPanel);
    updateSensitiveWordsHint(newApi);
    applyAsrBackendLocks();

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

        updateOpenRouterStreamingUi();
        updateLLMSettingsVisibility(apiSelect.value);
        updateSensitiveWordsHint(apiSelect.value);
        syncLLMTemplateKeySourceHintFromInputs();
    }, 100);
});

// 当设置改变时自动保存（延迟保存，避免频繁请求）
function onSettingChange(changedElement = null) {
    touchMainPanelUserEditedAt();
    applyAsrBackendLocks();
    applyAutoLanguageDetectorIfNeeded();
    updateDashscopeKeyFieldState();

    // 清除之前的定时器
    if (autoSaveTimer) {
        clearTimeout(autoSaveTimer);
    }

    // 200ms后自动保存（更快响应，静默保存）
    autoSaveTimer = setTimeout(async () => {
        // 先保存到本地浏览器
        saveConfigToLocalStorage();

        // 再保存到服务器
        const saveSucceeded = await saveConfig(true); // true表示是自动保存，不重启服务
        if (!saveSucceeded) {
            return;
        }

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
        applyAsrBackendLocks();

        // 确定实际的 API 类型（LLM + 流式开关 -> openrouter_streaming；混合模式保持独立值）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter_streaming_deepl_hybrid') {
            // 保持 hybrid
        } else if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                secondary_target_language: document.getElementById('secondary-target-language').value || null,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                llm_template: getSelectedLLMTemplateName(),
                llm_base_url: document.getElementById('llm-base-url').value.trim(),
                llm_model: document.getElementById('llm-model').value.trim(),
                llm_translation_formality: sanitizeLLMTranslationFormality(
                    document.getElementById('llm-translation-formality').value
                ),
                llm_translation_style: sanitizeLLMTranslationStyle(
                    document.getElementById('llm-translation-style').value
                ),
                openai_compat_extra_body_json: document.getElementById('openai-compat-extra-body-json').value.trim(),
                llm_parallel_fastest_mode: getLLMParallelFastestModeSelect(),
                source_language: getSourceLanguageEffective(),
                show_partial_results: document.getElementById('show-partial-results').checked,
                enable_furigana: document.getElementById('enable-furigana').checked,
                enable_pinyin: document.getElementById('enable-pinyin').checked,
                remove_trailing_period: document.getElementById('remove-trailing-period').checked,
                text_fancy_style: document.getElementById('text-fancy-style').value || 'none',
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
                preferred_backend: sanitizeAsrBackendValue(document.getElementById('asr-backend').value),
                enable_hot_words: document.getElementById('enable-hot-words').checked,
                enable_vad: document.getElementById('enable-vad').checked,
                vad_threshold: parseFloat(document.getElementById('vad-threshold').value),
                vad_silence_duration_ms: parseInt(document.getElementById('vad-silence-duration').value),
            },
            language_detector: {
                type: document.getElementById('language-detector').value,
            },
            smart_target_language: {
                primary_enabled: document.getElementById('smart-target-primary-enabled')?.checked ?? true,
                secondary_enabled: document.getElementById('smart-target-secondary-enabled')?.checked ?? false,
                strategy: document.getElementById('smart-target-strategy')?.value || 'auto',
                window_size: parseInt(document.getElementById('smart-target-window-size')?.value || '5'),
                exclude_self_language: document.getElementById('smart-target-exclude-self')?.checked ?? true,
                min_samples: parseInt(document.getElementById('smart-target-min-samples')?.value || '3'),
            },
            panel: {
                width: getNormalizedPanelWidth(),
            },
            osc: {
                send_target_port: getOscSendTargetPortFromForm(),
                compat_mode: isOscCompatModeEnabled(),
                compat_listen_port: getOscCompatListenPortFromForm(),
                bypass_udp_port_check:
                    document.getElementById('bypass-osc-udp-port-check')?.checked === true,
                send_error_messages:
                    document.getElementById('osc-send-error-messages')?.checked === true,
            },
            local_asr: isLocalAsrUiEnabled() ? getLocalAsrConfigFromForm() : null,
            api_keys: {
                llm: document.getElementById('llm-api-key').value.trim(),
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
            if (result.config_applied_at_ms != null && result.config_applied_at_ms !== undefined) {
                setMainConfigTouchMs(result.config_applied_at_ms);
            }
            if (!autoSave) {
                showMessage(t('msg.configSaved'), 'success');
            }
            return true;
        } else {
            const localizedMsg = localizeBackendMessage(result.message_id, result.message);
            showMessage(t('msg.saveConfigFailed') + ': ' + localizedMsg, 'error');
            return false;
        }
    } catch (error) {
        console.error('保存配置失败:', error);
        if (!autoSave) {
            showMessage(t('msg.saveConfigFailed'), 'error');
        }
        return false;
    }
}

// 更新服务状态
function getServiceLifecycle(status) {
    return status?.lifecycle || (status?.running ? 'running' : 'stopped');
}

function renderServiceLifecycle(status, t) {
    const lifecycle = getServiceLifecycle(status);
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    statusDot.classList.remove('running', 'starting', 'stopping');

    if (lifecycle === 'running') {
        statusText.textContent = t('status.running');
        statusDot.classList.add('running');
        startBtn.disabled = true;
        startBtn.textContent = t('btn.startService');
        stopBtn.disabled = false;
        stopBtn.textContent = t('btn.stopService');
        return;
    }

    if (lifecycle === 'starting') {
        statusText.textContent = t('status.starting');
        statusDot.classList.add('starting');
        startBtn.disabled = true;
        startBtn.textContent = t('btn.starting');
        stopBtn.disabled = true;
        stopBtn.textContent = t('btn.stopService');
        return;
    }

    if (lifecycle === 'stopping') {
        statusText.textContent = t('status.stopping');
        statusDot.classList.add('stopping');
        startBtn.disabled = true;
        startBtn.textContent = t('btn.startService');
        stopBtn.disabled = true;
        stopBtn.textContent = t('btn.stopping');
        return;
    }

    statusText.textContent = t('status.notRunning');
    startBtn.disabled = false;
    startBtn.textContent = t('btn.startService');
    stopBtn.disabled = true;
    stopBtn.textContent = t('btn.stopService');
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();

        const t = window.i18n ? window.i18n.t : (key) => key;
        renderServiceLifecycle(status, t);

        // 后端时间戳比大面板记录更新 → 有人（小面板等）在别处改了配置 → 拉到大面板
        const serverTs = Number(status.config_applied_at_ms) || 0;
        const mainTs = getMainConfigTouchMs();
        if (serverTs > mainTs && status.target_language) {
            const targetLangInput = document.getElementById('target-language');
            if (targetLangInput && targetLangInput.value !== status.target_language) {
                targetLangInput.value = status.target_language;
                renderLanguageComboMenus();
                saveConfigToLocalStorage();
            }
            setMainConfigTouchMs(serverTs);
        }
    } catch (error) {
        console.error('更新状态失败:', error);
    }
}

/** 根据 /api/udp-port-check 或带 udp_port_conflicts 的响应构建 UDP 端口冲突提示（无冲突则返回空字符串）。 */
function buildUdpPortConflictWarning(payload, t) {
    if (!payload || !Array.isArray(payload.udp_port_conflicts) || !payload.udp_port_conflicts.length) {
        return '';
    }
    const oscPort = payload.osc_udp_port ?? 9000;
    const programs = payload.udp_port_conflicts.map((p) => `${p.name} (PID ${p.pid})`).join(', ');
    return t('msg.udpPortConflictWarning', { port: oscPort, programs });
}

/** 冲突详情 + 已取消操作说明（供错误提示使用）。 */
function buildUdpPortBlockUserMessage(payload, t) {
    const detail = buildUdpPortConflictWarning(payload, t);
    if (!detail) {
        return '';
    }
    return `${detail} ${t('msg.udpPortBlockedCancel')}`;
}

async function fetchOscUdpPortCheck() {
    const res = await fetch(`${API_BASE}/udp-port-check`);
    if (!res.ok) {
        throw new Error(`udp-port-check HTTP ${res.status}`);
    }
    return res.json();
}

// 启动服务
async function startService() {
    const startBtn = document.getElementById('start-btn');
    const t = window.i18n ? window.i18n.t : (key) => key;

    applyAsrBackendLocks();

    startBtn.disabled = true;
    startBtn.textContent = t('btn.starting');
    pendingWarningMessage = null;

    const bypassOscUdp = shouldSkipOscUdpPortCheck();
    if (!bypassOscUdp) {
        try {
            const udpPayload = await fetchOscUdpPortCheck();
            const blockMsg = buildUdpPortBlockUserMessage(udpPayload, t);
            if (blockMsg) {
                showMessage('❌ ' + blockMsg, 'error');
                startBtn.disabled = false;
                startBtn.textContent = t('btn.startService');
                return;
            }
        } catch (e) {
            console.warn('OSC UDP port check failed:', e);
        }
    }

    try {
        const asrBackend = document.getElementById('asr-backend').value;
        const dashscopeKey = document.getElementById('dashscope-api-key').value.trim();
        const doubaoKey = document.getElementById('doubao-api-key').value.trim();
        const sonioxKey = document.getElementById('soniox-api-key').value.trim();
        const deeplKey = document.getElementById('deepl-api-key').value.trim();
        const llmKey = document.getElementById('llm-api-key').value.trim();
        const llmBaseUrl = document.getElementById('llm-base-url').value.trim();
        const llmModel = document.getElementById('llm-model').value.trim();
        const hasLLMKey = envStatus.llm.api_key_set || !!llmKey;

        const enableTranslation = document.getElementById('enable-translation').checked;
        const translationApiSelect = document.getElementById('translation-api-type');
        let translationApiType = translationApiSelect.value;

        const requiresDashscopeKey = currentConfigRequiresDashscopeKey();

        if (asrBackend === 'local') {
            if (!isLocalAsrUiEnabled()) {
                showMessage('❌ ' + t('msg.localAsrDisabledInBuild'), 'error');
                startBtn.disabled = false;
                startBtn.textContent = t('btn.startService');
                return;
            }
            const localResponse = await fetch(`${API_BASE}/local-asr/status`);
            const localPayload = await localResponse.json();
            const localEngine = document.getElementById('local-asr-engine')?.value || 'sensevoice';
            const engineStatus = localPayload?.engines?.[localEngine];
            if (!engineStatus || !engineStatus.ready) {
                if (!engineStatus) {
                    showMessage('❌ ' + t('msg.localAsrNotReady'), 'error');
                } else if (engineStatus.model_cached && Array.isArray(engineStatus.runtime_issues) && engineStatus.runtime_issues.length) {
                    showMessage(
                        '❌ ' + t('msg.localAsrNeedPythonDeps', { deps: engineStatus.runtime_issues.join(', ') }),
                        'error',
                    );
                } else if (engineStatus.model_cached === false) {
                    showMessage('❌ ' + t('msg.localAsrNotReady'), 'error');
                } else {
                    showMessage('❌ ' + t('msg.localAsrNeedSilero'), 'error');
                }
                ensureCollapsibleExpanded('local-asr-settings');
                startBtn.disabled = false;
                startBtn.textContent = t('btn.startService');
                return;
            }
        }

        if (requiresDashscopeKey && !dashscopeKey) {
            showMessage('❌ ' + t('msg.dashscopeRequired'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');

            // 展开API Keys配置区域，提示用户输入
            const apiKeysSection = document.getElementById('api-keys');
            if (apiKeysSection.classList.contains('collapsed')) {
                toggleCollapsible('api-keys');
            }

            // 高亮并震动 API Key 输入框
            highlightInput('dashscope-api-key');

            return;
        }

        if (asrBackend === 'doubao_file' && !doubaoKey) {
            showMessage('❌ ' + t('msg.doubaoKeyRequired'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');

            const apiKeysSection = document.getElementById('api-keys');
            if (apiKeysSection.classList.contains('collapsed')) {
                toggleCollapsible('api-keys');
            }
            highlightInput('doubao-api-key');

            return;
        }

        if (asrBackend === 'soniox' && !sonioxKey) {
            showMessage('❌ ' + t('msg.sonioxKeyRequired'), 'error');
            startBtn.disabled = false;
            startBtn.textContent = t('btn.startService');
            highlightInput('soniox-api-key');

            return;
        }

        if (requiresDashscopeKey) {
            const checkDashscopeResponse = await fetch(`${API_BASE}/check-api-key`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ api_key: dashscopeKey }),
            });

            const checkDashscopeResult = await checkDashscopeResponse.json();

            if (!checkDashscopeResult.valid) {
                const localizedMsg = localizeBackendMessage(checkDashscopeResult.message_id, checkDashscopeResult.message);
                showMessage('❌ ' + t('msg.dashscopeValidationFailed') + localizedMsg, 'error');
                startBtn.disabled = false;
                startBtn.textContent = t('btn.startService');
                highlightInput('dashscope-api-key');
                return;
            }
        }

        if (enableTranslation) {
            if (translationApiType === 'openrouter' || translationApiType === 'openrouter_streaming_deepl_hybrid') {
                const missingFields = [];
                const missingLabels = [];

                if (!llmBaseUrl) {
                    missingFields.push('llm-base-url');
                    missingLabels.push(t('label.llmBaseUrl'));
                }
                if (!llmModel) {
                    missingFields.push('llm-model');
                    missingLabels.push(t('label.llmModel'));
                }
                if (!hasLLMKey) {
                    missingFields.push('llm-api-key');
                    missingLabels.push(t('label.llmKey'));
                }

                if (missingFields.length > 0) {
                    showMessage('❌ ' + t('msg.llmFieldRequired', { field: missingLabels[0] }), 'error');
                    startBtn.disabled = false;
                    startBtn.textContent = t('btn.startService');
                    expandLLMSettingsPanel(missingFields);
                    return;
                }
            }

            const requiresDeeplKey = translationApiType === 'deepl' && !deeplKey;
            const requiresHybridLLMKey = translationApiType === 'openrouter_streaming_deepl_hybrid' && !hasLLMKey;
            const requiresHybridDeeplKey = translationApiType === 'openrouter_streaming_deepl_hybrid' && !deeplKey;

            if (requiresDeeplKey || requiresHybridLLMKey || requiresHybridDeeplKey) {
                translationApiType = 'google_dictionary';
                translationApiSelect.value = translationApiType;
                translationApiSelect.dispatchEvent(new Event('change'));
                saveConfigToLocalStorage();
                pendingWarningMessage = t('msg.autoSwitchToGoogle');
                showMessage('⚠️ ' + pendingWarningMessage, 'warning');
            }
        }

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

        const apiKeys = {
            dashscope: dashscopeKey,
            deepl: deeplKey,
            llm: llmKey,
            doubao: doubaoKey,
            soniox: sonioxKey,
        };

        const response = await fetch(`${API_BASE}/service/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                api_keys: apiKeys,
                bypass_osc_udp_port_check:
                    document.getElementById('bypass-osc-udp-port-check')?.checked === true,
            }),
        });

        const result = await response.json();

        if (result.success) {
            const acceleratorWarning = localizeBackendMessage(
                result.accelerator_warning_message_id,
                result.accelerator_warning_message,
            );
            const hasPendingWarning = !!pendingWarningMessage;
            const startMessageLines = [];
            if (hasPendingWarning) {
                startMessageLines.push('⚠️ ' + pendingWarningMessage);
            }
            startMessageLines.push('✅ ' + t('msg.serviceStarting'));
            if (acceleratorWarning) {
                startMessageLines.push('⚠️ ' + acceleratorWarning);
            }
            showMessage(startMessageLines.join('\n'), hasPendingWarning ? 'warning' : 'success');
            pendingWarningMessage = null;
            await updateStatus();
            setTimeout(updateStatus, 500);
        } else {
            if (Array.isArray(result.udp_port_conflicts) && result.udp_port_conflicts.length) {
                const blockMsg = buildUdpPortBlockUserMessage(result, t);
                showMessage('❌ ' + (blockMsg || localizeBackendMessage(result.message_id, result.message)), 'error');
            } else {
                const localizedMsg = localizeBackendMessage(result.message_id, result.message);
                showMessage('❌ ' + t('msg.serviceStartFailed') + localizedMsg, 'error');
            }
            startBtn.disabled = false;
        }
    } catch (error) {
        console.error('启动服务失败:', error);
        showMessage('❌ ' + t('msg.startServiceFailed'), 'error');
        startBtn.disabled = false;
    } finally {
        pendingWarningMessage = null;
    }
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
            const stopMessageKey = getServiceLifecycle(result) === 'stopped'
                ? 'msg.serviceStopSuccess'
                : 'msg.serviceStopping';
            showMessage(t(stopMessageKey), 'success');
            await updateStatus();
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
    }
}

// 重启服务
async function restartService() {
    try {
        const tr = window.i18n ? window.i18n.t : (key) => key;
        const bypassOscRestart = shouldSkipOscUdpPortCheck();
        if (!bypassOscRestart) {
            try {
                const udpPayload = await fetchOscUdpPortCheck();
                const blockMsg = buildUdpPortBlockUserMessage(udpPayload, tr);
                if (blockMsg) {
                    showMessage('❌ ' + blockMsg, 'error');
                    return;
                }
            } catch (e) {
                console.warn('OSC UDP port check failed:', e);
            }
        }

        const response = await fetch(`${API_BASE}/service/restart`, {
            method: 'POST',
        });

        const result = await response.json();

        if (result.success) {
            console.log('服务已重启');
            showMessage('✅ ' + tr('msg.serviceRestarted'), 'success');
            setTimeout(updateStatus, 500);
        } else if (Array.isArray(result.udp_port_conflicts) && result.udp_port_conflicts.length) {
            const blockMsg = buildUdpPortBlockUserMessage(result, tr);
            showMessage('❌ ' + (blockMsg || localizeBackendMessage(result.message_id, result.message)), 'error');
        } else {
            const localizedMsg = localizeBackendMessage(result.message_id, result.message);
            showMessage('❌ ' + localizedMsg, 'error');
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
        persistCurrentLLMTemplateState();
        const llmTemplateStorageSnapshot = snapshotLLMTemplateStorage();

        // 使用前端默认配置
        loadDefaultConfig();
        resetPanelFloatingModeSetting();
        resetQuickLanguageSettings();

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();

        // 先保存到本地浏览器
        saveConfigToLocalStorage();
        restoreLLMTemplateStorage(llmTemplateStorageSnapshot);

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

// 显示消息（不自动消失，仅被后续 showMessage 覆盖）
function showMessage(text, type) {
    const messageEl = document.getElementById('message');
    if (messageEl._replaceFlashTimer != null) {
        clearTimeout(messageEl._replaceFlashTimer);
        messageEl._replaceFlashTimer = null;
    }
    messageEl.textContent = text;
    messageEl.className = 'message ' + type;
    messageEl.classList.remove('message-replace-flash');
    /* 强制重排以便连续相同文案也能重播动画 */
    void messageEl.offsetWidth;
    messageEl.classList.add('message-replace-flash');
    messageEl._replaceFlashTimer = setTimeout(() => {
        messageEl.classList.remove('message-replace-flash');
        messageEl._replaceFlashTimer = null;
    }, 520);
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

function updateCollapsibleIcon(icon, collapsed) {
    if (!icon) return;
    icon.classList.toggle('collapsed', collapsed);
    icon.textContent = collapsed ? '▶' : '▼';
}

function syncCollapsibleContainerState(content) {
    if (!content) return;

    const container = content.parentElement;
    if (!container) return;

    container.classList.add('collapsible-container');

    container.classList.toggle(
        'collapsible-container-collapsed',
        content.classList.contains('collapsed'),
    );
}

function initializeCollapsibleStates() {
    document.querySelectorAll('.collapsible-content').forEach((content) => {
        bindCollapsibleTransitionCleanup(content);
        syncCollapsibleContainerState(content);

        if (!content.id) return;
        const icon = document.getElementById(`${content.id}-icon`);
        updateCollapsibleIcon(icon, content.classList.contains('collapsed'));
    });
}

function bindCollapsibleTransitionCleanup(content) {
    if (!content || content.dataset.collapseTransitionBound === 'true') {
        return;
    }

    content.addEventListener('transitionend', (event) => {
        if (event.propertyName !== 'max-height') {
            return;
        }

        if (content.classList.contains('collapsed')) {
            content.style.maxHeight = '0px';
            content.style.overflow = 'hidden';
            return;
        }

        content.style.maxHeight = 'none';
        content.style.overflow = 'visible';
    });

    content.dataset.collapseTransitionBound = 'true';
}

// 折叠/展开面板
function toggleCollapsible(id) {
    const content = document.getElementById(id);
    const icon = document.getElementById(id + '-icon');
    if (!content) return;

    bindCollapsibleTransitionCleanup(content);

    if (content.classList.contains('collapsed')) {
        content.style.overflow = 'hidden';
        content.classList.remove('collapsed');
        updateCollapsibleIcon(icon, false);
        syncCollapsibleContainerState(content);

        content.style.maxHeight = '0px';
        void content.offsetHeight;

        content.style.maxHeight = `${content.scrollHeight}px`;
        return;
    }

    content.style.overflow = 'hidden';
    content.style.maxHeight = `${content.scrollHeight}px`;
    void content.offsetHeight;

    content.classList.add('collapsed');
    updateCollapsibleIcon(icon, true);
    syncCollapsibleContainerState(content);
    content.style.maxHeight = '0px';
}

// 打开状态小面板
async function openMiniPanel() {
    const t = window.i18n ? window.i18n.t : (key) => key;

    try {
        // 先将当前配置同步到后端
        await saveConfig(true);

        // 收集 API Keys 一并发送给后端，让后端写入环境变量
        const apiKeys = {
            dashscope: document.getElementById('dashscope-api-key').value.trim(),
            deepl: document.getElementById('deepl-api-key').value.trim(),
            llm: document.getElementById('llm-api-key').value.trim(),
            doubao: document.getElementById('doubao-api-key').value.trim(),
            soniox: document.getElementById('soniox-api-key').value.trim(),
        };
        const floatingMode = document.getElementById('panel-floating-mode')?.checked ?? false;
        const quickLanguageSettings = getQuickLanguageSettingsForPanel();

        const response = await fetch('/api/open-panel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_keys: apiKeys,
                floating_mode: floatingMode,
                quick_language_settings: quickLanguageSettings,
            }),
        });
        const result = await response.json();

        if (result.success) {
            showMessage(t('msg.panelOpened') || '小面板已打开', 'success');
        } else {
            showMessage((t('msg.panelFailed') || '无法打开小面板: ') + result.error, 'error');
        }
    } catch (error) {
        showMessage('请求失败: ' + error, 'error');
    }
}

async function updateIpcStatus() {
    try {
        const response = await fetch(`${API_BASE}/ipc_status`);
        const status = await response.json();
        
        const textSpan = document.getElementById('ipc-status-text');
        const dot = document.getElementById('ipc-status-dot');
        const container = document.getElementById('ipc-status-indicator');
        const smartSection = document.getElementById('smart-target-language-section');
        
        if (!textSpan || !dot || !container) return;
        
        const t = window.i18n ? window.i18n.t : (key) => key;
        if (status.connected) {
            container.style.display = 'flex';
            if (status.mode === 'delegate') {
                textSpan.textContent = t('status.ipcConnectedDelegate');
            } else {
                textSpan.textContent = t('status.ipcConnected');
            }
            dot.style.backgroundColor = '#4caf50';
            if (smartSection) {
                smartSection.style.display = 'block';
                
                fetch(`${API_BASE}/smart-target-status`)
                    .then(r => r.json())
                    .then(status => {
                        const recentEl = document.getElementById('smart-target-recent-langs');
                        if (!recentEl) return;
                        if (status.recent_languages && status.recent_languages.length > 0) {
                            recentEl.innerHTML = status.recent_languages.map(lang => 
                                `<span style="background: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;">${lang}</span>`
                            ).join('');
                        } else {
                            const hintText = (typeof t === 'function') ? (t('hint.noRecentLanguages') || '暂无数据') : '暂无数据';
                            recentEl.innerHTML = `<span style="color: #888; font-size: 12px;">${hintText}</span>`;
                        }
                    })
                    .catch(() => {});
            }
        } else {
            container.style.display = 'none';
            if (smartSection) {
                smartSection.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('更新 IPC 状态失败:', error);
    }
}

function updateSmartTargetVisibility() {
    const primaryEnabled = document.getElementById('smart-target-primary-enabled')?.checked ?? true;
    const secondaryEnabled = document.getElementById('smart-target-secondary-enabled')?.checked ?? false;
    const details = document.getElementById('smart-target-settings');
    const advanced = document.getElementById('smart-target-advanced-settings');
    const badge = document.getElementById('smart-target-activated-badge');

    if (details) {
        details.style.display = '';
    }
    if (badge) {
        badge.hidden = !(primaryEnabled || secondaryEnabled);
    }
    if (details && !details.classList.contains('collapsed') && details.style.maxHeight !== 'none') {
        details.style.maxHeight = `${details.scrollHeight}px`;
    }
}

document.addEventListener('change', (e) => {
    if (e.target.id === 'smart-target-primary-enabled' || e.target.id === 'smart-target-secondary-enabled') {
        updateSmartTargetVisibility();
    }
});
