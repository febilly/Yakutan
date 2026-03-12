// API基础URL
const API_BASE = '/api';

// 自动保存定时器
let autoSaveTimer = null;

// 配置键名
const CONFIG_STORAGE_KEY = 'vrchat_translator_config';
const PANEL_FLOATING_MODE_STORAGE_KEY = 'panel_floating_mode';
const LLM_TEMPLATE_KEY_STORAGE_PREFIX = 'llm_template_key_';
const LLM_TEMPLATE_MODEL_STORAGE_PREFIX = 'llm_template_model_';

// 待显示的警告消息（用于自动切换翻译API）
let pendingWarningMessage = null;

// 环境变量状态（由后端提供）
let envStatus = {
    llm: {
        api_key_set: false,
    },
};

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

const LLM_TEMPLATE_CONFIGS = {
    'dashscope-qwen35': {
        baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        model: 'qwen3.5-plus',
        extraBody: '{"enable_thinking": false}',
        parallelFastestDefault: false,
        providerLabelKey: 'btn.llmTemplateDashscopeQwen',
        copyDashscopeKey: true,
    },
    openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        parallelFastestDefault: false,
        providerLabelKey: 'btn.llmTemplateOpenRouter',
    },
    longcat: {
        baseUrl: 'https://api.longcat.chat/openai/v1',
        model: 'LongCat-Flash-Lite',
        extraBody: '',
        parallelFastestDefault: true,
        providerLabelKey: 'btn.llmTemplateLongCat',
    },
    mercury2: {
        baseUrl: 'https://api.inceptionlabs.ai/v1',
        model: 'mercury-2',
        extraBody: '',
        parallelFastestDefault: false,
        providerLabelKey: 'btn.llmTemplateMercury2',
    },
};

let activeLLMTemplate = null;

function detectActiveLLMTemplate() {
    const baseUrl = (document.getElementById('llm-base-url')?.value || '').trim();

    for (const [templateName, templateConfig] of Object.entries(LLM_TEMPLATE_CONFIGS)) {
        if (baseUrl === templateConfig.baseUrl) {
            return templateName;
        }
    }

    return null;
}

function getLLMTemplateKeyStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_KEY_STORAGE_PREFIX}${templateName}` : null;
}

function getLLMTemplateModelStorageKey(templateName) {
    return templateName ? `${LLM_TEMPLATE_MODEL_STORAGE_PREFIX}${templateName}` : null;
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

function persistCurrentLLMTemplateKey() {
    const templateName = detectActiveLLMTemplate();
    const keyInput = document.getElementById('llm-api-key');
    if (!templateName || !keyInput) return;

    setStoredLLMTemplateKey(templateName, keyInput.value);
}

function persistCurrentLLMTemplateModel() {
    const templateName = detectActiveLLMTemplate();
    const modelInput = document.getElementById('llm-model');
    if (!templateName || !modelInput) return;

    if (templateName === 'openrouter') {
        setStoredLLMTemplateModel(templateName, modelInput.value);
    }
}

function snapshotLLMTemplateStorage() {
    const snapshot = {};

    Object.keys(LLM_TEMPLATE_CONFIGS).forEach((templateName) => {
        snapshot[getLLMTemplateKeyStorageKey(templateName)] = getStoredLLMTemplateKey(templateName);

        const modelStorageKey = getLLMTemplateModelStorageKey(templateName);
        snapshot[modelStorageKey] = getStoredLLMTemplateModel(templateName);
    });

    return snapshot;
}

function restoreLLMTemplateStorage(snapshot) {
    if (!snapshot) return;

    Object.entries(snapshot).forEach(([storageKey, value]) => {
        if (value) {
            localStorage.setItem(storageKey, value);
        } else {
            localStorage.removeItem(storageKey);
        }
    });
}

function resolveLLMTemplateKeySource(templateName) {
    const t = window.i18n ? window.i18n.t : (key) => key;
    const useInternationalEndpoint = document.getElementById('use-international-endpoint')?.checked ?? false;
    const templateConfig = LLM_TEMPLATE_CONFIGS[templateName];
    if (!templateConfig) return null;

    let url = '';
    if (templateName === 'dashscope-qwen35') {
        url = useInternationalEndpoint
            ? 'https://modelstudio.console.aliyun.com/ap-southeast-1?tab=doc#/api-key'
            : 'https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key';
    } else if (templateName === 'openrouter') {
        url = 'https://openrouter.ai/settings/keys';
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
        activeLLMTemplate = null;
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
    updateLLMTemplateKeySourceHint(detectActiveLLMTemplate());
}

function shouldShowLLMSettings(apiType) {
    return apiType === 'openrouter' || apiType === 'openrouter_streaming_deepl_hybrid';
}

function updateLLMSettingsVisibility(apiType = null, expandPanel = false) {
    const actualApiType = apiType || (document.getElementById('translation-api-type')
        ? document.getElementById('translation-api-type').value
        : 'qwen_mt');
    const wrapper = document.getElementById('llm-settings-wrapper');
    if (!wrapper) return;

    const shouldShow = shouldShowLLMSettings(actualApiType);
    wrapper.style.display = shouldShow ? 'block' : 'none';

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

function applyLLMTemplate(templateName) {
    const previousTemplateName = detectActiveLLMTemplate();
    const baseUrlInput = document.getElementById('llm-base-url');
    const modelInput = document.getElementById('llm-model');
    const keyInput = document.getElementById('llm-api-key');
    const extraBodyInput = document.getElementById('openai-compat-extra-body-json');
    const parallelFastestToggle = document.getElementById('enable-llm-parallel-fastest');
    const dashscopeKeyInput = document.getElementById('dashscope-api-key');
    const t = window.i18n ? window.i18n.t : (key) => key;
    const templateConfig = LLM_TEMPLATE_CONFIGS[templateName];

    if (!baseUrlInput || !modelInput || !keyInput || !extraBodyInput || !parallelFastestToggle || !templateConfig) {
        return;
    }

    if (previousTemplateName) {
        persistCurrentLLMTemplateKey();
        persistCurrentLLMTemplateModel();
    }

    baseUrlInput.value = templateConfig.baseUrl;
    if (Object.prototype.hasOwnProperty.call(templateConfig, 'model')) {
        modelInput.value = templateConfig.model;
    } else if (templateName === 'openrouter') {
        modelInput.value = getStoredLLMTemplateModel(templateName);
    }
    if (Object.prototype.hasOwnProperty.call(templateConfig, 'extraBody')) {
        extraBodyInput.value = templateConfig.extraBody;
    }
    parallelFastestToggle.checked = !!templateConfig.parallelFastestDefault;

    const storedKey = getStoredLLMTemplateKey(templateName);
    if (storedKey) {
        keyInput.value = storedKey;
        persistSecretInputValue('llm-api-key');
        envStatus.llm.api_key_set = true;
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
            renderLanguageComboMenu(combo);
            closeLanguageMenus();
            onSettingChange();
            input.focus();
        });

        menu.appendChild(noneButton);
    }

    LANGUAGE_OPTIONS.forEach((option) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'language-combo-option';
        if (option.code.toLowerCase() === currentValue) {
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
            renderLanguageComboMenu(combo);
            closeLanguageMenus();
            if (input.id === 'target-language') {
                updateFuriganaVisibility();
            }
            if (input.id.startsWith('quick-lang-')) {
                onQuickLangChange();
            } else {
                onSettingChange();
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

        combo.dataset.initialized = 'true';
    });

    renderLanguageComboMenus();
}

const QUICK_LANG_STORAGE_KEY = 'panel_quick_languages';
const QUICK_LANG_DEFAULTS = ['en', 'zh-CN', 'ja', 'ko'];
const QUICK_LANG_BAR_ENABLED_KEY = 'panel_quick_lang_bar_enabled';

function loadQuickLanguageSettings() {
    try {
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

    const toggle = document.getElementById('enable-quick-lang-bar');
    if (toggle) {
        const stored = localStorage.getItem(QUICK_LANG_BAR_ENABLED_KEY);
        toggle.checked = stored === null ? true : stored === 'true';
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
    localStorage.setItem(QUICK_LANG_STORAGE_KEY, JSON.stringify(QUICK_LANG_DEFAULTS));

    const toggle = document.getElementById('enable-quick-lang-bar');
    if (toggle) {
        toggle.checked = true;
    }
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
document.addEventListener('DOMContentLoaded', function () {
    // 先初始化 i18n 系统
    if (window.i18n) {
        window.i18n.initI18n();
    }

    setupLanguageComboboxes();

    document.addEventListener('click', function (event) {
        if (!event.target.closest('.language-combo')) {
            closeLanguageMenus();
        }
    });

    const targetLangInput = document.getElementById('target-language');
    if (targetLangInput) {
        targetLangInput.addEventListener('input', updateFuriganaVisibility);
    }

    loadConfigFromLocalStorage();
    loadPanelFloatingModeSetting();
    loadQuickLanguageSettings();
    loadAPIKeys();
    applyAsrBackendLocks();
    loadEnvStatus();
    refreshMicDevices(true);
    setupMicDeviceAutoRefresh();
    updateStatus();
    // 每2秒更新一次状态
    setInterval(updateStatus, 2000);

    // 显示配置保存提示
    showConfigStorageInfo();
});

document.addEventListener('i18n:languageChanged', function () {
    const useInternational = document.getElementById('use-international-endpoint')?.checked ?? false;
    updateAsrOptionsForInternational(useInternational);
    renderLanguageComboMenus();
    refreshMicDevices(true);
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
        defaultOpt.textContent = t('option.systemDefault');
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

    const activeTemplate = detectActiveLLMTemplate();
    const storedTemplateKey = getStoredLLMTemplateKey(activeTemplate);
    if (storedTemplateKey) {
        document.getElementById('llm-api-key').value = storedTemplateKey;
    } else if (llmKey) {
        document.getElementById('llm-api-key').value = llmKey;
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
            apiKeysIcon.classList.add('collapsed');
            apiKeysIcon.textContent = '▶';
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
            if (event.target.id === 'llm-model') {
                persistCurrentLLMTemplateModel();
            }
            syncLLMTemplateKeySourceHintFromInputs();
            onSettingChange(event.target);
        });
    });
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
    const isStreamingTranslation = enableTranslationToggle.checked
        && translationApiSelect.value === 'openrouter'
        && streamingModeToggle.checked;

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
                if (apiType === 'openrouter_streaming' || apiType === 'openrouter_streaming_deepl_hybrid') {
                    document.getElementById('translation-api-type').value = 'openrouter';
                    document.getElementById('openrouter-streaming-mode').checked = apiType === 'openrouter_streaming';
                } else {
                    document.getElementById('translation-api-type').value = apiType;
                    document.getElementById('openrouter-streaming-mode').checked = false;
                }
                document.getElementById('llm-base-url').value = config.translation.llm_base_url || '';
                document.getElementById('llm-model').value = config.translation.llm_model || '';
                document.getElementById('openai-compat-extra-body-json').value = config.translation.openai_compat_extra_body_json || '';
                document.getElementById('enable-llm-parallel-fastest').checked = config.translation.enable_llm_parallel_fastest
                    ?? (detectActiveLLMTemplate() === 'longcat');
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

            if (config.panel) {
                document.getElementById('panel-width').value = config.panel.width || 600;
                getNormalizedPanelWidth();
            }

            console.log('✓ 已从浏览器加载配置');
        } else {
            // 如果没有保存的配置，使用前端默认值
            loadDefaultConfig();
            loadConfigFromServer();
        }

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();
        updateFuriganaVisibility();
        updateLLMSettingsVisibility();
        updateSensitiveWordsHint();
        applyAsrBackendLocks();
        syncLLMTemplateKeySourceHintFromInputs();

    } catch (error) {
        console.error('加载本地配置失败:', error);
        // 出错时使用前端默认值
        loadDefaultConfig();
        toggleTranslationOptions();
        updateFuriganaVisibility();
        updateLLMSettingsVisibility();
        updateSensitiveWordsHint();
        applyAsrBackendLocks();
        syncLLMTemplateKeySourceHintFromInputs();
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
    document.getElementById('source-language').value = 'auto';
    document.getElementById('show-partial-results').checked = false;
    document.getElementById('enable-furigana').checked = false;
    document.getElementById('enable-pinyin').checked = false;
    document.getElementById('enable-reverse-translation').checked = true;
    document.getElementById('openrouter-streaming-mode').checked = false;
    document.getElementById('enable-llm-parallel-fastest').checked = false;
    document.getElementById('llm-base-url').value = '';
    document.getElementById('llm-model').value = '';
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
    document.getElementById('asr-backend').value = 'qwen';  // 可选: 'qwen', 'dashscope'
    document.getElementById('enable-hot-words').checked = true;
    document.getElementById('enable-vad').checked = true;
    document.getElementById('vad-threshold').value = 0.2;
    document.getElementById('vad-silence-duration').value = 800;
    document.getElementById('keepalive-interval').value = 30;
    document.getElementById('use-international-endpoint').checked = false;

    // 语言检测器
    document.getElementById('language-detector').value = 'cjke';

    // 小面板
    document.getElementById('panel-width').value = 600;

    console.log('✓ 已加载前端默认配置');
    updateFuriganaVisibility();
    updateLLMSettingsVisibility();
    updateSensitiveWordsHint();
    applyAsrBackendLocks();
    syncLLMTemplateKeySourceHintFromInputs();
}

// 从服务器加载配置（仅在本地无配置时使用）
async function loadConfigFromServer() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();

        // 填充表单
        document.getElementById('enable-translation').checked = config.translation.enable_translation;
        document.getElementById('target-language').value = config.translation.target_language;
        document.getElementById('secondary-target-language').value = config.translation.secondary_target_language || '';
        document.getElementById('fallback-language').value = config.translation.fallback_language || '';
        // 处理 LLM 流式模式的特殊情况
        const serverApiType = config.translation.api_type;
        if (serverApiType === 'openrouter_streaming' || serverApiType === 'openrouter_streaming_deepl_hybrid') {
            document.getElementById('translation-api-type').value = 'openrouter';
            document.getElementById('openrouter-streaming-mode').checked = serverApiType === 'openrouter_streaming';
        } else {
            document.getElementById('translation-api-type').value = serverApiType;
            document.getElementById('openrouter-streaming-mode').checked = false;
        }
        document.getElementById('llm-base-url').value = config.translation.llm_base_url || '';
        document.getElementById('llm-model').value = config.translation.llm_model || '';
        document.getElementById('openai-compat-extra-body-json').value = config.translation.openai_compat_extra_body_json || '';
        document.getElementById('enable-llm-parallel-fastest').checked = config.translation.enable_llm_parallel_fastest
            ?? (detectActiveLLMTemplate() === 'longcat');
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
        document.getElementById('panel-width').value = (config.panel && config.panel.width) || 600;
        getNormalizedPanelWidth();

        // 保存到本地
        saveConfigToLocalStorage();

        // 根据翻译开关显示/隐藏翻译选项
        toggleTranslationOptions();
        updateFuriganaVisibility();
        updateLLMSettingsVisibility();
        updateSensitiveWordsHint();
        applyAsrBackendLocks();
        syncLLMTemplateKeySourceHintFromInputs();

        console.log('已从服务器加载配置');

    } catch (error) {
        console.error('加载服务器配置失败:', error);
    }
}

// 保存配置到 localStorage
function saveConfigToLocalStorage() {
    try {
        applyAsrBackendLocks();

        // 确定实际的 API 类型（如果是 LLM 且启用了流式模式，使用 openrouter_streaming）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                secondary_target_language: document.getElementById('secondary-target-language').value || null,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                llm_base_url: document.getElementById('llm-base-url').value.trim(),
                llm_model: document.getElementById('llm-model').value.trim(),
                openai_compat_extra_body_json: document.getElementById('openai-compat-extra-body-json').value.trim(),
                enable_llm_parallel_fastest: document.getElementById('enable-llm-parallel-fastest').checked,
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
            },
            panel: {
                width: getNormalizedPanelWidth(),
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

    // 显示/隐藏 LLM 流式翻译模式选项
    if (newApi === 'openrouter') {
        streamingModeGroup.style.display = 'block';
    } else {
        streamingModeGroup.style.display = 'none';
        // 如果切换到不支持流式的API，自动关闭流式翻译开关
        document.getElementById('openrouter-streaming-mode').checked = false;
    }
    updateLLMSettingsVisibility(newApi, newApi === 'openrouter');
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

        // 初始化 LLM 流式模式选项的显示状态
        const streamingModeGroup = document.getElementById('openrouter-streaming-mode-group');
        if (apiSelect.value === 'openrouter') {
            streamingModeGroup.style.display = 'block';
        } else {
            streamingModeGroup.style.display = 'none';
        }
        updateLLMSettingsVisibility(apiSelect.value);
        updateSensitiveWordsHint(apiSelect.value);
        syncLLMTemplateKeySourceHintFromInputs();
    }, 100);
});

// 当设置改变时自动保存（延迟保存，避免频繁请求）
function onSettingChange(changedElement = null) {
    applyAsrBackendLocks();

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

        // 确定实际的 API 类型（如果是 LLM 且启用了流式模式，使用 openrouter_streaming）
        let actualApiType = document.getElementById('translation-api-type').value;
        if (actualApiType === 'openrouter' && document.getElementById('openrouter-streaming-mode').checked) {
            actualApiType = 'openrouter_streaming';
        }

        const config = {
            translation: {
                enable_translation: document.getElementById('enable-translation').checked,
                target_language: document.getElementById('target-language').value,
                secondary_target_language: document.getElementById('secondary-target-language').value || null,
                fallback_language: document.getElementById('fallback-language').value || null,
                api_type: actualApiType,
                llm_base_url: document.getElementById('llm-base-url').value.trim(),
                llm_model: document.getElementById('llm-model').value.trim(),
                openai_compat_extra_body_json: document.getElementById('openai-compat-extra-body-json').value.trim(),
                enable_llm_parallel_fastest: document.getElementById('enable-llm-parallel-fastest').checked,
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
            },
            panel: {
                width: getNormalizedPanelWidth(),
            },
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

        // 同步小面板对目标语言的更改到大菜单
        if (status.target_language) {
            const targetLangInput = document.getElementById('target-language');
            if (targetLangInput && targetLangInput.value !== status.target_language) {
                targetLangInput.value = status.target_language;
                renderLanguageComboMenus();
                saveConfigToLocalStorage();
            }
        }
    } catch (error) {
        console.error('更新状态失败:', error);
    }
}

// 启动服务
async function startService() {
    const startBtn = document.getElementById('start-btn');
    const t = window.i18n ? window.i18n.t : (key) => key;

    applyAsrBackendLocks();

    startBtn.disabled = true;
    startBtn.textContent = t('btn.starting');
    pendingWarningMessage = null;

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

        // 当前配置是否需要 DashScope Key
        const dashscopeRequiredByAsr = asrBackend === 'qwen' || asrBackend === 'dashscope';
        const dashscopeRequiredByTranslation = enableTranslation && translationApiType === 'qwen_mt';
        const requiresDashscopeKey = dashscopeRequiredByAsr || dashscopeRequiredByTranslation;

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
            if (translationApiType === 'openrouter') {
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
        persistCurrentLLMTemplateKey();
        persistCurrentLLMTemplateModel();
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
