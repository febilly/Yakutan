/**
 * 界面模式（简易/高级） + 与 main.js 的兼容层
 *
 * 简易模式：
 *   - 只显示 #simple-view（两张卡片），隐藏 #advanced-view
 *   - 把高级视图中的少量核心控件（原始 DOM 元素）移动到简易页插槽中，
 *     这样 main.js 通过 id 引用的逻辑完全不受影响
 *   - 进入简易模式时应用推荐默认值（Qwen 识别 + LLM 流式翻译 DeepSeek 等）
 *   - 仅语音识别（关闭翻译）时自动开启中间结果输出
 *   - API Key 的说明文字换成针对简易模式的简化说法
 * 高级模式：左右两栏卡片，所有功能可见。
 */
(function () {
    'use strict';

    const MODE_KEY = 'yakutan-ui-mode';

    const byId = (id) => document.getElementById(id);

    /* ---------------- 简易模式控件搬运 ---------------- */

    // [元素id, 目标插槽id]
    const SIMPLE_MOVES = [
        ['fg-enable-translation', 'simple-slot-main'],
        ['fg-mic-device', 'simple-slot-mic'],
        ['fg-target-language', 'simple-slot-langs'],
        ['fg-secondary-language', 'simple-slot-langs'],
        ['fg-show-original', 'simple-slot-langs'],
        ['fg-mic-control', 'simple-slot-switches'],
        ['fg-remove-period', 'simple-slot-switches'],
        ['dashscope-key-group', 'simple-slot-keys'],
        ['fg-llm-api-key', 'simple-slot-keys'],
    ];

    // 简易模式下，这些折叠区域对应的关键控件已被移动到简易页，
    // main.js 想展开它们时不需要切回高级模式
    const SIMPLE_SAFE_COLLAPSIBLES = new Set(['api-keys', 'llm-settings', 'translation-api']);

    const placeholders = new Map();
    let mode = 'simple';

    function moveToSimple() {
        SIMPLE_MOVES.forEach(([id, slotId]) => {
            const el = byId(id);
            const slot = byId(slotId);
            if (!el || !slot || placeholders.has(id)) return;
            const ph = document.createElement('span');
            ph.hidden = true;
            ph.dataset.simplePlaceholder = id;
            el.before(ph);
            placeholders.set(id, ph);
            slot.appendChild(el);
        });
    }

    function restoreFromSimple() {
        SIMPLE_MOVES.forEach(([id]) => {
            const el = byId(id);
            const ph = placeholders.get(id);
            if (!el || !ph || !ph.parentNode) return;
            ph.replaceWith(el);
            placeholders.delete(id);
        });
    }

    function syncSimpleTranslationModeControl() {
        const enableEl = byId('enable-translation');
        if (!enableEl) return;
        document.querySelectorAll('[data-simple-translation-mode]').forEach((button) => {
            const isTranslation = button.dataset.simpleTranslationMode === 'translation';
            const active = enableEl.checked === isTranslation;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    function setTranslationEnabled(enabled) {
        const enableEl = byId('enable-translation');
        if (!enableEl || enableEl.checked === enabled) {
            syncSimpleTranslationModeControl();
            return;
        }
        enableEl.checked = enabled;
        enableEl.dispatchEvent(new Event('change', { bubbles: true }));
        syncSimpleTranslationModeControl();
    }

    /** 仅语音识别（翻译关闭）时：隐藏语言选项 + 自动开启中间结果输出 */
    function syncSimpleTranslationUi() {
        if (mode !== 'simple') return;
        const enableEl = byId('enable-translation');
        if (!enableEl) return;
        const on = enableEl.checked;
        const block = byId('simple-translation-block');
        if (block) {
            block.style.display = on ? '' : 'none';
        }
        const partial = byId('show-partial-results');
        if (partial && partial.checked !== !on) {
            partial.checked = !on;
            partial.dispatchEvent(new Event('change', { bubbles: true }));
        }
        syncSimpleTranslationModeControl();
    }

    function setSelectValue(id, value) {
        const el = byId(id);
        if (el && el.value !== value) {
            el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    function setChecked(id, checked) {
        const el = byId(id);
        if (el && el.checked !== checked) {
            el.checked = checked;
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    /** 简易模式推荐默认值（幂等：只改动与目标不同的项） */
    function applySimpleDefaults() {
        // 语音识别：Qwen
        setSelectValue('asr-backend', 'qwen');
        // 翻译：LLM 流式 + DeepSeek 模板
        setSelectValue('translation-api-type', 'openrouter');
        setChecked('openrouter-streaming-mode', true);
        const tplBtn = document.querySelector('[data-llm-template="deepseek-v4-flash"]');
        if (tplBtn && tplBtn.getAttribute('aria-pressed') !== 'true'
            && typeof window.applyLLMTemplate === 'function') {
            window.applyLLMTemplate('deepseek-v4-flash');
        }
        // 正式程度中 + 标准语气
        setSelectValue('llm-translation-formality', 'medium');
        setSelectValue('llm-translation-style', 'standard');
        // 不使用自动目标语言推断
        setChecked('smart-target-primary-enabled', false);
        setChecked('smart-target-secondary-enabled', false);
        syncSimpleTranslationUi();
    }

    /* ---------------- 简易模式专属文案 ---------------- */

    // [元素id, 简易模式 i18n key, 高级模式 i18n key]
    const SIMPLE_COPY_SWAPS = [
        ['llm-api-key-label', 'label.deepseekKey', 'label.llmKey'],
        ['llm-api-key-hint', 'hint.translationOnlyKey', 'hint.llmKey'],
        ['dashscope-key-hint', 'hint.dashscopeKeySimple', 'hint.dashscopeKey'],
    ];

    function syncSimpleCopy() {
        const t = (window.i18n && typeof window.i18n.t === 'function')
            ? window.i18n.t
            : (key) => key;
        SIMPLE_COPY_SWAPS.forEach(([id, simpleKey, advancedKey]) => {
            const el = byId(id);
            if (!el) return;
            const key = mode === 'simple' ? simpleKey : advancedKey;
            el.setAttribute('data-i18n', key);
            el.textContent = t(key);
        });
        // 「获取 DeepSeek API Key」链接仅简易模式显示
        const dsHint = byId('llm-api-key-deepseek-hint');
        if (dsHint) dsHint.hidden = (mode !== 'simple');
    }

    document.addEventListener('i18n:languageChanged', syncSimpleCopy);

    /* ---------------- 模式切换 ---------------- */

    function updateModeButtons() {
        const simpleBtn = byId('mode-simple-btn');
        const advancedBtn = byId('mode-advanced-btn');
        if (simpleBtn) {
            simpleBtn.classList.toggle('active', mode === 'simple');
            simpleBtn.setAttribute('aria-pressed', mode === 'simple' ? 'true' : 'false');
        }
        if (advancedBtn) {
            advancedBtn.classList.toggle('active', mode === 'advanced');
            advancedBtn.setAttribute('aria-pressed', mode === 'advanced' ? 'true' : 'false');
        }
    }

    function setMode(newMode, options = {}) {
        const { save = true, applyDefaults = true } = options;
        mode = newMode === 'advanced' ? 'advanced' : 'simple';
        document.body.classList.toggle('mode-simple', mode === 'simple');

        if (mode === 'simple') {
            moveToSimple();
            if (applyDefaults) {
                applySimpleDefaults();
            }
            syncSimpleTranslationUi();
        } else {
            restoreFromSimple();
            const block = byId('simple-translation-block');
            if (block) block.style.display = '';
        }

        syncSimpleCopy();
        updateModeButtons();
        syncSimpleTranslationModeControl();
        if (typeof window.updateDashscopeKeyFieldState === 'function') {
            window.updateDashscopeKeyFieldState();
        }
        if (save) {
            try {
                localStorage.setItem(MODE_KEY, mode);
            } catch (e) { /* ignore */ }
        }
    }

    const simpleBtn = byId('mode-simple-btn');
    const advancedBtn = byId('mode-advanced-btn');
    if (simpleBtn) simpleBtn.addEventListener('click', () => setMode('simple'));
    if (advancedBtn) advancedBtn.addEventListener('click', () => setMode('advanced'));

    const enableTranslationEl = byId('enable-translation');
    if (enableTranslationEl) {
        enableTranslationEl.addEventListener('change', () => {
            syncSimpleTranslationUi();
            syncSimpleTranslationModeControl();
        });
    }

    document.querySelectorAll('[data-simple-translation-mode]').forEach((button) => {
        button.addEventListener('click', () => {
            setTranslationEnabled(button.dataset.simpleTranslationMode === 'translation');
        });
    });

    /* ---------------- 初始化 ---------------- */

    let initialMode = 'simple';
    try {
        initialMode = localStorage.getItem(MODE_KEY) || 'simple';
    } catch (e) { /* ignore */ }

    // 初始时不立即应用默认值：等 main.js 加载完配置后再做（见下方轮询）
    setMode(initialMode, { save: false, applyDefaults: false });

    // 等待 main.js 初始化完成（启动按钮脱离“加载中”状态）后，
    // 若处于简易模式则应用默认值
    (function waitForAppReady() {
        let tries = 0;
        const timer = setInterval(() => {
            tries += 1;
            const startBtn = byId('start-btn');
            const loadingText = (window.i18n && typeof window.i18n.t === 'function')
                ? window.i18n.t('btn.loading')
                : null;
            const ready = startBtn && (
                !startBtn.disabled
                || (loadingText && startBtn.textContent.trim() !== loadingText)
            );
            if (ready || tries > 60) {
                clearInterval(timer);
                if (mode === 'simple') {
                    applySimpleDefaults();
                }
                syncSimpleCopy();
            }
        }, 300);
    })();

    /* ---------------- main.js 兼容层 ---------------- */

    /**
     * main.js 在校验失败时会展开/高亮某个设置项。
     * 简易模式下若目标元素在高级视图里且不在简易页中，则自动切到高级模式。
     */
    function revealFor(element, collapsibleId) {
        if (!element || !element.closest) return;
        if (mode !== 'simple') return;
        if (collapsibleId && SIMPLE_SAFE_COLLAPSIBLES.has(collapsibleId)) return;
        if (element.closest('#simple-view')) return;
        if (element.closest('#advanced-view')) {
            setMode('advanced', { applyDefaults: false });
        }
    }

    function wrapGlobal(name, isCollapsible) {
        const original = window[name];
        if (typeof original !== 'function') return;
        window[name] = function (id, ...rest) {
            revealFor(byId(id), isCollapsible ? id : null);
            return original.call(this, id, ...rest);
        };
    }

    wrapGlobal('toggleCollapsible', true);
    wrapGlobal('ensureCollapsibleExpanded', true);
    wrapGlobal('highlightInput', false);
})();
