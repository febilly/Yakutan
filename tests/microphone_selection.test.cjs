const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../ui/static/js/main.js'), 'utf8');
const functions = source.slice(source.indexOf('function setMicDeviceValue('),
    source.indexOf('function setupMicDeviceAutoRefresh('));

function setup() {
    const select = {
        options: [{ value: '', textContent: 'Default' }], selected: '',
        appendChild(option) { this.options.push(option); },
        setAttribute(name, value) { this[name] = value; },
        set innerHTML(_) { this.options = []; this.selected = ''; },
        get value() { return this.selected; },
        set value(value) { this.selected = this.options.some(o => o.value === value) ? value : ''; },
    };
    const context = vm.createContext({
        document: { getElementById: () => select, createElement: () => ({}) },
        window: {}, API_BASE: '/api', console: { warn() {} },
        lastMicDefaultIdentity: null, resolveDefaultMicDisplayName: () => '',
        maybeRestartForNewSystemDefaultMic: async () => {}, showMessage() {},
    });
    vm.runInContext(functions, context);
    context.fetch = async () => ({ ok: true, json: async () => ({
        devices: [{ id: 'session:1', index: 1, name: 'USB Mic' },
                  { id: 'session:2', index: 2, name: 'USB Mic' }],
        selected_id: 'session:1',
    }) });
    return { context, select };
}

test('saved old index and expired token remain explicit instead of default', async () => {
    const { context, select } = setup();
    assert.equal(context.savedMicDeviceValue({ mic_device_index: 1 }), 'legacy:1');
    context.setMicDeviceValue('old-session:1');
    await context.refreshMicDevices(true);
    assert.equal(select.value, 'old-session:1');
    assert.match(select.options.find(o => o.value === select.value).textContent, /重新选择/);
});

test('same-name devices can be selected separately and survive polling', async () => {
    const { context, select } = setup();
    await context.refreshMicDevices(false);
    assert.equal(select.value, 'session:1');
    select.value = 'session:2';
    await context.refreshMicDevices(true);
    assert.equal(select.value, 'session:2');
});

test('explicit default is not replaced with server selection', async () => {
    const { context, select } = setup();
    await context.refreshMicDevices(true);
    assert.equal(select.value, '');
});

test('failed enumeration preserves the selected device', async () => {
    const { context, select } = setup();
    context.setMicDeviceValue('session:2');
    context.fetch = async () => ({ ok: false, json: async () => ({ error: 'busy' }) });
    await context.refreshMicDevices(true);
    assert.equal(select.value, 'session:2');
});

test('selection made while request is pending is preserved', async () => {
    const { context, select } = setup();
    await context.refreshMicDevices(false);
    const fetch = context.fetch;
    let release;
    context.fetch = () => new Promise(resolve => { release = () => resolve(fetch()); });
    const pending = context.refreshMicDevices(true);
    select.value = 'session:2';
    release();
    await pending;
    assert.equal(select.value, 'session:2');
});

test('WASAPI list refresh keeps stable ID without requiring a service restart', async () => {
    const { context, select } = setup();
    context.setMicDeviceValue('wasapi:endpoint-b');
    context.fetch = async () => ({ ok: true, json: async () => ({
        devices: [{ id: 'wasapi:endpoint-b', name: 'USB Mic', label_suffix: 'bbbbbbbb' },
                  { id: 'wasapi:endpoint-a', name: 'USB Mic', label_suffix: 'aaaaaaaa' }],
        default_id: 'wasapi:endpoint-a', supports_hotplug: true,
    }) });
    await context.refreshMicDevices(true);
    assert.equal(select.value, 'wasapi:endpoint-b');
    assert.equal(select['data-restart-required'], 'false');
    assert.equal(select.options.length, 3);
    assert.notEqual(select.options[1].textContent, select.options[2].textContent);
});

test('disconnected WASAPI device stays selected until it returns', async () => {
    const { context, select } = setup();
    context.setMicDeviceValue('wasapi:endpoint-b');
    await context.refreshMicDevices(true);
    assert.equal(select.value, 'wasapi:endpoint-b');
    assert.match(select.options.find(o => o.value === select.value).textContent, /未连接/);
});
