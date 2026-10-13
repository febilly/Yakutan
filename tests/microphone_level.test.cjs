const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/static/js/main.js'), 'utf8');
const meterCode = source.slice(source.indexOf('function renderMicLevel('), source.indexOf('// 显示配置保存信息'));

function setup(hidden = false) {
    const handlers = {};
    const connections = [];
    const select = { value: '', addEventListener: (name, fn) => { handlers[name] = fn; } };
    const meter = { setAttribute: (key, value) => { meter[key] = value; } };
    const fill = { style: {} };
    const elements = { 'mic-device': select, 'mic-level': meter, 'mic-level-fill': fill };
    const context = vm.createContext({
        document: { hidden, getElementById: id => elements[id] },
        window: { addEventListener: (name, fn) => { handlers[name] = fn; } },
        API_BASE: '/api', setInterval: fn => { handlers.tick = fn; return 1; }, clearInterval() {},
        EventSource: class {
            constructor(url) { this.url = url; connections.push(this); }
            close() { this.closed = true; }
        },
    });
    vm.runInContext(meterCode + '\nsetupMicLevelMeter();', context);
    return { handlers, connections, select, meter };
}

test('meters a hidden open page without requiring recognition to run', () => {
    const { connections, meter } = setup(true);
    connections[0].onmessage({ data: '{"level":0.1}' });
    assert.equal(meter['aria-valuenow'], '67');
});

test('device change closes old connection and ignores late audio', () => {
    const { handlers, connections, select, meter } = setup();
    const old = connections[0];
    select.value = 'wasapi:mic-b';
    handlers.change();
    assert.equal(old.closed, true);
    assert.match(connections[1].url, /wasapi%3Amic-b/);
    old.onmessage({ data: '{"level":1}' });
    assert.equal(meter['aria-valuenow'], '0');
});

test('closing page releases capture and restoring reconnects', () => {
    const { handlers, connections } = setup();
    handlers.pagehide();
    assert.equal(connections[0].closed, true);
    handlers.pageshow();
    assert.equal(connections.length, 2);
});

test('configuration hydration reconnects and failures clear meter', () => {
    const { handlers, connections, select, meter } = setup();
    select.value = 'wasapi:mic-c';
    handlers.tick();
    assert.equal(connections.length, 2);
    connections[1].onmessage({ data: '{"level":0.5}' });
    assert.equal(meter['aria-valuenow'], '90');
    connections[1].onerror();
    assert.equal(meter['aria-valuenow'], '0');
});
