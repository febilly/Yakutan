const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness({confirmed = true, response = {success: true}, ok = true} = {}) {
    const source = fs.readFileSync(path.join(__dirname, '../ui/static/js/main.js'), 'utf8');
    const start = source.indexOf('let legacyLocalModels = null;');
    const end = source.indexOf('\nasync function downloadLocalInferenceModels', start);
    const button = {}, size = {}, requests = [], alerts = [];
    const context = {document: {getElementById: id => id === 'local-legacy-cleanup-btn' ? button : size},
        window: {i18n: {t: key => key}}, API_BASE: '/api', confirm: () => confirmed,
        alert: value => alerts.push(value),
        fetch: async (...args) => { requests.push(args); return {ok, json: async () => response}; },
        refreshLocalInferenceStatus: async () => { context.refreshed = true; }};
    vm.createContext(context);
    vm.runInContext(source.slice(start, end), context);
    return {context, button, size, requests, alerts};
}

test('cleanup stays hidden without obsolete files and disabled before migration', () => {
    const h = harness();
    h.context.renderLegacyLocalModels({models: {legacy: {count: 0}}});
    assert.equal(h.button.hidden, true);
    h.context.renderLegacyLocalModels({models: {legacy: {count: 1, bytes: 1048576, can_cleanup: false}}});
    assert.equal(h.button.hidden, false);
    assert.equal(h.button.disabled, true);
    assert.equal(h.size.textContent, '1.0 MiB');
});

test('cancel never calls cleanup API', async () => {
    const h = harness({confirmed: false});
    h.context.renderLegacyLocalModels({models: {legacy: {count: 1, bytes: 1, can_cleanup: true}}});
    await h.context.cleanupLegacyLocalModels();
    assert.equal(h.requests.length, 0);
});

test('confirmed cleanup posts once and reports locked files', async () => {
    const h = harness({response: {success: true, failed: ['llama.dll']}});
    h.context.renderLegacyLocalModels({models: {legacy: {count: 1, bytes: 1, can_cleanup: true}}});
    await h.context.cleanupLegacyLocalModels();
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0][0], '/api/local-inference/legacy-cleanup');
    assert.equal(h.requests[0][1].method, 'POST');
    assert.equal(h.alerts.length, 1);
    assert.equal(h.context.refreshed, true);
});

test('download disables cleanup', () => {
    const h = harness();
    h.context.renderLegacyLocalModels({download: {running: true}, models: {legacy: {count: 1, bytes: 1, can_cleanup: true}}});
    assert.equal(h.button.disabled, true);
});
