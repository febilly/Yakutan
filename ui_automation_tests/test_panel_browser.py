from urllib.parse import urlencode

from playwright.sync_api import expect


def test_quick_lang_toggle_defaults_to_enabled(page, live_server):
    page.goto(live_server, wait_until='domcontentloaded')

    toggle = page.locator('#enable-quick-lang-bar')
    expect(toggle).to_be_checked()


def test_quick_lang_buttons_render_full_language_codes(page, live_server):
    query = urlencode([
        ('quick_lang_bar', '1'),
        ('quick_lang', 'en-GB'),
        ('quick_lang', 'zh-Hant-TW'),
        ('quick_lang', 'pt-BR'),
        ('quick_lang', 'fr-CA'),
    ])
    page.goto(f'{live_server}/panel?{query}', wait_until='domcontentloaded')

    buttons = page.locator('.quick-lang-btn')
    expect(buttons).to_have_count(4)
    expect(buttons.nth(0)).to_have_text('en-GB')
    expect(buttons.nth(1)).to_have_text('zh-Hant-TW')
    expect(buttons.nth(2)).to_have_text('pt-BR')
    expect(buttons.nth(3)).to_have_text('fr-CA')


def test_quick_lang_bar_can_be_hidden_from_initial_snapshot(page, live_server):
    query = urlencode([
        ('quick_lang_bar', '0'),
        ('quick_lang', 'en'),
        ('quick_lang', 'zh-CN'),
        ('quick_lang', 'ja'),
        ('quick_lang', 'ko'),
    ])
    page.goto(f'{live_server}/panel?{query}', wait_until='domcontentloaded')

    bar = page.locator('#quick-lang-bar')
    expect(bar).to_be_hidden()
    expect(page.locator('.quick-lang-btn')).to_have_count(0)