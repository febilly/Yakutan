"""Regression coverage for the desktop-app version shown in the control-panel footer."""

import config


def test_index_footer_shows_configured_app_version():
    from ui import app as ui_app

    response = ui_app.app.test_client().get('/')

    assert response.status_code == 200
    assert f'v{config.APP_VERSION}' in response.get_data(as_text=True)
