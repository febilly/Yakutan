"""Regression coverage for browser-managed settings versus legacy CLI .env."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import config


def _read_config_in_fresh_process(env_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)
    script = (
        "import json, config; "
        "print(json.dumps({"
        "'api_type': config.TRANSLATION_API_TYPE,"
        "'template': config.LLM_TEMPLATE,"
        "'base_url': config.LLM_BASE_URL,"
        "'model': config.LLM_MODEL,"
        "'extra_body': config.OPENAI_COMPAT_EXTRA_BODY_JSON,"
        "'llm_key': config.LLM_API_KEY,"
        "'osc_port': config.OSC_SEND_TARGET_PORT"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_plain_config_import_ignores_user_environment():
    values = _read_config_in_fresh_process(
        {
            "TRANSLATION_API_TYPE": "qwen_mt",
            "LLM_TEMPLATE": "custom1",
            "LLM_BASE_URL": "https://environment.invalid/v1",
            "LLM_MODEL": "environment-model",
            "OPENAI_COMPAT_EXTRA_BODY_JSON": '{"from": "environment"}',
            "LLM_API_KEY": "environment-key",
            "OSC_SEND_TARGET_PORT": "12345",
        }
    )

    assert values == {
        "api_type": "openrouter_streaming",
        "template": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "extra_body": '{"thinking": {"type": "disabled"}}',
        "llm_key": "",
        "osc_port": 9000,
    }


def test_cli_environment_overlay_is_explicit(monkeypatch):
    tracked = {
        name: getattr(config, name)
        for name in (
            "TRANSLATION_API_TYPE",
            "LLM_TEMPLATE",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "OPENAI_COMPAT_EXTRA_BODY_JSON",
            "LLM_API_KEY",
            "OSC_SEND_TARGET_PORT",
        )
    }
    monkeypatch.setenv("TRANSLATION_API_TYPE", "qwen_mt")
    monkeypatch.setenv("LLM_TEMPLATE", "custom1")
    monkeypatch.setenv("LLM_BASE_URL", "https://cli.example/v1")
    monkeypatch.setenv("LLM_MODEL", "cli-model")
    monkeypatch.setenv("OPENAI_COMPAT_EXTRA_BODY_JSON", '{"cli": true}')
    monkeypatch.setenv("LLM_API_KEY", "cli-key")
    monkeypatch.setenv("OSC_SEND_TARGET_PORT", "12345")

    try:
        config.apply_cli_env()

        assert config.TRANSLATION_API_TYPE == "qwen_mt"
        assert config.LLM_TEMPLATE == "custom1"
        assert config.LLM_BASE_URL == "https://cli.example/v1"
        assert config.LLM_MODEL == "cli-model"
        assert config.OPENAI_COMPAT_EXTRA_BODY_JSON == '{"cli": true}'
        assert config.LLM_API_KEY == "cli-key"
        assert config.OSC_SEND_TARGET_PORT == 12345
    finally:
        for name, value in tracked.items():
            setattr(config, name, value)


def test_web_defaults_are_complete_deepseek_profile():
    defaults = config.get_default_ui_config()
    translation = defaults["translation"]

    assert translation["enable_translation"] is True
    assert translation["api_type"] == "openrouter_streaming"
    assert translation["llm_template"] == "deepseek-v4-flash"
    assert translation["llm_base_url"] == "https://api.deepseek.com"
    assert translation["llm_model"] == "deepseek-v4-flash"
    assert translation["openai_compat_extra_body_json"] == (
        '{"thinking": {"type": "disabled"}}'
    )
    assert translation["llm_translation_formality"] == "medium"
    assert translation["llm_translation_style"] == "standard"
    assert translation["llm_parallel_fastest_mode"] == "off"


def test_web_config_update_rolls_back_settings_and_credentials(monkeypatch):
    from ui import app as ui_app

    monkeypatch.setattr(config, "TARGET_LANGUAGE", "ja")
    monkeypatch.setattr(config, "LLM_API_KEY", "before-key")

    success, message_id, _message = ui_app.update_config(
        {
            "api_keys": {"llm": "new-key"},
            "translation": {
                "target_language": "fr",
                "openai_compat_extra_body_json": "[]",
            },
        }
    )

    assert success is False
    assert message_id == "msg.invalidExtraBodyJson"
    assert config.TARGET_LANGUAGE == "ja"
    assert config.LLM_API_KEY == "before-key"


def test_runtime_credentials_replace_and_clear_without_environment_fallback(monkeypatch):
    from ui import app as ui_app

    monkeypatch.setenv("LLM_API_KEY", "inherited-key")
    monkeypatch.setattr(config, "LLM_API_KEY", "")

    ui_app.update_runtime_credentials({"llm": "browser-key"})
    assert config.LLM_API_KEY == "browser-key"

    ui_app.update_runtime_credentials({"llm": ""})
    assert config.LLM_API_KEY == ""

    response = ui_app.app.test_client().get("/api/credentials/status")
    assert response.status_code == 200
    assert response.get_json()["llm"]["api_key_set"] is False


def test_defaults_endpoint_uses_canonical_deepseek_profile():
    from ui import app as ui_app

    response = ui_app.app.test_client().get("/api/config/defaults")
    assert response.status_code == 200

    translation = response.get_json()["translation"]
    assert translation["api_type"] == config.DEFAULT_TRANSLATION_API_TYPE
    assert translation["llm_template"] == config.DEFAULT_LLM_TEMPLATE
    assert translation["llm_base_url"] == config.DEFAULT_LLM_BASE_URL
    assert translation["llm_model"] == config.DEFAULT_LLM_MODEL
    assert (
        translation["openai_compat_extra_body_json"]
        == config.DEFAULT_LLM_EXTRA_BODY_JSON
    )
    assert translation["translate_partial_results"] is True


def test_web_config_update_hymt2_streaming_toggle(monkeypatch):
    from ui import app as ui_app

    monkeypatch.setattr(config, "TRANSLATION_API_TYPE", "openrouter_streaming")
    monkeypatch.setattr(config, "TRANSLATE_PARTIAL_RESULTS", True)

    # 1. 切到 hymt2 并显式关闭流式
    success, message_id, _ = ui_app.update_config(
        {
            "translation": {
                "api_type": "hymt2",
                "hymt2_websocket_url": "ws://127.0.0.1:18765",
                "translate_partial_results": False,
            }
        }
    )
    assert success is True
    assert config.TRANSLATION_API_TYPE == "hymt2"
    assert config.TRANSLATE_PARTIAL_RESULTS is False

    # 2. 重新开启流式
    success, message_id, _ = ui_app.update_config(
        {
            "translation": {
                "api_type": "hymt2",
                "translate_partial_results": True,
            }
        }
    )
    assert success is True
    assert config.TRANSLATE_PARTIAL_RESULTS is True


def test_recognizer_credentials_ignore_inherited_environment(monkeypatch):
    from speech_recognizers import recognizer_factory

    monkeypatch.setenv("DASHSCOPE_API_KEY", "inherited-dashscope")
    monkeypatch.setenv("SONIOX_API_KEY", "inherited-soniox")
    monkeypatch.setenv("DOUBAO_API_KEY", "inherited-doubao")
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "")
    monkeypatch.setattr(config, "SONIOX_API_KEY", "")
    monkeypatch.setattr(config, "DOUBAO_API_KEY", "")
    monkeypatch.setattr(config, "DOUBAO_APP_ID", "")
    monkeypatch.setattr(config, "DOUBAO_ACCESS_KEY", "")
    monkeypatch.setattr(recognizer_factory, "SonioxSpeechRecognizer", object)
    monkeypatch.setattr(recognizer_factory, "WEBSOCKETS_AVAILABLE", True)

    recognizer_factory.init_dashscope_api_key()

    assert recognizer_factory.dashscope.api_key == "<your-dashscope-api-key>"
    assert recognizer_factory.is_backend_available("soniox") is False
    assert recognizer_factory._resolve_doubao_credentials() == (None, None, None)
