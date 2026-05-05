from __future__ import annotations

import json

import vrcx_context_bridge as bridge


class TestVrcxContextBridge:
    def test_build_console_script_fills_endpoint_and_token(self):
        script = bridge.build_console_script("http://127.0.0.1:5001/vrcx/context")

        assert "__VRCX_CONTEXT_ENDPOINT__" not in script
        assert "__VRCX_CONTEXT_TOKEN__" not in script
        assert "http://127.0.0.1:5001/vrcx/context" in script
        assert bridge.get_token() in script
        assert "1.5-compact-players" in script
        assert "heartbeatIntervalMs: 30000" in script
        assert "printContextOnPush: true" in script
        assert "sent context #" in script
        assert 'reason: unchanged ? "heartbeat" : reason' in script
        assert "Pronouns" in script
        assert "renderUserLine" in script
        assert "collectOtherPlayerNames" in script
        assert "Other players in instance" in script
        assert "Known players in instance" not in script
        assert "worldRef.description" not in script
        assert "worldRef.authorName" not in script
        assert "entry.pronounce" in script

    def test_store_payload_rejects_invalid_token(self):
        ok, reason = bridge.store_payload(
            "bad-token",
            json.dumps({"contextText": "World: Test"}).encode("utf-8"),
        )

        assert ok is False
        assert reason == "invalid token"

    def test_store_payload_makes_context_available_for_prompt(self):
        payload = {
            "sequence": 7,
            "context": {
                "ok": True,
                "world": {"name": "Test World"},
                "friends": [{"name": "Alice"}],
                "players": [{"name": "Bob"}],
            },
            "contextText": "World: Test World\nKnown players here: Alice; Bob",
        }

        ok, reason = bridge.store_payload(
            bridge.get_token(),
            json.dumps(payload).encode("utf-8"),
        )

        assert ok is True
        assert reason == "ok"
        assert "Test World" in bridge.get_latest_context_text()

        prefix = bridge.build_translation_context_prefix("Base VRChat prompt")
        assert "Base VRChat prompt" in prefix
        assert "<VRCHAT_CONTEXT>" in prefix
        assert "Alice" in prefix
        assert "Do not translate, output, or disclose this context" in prefix

        status = bridge.get_status()
        assert status["connected"] is True
        assert status["latestSequence"] == 7

    def test_build_asr_context_text_and_terms(self):
        payload = {
            "sequence": 8,
            "context": {
                "ok": True,
                "self": {"name": "SelfUser"},
                "world": {"name": "Cozy World"},
                "friends": [{"name": "Alice"}],
                "players": [{"name": "Bob"}, {"name": "Alice"}],
            },
            "contextText": "World: Cozy World\nKnown players here: Alice; Bob",
        }

        ok, reason = bridge.store_payload(
            bridge.get_token(),
            json.dumps(payload).encode("utf-8"),
        )

        assert ok is True
        assert reason == "ok"

        asr_context = bridge.build_asr_context_text("HotTerm")
        assert "HotTerm" in asr_context
        assert "VRChat ASR hints" in asr_context
        assert "Cozy World" in asr_context
        assert "Bob" in asr_context

        terms = bridge.get_asr_context_terms()
        assert terms == ["Cozy World", "SelfUser", "Alice", "Bob"]

    def test_store_payload_uses_js_context_text_verbatim(self):
        context_text = "\n".join([
            "[VRChat/VRCX local context]",
            "## World",
            "- Name: Detailed World",
            "## Friends in instance",
            "- Name: Alice; Status: ask me; Status note: friend note; Pronouns: she/her",
            "## Other players in instance",
            "- Names: Bob; Carol",
            "## JS-rendered sentinel",
            "- Format: owned by the console script",
        ])
        payload = {
            "sequence": 9,
            "context": {
                "ok": True,
                "self": {"name": "SelfUser"},
                "world": {"name": "Detailed World"},
                "friends": [
                    {
                        "name": "Alice",
                        "status": "ask me",
                        "statusDescription": "friend note",
                        "pronouns": "she/her",
                    }
                ],
                "players": [
                    {"name": "SelfUser", "friend": False},
                    {"name": "Alice", "friend": True},
                    {"name": "Bob", "friend": False},
                    {"name": "Carol", "friend": False},
                ],
                "counts": {"players": 4, "friends": 1, "capacity": 16},
            },
            "contextText": context_text,
        }

        ok, reason = bridge.store_payload(
            bridge.get_token(),
            json.dumps(payload).encode("utf-8"),
        )

        assert ok is True
        assert reason == "ok"

        assert bridge.get_latest_context_text() == context_text
        assert bridge.get_latest_context() == payload["context"]

        status = bridge.get_status()
        assert status["latestContextText"] == context_text
        assert status["latestContext"] == payload["context"]
