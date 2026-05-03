from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from typing import Any, Optional


CONTEXT_STALE_MS = 60_000
MAX_CONTEXT_TEXT_CHARS = 3_000
MAX_ASR_CONTEXT_TEXT_CHARS = 2_000
MAX_REQUEST_BODY_BYTES = 256 * 1024

_token = secrets.token_urlsafe(24)
_lock = threading.RLock()
_latest_payload: Optional[dict[str, Any]] = None
_latest_context: Optional[dict[str, Any]] = None
_latest_context_text = ""
_latest_received_at_ms = 0
_latest_sequence = 0
_latest_hash = ""


VRCX_CONSOLE_SCRIPT_TEMPLATE = r"""
(function () {
    const CONFIG = {
        endpoint: "__VRCX_CONTEXT_ENDPOINT__",
        token: "__VRCX_CONTEXT_TOKEN__",
        checkIntervalMs: 1000,
        minPushIntervalMs: 1000,
        heartbeatIntervalMs: 30000,
        pushImmediately: true,
        fireAndForget: true,
        maxConsecutiveFailures: 5,
        maxPlayers: 50,
        maxFriends: 30,
        maxWorldDescriptionChars: 280,
        printContextOnPush: true,
        verbose: false
    };

    function unref(value) {
        return value && typeof value === "object" && value.__v_isRef ? value.value : value;
    }

    function safeText(value, maxLen) {
        if (value === null || value === undefined) return "";
        return String(value).replace(/\s+/g, " ").trim().slice(0, maxLen || 120);
    }

    function firstText(maxLen) {
        for (let i = 1; i < arguments.length; i++) {
            const text = safeText(arguments[i], maxLen);
            if (text) return text;
        }
        return "";
    }

    function valuesOf(m) {
        m = unref(m);
        if (!m) return [];
        if (m instanceof Map) return Array.from(m.values());
        if (Array.isArray(m)) return m;
        if (typeof m.values === "function") {
            try {
                return Array.from(m.values());
            } catch (_) {}
        }
        return typeof m === "object" ? Object.values(m) : [];
    }

    function getFrom(m, key) {
        m = unref(m);
        if (!m || !key) return undefined;
        if (m instanceof Map) return m.get(key);
        if (typeof m.get === "function") {
            try {
                return m.get(key);
            } catch (_) {}
        }
        return m[key];
    }

    function sizeOf(m) {
        m = unref(m);
        if (!m) return 0;
        if (typeof m.size === "number") return m.size;
        return valuesOf(m).length;
    }

    function getStores() {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
        let vueApp = null;
        while (walker.nextNode()) {
            if (walker.currentNode.__vue_app__) {
                vueApp = walker.currentNode.__vue_app__;
                break;
            }
        }
        if (!vueApp) return null;

        const provides = vueApp._context && vueApp._context.provides;
        const pinia =
            vueApp.config.globalProperties.$pinia ||
            (provides && (provides.pinia || provides[Symbol.for("pinia")]));
        if (!pinia || !pinia._s) return null;

        return {
            locationStore: pinia._s.get("Location"),
            userStore: pinia._s.get("User"),
            friendStore: pinia._s.get("Friend"),
            worldStore: pinia._s.get("World"),
            instanceStore: pinia._s.get("Instance"),
            photonStore: pinia._s.get("Photon"),
            gameStore: pinia._s.get("Game")
        };
    }

    function parseLocation(raw) {
        let tag = String(raw || "");
        const out = {
            isOffline: tag === "offline" || tag === "offline:offline",
            isPrivate: tag === "private" || tag === "private:private",
            isTraveling: tag === "traveling" || tag === "traveling:traveling",
            isRealInstance: false,
            worldId: "",
            instanceName: "",
            accessType: "",
            accessLabel: "",
            region: ""
        };
        if (!tag || out.isOffline || out.isPrivate || out.isTraveling || tag.startsWith("local")) {
            return out;
        }

        const shortNameIndex = tag.indexOf("&shortName=");
        if (shortNameIndex >= 0) tag = tag.slice(0, shortNameIndex);
        const sep = tag.indexOf(":");
        out.isRealInstance = sep >= 0;
        out.worldId = sep >= 0 ? tag.slice(0, sep) : tag;
        if (sep < 0) return out;

        const instanceId = tag.slice(sep + 1);
        const parts = instanceId.split("~");
        out.instanceName = safeText(parts[0], 60);
        out.accessType = "public";

        for (let i = 1; i < parts.length; i++) {
            const s = parts[i];
            const a = s.indexOf("(");
            const z = a >= 0 ? s.lastIndexOf(")") : -1;
            const key = z >= 0 ? s.slice(0, a) : s;
            const value = a < z ? s.slice(a + 1, z) : "";
            if (key === "hidden") out.accessType = "friends+";
            else if (key === "friends") out.accessType = "friends";
            else if (key === "private") out.accessType = "invite";
            else if (key === "canRequestInvite" && out.accessType === "invite") out.accessType = "invite+";
            else if (key === "group") out.accessType = "group";
            else if (key === "groupAccessType" && value === "plus") out.accessType = "group+";
            else if (key === "region") out.region = safeText(value, 24);
        }

        const labels = {
            public: "Public",
            friends: "Friends",
            "friends+": "Friends+",
            invite: "Invite",
            "invite+": "Invite+",
            group: "Group",
            "group+": "Group+"
        };
        out.accessLabel = labels[out.accessType] || out.accessType;
        return out;
    }

    function normalizeUser(entry, stores) {
        entry = unref(entry) || {};
        const ref0 = unref(entry.ref) || {};
        const id = entry.userId || entry.id || ref0.id || ref0.userId || "";
        const cached = getFrom(stores.userStore && stores.userStore.cachedUsers, id) || {};
        const friendCtx = getFrom(stores.friendStore && stores.friendStore.friends, id) || null;
        const friendRef = unref(friendCtx && friendCtx.ref) || {};
        const ref = Object.assign({}, cached, friendRef, ref0);
        const name = safeText(
            entry.displayName ||
            entry.name ||
            ref.displayName ||
            (friendCtx && friendCtx.name) ||
            "",
            80
        );
        if (!name) return null;
        return {
            name: name,
            friend: Boolean(friendCtx || entry.isFriend),
            status: safeText(ref.status || (friendCtx && friendCtx.state) || entry.status || "", 24),
            statusDescription: safeText(ref.statusDescription || "", 80),
            pronouns: firstText(
                80,
                entry.pronouns,
                entry.pronoun,
                entry.pronounsText,
                entry.pronounce,
                entry.pronunciation,
                ref.pronouns,
                ref.pronoun,
                ref.pronounsText,
                ref.pronounce,
                ref.pronunciation,
                friendCtx && friendCtx.pronouns,
                friendCtx && friendCtx.pronoun,
                friendCtx && friendCtx.pronounsText,
                friendRef.pronouns,
                friendRef.pronoun,
                friendRef.pronounsText
            ),
            master: Boolean(entry.isMaster),
            moderator: Boolean(entry.isModerator)
        };
    }

    function dedupeUsers(users) {
        const seen = new Set();
        const out = [];
        for (const user of users) {
            if (!user || !user.name || seen.has(user.name)) continue;
            seen.add(user.name);
            out.push(user);
        }
        return out;
    }

    function buildContext() {
        const stores = getStores();
        if (!stores || !stores.locationStore || !stores.userStore) {
            return { ok: false, error: "VRCX Pinia stores unavailable" };
        }

        const locationStore = stores.locationStore;
        const userStore = stores.userStore;
        const worldStore = stores.worldStore;
        const instanceStore = stores.instanceStore;
        const photonStore = stores.photonStore;
        const gameStore = stores.gameStore;

        const currentUser = unref(userStore.currentUser) || {};
        const lastLocation = unref(locationStore.lastLocation) || {};
        const rawLocation =
            lastLocation.location === "traveling"
                ? unref(locationStore.lastLocationDestination) || lastLocation.location
                : lastLocation.location || "";
        const parsed = parseLocation(rawLocation);

        const currentInstanceWorld = unref(instanceStore && instanceStore.currentInstanceWorld) || {};
        const worldRef =
            unref(currentInstanceWorld.ref) ||
            getFrom(worldStore && worldStore.cachedWorlds, parsed.worldId) ||
            {};
        const instanceRef = unref(currentInstanceWorld.instance) || {};
        const instanceUsers = valuesOf(instanceStore && instanceStore.currentInstanceUsersData);

        const friends = dedupeUsers(
            valuesOf(lastLocation.friendList).map(function (x) {
                return normalizeUser(x, stores);
            })
        ).slice(0, CONFIG.maxFriends);

        const players = dedupeUsers(
            instanceUsers.concat(valuesOf(lastLocation.playerList)).map(function (x) {
                return normalizeUser(x, stores);
            })
        ).slice(0, CONFIG.maxPlayers);

        const capacity = instanceRef.capacity || worldRef.capacity || worldRef.recommendedCapacity || null;
        return {
            ok: true,
            self: {
                name: safeText(currentUser.displayName, 80),
                status: safeText(currentUser.status, 24),
                statusDescription: safeText(currentUser.statusDescription, 100),
                pronouns: firstText(
                    80,
                    currentUser.pronouns,
                    currentUser.pronoun,
                    currentUser.pronounsText,
                    currentUser.pronounce,
                    currentUser.pronunciation
                )
            },
            game: {
                running: Boolean(unref(gameStore && gameStore.isGameRunning)),
                noVR: Boolean(unref(gameStore && gameStore.isGameNoVR)),
                hmdAfk: Boolean(unref(gameStore && gameStore.isHmdAfk))
            },
            location: {
                offline: parsed.isOffline,
                private: parsed.isPrivate,
                traveling: lastLocation.location === "traveling" || parsed.isTraveling,
                type: parsed.accessLabel,
                instance: parsed.instanceName,
                region: parsed.region || safeText(instanceRef.region || "", 24)
            },
            world: {
                name: safeText(
                    worldRef.name ||
                    lastLocation.name ||
                    (currentUser.presence && currentUser.presence.world) ||
                    "",
                    120
                ),
                description: safeText(worldRef.description || "", CONFIG.maxWorldDescriptionChars),
                author: safeText(worldRef.authorName || "", 80)
            },
            counts: {
                players: sizeOf(lastLocation.playerList),
                friends: sizeOf(lastLocation.friendList),
                photon: sizeOf(photonStore && photonStore.photonLobby),
                capacity: capacity
            },
            friends: friends,
            players: players
        };
    }

    function renderContextText(ctx) {
        if (!ctx.ok) return "VRCX context unavailable: " + ctx.error;

        const lines = [];
        lines.push("[VRChat/VRCX local context]");

        function addField(fields, label, value) {
            const text = safeText(value, 180);
            if (text) fields.push(label + ": " + text);
        }

        function renderUserLine(user) {
            const fields = [];
            const tags = [];
            addField(fields, "Name", user && user.name);
            addField(fields, "Status", user && user.status);
            addField(fields, "Status note", user && user.statusDescription);
            addField(fields, "Pronouns", user && user.pronouns);
            if (user && user.friend) tags.push("friend");
            if (user && user.master) tags.push("master");
            if (user && user.moderator) tags.push("moderator");
            if (tags.length) fields.push("Tags: " + tags.join(", "));
            return "- " + (fields.length ? fields.join("; ") : "Name: unknown");
        }

        lines.push("## Self");
        lines.push(renderUserLine(ctx.self));

        lines.push("## World");
        const worldFields = [];
        addField(worldFields, "Name", ctx.world.name || "unknown world");
        addField(worldFields, "Author", ctx.world.author);
        lines.push("- " + worldFields.join("; "));
        if (ctx.world.description) lines.push("- Description: " + ctx.world.description);

        const locFields = [];
        addField(locFields, "Access", ctx.location.type);
        addField(locFields, "Instance", ctx.location.instance);
        addField(locFields, "Region", ctx.location.region);
        if (ctx.location.traveling) locFields.push("State: traveling");
        if (ctx.location.offline) locFields.push("State: offline");
        if (locFields.length) {
            lines.push("## Instance");
            lines.push("- " + locFields.join("; "));
        }

        lines.push(
            "## Counts\n" +
            "- Players: " +
            (ctx.counts.players || 0) +
            "/" +
            (ctx.counts.capacity || "?") +
            "; Friends: " +
            (ctx.counts.friends || 0)
        );

        if (ctx.friends.length) {
            lines.push("## Friends in instance");
            ctx.friends.forEach(function (f) {
                lines.push(renderUserLine(f));
            });
        }

        if (ctx.players.length) {
            lines.push("## Known players in instance");
            ctx.players.forEach(function (p) {
                lines.push(renderUserLine(p));
            });
        }

        return lines.join("\n");
    }

    function stableStringify(value) {
        const seen = new WeakSet();
        function walk(v) {
            if (v === null || typeof v !== "object") return v;
            if (seen.has(v)) return "[Circular]";
            seen.add(v);
            if (Array.isArray(v)) return v.map(walk);
            const out = {};
            for (const key of Object.keys(v).sort()) {
                if (typeof v[key] !== "function") out[key] = walk(v[key]);
            }
            return out;
        }
        return JSON.stringify(walk(value));
    }

    function hashString(str) {
        let h = 2166136261;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
        }
        return (h >>> 0).toString(16);
    }

    function makeUrl() {
        const url = new URL(CONFIG.endpoint);
        url.searchParams.set("token", CONFIG.token);
        return url.toString();
    }

    let timer = null;
    let inFlight = false;
    let stopped = false;
    let lastHash = "";
    let lastPushAt = 0;
    let sequence = 0;
    let latestContext = null;
    let latestContextText = "";
    let consecutiveFailures = 0;

    function stop(reason) {
        if (stopped) return;
        stopped = true;
        if (timer) clearInterval(timer);
        timer = null;
        console.warn("[VRCXLocalContextBridge] stopped: " + (reason || "manual"));
    }

    async function pushNow(reason) {
        if (stopped || inFlight) return;
        if (reason !== "manual" && Date.now() - lastPushAt < CONFIG.minPushIntervalMs) return;

        let context;
        try {
            context = buildContext();
        } catch (e) {
            consecutiveFailures += 1;
            console.error("[VRCXLocalContextBridge] failed to build context:", e);
            if (consecutiveFailures >= CONFIG.maxConsecutiveFailures) {
                stop("too many build failures");
            }
            return;
        }

        latestContext = context;
        latestContextText = renderContextText(context);
        const currentHash = hashString(stableStringify(context));
        const heartbeatDue =
            !lastPushAt ||
            Date.now() - lastPushAt >= CONFIG.heartbeatIntervalMs;
        const unchanged = currentHash === lastHash;
        if (reason !== "manual" && unchanged && !heartbeatDue) return;

        const payload = {
            sequence: ++sequence,
            reason: unchanged ? "heartbeat" : reason,
            hash: currentHash,
            context: context,
            contextText: latestContextText
        };

        try {
            inFlight = true;
            await fetch(makeUrl(), {
                method: "POST",
                mode: CONFIG.fireAndForget ? "no-cors" : "cors",
                cache: "no-store",
                headers: { "Content-Type": "text/plain;charset=UTF-8" },
                body: JSON.stringify(payload)
            });
            lastHash = currentHash;
            lastPushAt = Date.now();
            consecutiveFailures = 0;
            if (CONFIG.printContextOnPush) {
                console.log(
                    "[VRCXLocalContextBridge] sent context #" +
                    payload.sequence +
                    " (" +
                    payload.reason +
                    ")\n" +
                    latestContextText
                );
            }
            if (CONFIG.verbose) {
                console.log("[VRCXLocalContextBridge] pushed #" + sequence);
            }
        } catch (e) {
            consecutiveFailures += 1;
            console.error(
                "[VRCXLocalContextBridge] push failed " +
                consecutiveFailures +
                "/" +
                CONFIG.maxConsecutiveFailures,
                e
            );
            if (consecutiveFailures >= CONFIG.maxConsecutiveFailures) {
                stop("too many push failures");
            }
        } finally {
            inFlight = false;
        }
    }

    timer = setInterval(function () {
        pushNow("changed");
    }, CONFIG.checkIntervalMs);

    window.VRCXLocalContextBridge = {
        version: "1.4-labeled",
        getContext: function () {
            latestContext = buildContext();
            latestContextText = renderContextText(latestContext);
            return latestContext;
        },
        getContextText: function () {
            if (!latestContextText) {
                latestContext = buildContext();
                latestContextText = renderContextText(latestContext);
            }
            return latestContextText;
        },
        pushNow: function () {
            return pushNow("manual");
        },
        stop: function () {
            stop("manual");
        },
        getStatus: function () {
            return {
                version: "1.4-labeled",
                stopped: stopped,
                sequence: sequence,
                lastPushAt: lastPushAt ? new Date(lastPushAt).toISOString() : "",
                consecutiveFailures: consecutiveFailures,
                heartbeatIntervalMs: CONFIG.heartbeatIntervalMs,
                endpoint: CONFIG.endpoint
            };
        }
    };

    if (CONFIG.pushImmediately) pushNow("startup");
    console.log("[VRCXLocalContextBridge] ready. Use window.VRCXLocalContextBridge.stop() to stop.");
})();
"""


def get_token() -> str:
    return _token


def build_console_script(endpoint: str) -> str:
    return (
        VRCX_CONSOLE_SCRIPT_TEMPLATE
        .replace("__VRCX_CONTEXT_ENDPOINT__", endpoint)
        .replace("__VRCX_CONTEXT_TOKEN__", get_token())
        .strip()
    )


def store_payload(token: str, body: bytes) -> tuple[bool, str]:
    if token != get_token():
        return False, "invalid token"
    if len(body or b"") > MAX_REQUEST_BODY_BYTES:
        return False, "request body too large"

    try:
        payload = json.loads((body or b"{}").decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return False, "invalid json"
    if not isinstance(payload, dict):
        return False, "payload must be an object"

    context = payload.get("context")
    if not isinstance(context, dict):
        context = None

    context_text = str(payload.get("contextText") or "").strip()
    if len(context_text) > MAX_CONTEXT_TEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_TEXT_CHARS].rstrip() + "\n..."

    sequence = payload.get("sequence")
    try:
        sequence_int = int(sequence)
    except (TypeError, ValueError):
        sequence_int = 0

    digest = str(payload.get("hash") or "").strip()
    if not digest:
        digest = hashlib.sha256(context_text.encode("utf-8")).hexdigest()[:16]

    now_ms = int(time.time() * 1000)
    with _lock:
        global _latest_payload, _latest_context, _latest_context_text
        global _latest_received_at_ms, _latest_sequence, _latest_hash
        _latest_payload = payload
        _latest_context = context
        _latest_context_text = context_text
        _latest_received_at_ms = now_ms
        _latest_sequence = sequence_int
        _latest_hash = digest

    return True, "ok"


def _age_ms(now_ms: Optional[int] = None) -> Optional[int]:
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    with _lock:
        if not _latest_received_at_ms:
            return None
        return max(0, now_ms - _latest_received_at_ms)


def get_latest_context_text(max_age_ms: int = CONTEXT_STALE_MS) -> str:
    with _lock:
        text = _latest_context_text
    if not text:
        return ""
    age = _age_ms()
    if age is None or age > max_age_ms:
        return ""
    return text


def get_latest_context(max_age_ms: int = CONTEXT_STALE_MS) -> Optional[dict[str, Any]]:
    age = _age_ms()
    if age is None or age > max_age_ms:
        return None
    with _lock:
        return _latest_context


def _trim_text(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n..."
    return text


def build_asr_context_text(base_context: str = "") -> str:
    context_text = get_latest_context_text()
    base = (base_context or "").strip()
    if not context_text:
        return base

    asr_context = _trim_text(context_text, MAX_ASR_CONTEXT_TEXT_CHARS)
    vrcx_block = f"VRChat ASR hints:\n{asr_context}"
    if not base:
        return vrcx_block
    return f"{base}\n\n{vrcx_block}"


def _append_term(terms: list[str], seen: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if not text or text in seen:
        return
    seen.add(text)
    terms.append(text)


def get_asr_context_terms(max_terms: int = 80) -> list[str]:
    context = get_latest_context()
    if not isinstance(context, dict):
        return []

    terms: list[str] = []
    seen: set[str] = set()

    world = context.get("world")
    if isinstance(world, dict):
        _append_term(terms, seen, world.get("name"))
        _append_term(terms, seen, world.get("author"))

    self_info = context.get("self")
    if isinstance(self_info, dict):
        _append_term(terms, seen, self_info.get("name"))

    for key in ("friends", "players"):
        entries = context.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                _append_term(terms, seen, entry.get("name"))
                if len(terms) >= max_terms:
                    return terms

    return terms[:max_terms]


def build_translation_context_prefix(base_prefix: str = "") -> str:
    context_text = get_latest_context_text()
    if not context_text:
        return base_prefix

    vrcx_block = (
        "VRChat/VRCX local context for disambiguating world names, player names, "
        "places and references. Do not translate, output, or disclose this context.\n"
        "<VRCHAT_CONTEXT>\n"
        f"{context_text}\n"
        "</VRCHAT_CONTEXT>"
    )
    base = (base_prefix or "").strip()
    if not base:
        return vrcx_block
    return f"{base}\n\n{vrcx_block}"


def get_status() -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    with _lock:
        age = None if not _latest_received_at_ms else max(0, now_ms - _latest_received_at_ms)
        stale = age is None or age > CONTEXT_STALE_MS
        return {
            "ok": True,
            "connected": age is not None and not stale,
            "ageMs": age,
            "stale": stale,
            "latestSequence": _latest_sequence,
            "latestHash": _latest_hash,
            "contextTextChars": len(_latest_context_text),
            "latestContextText": _latest_context_text if not stale else "",
            "latestContext": _latest_context if not stale else None,
        }
