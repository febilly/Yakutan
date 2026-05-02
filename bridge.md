这是实现“VRCX 本地环境上下文桥接”的参考说明，不是完整 prompt。目标是让外部翻译程序把用户当前 VRChat/VRCX 环境信息作为翻译 LLM 的辅助 context。

====================
一、整体架构
====================

推荐架构是“外部程序监听 localhost，VRCX 控制台脚本主动推送”。

不要依赖 VRCX 控制台脚本自己开 HTTP server，因为 VRCX/Electron/浏览器控制台里不一定能访问 Node require。更稳的是：

外部程序启动：
1. 监听 127.0.0.1。
2. 向系统申请随机可用端口，例如 Node.js 中 server.listen(0, "127.0.0.1")。
3. 生成随机 token，例如 crypto.randomBytes(24).toString("hex")。
4. 把随机端口和 token 填入 VRCX 控制台脚本模板。
5. 自动复制填好的完整脚本到剪贴板。
6. 提示用户：“脚本已复制，请打开 VRCX 主界面控制台粘贴运行。”
7. 如果无法访问剪贴板，则把脚本显示出来，让用户手动复制。

VRCX 控制台脚本运行：
1. 只读取 VRCX 本机 Pinia store 状态。
2. 不调用 window.webApiService。
3. 不请求 https://api.vrchat.cloud。
4. 不主动请求任何 VRChat API。
5. 定期构建 context。
6. 内容变化时 POST 到外部程序。
7. 连续推送失败达到上限后自动停止，避免刷错误。

外部程序收到 context：
1. 缓存 latestPayload。
2. 缓存 latestContext。
3. 缓存 latestContextText。
4. 缓存 latestReceivedAt、latestSequence、latestHash。
5. 翻译时，如果 latestContextText 存在且没有过期，就加入 LLM prompt 作为辅助环境上下文。

====================
二、外部程序 HTTP 接口
====================

外部程序应至少提供两个接口：

POST /vrcx/context?token=<token>

用途：
接收 VRCX 控制台脚本推送的 context payload。

处理逻辑：
1. 校验 token。
2. 读取 body。
3. JSON.parse body。
4. 保存：
   - latestPayload = payload
   - latestContext = payload.context
   - latestContextText = payload.contextText
   - latestReceivedAt = Date.now()
   - latestSequence = payload.sequence
   - latestHash = payload.hash
5. 返回 204。

注意：
VRCX 脚本默认使用 fetch no-cors + text/plain，所以服务端要把 text/plain body 当 JSON 字符串解析。

GET /vrcx/context/latest

用途：
调试或内部读取最新 context。

返回示例：
{
  "ok": true,
  "ageMs": 1234,
  "stale": false,
  "latestSequence": 12,
  "latestHash": "abc123",
  "latestContext": {...},
  "latestContextText": "..."
}

====================
三、翻译 LLM context 注入方式
====================

翻译时，如果 latestContextText 存在且未过期，例如 ageMs < 60000，把它加入 system/developer prompt。

注意：
这个 context 是辅助信息，不是待翻译文本。

推荐 prompt 结构：

你是一个实时语音/文本翻译助手。你的任务是把用户实际说的话从 {sourceLanguage} 翻译成 {targetLanguage}。

下面的 VRChat/VRCX 环境上下文只用于帮助理解场景、世界名、玩家名、好友关系、地点、称呼和指代。不要翻译或输出上下文本身。不要泄露上下文中的隐私信息。不要根据上下文编造原文没有表达的内容。

<VRCHAT_CONTEXT>
{latestContextText}
</VRCHAT_CONTEXT>

翻译要求：
- 只输出用户实际输入内容的译文。
- 保留原文语气、情绪、粗口强度和口语风格。
- 如果原文提到玩家名、世界名或上下文中的实体，尽量保持一致。
- 如果原文有歧义，可以利用上下文消歧。
- 不要添加解释。
- 不要输出引号，除非原文需要。

如果没有可用 context，则使用普通翻译 prompt，不要阻塞翻译流程。

====================
四、用户引导流程
====================

外部程序启动后：

1. 启动本机 HTTP 接收端。
2. 得到随机端口。
3. 生成随机 token。
4. 填充 VRCX 控制台脚本模板：
   - 替换 __VRCX_CONTEXT_PORT__
   - 替换 __VRCX_CONTEXT_TOKEN__
5. 复制到剪贴板。
6. UI 提示：

“VRCX 上下文桥接脚本已复制到剪贴板。
请打开 VRCX 主界面，按 F12 或打开开发者控制台，把脚本粘贴进去运行。
运行后，本程序会自动接收当前世界、实例、好友和玩家上下文，并用于辅助翻译。”

如果外部程序重启，端口和 token 会变化，需要重新复制粘贴脚本。
如果 VRCX 刷新或重启，也需要重新粘贴脚本。

====================
五、VRCX 控制台脚本模板
====================

外部程序需要把下面模板中的：
- __VRCX_CONTEXT_PORT__
- __VRCX_CONTEXT_TOKEN__

替换成真实运行时端口和 token。

const VRCX_CONSOLE_SCRIPT_TEMPLATE = String.raw`
(function () {
    console.log("🚀 [VRCXLocalContextBridge v1.1] 启动...");

    const CONFIG = {
        endpoint: "http://127.0.0.1:__VRCX_CONTEXT_PORT__/vrcx/context",
        token: "__VRCX_CONTEXT_TOKEN__",

        checkIntervalMs: 300,
        minPushIntervalMs: 500,
        pushImmediately: true,

        // true = 使用 no-cors + text/plain，避免 CORS 预检，适合 localhost 推送。
        fireAndForget: true,

        // 连续构建/推送失败达到该次数后自动停止。
        maxConsecutiveFailures: 5,

        includeFriendList: true,
        includePlayerList: true,

        // 默认不要把用户 ID、实例 private/group/friends ID 暴露给 LLM。
        includeUserIds: false,
        includeRawLocation: false,

        maxPlayers: 80,
        maxFriends: 80,
        maxWorldDescriptionChars: 900,

        verbose: true
    };

    function unref(value) {
        if (value && typeof value === "object" && value.__v_isRef) {
            return value.value;
        }
        return value;
    }

    function getServices() {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
        let vueApp = null;

        while (walker.nextNode()) {
            if (walker.currentNode.__vue_app__) {
                vueApp = walker.currentNode.__vue_app__;
                break;
            }
        }

        if (!vueApp) return null;

        const pinia =
            vueApp.config.globalProperties.$pinia ||
            (
                vueApp._context &&
                vueApp._context.provides &&
                (
                    vueApp._context.provides.pinia ||
                    vueApp._context.provides[Symbol.for("pinia")]
                )
            );

        if (!pinia || !pinia._s) return null;

        return {
            pinia,
            locationStore: pinia._s.get("Location"),
            userStore: pinia._s.get("User"),
            friendStore: pinia._s.get("Friend"),
            worldStore: pinia._s.get("World"),
            instanceStore: pinia._s.get("Instance"),
            photonStore: pinia._s.get("Photon"),
            gameStore: pinia._s.get("Game")
        };
    }

    function safeText(value, maxLen) {
        if (value === null || value === undefined) return "";
        return String(value)
            .replace(/\\s+/g, " ")
            .trim()
            .slice(0, maxLen || 240);
    }

    function mapValues(m) {
        m = unref(m);
        if (!m) return [];
        if (m instanceof Map) return Array.from(m.values());
        if (Array.isArray(m)) return m;

        if (typeof m.values === "function") {
            try {
                return Array.from(m.values());
            } catch (_) {}
        }

        if (typeof m === "object") return Object.values(m);
        return [];
    }

    function mapGet(m, key) {
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

    function mapSize(m) {
        m = unref(m);
        if (!m) return 0;
        if (typeof m.size === "number") return m.size;
        return mapValues(m).length;
    }

    function formatDuration(ms) {
        if (!Number.isFinite(ms) || ms <= 0) return "";
        const sec = Math.floor(ms / 1000);
        const min = Math.floor(sec / 60);
        const hour = Math.floor(min / 60);

        if (hour > 0) return hour + "h " + (min % 60) + "m";
        if (min > 0) return min + "m";
        return sec + "s";
    }

    function parseLocation(tag) {
        let _tag = String(tag || "");

        const ctx = {
            tag: _tag,
            isOffline: false,
            isPrivate: false,
            isTraveling: false,
            isRealInstance: false,
            worldId: "",
            instanceId: "",
            instanceName: "",
            accessType: "",
            accessTypeName: "",
            region: "",
            shortName: "",
            userId: null,
            hiddenId: null,
            privateId: null,
            friendsId: null,
            groupId: null,
            groupAccessType: null,
            canRequestInvite: false,
            strict: false,
            ageGate: false
        };

        if (_tag === "offline" || _tag === "offline:offline") {
            ctx.isOffline = true;
            return ctx;
        }

        if (_tag === "private" || _tag === "private:private") {
            ctx.isPrivate = true;
            return ctx;
        }

        if (_tag === "traveling" || _tag === "traveling:traveling") {
            ctx.isTraveling = true;
            return ctx;
        }

        if (!tag || _tag.startsWith("local")) {
            return ctx;
        }

        ctx.isRealInstance = true;

        const sep = _tag.indexOf(":");
        const shortNameQualifier = "&shortName=";
        const shortNameIndex = _tag.indexOf(shortNameQualifier);

        if (shortNameIndex >= 0) {
            ctx.shortName = _tag.substr(shortNameIndex + shortNameQualifier.length);
            _tag = _tag.substr(0, shortNameIndex);
        }

        if (sep < 0) {
            ctx.worldId = _tag;
            return ctx;
        }

        ctx.worldId = _tag.substr(0, sep);
        ctx.instanceId = _tag.substr(sep + 1);

        ctx.instanceId.split("~").forEach(function (s, i) {
            if (!i) {
                ctx.instanceName = s;
                return;
            }

            const A = s.indexOf("(");
            const Z = A >= 0 ? s.lastIndexOf(")") : -1;
            const key = Z >= 0 ? s.substr(0, A) : s;
            const value = A < Z ? s.substr(A + 1, Z - A - 1) : "";

            if (key === "hidden") ctx.hiddenId = value;
            else if (key === "private") ctx.privateId = value;
            else if (key === "friends") ctx.friendsId = value;
            else if (key === "canRequestInvite") ctx.canRequestInvite = true;
            else if (key === "region") ctx.region = value;
            else if (key === "group") ctx.groupId = value;
            else if (key === "groupAccessType") ctx.groupAccessType = value;
            else if (key === "strict") ctx.strict = true;
            else if (key === "ageGate") ctx.ageGate = true;
        });

        ctx.accessType = "public";

        if (ctx.privateId !== null) {
            ctx.accessType = ctx.canRequestInvite ? "invite+" : "invite";
            ctx.userId = ctx.privateId;
        } else if (ctx.friendsId !== null) {
            ctx.accessType = "friends";
            ctx.userId = ctx.friendsId;
        } else if (ctx.hiddenId !== null) {
            ctx.accessType = "friends+";
            ctx.userId = ctx.hiddenId;
        } else if (ctx.groupId !== null) {
            ctx.accessType = "group";
        }

        ctx.accessTypeName = ctx.accessType;

        if (ctx.groupAccessType !== null) {
            if (ctx.groupAccessType === "public") ctx.accessTypeName = "groupPublic";
            else if (ctx.groupAccessType === "plus") ctx.accessTypeName = "groupPlus";
        }

        return ctx;
    }

    function accessTypeLabel(type) {
        const labels = {
            public: "Public",
            friends: "Friends",
            "friends+": "Friends+",
            invite: "Invite",
            "invite+": "Invite+",
            group: "Group",
            groupPublic: "Group Public",
            groupPlus: "Group+"
        };

        return labels[type] || type || "";
    }

    function normalizeUser(entry, services) {
        entry = unref(entry) || {};

        const userStore = services.userStore;
        const friendStore = services.friendStore;

        const ref0 = unref(entry.ref) || {};
        const id =
            entry.userId ||
            entry.id ||
            ref0.id ||
            ref0.userId ||
            "";

        const cachedUser = mapGet(userStore && userStore.cachedUsers, id) || {};
        const friendCtx = mapGet(friendStore && friendStore.friends, id) || null;
        const friendRef = unref(friendCtx && friendCtx.ref) || {};

        const ref = Object.assign({}, cachedUser, friendRef, ref0);

        const joinAtMs =
            entry.joinTime ||
            entry.timer ||
            ref.$location_at ||
            ref.$online_for ||
            null;

        const result = {
            displayName: safeText(
                entry.displayName ||
                entry.name ||
                ref.displayName ||
                (friendCtx && friendCtx.name) ||
                "Unknown",
                80
            ),
            isFriend: Boolean(friendCtx || entry.isFriend),
            status: safeText(ref.status || (friendCtx && friendCtx.state) || entry.status || "", 40),
            statusDescription: safeText(ref.statusDescription || "", 160),
            trustLevel: safeText(ref.$trustLevel || "", 40),
            platform: safeText(
                ref.last_platform ||
                (ref.presence && ref.presence.platform) ||
                "",
                40
            ),
            inVRMode: Boolean(entry.inVRMode),
            isMaster: Boolean(entry.isMaster),
            isModerator: Boolean(entry.isModerator),
            joinedAt: joinAtMs ? new Date(Number(joinAtMs)).toISOString() : "",
            joinedAgo: joinAtMs ? formatDuration(Date.now() - Number(joinAtMs)) : ""
        };

        if (CONFIG.includeUserIds && id) {
            result.id = id;
        }

        return result;
    }

    function dedupeUsers(users) {
        const seen = new Set();
        const out = [];

        for (const user of users) {
            if (!user || !user.displayName || user.displayName === "Unknown") continue;

            const key = user.id || user.displayName;
            if (seen.has(key)) continue;

            seen.add(key);
            out.push(user);
        }

        return out;
    }

    function buildContextObject() {
        const services = getServices();

        if (!services || !services.locationStore || !services.userStore) {
            return {
                ok: false,
                generatedAt: new Date().toISOString(),
                error: "无法连接 VRCX Pinia stores。请确认脚本运行在 VRCX 主界面控制台。"
            };
        }

        const locationStore = services.locationStore;
        const userStore = services.userStore;
        const worldStore = services.worldStore;
        const instanceStore = services.instanceStore;
        const photonStore = services.photonStore;
        const gameStore = services.gameStore;

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
            mapGet(worldStore && worldStore.cachedWorlds, parsed.worldId) ||
            {};

        const instanceRef = unref(currentInstanceWorld.instance) || {};
        const currentInstanceUsersData = mapValues(instanceStore && instanceStore.currentInstanceUsersData);

        const playerListRaw = mapValues(lastLocation.playerList);
        const friendListRaw = mapValues(lastLocation.friendList);

        const friends = dedupeUsers(
            friendListRaw.map(function (x) {
                return normalizeUser(x, services);
            })
        ).slice(0, CONFIG.maxFriends);

        const playersFromLocation = playerListRaw.map(function (x) {
            return normalizeUser(x, services);
        });

        const playersFromInstance = currentInstanceUsersData.map(function (x) {
            return normalizeUser(x, services);
        });

        const knownPlayers = dedupeUsers(
            playersFromInstance.concat(playersFromLocation)
        ).slice(0, CONFIG.maxPlayers);

        const capacity =
            instanceRef.capacity ||
            worldRef.capacity ||
            worldRef.recommendedCapacity ||
            null;

        const context = {
            ok: true,
            generatedAt: new Date().toISOString(),
            source: "VRCX local Pinia stores only; no VRChat API request",

            self: {
                displayName: safeText(currentUser.displayName, 80),
                status: safeText(currentUser.status, 40),
                statusDescription: safeText(currentUser.statusDescription, 200),
                platform: safeText(
                    currentUser.last_platform ||
                    (currentUser.presence && currentUser.presence.platform) ||
                    "",
                    40
                )
            },

            game: {
                isGameRunning: Boolean(unref(gameStore && gameStore.isGameRunning)),
                isGameNoVR: Boolean(unref(gameStore && gameStore.isGameNoVR)),
                isSteamVRRunning: Boolean(unref(gameStore && gameStore.isSteamVRRunning)),
                isHmdAfk: Boolean(unref(gameStore && gameStore.isHmdAfk)),
                lastLocationTime: lastLocation.date || null
            },

            location: {
                worldId: parsed.worldId || "",
                instanceName: safeText(parsed.instanceName, 80),
                accessType: parsed.accessType,
                accessTypeLabel: accessTypeLabel(parsed.accessTypeName || parsed.accessType),
                region: safeText(parsed.region || (parsed.instanceId ? "us/default" : ""), 40),
                shortName: safeText(parsed.shortName, 80),
                groupAccessType: safeText(parsed.groupAccessType, 40),
                isOffline: parsed.isOffline,
                isPrivate: parsed.isPrivate,
                isTraveling: lastLocation.location === "traveling" || parsed.isTraveling,
                isRealInstance: parsed.isRealInstance,
                canRequestInvite: parsed.canRequestInvite,
                strict: parsed.strict,
                ageGate: parsed.ageGate
            },

            world: {
                name: safeText(
                    worldRef.name ||
                    lastLocation.name ||
                    (currentUser.presence && currentUser.presence.world) ||
                    "",
                    160
                ),
                description: safeText(worldRef.description || "", CONFIG.maxWorldDescriptionChars),
                authorName: safeText(worldRef.authorName || "", 100),
                releaseStatus: safeText(worldRef.releaseStatus || "", 40),
                capacity: capacity,
                recommendedCapacity: worldRef.recommendedCapacity || null,
                occupants: instanceRef.n_users || instanceRef.userCount || null,
                visits: worldRef.visits || null,
                favorites: worldRef.favorites || null,
                heat: worldRef.heat || null,
                popularity: worldRef.popularity || null,
                publicationDate: worldRef.publicationDate || "",
                updated_at: worldRef.updated_at || ""
            },

            instance: {
                accessType: parsed.accessType,
                accessTypeLabel: accessTypeLabel(parsed.accessTypeName || parsed.accessType),
                region: safeText(instanceRef.region || parsed.region || "", 40),
                queueEnabled: Boolean(instanceRef.queueEnabled),
                canRequestInvite: Boolean(parsed.canRequestInvite)
            },

            counts: {
                knownPlayersInInstance: mapSize(lastLocation.playerList),
                knownFriendsInInstance: mapSize(lastLocation.friendList),
                currentInstanceUsersData: currentInstanceUsersData.length,
                photonLobby: mapSize(photonStore && photonStore.photonLobby),
                capacity: capacity
            },

            friendsInSameInstance: CONFIG.includeFriendList ? friends : [],
            knownPlayersInSameInstance: CONFIG.includePlayerList ? knownPlayers : []
        };

        if (CONFIG.includeUserIds) {
            if (currentUser.id) context.self.id = currentUser.id;
            context.instance.ownerId = safeText(instanceRef.ownerId || parsed.userId || "", 100);
            context.instance.groupId = safeText(parsed.groupId || "", 100);
        }

        if (CONFIG.includeRawLocation) {
            context.location.raw = rawLocation;
            context.location.rawFromLastLocation = lastLocation.location || "";
            context.instance.fullLocationRaw = rawLocation;
        }

        return context;
    }

    function renderContextText(ctx) {
        if (!ctx.ok) {
            return "[VRCX Context]\\nERROR: " + ctx.error + "\\nGenerated: " + ctx.generatedAt;
        }

        const lines = [];

        lines.push("[VRChat / VRCX 当前环境上下文]");
        lines.push("生成时间: " + ctx.generatedAt);
        lines.push("数据来源: " + ctx.source);
        lines.push("");

        lines.push("## 当前用户");
        lines.push("- 名称: " + (ctx.self.displayName || "未知"));
        if (ctx.self.status) lines.push("- 状态: " + ctx.self.status);
        if (ctx.self.statusDescription) lines.push("- 状态简介: " + ctx.self.statusDescription);
        if (ctx.self.platform) lines.push("- 平台: " + ctx.self.platform);
        lines.push("");

        lines.push("## 当前世界/实例");
        lines.push("- 世界名: " + (ctx.world.name || "未知/尚未缓存"));
        if (ctx.world.description) lines.push("- 世界简介: " + ctx.world.description);
        if (ctx.world.authorName) lines.push("- 作者: " + ctx.world.authorName);
        if (ctx.world.releaseStatus) lines.push("- 发布状态: " + ctx.world.releaseStatus);
        lines.push("- 实例类型: " + (ctx.location.accessTypeLabel || "未知"));
        if (ctx.location.instanceName) lines.push("- 实例编号: " + ctx.location.instanceName);
        if (ctx.location.region) lines.push("- 区域: " + ctx.location.region);
        if (ctx.location.groupAccessType) lines.push("- Group 访问类型: " + ctx.location.groupAccessType);
        if (ctx.location.ageGate) lines.push("- 年龄门槛: 是");
        if (ctx.location.strict) lines.push("- Strict: 是");
        if (ctx.location.isTraveling) lines.push("- 当前状态: Traveling");
        lines.push(
            "- 人数: 已知玩家 " +
            ctx.counts.knownPlayersInInstance +
            " / 容量 " +
            (ctx.counts.capacity || "未知") +
            "，同实例好友 " +
            ctx.counts.knownFriendsInInstance
        );
        lines.push("");

        lines.push("## 同实例好友");
        if (!ctx.friendsInSameInstance.length) {
            lines.push("- 暂无已知好友，或 VRCX 当前本地状态尚未记录。");
        } else {
            for (const f of ctx.friendsInSameInstance) {
                const attrs = [];
                if (f.status) attrs.push(f.status);
                if (f.trustLevel) attrs.push(f.trustLevel);
                if (f.platform) attrs.push(f.platform);
                if (f.joinedAgo) attrs.push("joined " + f.joinedAgo + " ago");

                lines.push(
                    "- " +
                    f.displayName +
                    (attrs.length ? " (" + attrs.join(", ") + ")" : "")
                );
            }
        }
        lines.push("");

        lines.push("## 同实例已知玩家");
        if (!ctx.knownPlayersInSameInstance.length) {
            lines.push("- 暂无已知玩家列表，或 VRCX 当前本地状态尚未记录。");
        } else {
            for (const p of ctx.knownPlayersInSameInstance) {
                const attrs = [];
                if (p.isFriend) attrs.push("friend");
                if (p.status) attrs.push(p.status);
                if (p.trustLevel) attrs.push(p.trustLevel);
                if (p.platform) attrs.push(p.platform);
                if (p.isMaster) attrs.push("master");
                if (p.isModerator) attrs.push("moderator");
                if (p.joinedAgo) attrs.push("joined " + p.joinedAgo + " ago");

                lines.push(
                    "- " +
                    p.displayName +
                    (attrs.length ? " (" + attrs.join(", ") + ")" : "")
                );
            }
        }

        lines.push("");
        lines.push("## 注意");
        lines.push("- 这些信息全部来自 VRCX 本机已缓存状态，不会请求 VRChat API。");
        lines.push("- 世界详情为空通常表示 VRCX 当前还没有缓存该世界详情。");
        lines.push("- 玩家列表基于 VRCX/game log/Photon 本地状态，可能不是服务器完整实时列表。");

        return lines.join("\\n");
    }

    function makeComparable(value) {
        const volatileKeys = new Set([
            "generatedAt",
            "joinedAgo"
        ]);

        function walk(v) {
            if (v === null || typeof v !== "object") return v;

            if (Array.isArray(v)) {
                return v.map(walk);
            }

            const out = {};
            for (const key of Object.keys(v).sort()) {
                if (volatileKeys.has(key)) continue;
                out[key] = walk(v[key]);
            }

            return out;
        }

        return walk(value);
    }

    function stableStringify(value) {
        const seen = new WeakSet();

        function walk(v) {
            if (v === null || typeof v !== "object") return v;

            if (seen.has(v)) return "[Circular]";
            seen.add(v);

            if (Array.isArray(v)) {
                return v.map(walk);
            }

            const out = {};
            for (const key of Object.keys(v).sort()) {
                if (typeof v[key] !== "function") {
                    out[key] = walk(v[key]);
                }
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
        if (CONFIG.token) {
            url.searchParams.set("token", CONFIG.token);
        }
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

        if (timer) {
            clearInterval(timer);
            timer = null;
        }

        console.warn("🛑 [VRCXLocalContextBridge] 已停止。原因: " + (reason || "manual"));
    }

    async function pushNow(reason) {
        if (stopped) return;
        if (inFlight) return;

        const now = Date.now();

        if (reason !== "manual" && now - lastPushAt < CONFIG.minPushIntervalMs) {
            return;
        }

        let context;

        try {
            context = buildContextObject();
        } catch (e) {
            consecutiveFailures += 1;
            console.error("❌ [VRCXLocalContextBridge] 构建 context 失败：", e);

            if (consecutiveFailures >= CONFIG.maxConsecutiveFailures) {
                stop("连续 " + consecutiveFailures + " 次构建/推送失败");
            }

            return;
        }

        const comparable = makeComparable(context);
        const comparableString = stableStringify(comparable);
        const currentHash = hashString(comparableString);

        latestContext = context;
        latestContextText = renderContextText(context);

        if (reason !== "manual" && currentHash === lastHash) {
            return;
        }

        const payload = {
            type: "vrcx.context.changed",
            version: "1.1",
            sequence: ++sequence,
            reason: reason,
            pushedAt: new Date().toISOString(),
            hash: currentHash,
            context: context,
            contextText: latestContextText
        };

        const body = JSON.stringify(payload);

        try {
            inFlight = true;

            if (CONFIG.fireAndForget) {
                await fetch(makeUrl(), {
                    method: "POST",
                    mode: "no-cors",
                    cache: "no-store",
                    headers: {
                        "Content-Type": "text/plain;charset=UTF-8"
                    },
                    body: body
                });
            } else {
                const res = await fetch(makeUrl(), {
                    method: "POST",
                    mode: "cors",
                    cache: "no-store",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: body
                });

                if (!res.ok) {
                    throw new Error("HTTP " + res.status + " " + res.statusText);
                }
            }

            lastHash = currentHash;
            lastPushAt = Date.now();
            consecutiveFailures = 0;

            if (CONFIG.verbose) {
                const worldName = context && context.world && context.world.name
                    ? context.world.name
                    : "unknown world";
                const friends = context && context.counts
                    ? context.counts.knownFriendsInInstance
                    : "?";
                const players = context && context.counts
                    ? context.counts.knownPlayersInInstance
                    : "?";

                console.log(
                    "✅ [VRCXLocalContextBridge] 已推送 #" +
                    sequence +
                    ": " +
                    worldName +
                    ", friends=" +
                    friends +
                    ", players=" +
                    players +
                    ", reason=" +
                    reason
                );
            }
        } catch (e) {
            consecutiveFailures += 1;

            console.error(
                "❌ [VRCXLocalContextBridge] 推送失败 " +
                consecutiveFailures +
                "/" +
                CONFIG.maxConsecutiveFailures +
                ":",
                e
            );

            if (consecutiveFailures >= CONFIG.maxConsecutiveFailures) {
                stop("连续 " + consecutiveFailures + " 次推送失败");
            }
        } finally {
            inFlight = false;
        }
    }

    function tick() {
        pushNow("changed");
    }

    timer = setInterval(tick, CONFIG.checkIntervalMs);

    window.VRCXLocalContextBridge = {
        version: "1.1",
        config: CONFIG,

        getContext: function () {
            latestContext = buildContextObject();
            latestContextText = renderContextText(latestContext);
            return latestContext;
        },

        getContextText: function () {
            if (!latestContextText) {
                latestContext = buildContextObject();
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
                version: "1.1",
                stopped: stopped,
                sequence: sequence,
                lastHash: lastHash,
                lastPushAt: lastPushAt ? new Date(lastPushAt).toISOString() : "",
                consecutiveFailures: consecutiveFailures,
                maxConsecutiveFailures: CONFIG.maxConsecutiveFailures,
                endpoint: CONFIG.endpoint
            };
        }
    };

    if (CONFIG.pushImmediately) {
        pushNow("startup");
    }

    console.log("✅ [VRCXLocalContextBridge] 已就绪。");
    console.log("手动推送: window.VRCXLocalContextBridge.pushNow()");
    console.log("读取上下文: window.VRCXLocalContextBridge.getContextText()");
    console.log("查看状态: window.VRCXLocalContextBridge.getStatus()");
    console.log("停止脚本: window.VRCXLocalContextBridge.stop()");
})();
`;

====================
六、失败停止策略
====================

VRCX 控制台脚本必须有连续失败自动停止机制：

- consecutiveFailures 从 0 开始。
- 每次构建 context 或 fetch 推送抛错，consecutiveFailures += 1。
- 每次推送成功，consecutiveFailures = 0。
- 如果 consecutiveFailures >= maxConsecutiveFailures，调用 stop()。
- 默认 maxConsecutiveFailures = 5。

注意：
如果使用 fetch no-cors，浏览器无法可靠读取 HTTP 状态码，所以 401/500 这类服务端状态可能不会表现为失败。只有网络层失败、端口无法连接、fetch 抛错等情况才会被计入失败。服务端仍然必须校验 token。

====================
七、隐私和安全约束
====================

1. 外部程序只监听 127.0.0.1。
2. 不要监听 0.0.0.0。
3. token 每次启动随机生成。
4. VRCX 脚本默认 includeUserIds = false。
5. VRCX 脚本默认 includeRawLocation = false。
6. 不要请求 VRChat API。
7. 不要上传完整原始 JSON 到第三方。
8. 给翻译 LLM 的 context 应优先使用 latestContextText，而不是完整 raw JSON。
9. context 只用于辅助翻译，不应该直接出现在译文中。