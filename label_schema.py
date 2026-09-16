"""Shared medication-action label vocabulary.

Keep this module dependency-free so the offline smoke tests, the production
worker and the Qwen comparison scripts use exactly the same label contract.
The default prompt exposes the nine observable v4 business actions; robot
execution primitives and deferred cognitive labels stay outside this contract.
"""

from __future__ import annotations


ACTION_LABEL_SCHEMA_VERSION = "p0-atomic-action.v4"

CORE_LABEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "acquire_item": {
        "name": "拿取容器/工具",
        "desc": "手明确到达并取得将参与配药的容器或工具，物体离开原稳定位置并进入操作区；不标持续持有、调整位置或重复抓握",
    },
    "open_container": {
        "name": "开启容器",
        "desc": "明确打开药品主容器，如掰开安瓿或移除西林瓶/输液容器的保护部件；不包括拆注射器包装",
    },
    "disinfect_access": {
        "name": "消毒容器接口",
        "desc": "看到棉签或其他消毒用品实际擦拭瓶塞、安瓿颈或容器接口；仅出现用品不构成该动作",
    },
    "insert_needle": {
        "name": "针头进入容器",
        "desc": "清楚看到针头穿过瓶塞或接口并进入目标药品容器；仅靠近或被遮挡时不标注",
    },
    "draw_liquid": {
        "name": "抽取药液",
        "desc": "针头已进入容器且活塞向后移动，能看到药液进入注射器；仅进针没有抽液证据时不标注",
    },
    "transfer_liquid": {
        "name": "转移药液",
        "desc": "使用注射器把药液从一个容器推入另一个目标容器；要求能确认来源、目标和推注方向，不表示面向患者给药",
    },
    "agitate_solution": {
        "name": "混匀药液",
        "desc": "明确摇晃、翻转或振荡装有药液的容器以混匀；仅拿着或移动容器不标注",
    },
    "discard_sharps": {
        "name": "处置锐器",
        "desc": "明确把针头、安瓿或其他锐器放入锐器盒；仅出现锐器盒或丢弃普通耗材不标注",
    },
    "label_prepared_medication": {
        "name": "给成品贴签",
        "desc": "看到标签实际粘贴到注射器、药瓶或成品输液容器；仅查看、拿着标签或成品包装不标注",
    },
}

# Deferred labels such as ``inspect_medication`` and ``place_item`` are not in
# the v4 schema. Add them only through a new platform schema revision after
# their visual evidence and acceptance rules are defined.
OPTIONAL_LABEL_DEFINITIONS: dict[str, dict[str, str]] = {}

# 历史任务可能仍冻结这些 ID。只在消息明确携带且平台仍允许时兼容，
# 不把它们放入默认 schema。
LEGACY_LABEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "open_cap": {
        "name": "开启瓶塞/针帽",
        "desc": "明确开启或掰开药品容器，或拔下注射器针帽；只标注实际开启动作",
    },
    "inject_liquid": {
        "name": "转移药液",
        "desc": "推动活塞将药液推入另一药瓶、输液袋或目标容器；不表示面向患者给药",
    },
    "mix_solution": {
        "name": "混匀药液",
        "desc": "可见摇晃、翻转或振荡容器以混匀药液",
    },
    "label_seal": {
        "name": "贴签/封装",
        "desc": "可见粘贴成品标签或封闭成品包装；不把拿取、持物或查看标签算作此动作",
    },
}

DEFAULT_LABEL_DEFINITIONS: dict[str, dict[str, str]] = dict(CORE_LABEL_DEFINITIONS)
ALL_LABEL_DEFINITIONS: dict[str, dict[str, str]] = {
    **CORE_LABEL_DEFINITIONS,
    **OPTIONAL_LABEL_DEFINITIONS,
    **LEGACY_LABEL_DEFINITIONS,
}

# 这些旧标签无法稳定表示一次需要监测的物理配药动作，或会覆盖大段持物/移动过程。
# 即使旧 platform allowed_labels 仍带有它们，worker 也会在送入 prompt 前过滤。
IGNORED_LABELS = frozenset({"retrieve_medicine", "check_info"})

# 拿取动作涉及的目标物品受体类别（用于模型受体锚定与分类识别）
RECOGNIZED_ITEM_TYPES: dict[str, str] = {
    "vial": "西林瓶/粉针瓶",
    "ampoule": "安瓿瓶",
    "syringe": "注射器",
    "infusion_bottle": "输液瓶/输液袋",
    "needle": "针头/穿刺针",
    "cotton_swab": "棉签",
    "packaging": "包装耗材",
    "other_tool": "其他工具/器械",
}

# 拿取动作（瞬态物理事件）的最大允许持续时间（秒），防止拼接出超长持物或空镜循环
MAX_ACQUIRE_DURATION_SEC: int = 5

