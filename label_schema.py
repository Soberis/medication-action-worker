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
        "desc": "手伸向原静止物品并将其拿起离开台面的瞬间（通常仅1~3秒），物体进入操作区；持续握持不动、调整握姿、拆包装或未使物体发生位移绝不标注",
    },
    "open_container": {
        "name": "开启容器",
        "desc": "明确打开药品主容器或包装密封，如掰开安瓿瓶、取下西林瓶铝盖/塑料塞或拔下注射器针帽；不包括拆外层纸塑包装",
    },
    "disinfect_access": {
        "name": "消毒容器接口",
        "desc": "看到棉签或其他消毒用品实际反复擦拭瓶塞、安瓿颈或容器接口（持续擦拭2秒以上）；仅拿着棉签或轻微靠近不标注",
    },
    "insert_needle": {
        "name": "针头进入容器",
        "desc": "清楚看到针头穿过瓶塞或接口并进入目标药品容器且保持进针状态；仅靠近、悬停或被遮挡时不标注",
    },
    "draw_liquid": {
        "name": "抽取药液",
        "desc": "针头已在容器内且活塞向后移动，能清楚看到药液进入注射器针筒；仅进针但没有抽吸液体证据时不标注",
    },
    "transfer_liquid": {
        "name": "转移药液",
        "desc": "使用注射器把药液推入另一目标药瓶、输液袋或容器；必须能确认推注方向与活塞前移，不表示面向患者给药",
    },
    "agitate_solution": {
        "name": "混匀药液",
        "desc": "明确上下翻转输液袋或反复摇晃/振荡药瓶以混匀药液；水剂抽吸无此动作，仅拿着或微移容器绝不标注",
    },
    "discard_sharps": {
        "name": "处置锐器",
        "desc": "明确看到将针头、安瓿等锐器投入硬质利器盒（Sharps Container）；手部移开、放下物品或将垃圾扔进普通垃圾袋绝不标注",
    },
    "label_prepared_medication": {
        "name": "给成品贴签",
        "desc": "看到配药标签实际粘贴到注射器、药瓶或成品输液容器；仅查看标签、拿着标签或拿取成品包装绝不标注",
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

