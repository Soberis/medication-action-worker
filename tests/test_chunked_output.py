"""无 GPU 测试分块评测脚本的模型输出容错。"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import re
import runpy
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "run_35_chunked.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
names = {
    "_scan_balanced",
    "_loads_lenient",
    "_coerce_action_value",
    "parse_model_output",
    "_token_limits",
    "validate_local_actions",
    "merge_actions",
}
functions = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name in names
]
namespace: dict[str, Any] = {
    "Any": Any, "json": json, "re": re, "math": math,
    "RETRY_MAX_NEW_TOKENS": 1024,
}
namespace["_TRAILING_COMMA_RE"] = re.compile(r",\s*([\]}])")
namespace["LABEL_DEFINITIONS"] = runpy.run_path(
    str(ROOT / "label_schema.py")
)["DEFAULT_LABEL_DEFINITIONS"]
exec(compile(ast.Module(body=functions, type_ignores=[]), "<chunked-output-test>", "exec"), namespace)
parse_model_output = namespace["parse_model_output"]
token_limits = namespace["_token_limits"]
validate_local_actions = namespace["validate_local_actions"]
merge_actions = namespace["merge_actions"]


checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, condition))
    print(f"[{'OK ' if condition else 'FAIL'}] {name}")


check(
    "标准数组",
    parse_model_output('[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}]')
    == [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}],
)
check(
    "Markdown 围栏",
    parse_model_output('```json\n[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"}]\n```')
    == [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}],
)
check(
    "actions 外层包装",
    parse_model_output('{"actions": [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"}]}')
    == [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}],
)
check(
    "单个动作对象",
    parse_model_output('{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"}')
    == [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}],
)
check(
    "末尾多余逗号",
    parse_model_output('[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid",},]')
    == [{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}],
)
check(
    "截断数组拒绝",
    parse_model_output('[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9')
    is None,
)
check(
    "不从截断数组中捡取前缀动作",
    parse_model_output(
        '[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"}, '
        '{"start_sec": 4'
    ) is None,
)
check(
    "截断对象拒绝",
    parse_model_output('{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"') is None,
)
check(
    "紧凑数组行",
    parse_model_output('[[1, 3, "draw_liquid", 0.9], [4, 6, "transfer_liquid", 0.8]]')
    == [
        {"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9},
        {"start_sec": 4, "end_sec": 6, "label": "transfer_liquid", "confidence": 0.8},
    ],
)
check("明确空结果", parse_model_output("[]") == [])
check("无关文本不伪装成空结果", parse_model_output("无法识别") is None)
check("说明文字中的空数组不算协议成功", parse_model_output("无法识别，[]") is None)
check("512 截断时扩大到 1024", token_limits(512) == [512, 1024])
check("1024 不重复重试", token_limits(1024) == [1024])

overlap_input = [
    {"start_sec": 3, "end_sec": 5, "label": "insert_needle", "confidence": 0.98},
    {"start_sec": 0, "end_sec": 4, "label": "acquire_item", "confidence": 0.98},
]
validated = validate_local_actions(overlap_input, 10)
check(
    "重叠/乱序候选先排序而不崩溃",
    [(item["start_sec"], item["end_sec"]) for item in validated] == [(0, 4), (3, 5)],
)
merged = merge_actions([dict(item) for item in validated], 10)
check(
    "重叠候选由统一收尾切成非重叠区间",
    len(merged) == 2 and merged[0]["end_sec"] <= merged[1]["start_sec"],
)

failed = [name for name, passed in checks if not passed]
print(f"\n共 {len(checks)} 项，通过 {len(checks) - len(failed)}，失败 {len(failed)}")
if failed:
    print("失败项: " + ", ".join(failed))
    raise SystemExit(1)
