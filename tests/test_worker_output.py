"""无 GPU 测试生产 worker 的 JSON 截断边界。"""

from __future__ import annotations

import ast
import json
import logging
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "action_worker.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))


def find_function(name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


namespace: dict[str, Any] = {
    "Any": Any,
    "json": json,
    "re": re,
    "logger": logging.getLogger("worker-output-test"),
}
namespace["_TRAILING_COMMA_RE"] = re.compile(r",\s*([\]}])")
namespace["RETRY_MAX_NEW_TOKENS"] = 1024
namespace["CONTINUATION_GAP_SEC"] = 0
for name in (
    "_scan_balanced", "_loads_lenient", "_output_needs_token_retry",
    "_is_explicit_empty_output", "_token_limits", "_can_continue", "_merge_same_label",
):
    node = find_function(name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<worker-output-test>", "exec"), namespace)

collect_node = find_function("_collect_raw_items")
class_node = ast.ClassDef(
    name="Parser",
    bases=[],
    keywords=[],
    body=[collect_node],
    decorator_list=[],
)
ast.fix_missing_locations(class_node)
exec(compile(ast.Module(body=[class_node], type_ignores=[]), "<worker-output-test>", "exec"), namespace)

collect = namespace["Parser"]._collect_raw_items
needs_retry = namespace["_output_needs_token_retry"]
is_empty = namespace["_is_explicit_empty_output"]
token_limits = namespace["_token_limits"]
can_continue = namespace["_can_continue"]
merge_same_label = namespace["_merge_same_label"]


checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, condition))
    print(f"[{'OK ' if condition else 'FAIL'}] {name}")


complete = '[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid", "confidence": 0.9}]'
truncated = '[{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"}, {"start_sec": 4'
truncated_object = '{"start_sec": 1, "end_sec": 3, "label": "draw_liquid"'

check("完整数组保留动作", len(collect(complete)) == 1)
check("截断数组不捡取前缀", collect(truncated) == [])
check("截断对象不捡取前缀", collect(truncated_object) == [])
check("截断数组触发重试判定", needs_retry(truncated) is True)
check("截断对象触发重试判定", needs_retry(truncated_object) is True)
check("完整数组不触发重试判定", needs_retry(complete) is False)
check("完整空数组可接受", is_empty("[]") is True)
check("说明文字中的空数组不算成功", is_empty("无法识别，[]") is False)
check("512 截断时扩大到 1024", token_limits(512) == [512, 1024])
check("1024 不重复重试", token_limits(1024) == [1024])

first = {"label": "acquire_item", "start_sec": 1, "end_sec": 2, "confidence": 0.9, "source_window": 1}
second = {"label": "acquire_item", "start_sec": 3, "end_sec": 4, "confidence": 0.9, "source_window": 2}
check("同标签有 1 秒空档时不续接", can_continue(first, second) is False)
check("同标签首尾相接时可续接", can_continue(first, {**second, "start_sec": 2}) is True)

merged = dict(first)
merged.update({"stitched_windows": [1]})
check(
    "worker 不把有空档的同标签动作合并",
    merge_same_label(merged, second) is False and merged["end_sec"] == 2,
)

failed = [name for name, passed in checks if not passed]
print(f"\n共 {len(checks)} 项，通过 {len(checks) - len(failed)}，失败 {len(failed)}")
if failed:
    print("失败项: " + ", ".join(failed))
    raise SystemExit(1)
