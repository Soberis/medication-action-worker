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


import math

namespace: dict[str, Any] = {
    "Any": Any,
    "json": json,
    "re": re,
    "math": math,
    "logger": logging.getLogger("worker-output-test"),
}
namespace["_TRAILING_COMMA_RE"] = re.compile(r",\s*([\]}])")
namespace["RETRY_MAX_NEW_TOKENS"] = 1024
namespace["CONTINUATION_GAP_SEC"] = 0
namespace["MAX_ACQUIRE_DURATION_SEC"] = 5
for name in (
    "_scan_balanced", "_loads_lenient", "_output_needs_token_retry",
    "_is_explicit_empty_output", "_token_limits", "_can_continue", "_merge_same_label",
    "_suppress_micro_oscillations",
):
    node = find_function(name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<worker-output-test>", "exec"), namespace)

collect_node = find_function("_collect_raw_items")
salvage_node = find_function("_salvage_truncated_output")
norm_node = find_function("_normalize_confidence")
class_node = ast.ClassDef(
    name="Parser",
    bases=[],
    keywords=[],
    body=[collect_node, salvage_node, norm_node],
    decorator_list=[],
)
ast.fix_missing_locations(class_node)
exec(compile(ast.Module(body=[class_node], type_ignores=[]), "<worker-output-test>", "exec"), namespace)

parser = namespace["Parser"]()
collect = parser._collect_raw_items
salvage = parser._salvage_truncated_output
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
check("截断数组在保底抢救模式下提取有效动作", len(salvage(truncated, 30, {"draw_liquid": {"name": "抽取药液"}})) == 1)
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
check(
    "acquire_item 超过 5 秒不续接合并",
    can_continue(
        {"label": "acquire_item", "start_sec": 1, "end_sec": 5, "confidence": 0.9},
        {"label": "acquire_item", "start_sec": 5, "end_sec": 7, "confidence": 0.9},
    ) is False,
)
check(
    "不同物品的 acquire_item 不合并",
    can_continue(
        {"label": "acquire_item", "item": "syringe", "start_sec": 1, "end_sec": 2, "confidence": 0.9},
        {"label": "acquire_item", "item": "vial", "start_sec": 2, "end_sec": 4, "confidence": 0.9},
    ) is False,
)
check(
    "相同物品的 acquire_item 在时限内可合并",
    can_continue(
        {"label": "acquire_item", "item": "vial", "start_sec": 1, "end_sec": 2, "confidence": 0.9},
        {"label": "acquire_item", "item": "vial", "start_sec": 2, "end_sec": 4, "confidence": 0.9},
    ) is True,
)
with_item = '[{"start_sec": 1, "end_sec": 3, "label": "acquire_item", "item": "syringe", "confidence": 0.92}]'
check("解析保留 item 字段", collect(with_item)[0].get("item") == "syringe")

merged = dict(first)
merged.update({"stitched_windows": [1]})
check(
    "worker 不把有空档的同标签动作合并",
    merge_same_label(merged, second) is False and merged["end_sec"] == 2,
)

suppress_fn = namespace["_suppress_micro_oscillations"]
normal_seq = [
    {"start_sec": 1, "end_sec": 2, "label": "acquire_item", "confidence": 0.9},
    {"start_sec": 5, "end_sec": 10, "label": "draw_liquid", "confidence": 0.95},
]
check("正常非抖动序列不被抑制", len(suppress_fn(normal_seq)) == 2)

chatter_seq = [
    {"start_sec": 0, "end_sec": 1, "label": "insert_needle", "confidence": 0.90},
    {"start_sec": 1, "end_sec": 2, "label": "draw_liquid", "confidence": 0.96},
    {"start_sec": 2, "end_sec": 3, "label": "discard_sharps", "confidence": 0.85},
    {"start_sec": 3, "end_sec": 4, "label": "acquire_item", "confidence": 0.88},
    {"start_sec": 10, "end_sec": 15, "label": "draw_liquid", "confidence": 0.95},
]
suppressed_res = suppress_fn(chatter_seq)
check(
    "连续 3 个以上 1 秒高频交替抖动被平滑抑制",
    len(suppressed_res) == 2 and suppressed_res[0]["label"] == "draw_liquid",
)

distant_micro_seq = [
    {"start_sec": 0, "end_sec": 1, "label": "open_container", "confidence": 0.90},
    {"start_sec": 30, "end_sec": 31, "label": "insert_needle", "confidence": 0.92},
    {"start_sec": 60, "end_sec": 61, "label": "draw_liquid", "confidence": 0.95},
]
check("远距离非相邻的独立短动作不被错误串联", len(suppress_fn(distant_micro_seq)) == 3)

same_label_micro_seq = [
    {"start_sec": 1, "end_sec": 2, "label": "acquire_item", "item": "vial", "confidence": 0.90},
    {"start_sec": 2, "end_sec": 3, "label": "acquire_item", "item": "syringe", "confidence": 0.92},
    {"start_sec": 3, "end_sec": 4, "label": "acquire_item", "item": "needle", "confidence": 0.95},
]
check("同标签不同受体的连续微动作不被错误当成震荡平滑", len(suppress_fn(same_label_micro_seq)) == 3)

failed = [name for name, passed in checks if not passed]
print(f"\n共 {len(checks)} 项，通过 {len(checks) - len(failed)}，失败 {len(failed)}")
if failed:
    print("失败项: " + ", ".join(failed))
    raise SystemExit(1)
