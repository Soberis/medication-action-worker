"""标签定义解析的无 GPU 单元测试。

只通过 AST 抽取 worker 的纯标签解析方法，避免测试启动 torch、Transformers、
RabbitMQ 或连接 Core API。
"""

from __future__ import annotations

import ast
import logging
import pathlib
import runpy
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "action_worker.py"
tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))


def extract_function(name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


namespace = runpy.run_path(str(ROOT / "label_schema.py"))
namespace.update({"Any": Any, "logger": logging.getLogger("label-resolution-test")})
exec(
    compile(
        ast.Module(
            body=[extract_function("_canonical_label_key"), extract_function("_resolve_label_definitions")],
            type_ignores=[],
        ),
        "<label-resolution-test>",
        "exec",
    ),
    namespace,
)


class FakeWorker:
    api_base_url = ""
    trust_env = False

    def _read_bearer_token(self) -> str:
        return ""


resolve = namespace["_resolve_label_definitions"]
worker = FakeWorker()
failures: list[str] = []


def check(name: str, condition: bool) -> None:
    print(f"[{'OK ' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(name)


check(
    "无标签限制时使用 v4 默认词表",
    set(resolve(worker, {})) == set(namespace["DEFAULT_LABEL_DEFINITIONS"]),
)
check(
    "allowed_labels 是严格白名单",
    set(resolve(worker, {"allowed_labels": ["DRAW_LIQUID"]})) == {"draw_liquid"},
)
check(
    "未知标签没有定义时 fail-closed",
    resolve(worker, {"allowed_labels": ["new_action"]}) == {},
)
check(
    "空 description 不会被字符串化成可用判据",
    resolve(worker, {
        "allowed_labels": ["new_action"],
        "label_definitions": [{"label_id": "new_action", "description": None}],
    }) == {},
)

custom_payload = {
    "allowed_labels": ["new_action"],
    "label_definitions": [{
        "label_id": "new_action",
        "display_name": "新动作",
        "positive_evidence": ["明确看到新动作发生"],
        "exclusion_criteria": ["仅出现相关物品不算"],
    }],
}
custom = resolve(worker, custom_payload)
check("新标签带正面证据可进入 prompt", set(custom) == {"new_action"})
check("新标签排除条件被保留", "排除条件" in custom.get("new_action", {}).get("desc", ""))
check(
    "泛化旧标签仍被忽略",
    resolve(worker, {"allowed_labels": ["retrieve_medicine"]}) == {},
)

if failures:
    print(f"失败 {len(failures)} 项: {', '.join(failures)}")
    raise SystemExit(1)
print("标签解析测试全部通过")
