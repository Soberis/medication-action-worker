"""消息失败处理的单元测试（不依赖 torch / GPU / 网络）。

覆盖的正是最贵的一类 bug：消息被默默丢掉。平台只回收「已认领且租约过期」的尝试，
从未被认领的 QUEUED 作业不会自愈 —— 所以 401/403 时一旦把消息丢弃，那个作业就永久卡住。

为什么用 AST 抽取而不是 import：本模块被测的是纯逻辑，而 action_worker.py 顶层会
import torch / transformers。用 AST 只取出被测定义，就能在没有 GPU 的机器上跑。

    python tests/test_message_handling.py
"""
import ast
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
import threading
from datetime import UTC, datetime
from typing import Callable

logging.disable(logging.CRITICAL)

SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / "action_worker.py"
tree = ast.parse(SRC_PATH.read_text(encoding="utf-8"))

WANT_FUNCS = {
    "_lease_error_hint", "_result_error_hint", "_remove_quietly",
    "_response_excerpt", "_download_error_hint", "_handle_message", "_validate_lease_grant",
    "_parse_instant", "_safe_job_id",
}
WANT_CLASSES = {"JobFailure", "LeaseHeartbeat"}
WANT_CONSTS = {
    "RECONNECT_BACKOFF_MAX_SEC", "LEASE_TOKEN_PATTERN", "LOCAL_DEFAULT_WORKER_ID",
    "DEFAULT_LABEL_DEFINITIONS", "IGNORED_LABELS",
}

ns = {"Any": object, "os": os, "json": json, "int": int, "str": str, "bytes": bytes,
      "logger": logging.getLogger("silent"), "threading": threading, "re": re,
      "datetime": datetime, "UTC": UTC, "Callable": Callable}
# LeaseHeartbeat.__enter__ 的返回注解在类体定义时就会求值并引用类名自身，先放占位避免 NameError
ns["LeaseHeartbeat"] = object

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
    elif isinstance(node, ast.ClassDef) and node.name in WANT_CLASSES:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in WANT_CONSTS:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id in WANT_CONSTS:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)

JobFailure = ns["JobFailure"]
LeaseHeartbeat = ns["LeaseHeartbeat"]
handle_message = ns["_handle_message"]
validate_lease_grant = ns["_validate_lease_grant"]
MAX_BACKOFF = ns["RECONNECT_BACKOFF_MAX_SEC"]
LEASE_TOKEN_PATTERN = ns["LEASE_TOKEN_PATTERN"]

results = []


def check(name, ok, extra=""):
    results.append((name, ok, extra))
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra and not ok else ""))


class FakeChannel:
    def __init__(self):
        self.calls = []

    def basic_ack(self, delivery_tag):
        self.calls.append(("ack", delivery_tag))

    def basic_nack(self, delivery_tag, requeue):
        self.calls.append(("nack", delivery_tag, requeue))


class FakeMethod:
    def __init__(self, redelivered=False, delivery_tag=7):
        self.redelivered = redelivered
        self.delivery_tag = delivery_tag


class FakeWorker:
    worker_id = "ai-worker-local-1"
    sampling_fps = 1.0

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self._reconnect_backoff = 5.0

    def process_job(self, payload):
        self._behaviour(payload)


def run(behaviour, body, redelivered=False):
    channel, method = FakeChannel(), FakeMethod(redelivered=redelivered)
    worker = FakeWorker(behaviour)
    raised = None
    try:
        handle_message(worker, channel, method, body)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    return worker, channel, raised


def ok_job(payload):
    return None


# ---------- 1. JobFailure 分档 ----------
cred = JobFailure("Core API 租约申请", "d", status=403, core_api=True)
check("Core API 403 判为凭证错误", cred.is_credential_error is True)
check("403 不可重投", cred.retryable is False)
check("Core API 401 判为凭证错误",
      JobFailure("Core API 租约申请", "d", status=401, core_api=True).is_credential_error is True)
check("同状态码但来自对象存储 → 不算凭证错误",
      JobFailure("媒体下载", "d", status=403).is_credential_error is False)
check("409 非凭证错误且不可重投",
      JobFailure("op", "d", status=409).is_credential_error is False
      and JobFailure("op", "d", status=409).retryable is False)
check("500 可重投", JobFailure("op", "d", status=500).retryable is True)
check("429 可重投", JobFailure("op", "d", status=429).retryable is True)
check("显式 retryable 覆盖状态码推断",
      JobFailure("op", "d", status=200, retryable=False).retryable is False)
check("hint 写进消息", "→" in str(JobFailure("op", "d", status=403, hint="照着做")))

# ---------- 2. _handle_message 路由 ----------
body = json.dumps({"job_id": "j1", "message_id": "m1"}).encode()
w, ch, raised = run(ok_job, body)
check("成功路径 → ack", ch.calls == [("ack", 7)] and raised is None)


def fail_403(_):
    raise JobFailure("Core API 租约申请", "bad", status=403, hint="h", core_api=True)


w, ch, raised = run(fail_403, body)
check("Core API 403 → 不 ack 也不丢弃", ch.calls == [])
check("Core API 403 → 抛出以便断开重连", isinstance(raised, JobFailure))
check("Core API 403 → 退避拉满，避免立刻重拿同一个 403", w._reconnect_backoff == MAX_BACKOFF)


def fail_storage_403(_):
    # 对象存储拒绝下载时也返回 403（预签名 URL 过期 / SigV4 host 不匹配）。
    # 它不是 Core API 的凭证问题：改配置不会让旧 URL 复活，重投也不会自愈 ——
    # 一旦被判成凭证问题就会「不 ACK 不丢弃」，这条消息被无限重投、worker 永久卡死。
    raise JobFailure("媒体下载", "signature mismatch", status=403,
                     hint=ns["_download_error_hint"](403))


w, ch, raised = run(fail_storage_403, body)
check("对象存储 403 → 丢弃(requeue=False)，不得无限重投",
      ch.calls == [("nack", 7, False)] and raised is None)


def fail_409(_):
    raise JobFailure("Core API 租约申请", "stale", status=409)


w, ch, raised = run(fail_409, body)
check("409 → 丢弃(requeue=False)", ch.calls == [("nack", 7, False)] and raised is None)


def fail_500(_):
    raise JobFailure("结果回传", "boom", status=500)


w, ch, _ = run(fail_500, body, redelivered=False)
check("500 首次 → 重投一次", ch.calls == [("nack", 7, True)])
w, ch, _ = run(fail_500, body, redelivered=True)
check("500 重投后 → 丢弃", ch.calls == [("nack", 7, False)])


def fail_contract(_):
    raise JobFailure("任务契约校验", "缺 message_id")


w, ch, _ = run(fail_contract, body)
check("本地契约失败 → 直接丢弃", ch.calls == [("nack", 7, False)])

w, ch, _ = run(ok_job, b"{not json")
check("非法 JSON → 丢弃而非重投", ch.calls == [("nack", 7, False)])


def fail_generic(_):
    raise RuntimeError("unexpected")


w, ch, _ = run(fail_generic, body, redelivered=False)
check("未知异常首次 → 重投一次", ch.calls == [("nack", 7, True)])
w, ch, _ = run(fail_generic, body, redelivered=True)
check("未知异常重投后 → 丢弃", ch.calls == [("nack", 7, False)])

w, ch, _ = run(ok_job, json.dumps(["not", "an", "object"]).encode())
check("消息体不是对象 → 丢弃", ch.calls == [("nack", 7, False)])

# ---------- 3. 租约校验 ----------
FULL = {"job_id": "j1", "attempt": 1, "input_url": "u", "expires_at": "e", "lease_until": "l",
        "heartbeat_interval_seconds": 300, "input_sha256": "a" * 64, "input_size_bytes": 10,
        "content_type": "video/mp4", "model_sha256": "m", "sampling_fps": 1, "profile_sha256": "p"}
payload = {"job_id": "j1", "attempt": 1, "input_size_bytes": 10, "sampling_fps": 1}
try:
    validate_lease_grant(FakeWorker(ok_job), payload, dict(FULL))
    check("完整租约 → 通过", True)
except Exception as exc:  # noqa: BLE001
    check("完整租约 → 通过", False, repr(exc))

try:
    validate_lease_grant(FakeWorker(ok_job), payload, {k: v for k, v in FULL.items() if k != "expires_at"})
    check("缺字段 → 报错", False)
except JobFailure as exc:
    check("缺字段 → 报错并点名", "expires_at" in str(exc) and exc.retryable is False)

try:
    validate_lease_grant(FakeWorker(ok_job), payload, {**FULL, "input_size_bytes": 999})
    check("大小不一致 → 报错并给出两侧取值", False)
except JobFailure as exc:
    check("大小不一致 → 报错并给出两侧取值", "999" in str(exc) and "10" in str(exc))

try:
    validate_lease_grant(FakeWorker(ok_job), payload, {**FULL, "sampling_fps": 2})
    check("信封与租约的采样率不一致 → 报错", False)
except JobFailure as exc:
    check("信封与租约的采样率不一致 → 点名两侧取值",
          "sampling_fps" in str(exc) and "2" in str(exc) and exc.retryable is False)

payload_nofps = {k: v for k, v in payload.items() if k != "sampling_fps"}
try:
    validate_lease_grant(FakeWorker(ok_job), payload_nofps, {**FULL, "sampling_fps": 2})
    check("租约与本机配置的采样率不一致 → 报错", False)
except JobFailure as exc:
    check("租约与本机配置的采样率不一致 → 报错", "采样率" in str(exc) and exc.retryable is False)

# ---------- 4. 契约常量与租约止损 ----------
check("lease_token 合法值通过", bool(LEASE_TOKEN_PATTERN.fullmatch("A" * 43)))
check("lease_token 过短被拒", LEASE_TOKEN_PATTERN.fullmatch("A" * 42) is None)
check("lease_token 含非法字符被拒", LEASE_TOKEN_PATTERN.fullmatch("A" * 42 + "!") is None)
check("lease_token 超长被拒", LEASE_TOKEN_PATTERN.fullmatch("A" * 129) is None)

check("ISO 时间戳 Z 后缀可解析", ns["_parse_instant"]("2026-09-11T12:00:00Z") is not None)
check("空/非法时间戳返回 None",
      ns["_parse_instant"]("") is None and ns["_parse_instant"]("nope") is None)

hb = LeaseHeartbeat("http://api", "tok", "w", "j1", 1, "L" * 43, 30,
                    lease_until="2026-09-11T12:00:00Z")
try:
    hb.raise_if_lost("窗口 1")
    check("租约未失效 → 不中止", True)
except JobFailure:  # noqa: BLE001
    check("租约未失效 → 不中止", False)

hb._mark_lost("心跳返回 409")
try:
    hb.raise_if_lost("窗口 2")
    check("租约失效 → 中止推理", False)
except JobFailure as exc:
    check("租约失效 → 抛 JobFailure 且不可重投",
          "租约已失效" in str(exc) and exc.retryable is False)
check("租约失效原因被完整保留", "409" in hb.lost_reason)

# ---------- 5. 小工具 ----------
tmp = os.path.join(tempfile.gettempdir(), "_aw_selftest_del.txt")
pathlib.Path(tmp).write_text("x", encoding="utf-8")
ns["_remove_quietly"](tmp)
check("_remove_quietly 删掉已存在文件", not os.path.exists(tmp))
ns["_remove_quietly"](tmp)
ns["_remove_quietly"](None)
check("_remove_quietly 对缺失/None 不抛异常", True)


class FakeResp:
    status_code = 403

    def read(self):
        return b'<Error><Code>SignatureDoesNotMatch</Code></Error>'


excerpt = ns["_response_excerpt"](FakeResp())
check("_response_excerpt 截取响应体", "SignatureDoesNotMatch" in excerpt)
check("403 下载提示点到签名与 host", "host" in ns["_download_error_hint"](403))
check("local Worker 默认身份与平台部署模板一致",
      ns["LOCAL_DEFAULT_WORKER_ID"] == "ai-worker-local-1")
check("租约 403 提示点到 worker_id 与 ai-worker-local-1",
      "worker_id" in ns["_lease_error_hint"](403, "ai-worker-local-1")
      and "ai-worker-local-1" in ns["_lease_error_hint"](403, "ai-worker-local-1"))
check("租约 401 提示点到 32 字节", "32" in ns["_lease_error_hint"](401, "w"))

# ---------- 6. 配药动作词表：只保留可由视频证明的实体操作 ----------
active_labels = {
    "acquire_item", "open_container", "disinfect_access", "insert_needle", "draw_liquid", "transfer_liquid",
    "agitate_solution", "discard_sharps", "label_prepared_medication"
}
ignored_labels = {"retrieve_medicine", "check_info"}
worker_default_labels = ns.get("DEFAULT_LABEL_DEFINITIONS", {})
worker_keys_lower = {k.lower().strip() for k in worker_default_labels.keys()}
worker_ignored_labels = {str(k).lower().strip() for k in ns.get("IGNORED_LABELS", ())}

check(
    "Worker 默认标签集只保留实体配药动作",
    worker_keys_lower == active_labels and not (worker_keys_lower & ignored_labels),
    f"unexpected: {worker_keys_lower ^ active_labels}"
)
check("Worker 默认标签 ID 使用统一小写形式", set(worker_default_labels.keys()) == active_labels)
check("旧泛化标签被明确列入忽略集合", worker_ignored_labels == ignored_labels)

all_have_criteria = all(
    isinstance(info, dict) and bool(info.get("name")) and bool(info.get("desc"))
    for info in worker_default_labels.values()
)
check("Worker 默认标签定义均包含有效的 name 与 desc 判据", all_have_criteria and len(worker_default_labels) == 9)

failed = [r for r in results if not r[1]]
print(f"\n共 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
if failed:
    for name, _, extra in failed:
        print(f"  FAIL: {name} {extra}")
    sys.exit(1)
print("全部通过")
