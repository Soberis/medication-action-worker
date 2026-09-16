"""Action Candidate Worker Daemon for Qwen3.5/Qwen3-VL on RTX 4090.

1. Model resident in VRAM (zero cold start).
2. RabbitMQ consumer with prefetch_count=1 (strict serial execution, no concurrent OOM).
3. Frame-accurate chunked inference with cross-window continuation stitching.
4. Result submission to Core API.

生产任务在推理期间由独立线程续期租约；媒体下载完成后还会按租约声明校验字节数和 SHA-256。
"""

# ruff: noqa: E402
# The CUDA allocator environment must be configured before importing torch.
from __future__ import annotations

import importlib.util
import os


def _load_dotenv() -> None:
    """自动加载 .env 文件，保证无论直接 `python action_worker.py` 还是通过脚本启动都能读到配置。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join(os.getcwd(), ".env"), os.path.join(script_dir, ".env")):
            if os.path.isfile(p):
                load_dotenv(p, override=False)
                return
    except ImportError:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for env_path in (os.path.join(os.getcwd(), ".env"), os.path.join(script_dir, ".env")):
        if not os.path.isfile(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
        break


_load_dotenv()

# Worker 身份不该由部署者手填。平台在 local/test 下用一个固定的 AI_WORKER_ID
# （默认 ai-worker-local-1）合成身份，生产下身份就是 JWT 的 sub —— 两种模式下这个值
# 都不是"部署配置"，所以在未显式指定时自动推导：能读到令牌的 sub 就用 sub，
# 否则回落到平台的本地默认值。
LOCAL_DEFAULT_WORKER_ID = "ai-worker-local-1"

# 平台契约里 model_name 是「人类可读的模型身份」（用于 provenance），不是摘要。
# 正常从本地模型目录名推导；目录名取不到时回落到平台侧的模型名常量。
ACTION_MODEL_FALLBACK_NAME = "qwen3.5-4b"


def _jwt_subject(token: str) -> str:
    """从未经验证的 JWT 载荷读取 sub，仅用于自报身份；校验始终在平台侧完成。"""
    parts = (token or "").split(".")
    if len(parts) != 3 or not parts[1]:
        return ""
    try:
        import base64
        import json

        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        return ""
    subject = payload.get("sub") if isinstance(payload, dict) else None
    return subject.strip() if isinstance(subject, str) else ""


# 必须在 import torch 之前设置：CUDA 缓存分配器在首次分配时就固化参数，
# 放到 import torch 之后（哪怕只是 import 之后、还没用到 CUDA）也不保险。
# 用 setdefault 保证 run.sh / .env 里的显式配置优先。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import threading
import time
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc
from pathlib import Path
from typing import Any, Callable

import httpx
from label_schema import (
    ACTION_LABEL_SCHEMA_VERSION,
    ALL_LABEL_DEFINITIONS,
    DEFAULT_LABEL_DEFINITIONS,
    IGNORED_LABELS,
    MAX_ACQUIRE_DURATION_SEC,
    RECOGNIZED_ITEM_TYPES,
)
import pika
import torch
from transformers import AutoConfig, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("action_worker")


def _resolve_model_class(model_path: str) -> tuple[type[Any], str]:
    """根据本地模型 config 选择对应的多模态生成类。

    Qwen3.5 与 Qwen3-VL 都接受视频输入，但不是同一个模型架构；不能因为
    当前环境里能导入 Qwen3VLForConditionalGeneration，就把所有模型都交给它。
    这里按 config.model_type 路由，并把 Qwen3.5 的导入延迟到真正使用该模型时，
    以保证旧的 Python/Transformers 环境仍然可以运行 Qwen3-VL baseline。
    """
    config = AutoConfig.from_pretrained(model_path)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = getattr(config, "architectures", None) or []
    architecture_names = {str(name).lower() for name in architectures}

    if model_type == "qwen3_5" or "qwen3_5forconditionalgeneration" in architecture_names:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "当前 Transformers 不支持 Qwen3.5；请使用 Python 3.10+ 与包含 "
                "Qwen3_5ForConditionalGeneration 的版本。"
            ) from exc
        return Qwen3_5ForConditionalGeneration, "qwen3_5"

    if model_type == "qwen3_vl" or "qwen3vlforconditionalgeneration" in architecture_names:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration, "qwen3_vl"
        except ImportError:
            # 兼容只提供 AutoModelForImageTextToText 的 Transformers 版本。
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText, "qwen3_vl"

    from transformers import AutoModelForImageTextToText
    logger.warning(
        "未识别模型类型 %r，将使用 AutoModelForImageTextToText；architectures=%s",
        model_type or "<empty>", architectures,
    )
    return AutoModelForImageTextToText, model_type or "auto"

def _canonical_label_key(value: Any) -> str:
    """将平台或旧消息中的标签 ID 收敛为统一的小写 snake_case 键。"""
    return str(value or "").strip().lower()

CORE_CHUNK_SEC = 24.0
CONTEXT_PAD_SEC = 3.0

# 断线重连退避：避免网络不通时以固定 5s 频率刷屏（截图里就是这个现象）
RECONNECT_BACKOFF_MIN_SEC = 5.0
RECONNECT_BACKOFF_MAX_SEC = 60.0

# Core API 契约对 lease_token 的约束（contracts/events/ai-action-job-v1.schema.json）
LEASE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43,128}")


def _env_int(name: str, default: int) -> int:
    """读取整型环境变量，空值/非法值一律回落默认值，避免 import 期崩溃。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法整数，使用默认值 %s。", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法数字，使用默认值 %s。", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("环境变量 %s=%r 不是合法布尔值，使用默认值 %s。", name, raw, default)
    return default


# BlockingConnection 在回调里跑 ffmpeg + 多窗口推理时会阻塞数分钟，
# 期间无法 pump 事件循环，因此默认关闭 AMQP 心跳（由 TCP keepalive 兜底）。
DEFAULT_HEARTBEAT_SEC = _env_int("RABBITMQ_HEARTBEAT", 0)
DEFAULT_BLOCKED_TIMEOUT_SEC = _env_float("RABBITMQ_BLOCKED_TIMEOUT", 300.0)

# httpx 默认 trust_env=True，会读取 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY。
# 本 worker 只访问内网 Core API（如 192.168.1.7:8080），走代理必然是错的，
# 且症状是「拿到代理自己的 5xx」而不是连接失败，非常难查。默认不信任环境代理。
DEFAULT_TRUST_ENV = _env_bool("PLATFORM_TRUST_ENV", False)

# 单个窗口通常只需要输出短 JSON 数组；密集窗口可能超过 512/1024 tokens，
# 默认给 1024。若发生疑似截断，推理层会自动重试更高上限（最高 2048）。
RETRY_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 1024)


def _token_limits(initial: int) -> list[int]:
    """返回有限的生成上限序列：若发生疑似截断，尝试阶梯放大上限重试，最高到 RETRY_MAX_NEW_TOKENS。"""
    limits = [initial]
    if initial < RETRY_MAX_NEW_TOKENS:
        step = 1024 if initial < 1024 else RETRY_MAX_NEW_TOKENS
        limits.append(step)
        if step < RETRY_MAX_NEW_TOKENS:
            limits.append(RETRY_MAX_NEW_TOKENS)
    dedup: list[int] = []
    for x in limits:
        if x not in dedup:
            dedup.append(x)
    return dedup


# Qwen3.5 的 Hub kernels 是实验性 opt-in：它们可能需要 trust_remote_code，
# 且必须和当前 Transformers/PyTorch 组合一起回归。默认关闭，避免升级依赖时
# 让一个原本可用的 worker 在启动阶段突然失败。
DEFAULT_USE_HUB_KERNELS = _env_bool("QWEN_USE_HUB_KERNELS", False)

# 每帧视觉 token 的像素预算。保留现有基线，后续可在独立 benchmark 中调整。
PIXELS_PER_UNIT = 28 * 28
MIN_PIXELS = 128 * PIXELS_PER_UNIT
MAX_PIXELS = 256 * PIXELS_PER_UNIT

# 显存空闲自动卸载超时秒数 (弹性显存引擎):
#   -1: 保持原版常驻 17GB 显存 (永不卸载)
#    0: 极严格即用即卸模式 (单个任务处理完立即释放)
#  >0: 弹性按需保活 (平时 0 MB；首个任务动态唤醒；空闲 N 秒无新任务自动归还显存；默认 300s = 5 分钟)
DEFAULT_IDLE_OFFLOAD_SEC = _env_int("IDLE_OFFLOAD_SECONDS", 300)
if os.getenv("LAZY_LOAD") == "0" and "IDLE_OFFLOAD_SECONDS" not in os.environ:
    DEFAULT_IDLE_OFFLOAD_SEC = -1



def _warn_on_env_proxy() -> None:
    """启动时提示：环境里设了代理变量，但内网调用不会走代理。"""
    proxy_vars = [v for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
                  if os.getenv(v)]
    if proxy_vars and not DEFAULT_TRUST_ENV:
        logger.warning(
            "检测到环境变量 %s 已设置，但 PLATFORM_TRUST_ENV=0（默认），"
            "worker 访问 Core API 时不会走这些代理。如需走代理请设 PLATFORM_TRUST_ENV=1。",
            "/".join(proxy_vars),
        )

# LLM 常见的数组/对象末尾多余逗号
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> list[str]:
    """按括号配对切出所有最外层完整片段，正确跳过字符串字面量与转义。

    不能用 re.search(r"\\[.*\\]")：贪婪匹配会把 "结果 [1,2] 和 [{...}]" 整段吞下，
    再因为中间夹着非 JSON 文本而解析失败，于是一条真实动作被静默丢弃。
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


def _loads_lenient(snippet: str) -> Any:
    """json.loads 的宽松版：额外容忍末尾多余逗号。失败返回 None。"""
    try:
        return json.loads(snippet)
    except Exception:
        pass
    repaired = _TRAILING_COMMA_RE.sub(r"\1", snippet)
    if repaired != snippet:
        try:
            return json.loads(repaired)
        except Exception:
            pass
    return None


def _output_needs_token_retry(text: str) -> bool:
    """识别空输出或未闭合 JSON 容器，供一次性扩大输出上限。"""
    cleaned = text.strip()
    if not cleaned:
        return True
    if "[" in cleaned and not _scan_balanced(cleaned, "[", "]"):
        return True
    if "{" in cleaned and not _scan_balanced(cleaned, "{", "}"):
        return True
    return False


def _is_explicit_empty_output(text: str) -> bool:
    """仅接受完整的 []（或完整 wrapper）作为无动作结果。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = _loads_lenient(cleaned)
    if parsed == []:
        return True
    return isinstance(parsed, dict) and any(value == [] for value in parsed.values())


def _scratch_dir() -> str:
    """临时文件落盘目录：优先 Linux 共享内存（/dev/shm），避免打满磁盘 IO；不可用则回落当前目录。

    临时切片与下载的视频走同一处，避免两套「优先 shm」逻辑各写一遍。
    """
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return "."


def _safe_job_id(job_id: Any) -> str:
    """把上游 job_id 收敛成可安全拼进文件名的形式。

    job_id 来自消息体，不可信。含 `/` 或 `..` 时直接拼进路径会把文件写到临时目录之外。
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(job_id)).strip("._")[:80]
    return cleaned or "job"


def _remove_quietly(path: str | None) -> None:
    """删除临时文件。删除失败不能盖掉真正要报的那个错。"""
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _response_excerpt(resp: Any, limit: int = 300) -> str:
    """尽力取一小段响应体用于日志。取不到就退回状态码，绝不因为写日志再抛一个异常。"""
    try:
        body = resp.read()
    except Exception:
        return f"HTTP {getattr(resp, 'status_code', '?')}"
    text = bytes(body)[:limit].decode("utf-8", "replace").strip()
    return text or f"HTTP {getattr(resp, 'status_code', '?')}"


def _download_error_hint(status: int) -> str:
    """对象存储拒绝下载时的排查方向。注意不要把预签名 URL 本身写进日志。"""
    if status in (401, 403):
        return ("对象存储拒绝了该预签名 URL：签名绑定 Host，客户端侧任何改写 URL host 的操作"
                "都会破坏 SigV4 签名；也可能是授权已过期（默认 15 分钟）。"
                "正解是让平台的 S3_ENDPOINT 与 worker 用同一个可达地址，然后重新申请租约。")
    if status == 404:
        return "对象存储里找不到该对象：媒体可能已被删除，平台侧的删除应当同时取消该作业。"
    return ""


def _parse_instant(raw: Any) -> datetime | None:
    """解析 Core API 返回的 ISO-8601 时间戳；无法解析时返回 None（调用方按「未提供死线」处理）。"""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("无法解析时间戳 %r，按「未提供」处理。", text)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# 同标签只在相交或首尾相接时合并；存在 1 秒空档也可能是两个独立动作。
CONTINUATION_GAP_SEC = 0


def _can_continue(acc: dict[str, Any], seg: dict[str, Any]) -> bool:
    """seg 能否作为 acc 的同标签续段（同标签且时间相邻或相交）。"""
    if acc["label"] != seg["label"]:
        return False
    if seg["start_sec"] > acc["end_sec"] + CONTINUATION_GAP_SEC:
        return False
    # 针对瞬态的 acquire_item 施加最大合并时长上限，防止将长段持物或空镜退化误报无脑拼接成几十秒巨段
    if acc["label"] == "acquire_item":
        max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
        if max(acc["end_sec"], seg["end_sec"]) - acc["start_sec"] > max_dur:
            return False
        if acc.get("item") and seg.get("item") and acc["item"] != seg["item"]:
            return False
    return True


def _merge_same_label(acc: dict[str, Any], seg: dict[str, Any]) -> bool:
    """同标签且相邻时把 seg 并入 acc，返回是否发生合并。

    这是**唯一**的「跨窗续段」合并入口：窗口内续接与全局收尾都调用它。
    判据只写在 `_can_continue` 一处，避免同一规则在两处各写一遍、日后悄悄漂移。
    """
    if not _can_continue(acc, seg):
        return False
    acc["end_sec"] = max(acc["end_sec"], seg["end_sec"])
    acc["confidence"] = round(max(acc["confidence"], seg["confidence"]), 2)
    if not acc.get("item") and seg.get("item"):
        acc["item"] = seg["item"]
    windows = acc.setdefault("stitched_windows", [acc["source_window"]])
    for w in seg.get("stitched_windows", []):
        if w not in windows:
            windows.append(w)
    return True


class JobFailure(RuntimeError):
    """当前这条消息无法完成。覆盖 Core API 的响应错误，也覆盖本地的契约校验失败。

    必须区分「重投就能好」和「重投一万次也一样」：
      - 不区分 → 永久性错误会白烧一次完整的 GPU 推理；
      - 把 Core API 的 401/403 当普通失败丢掉更糟 → 平台只回收「已认领且租约过期」的尝试，
        从未被认领的 QUEUED 作业不会自愈，消息一丢这个作业就永久卡住。

    分档（见 `_handle_message`）：
      - Core API 认证层 401/403 → 重投无用，但绝不能丢消息，等配置修好自动恢复；
      - 其余 400/404/409 与本地契约校验 → 丢弃即可，平台会按新 attempt 重新派发；
      - 5xx / 429 / 网络抖动 → 重投一次。
    """

    def __init__(self, operation: str, detail: str, *, status: int | None = None, hint: str = "",
                 retryable: bool | None = None, core_api: bool = False) -> None:
        head = f"{operation} 失败" + (f" (HTTP {status})" if status is not None else "")
        message = f"{head}: {detail}" if detail else head
        if hint:
            message = f"{message}\n    → {hint}"
        super().__init__(message)
        self.operation = operation
        self.detail = detail
        self.status = status
        self.hint = hint
        self.core_api = core_api
        if retryable is None:
            retryable = status is not None and (status >= 500 or status == 429)
        self.retryable = retryable

    @property
    def is_credential_error(self) -> bool:
        """只有「Core API 认证层」的失败才算凭证问题。

        对象存储拒绝下载时同样返回 403（预签名 URL 过期 / SigV4 host 不匹配），
        但语义完全不同 —— 那是「这个 URL 不能用了」，改配置也不会让旧 URL 复活，
        重投更不会自愈。若把它也判成凭证问题，`_handle_message` 就会既不 ACK 也不丢弃，
        这条消息被 broker 无限重投，worker 从此卡死在它上面。
        """
        return self.core_api and self.status in (401, 403)


def _lease_error_hint(status: int, worker_id: str) -> str:
    """把 /lease 的状态码翻译成可以直接照做的排查动作。"""
    if status == 401:
        return ("认证没通过：Token 与 Core API 不匹配、长度不足 32 字节，或 Token 文件内容为空。"
                "平台 local/test 模式下 Worker 端点用的是 AI_WORKER_BEARER_TOKEN(_FILE)，"
                "而不是项目级的合成 Token。")
    if status == 403:
        return (f"认证已通过，但请求体自报的 worker_id 与平台认定的身份不相等。"
                f"本进程自报 worker_id={worker_id!r}，请确认它与平台侧 AI_WORKER_ID 一致"
                f"（local 默认 ai-worker-local-1）。身份默认自动推导，不要手写一个自造的名字。")
    if status == 404:
        return "作业不存在：该消息可能已过期，或 worker 连到了另一套数据库的 Core API。"
    if status == 409:
        return ("租约冲突 / 尝试已过期：同一 attempt 的租约属于其它 Worker 且仍有效，"
                "或该尝试已被平台回收。本条消息丢弃即可，平台会按新的 attempt 重新派发。")
    if status == 400:
        return "请求不符合平台契约（字段缺失 / 长度 / 格式），先确认 worker 与平台的版本一致。"
    return ""


def _result_error_hint(status: int) -> str:
    """把 action-result 的状态码翻译成排查动作。"""
    if status == 401:
        return "认证没通过：检查 Token 是否与 Core API 匹配、是否已经过期。"
    if status == 403:
        return ("认证已通过但身份不匹配：该 attempt 的租约属于另一个 Worker，"
                "或本进程自报的 worker_id 与平台认定值不一致。")
    if status == 400:
        return ("结果体不符合契约：最常见是标签不在平台 allowed_labels 内，"
                "或 start_ms/end_ms 越界、重叠。平台开了 fail-on-unknown-properties，"
                "多带一个字段也会直接 400。")
    if status == 409:
        return "租约已过期、作业已不在 RUNNING，或该 message_id 已被用过但内容不同。"
    if status == 404:
        return "作业或尝试不存在：这条消息已经过期。"
    return ""


class LeaseHeartbeat:
    """在后台线程中定期发送租约保活心跳，防止长时间任务导致 Core API 租约过期。

    心跳失败不能只打一条日志就算完：下载 24GB 媒体、首次装载 17GB 模型、跑完多个推理
    窗口，任何一段都可能长达几十分钟。一旦租约已被平台回收（404/409）或 Core API 已不可达
    且越过了 lease_until，再跑下去必然是白烧一次完整推理、并且回传一定失败。
    所以这里把「租约确定失效」暴露成 lost_event，由主流程在窗口之间检查并提前中止。
    """

    def __init__(
        self,
        api_base_url: str,
        bearer_token: str | Callable[[], str],
        worker_id: str,
        job_id: str,
        attempt: int,
        lease_token: str,
        interval_sec: float,
        lease_until: str = "",
        trust_env: bool = False,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.worker_id = worker_id
        self.job_id = job_id
        self.attempt = attempt
        self.lease_token = lease_token
        self.interval_sec = max(5.0, interval_sec) if interval_sec > 0 else 30.0
        self.lease_until = _parse_instant(lease_until)
        self.trust_env = trust_env
        self.lost_event = threading.Event()
        self.lost_reason = ""
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"lease-heartbeat-{_safe_job_id(job_id)}",
        )

    def __enter__(self) -> LeaseHeartbeat:
        if self.lease_token and self.api_base_url:
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def raise_if_lost(self, where: str) -> None:
        """由主流程在耗时步骤之间调用：租约已失效就没必要继续把 GPU 时间花下去。"""
        if self.lost_event.is_set():
            raise JobFailure("租约保活", f"{where}时租约已失效: {self.lost_reason}")

    def _mark_lost(self, reason: str) -> None:
        if self.lost_event.is_set():
            return
        self.lost_reason = reason
        self.lost_event.set()
        logger.error("[%s] 租约已失效（%s）。将在当前步骤结束后中止，不再继续消耗 GPU。",
                     self.worker_id, reason)

    def _check_deadline(self, reason: str) -> None:
        """Core API 抖一下不该立刻放弃；只有越过 lease_until 这条死线才认定失效。"""
        logger.warning("[%s] 租约心跳失败: %s", self.worker_id, reason)
        if self.lease_until is not None and datetime.now(UTC) >= self.lease_until:
            self._mark_lost(f"{reason}，且已越过 lease_until={self.lease_until.isoformat()}")

    def _run(self) -> None:
        base = self.api_base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{self.job_id}/attempts/{self.attempt}/heartbeat"
        body = {"worker_id": self.worker_id}

        while not self._stop_event.wait(self.interval_sec):
            try:
                token = self.bearer_token() if callable(self.bearer_token) else self.bearer_token
                headers = {
                    "Content-Type": "application/json",
                    "X-AI-Lease-Token": self.lease_token,
                }
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                with httpx.Client(timeout=10.0, trust_env=self.trust_env) as client:
                    resp = client.post(url, json=body, headers=headers)
                    status_code = resp.status_code
                    excerpt = resp.text[:200]
            except Exception as exc:
                self._check_deadline(f"心跳请求异常: {exc}")
                continue

            if status_code == 200:
                logger.debug("[%s] 租约心跳成功 (job: %s)", self.worker_id, self.job_id)
            elif status_code in (404, 409):
                # 明确失效：作业/尝试已不存在，或租约已被回收、被其它 Worker 接管。
                self._mark_lost(f"心跳返回 {status_code}: {excerpt}")
                return
            elif status_code in (401, 403):
                # 凭证本地修不了，但结果也一样回传不上去。留给提交时的凭证分支处理
                # （那里不 ACK 不丢弃，等配置修好后自动恢复），这里只留痕、不算失效。
                logger.warning("[%s] 租约心跳认证失败 status=%s: %s", self.worker_id, status_code, excerpt)
            else:
                self._check_deadline(f"心跳返回 {status_code}: {excerpt}")


class ActionWorker:
    def __init__(
        self,
        model_path: str,
        api_base_url: str,
        rabbitmq_url: str,
        queue_name: str = "platform.ai.action-candidate.v1",
        worker_id: str = "",
        bearer_token: str = "",
        bearer_token_file: str = "",
        sampling_fps: float = 1.0,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        blocked_connection_timeout: float = DEFAULT_BLOCKED_TIMEOUT_SEC,
        trust_env: bool = DEFAULT_TRUST_ENV,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        idle_offload_sec: int = DEFAULT_IDLE_OFFLOAD_SEC,
        use_hub_kernels: bool = DEFAULT_USE_HUB_KERNELS,
    ) -> None:
        self.model_path = model_path
        self.api_base_url = api_base_url.rstrip("/")
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.bearer_token = (bearer_token or os.getenv("PLATFORM_WORKER_BEARER_TOKEN")
                             or os.getenv("AI_WORKER_BEARER_TOKEN") or "").strip()
        self.bearer_token_file = (bearer_token_file or os.getenv("PLATFORM_WORKER_BEARER_TOKEN_FILE")
                                  or os.getenv("AI_WORKER_BEARER_TOKEN_FILE") or "").strip()
        if self.bearer_token and self.bearer_token_file:
            raise ValueError("bearer_token and bearer_token_file are mutually exclusive")
        if not self.bearer_token and not self.bearer_token_file:
            raise ValueError("Core API Bearer Token is required; configure --bearer-token or --bearer-token-file")
        self._short_token_warned = False
        # 回传给平台的模型身份：取本地模型目录名（如 Qwen3-VL-8B-Instruct）。
        # 之前这里填的是 payload 里的 model_sha256，把 64 位摘要当成了模型名。
        self.model_name = os.path.basename(str(model_path).rstrip("/\\")) or ACTION_MODEL_FALLBACK_NAME
        self.worker_id, worker_id_source = self._resolve_worker_id(worker_id)
        logger.info("[%s] Core API 鉴权就绪 (source=%s, worker_id_source=%s)", self.worker_id,
                    "file" if self.bearer_token_file else "inline", worker_id_source)
        self.heartbeat_sec = heartbeat_sec
        self.blocked_connection_timeout = blocked_connection_timeout
        self.trust_env = trust_env
        self.idle_offload_sec = idle_offload_sec
        self.use_hub_kernels = bool(use_hub_kernels)
        self.last_active_time = time.time()
        self.model: Any = None
        if max_new_tokens < 32:
            # 设得过小会让每个窗口都输出空结果，而且现场看不出是配置问题
            logger.warning("max_new_tokens=%s 过小，已抬到 32。", max_new_tokens)
            max_new_tokens = 32
        self.max_new_tokens = max_new_tokens
        if sampling_fps <= 0:
            # 采样率 0 等于让模型看空视频，只会稳定输出 []，现场同样看不出是配置问题
            logger.warning("sampling_fps=%s 非法（须 > 0），已回退 1.0。", sampling_fps)
            sampling_fps = 1.0
        elif sampling_fps != int(sampling_fps):
            # 回传契约里 sampling_fps 是整型，非整数会被静默截断，先提醒
            logger.warning("sampling_fps=%s 非整数，回传时会被截断为 %s。", sampling_fps, int(sampling_fps))
        self.sampling_fps = sampling_fps
        self._reconnect_backoff = RECONNECT_BACKOFF_MIN_SEC

        logger.info("Initializing Action Worker with processor for model: %s", model_path)
        self.model_class, self.model_family = _resolve_model_class(model_path)
        logger.info("模型架构路由: model_family=%s, loader=%s", self.model_family, self.model_class.__name__)
        # min/max_pixels 是 processor 的静态默认预算；逐帧 cap 在真正调用
        # apply_chat_template 时通过 processor_kwargs 传入（见 _infer_single_window）。
        # 不把调用期参数塞进 from_pretrained，兼容旧 Qwen3-VL processor。
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        # 动态探测与自适应注意力内核 (FlashAttention-2 / SDPA)
        attn_impl = "sdpa"
        try:
            flash_attn_available = importlib.util.find_spec("flash_attn") is not None
        except (ImportError, ValueError):
            flash_attn_available = False
        if flash_attn_available:
            attn_impl = "flash_attention_2"
            logger.info("Detected flash-attn, enabled FlashAttention-2 hardware acceleration.")
        else:
            logger.info("flash-attn not found, using PyTorch native SDPA kernel.")
        self.attn_impl = attn_impl

        # 显存策略分支
        if self.idle_offload_sec < 0:
            logger.info("[%s] 显存策略: 永久常驻模式 (IDLE_OFFLOAD_SECONDS=%s，启动即常驻 VRAM)。", self.worker_id, self.idle_offload_sec)
            self._ensure_model_loaded()
        elif self.idle_offload_sec == 0:
            logger.info("[%s] 显存策略: 极严格即用即卸模式 (平时显存 0 MB，单任务处理完立即彻底释放)。", self.worker_id)
        else:
            logger.info(
                "[%s] 显存策略: 弹性显存引擎已就绪 (平时显存 0 MB，首发动态唤醒，空闲 %ss 自动卸载)。",
                self.worker_id, self.idle_offload_sec,
            )

    def _ensure_model_loaded(self) -> None:
        """确保模型已装载入 GPU 显存。若已加载则直接复用（0 延迟），未加载则动态唤醒。"""
        if self.model is not None:
            return
        t0 = time.perf_counter()
        logger.info("[%s] 正在唤醒加载模型至 GPU 显存 (BF16, attn=%s)...", self.worker_id, self.attn_impl)
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": self.attn_impl,
            "device_map": "cuda",
        }
        if self.model_family == "qwen3_5" and self.use_hub_kernels:
            # Transformers 官方 Qwen3.5 文档要求显式 opt-in；当前 Hub kernel
            # 可能需要 remote-code 信任，只有部署者主动打开时才启用。
            model_kwargs.update(use_kernels=True, trust_remote_code=True)
            logger.warning("[%s] 已启用 Qwen3.5 Hub kernels；正在执行实验性加速路径。", self.worker_id)
        self.model = self.model_class.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()
        cost = time.perf_counter() - t0
        logger.info("[%s] 模型唤醒加载完成，耗时: %.2fs，GPU 显存已就绪。", self.worker_id, cost)

    def _offload_model(self) -> None:
        """卸载模型并彻底清空 CUDA 缓存，将 4090 显存完全归还给系统。"""
        if self.model is None:
            return
        logger.info(
            "[%s] 空闲超时 (已持续 %ss 无任务)，正在自动卸载模型并释放 GPU 显存...",
            self.worker_id, self.idle_offload_sec,
        )
        del self.model
        self.model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[%s] GPU 显存已完全释放至 0 MB (待命中，等待新任务唤醒)。", self.worker_id)

    def run(self) -> None:
        while True:
            try:
                self._consume()
            except KeyboardInterrupt:
                logger.info("收到中断信号，Action Worker 已安全退出。")
                return
            except Exception as exc:
                delay = self._reconnect_backoff
                logger.error(
                    "连接或消费循环异常: %s。%.0fs 后重试（退避上限 %.0fs）。",
                    exc, delay, RECONNECT_BACKOFF_MAX_SEC,
                )
                logger.error(
                    "排查建议: nc -zv <broker_host> <port> 测连通性；确认 rabbitmq 监听 0.0.0.0; "
                    "确认防火墙放行该端口; 确认 vhost / 账号密码正确。"
                )
                try:
                    time.sleep(delay)
                except KeyboardInterrupt:
                    logger.info("收到中断信号，Action Worker 已安全退出。")
                    return
                # 只有「连不上」才需要退避；连上过再掉线应尽快重连。
                self._reconnect_backoff = min(delay * 2.0, RECONNECT_BACKOFF_MAX_SEC)

    def _build_connection_params(self) -> pika.URLParameters:
        params = pika.URLParameters(self.rabbitmq_url)
        # 回调内要跑 ffmpeg 切片 + 多窗口 generate()，期间无法 pump 事件循环，
        # 默认 60s 心跳必然被 broker 判定为失联而断开，故默认 heartbeat=0。
        params.heartbeat = self.heartbeat_sec
        params.blocked_connection_timeout = self.blocked_connection_timeout
        return params

    def _consume(self) -> None:
        params = self._build_connection_params()
        connection = pika.BlockingConnection(params)
        # 连接建立成功即重置退避，保证「掉线后立刻重连」而不是等满 60s。
        self._reconnect_backoff = RECONNECT_BACKOFF_MIN_SEC
        channel = None
        try:
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            channel.basic_qos(prefetch_count=1)

            logger.info(
                "[%s] 已连接 %s，等待队列 '%s' 的任务 (prefetch=1, heartbeat=%ss, idle_offload=%ss)...",
                self.worker_id, params.host, self.queue_name, self.heartbeat_sec, self.idle_offload_sec,
            )

            for method, _properties, body in channel.consume(self.queue_name, inactivity_timeout=1.0):
                if method is None:
                    # 队列空闲滴答：检查是否达到空闲卸载阈值
                    if self.idle_offload_sec >= 0 and self.model is not None:
                        if time.time() - self.last_active_time >= self.idle_offload_sec:
                            self._offload_model()
                    continue

                # 收到任务消息
                self._handle_message(channel, method, body)
                self.last_active_time = time.time()
                if self.idle_offload_sec == 0:
                    self._offload_model()
        finally:
            if channel is not None:
                try:
                    channel.cancel()
                except Exception:
                    pass
            try:
                if connection.is_open:
                    connection.close()
            except Exception:
                pass

    def _handle_message(self, channel: Any, method: Any, body: bytes) -> None:
        job_id = "<unknown>"
        message_id = "<unknown>"
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise JobFailure("任务契约校验", "消息体不是 JSON 对象")
            job_id = payload.get("job_id") or job_id
            message_id = payload.get("message_id") or message_id
            logger.info("[%s] 收到任务: %s (message=%s)", self.worker_id, job_id, message_id)
            self.process_job(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("[%s] 消息体不是合法 JSON，丢弃该消息 (message=%s): %s",
                         self.worker_id, message_id, exc)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        except JobFailure as exc:
            if exc.is_credential_error:
                # 这里绝不 ACK 也绝不丢弃：平台只回收「已认领且租约过期」的尝试，
                # 从未被认领的 QUEUED 作业不会自愈 —— 消息一丢，这个作业就永久卡住。
                # 保留未确认状态，连接断开后由 broker 重投，等配置修好即自动恢复。
                logger.error(
                    "[%s] 任务 %s (message=%s) 已中止（Core API 凭证 / 身份问题）:\n%s\n"
                    "    该消息未被确认、也未被丢弃；修好上面的配置后本进程会自动重连并继续处理它。",
                    self.worker_id, job_id, message_id, exc,
                )
                # 唯一需要退避的失败：立刻重连只会再拿一次同样的 401/403，所以按最大退避等。
                self._reconnect_backoff = RECONNECT_BACKOFF_MAX_SEC
                raise
            if exc.retryable and not method.redelivered:
                logger.warning("[%s] 任务 %s (message=%s) 可重试失败，重投一次:\n%s",
                               self.worker_id, job_id, message_id, exc)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                return
            logger.error(
                "[%s] 任务 %s (message=%s) 不可恢复，丢弃该消息（不再重投）:\n%s\n"
                "    平台的租约回收会把这个作业重新派发成新的 attempt，本条消息对应的旧 attempt 已无价值。",
                self.worker_id, job_id, message_id, exc,
            )
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        except Exception as exc:
            logger.error("[%s] 任务 %s (message=%s) 处理失败: %s",
                         self.worker_id, job_id, message_id, exc, exc_info=True)
            if method.redelivered:
                # 已经重投过一次仍然失败 -> 停止 requeue，避免毒药消息无限死循环
                logger.error(
                    "[%s] 任务 %s (message=%s) 重投后仍失败，丢弃该消息；"
                    "平台侧的租约回收会重新派发该作业。",
                    self.worker_id, job_id, message_id,
                )
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            else:
                logger.warning("[%s] 任务 %s (message=%s) 将重投一次 (requeue=True)。",
                               self.worker_id, job_id, message_id)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return

        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("[%s] 任务 %s (message=%s) 处理完成并 ACK。", self.worker_id, job_id, message_id)

    def _read_bearer_token(self) -> str:
        """每次请求都重读文件，使生产环境轮换 JWT 时无需重启本进程。"""
        if self.bearer_token_file:
            path = Path(self.bearer_token_file)
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                # 裸的 FileNotFoundError 在日志里看不出「哪个配置项指向了哪个坏路径」
                raise RuntimeError(
                    f"无法读取 Token 文件 {path}: {exc}。请确认 PLATFORM_WORKER_BEARER_TOKEN_FILE "
                    f"指向的文件存在，且本进程有读权限。"
                ) from exc
        else:
            token = self.bearer_token
        if not token:
            raise RuntimeError("Core API Bearer Token 为空：Token 文件内容为空，或环境变量未配置。")
        if len(token) < 32 and not getattr(self, "_short_token_warned", False):
            self._short_token_warned = True
            logger.warning(
                "Bearer Token 只有 %d 字节；平台 local/test 模式要求 ≥ 32 字节，否则一律返回 401。",
                len(token),
            )
        return token

    def _resolve_worker_id(self, requested: str) -> tuple[str, str]:
        """推导自报身份，优先级：显式配置 > 令牌的 sub > 平台本地默认值。

        平台会用认证主体（生产=JWT sub，local=替身常量）与请求体里的 worker_id
        做严格相等校验，不一致会返回 403。把默认值猜成别的字符串只会让默认部署
        必然 403，因此这里不再假设一个自造的名字。
        """
        explicit = (requested or "").strip()
        if explicit:
            return explicit, "configured"
        subject = _jwt_subject(self._read_bearer_token())
        if subject:
            return subject, "jwt-sub"
        return LOCAL_DEFAULT_WORKER_ID, "local-default"

    def _acquire_lease(self, job_id: str, attempt: int, lease_token: str) -> dict[str, Any]:
        """向 Core API 申请任务租约并获取输入媒体授权 URL。"""
        base = self.api_base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{job_id}/attempts/{attempt}/lease"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": lease_token,
        }
        headers["Authorization"] = f"Bearer {self._read_bearer_token()}"
        body = {"worker_id": self.worker_id}

        logger.info("[%s] 正在向 Core API 申请租约: %s", self.worker_id, url)
        with httpx.Client(timeout=30.0, trust_env=self.trust_env) as client:
            try:
                resp = client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                raise JobFailure("Core API 租约申请", f"网络不可达: {exc}", retryable=True) from exc
            if resp.status_code != 200:
                raise JobFailure(
                    "Core API 租约申请", resp.text[:300].strip(),
                    status=resp.status_code, hint=_lease_error_hint(resp.status_code, self.worker_id),
                    core_api=True,
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise JobFailure("Core API 租约申请", f"响应不是合法 JSON: {exc}") from exc
        if not isinstance(data, dict) or not data.get("input_url"):
            raise JobFailure("Core API 租约申请", "响应缺少 input_url（worker 与平台的契约不一致）")
        return data

    def _validate_lease_grant(self, payload: dict[str, Any], grant: dict[str, Any]) -> None:
        """租约是平台冻结的事实集合，必须与本条消息、与本机配置三方一致。

        这里失败一律按「不可重投」处理：平台冻结的摘要 / 模型 / 采样率不会因为再试
        一次而改变，硬着头皮推理只会把一个已知不一致的结果回传上去。
        """
        required = ("job_id", "attempt", "input_url", "expires_at", "lease_until", "heartbeat_interval_seconds",
                    "input_sha256", "input_size_bytes", "content_type", "model_sha256", "sampling_fps",
                    "profile_sha256")
        missing = [field for field in required if field not in grant]
        if missing:
            raise JobFailure("租约校验", f"租约响应缺少字段: {', '.join(missing)}")
        if str(grant["job_id"]) != str(payload.get("job_id")) or int(grant["attempt"]) != int(payload.get("attempt", 1)):
            raise JobFailure("租约校验", "租约的 job_id / attempt 与本条消息不一致")
        for field in ("input_sha256", "input_size_bytes", "content_type", "model_sha256", "sampling_fps", "profile_sha256"):
            if field in payload and str(payload[field]) != str(grant[field]):
                raise JobFailure(
                    "租约校验",
                    f"字段 {field} 不一致: 消息={payload[field]!r} 租约={grant[field]!r}",
                )
        try:
            size = int(grant["input_size_bytes"])
            heartbeat_interval = int(grant["heartbeat_interval_seconds"])
            grant_fps = float(grant["sampling_fps"])
        except (TypeError, ValueError) as exc:
            raise JobFailure("租约校验", f"租约里的数值字段无法解析: {exc}") from exc
        if size < 1:
            raise JobFailure("租约校验", f"input_size_bytes={grant['input_size_bytes']!r} 必须为正数")
        if heartbeat_interval < 1:
            raise JobFailure("租约校验",
                             f"heartbeat_interval_seconds={grant['heartbeat_interval_seconds']!r} 必须为正数")
        if grant_fps != float(self.sampling_fps):
            raise JobFailure(
                "租约校验",
                f"采样率不一致: 作业冻结 {grant['sampling_fps']}，本机配置 {self.sampling_fps}。"
                f"worker 必须按作业冻结的采样率运行，改 SAMPLING_FPS 前先确认平台侧的取值。",
            )

    def process_job(self, payload: dict[str, Any]) -> None:
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise JobFailure("任务契约校验", "消息缺少必需字段 job_id")
        message_id = str(payload.get("message_id") or "").strip()
        if not message_id:
            # 平台按 message_id 做幂等与身份核对；这里编一个 uuid 只会稳定拿到 409，
            # 先白付一次完整推理。契约要求它必填，缺失即消息不合法，立即失败。
            raise JobFailure("任务契约校验", "消息缺少必需字段 message_id（自造 uuid 必然被平台拒绝）")
        try:
            attempt = int(payload.get("attempt", 1))
        except (TypeError, ValueError) as exc:
            raise JobFailure("任务契约校验", f"attempt 不是整数: {payload.get('attempt')!r}") from exc
        if attempt < 1:
            raise JobFailure("任务契约校验", f"attempt 必须 >= 1，实际 {attempt}")

        lease_token = str(payload.get("lease_token") or "").strip()
        # 契约 pattern ^[A-Za-z0-9_-]{43,128}$。坏凭证必须在这里判掉：带着它去请求只会拿到
        # 401/403，而 401/403 的处置是「不 ACK 不丢弃」，这条消息会被 broker 无限重投。
        if not LEASE_TOKEN_PATTERN.fullmatch(lease_token):
            raise JobFailure(
                "任务契约校验",
                f"lease_token 不符合契约 pattern（长度 {len(lease_token)}，应为 43-128 位 URL-safe 字符）",
            )
        if not self.api_base_url:
            raise JobFailure("任务契约校验", "未配置 Core API 地址，无法申请租约")

        allowed_labels = payload.get("allowed_labels")
        label_schema_payload = payload.get("label_schema")
        has_label_snapshot = (
            (isinstance(payload.get("labels"), list) and bool(payload["labels"]))
            or (isinstance(payload.get("label_definitions"), list) and bool(payload["label_definitions"]))
            or (
                isinstance(label_schema_payload, dict)
                and isinstance(label_schema_payload.get("labels"), list)
                and bool(label_schema_payload["labels"])
            )
        )

        # 输入只能来自租约。ai-action-job.v1 是 additionalProperties:false 且不含任何 URL /
        # 磁盘路径字段，平台永不发送。曾有一条直传 video_url / video_path 的旁路，它会让
        # expected_size 与 expected_sha256 双双为 None —— 摘要校验整条失效，且完全绕过租约。
        lease_grant = self._acquire_lease(job_id, attempt, lease_token)
        self._validate_lease_grant(payload, lease_grant)

        hb_interval = float(lease_grant["heartbeat_interval_seconds"])
        tmp_download = os.path.join(_scratch_dir(), f"_download_{_safe_job_id(job_id)}.mp4")

        # 心跳覆盖「下载 → 加载模型 → 推理 → 回传」全程：24GB 下载与 17GB 模型首次装载
        # 都可能超过租约 TTL，而这两段原先都在心跳范围之外，租约会在静默中过期。
        with LeaseHeartbeat(
            api_base_url=self.api_base_url,
            bearer_token=self._read_bearer_token,
            worker_id=self.worker_id,
            job_id=job_id,
            attempt=attempt,
            lease_token=lease_token,
            interval_sec=hb_interval,
            lease_until=lease_grant.get("lease_until", ""),
            trust_env=self.trust_env,
        ) as heartbeat:
            local_video = self._download_media(lease_grant, tmp_download, heartbeat)
            try:
                try:
                    label_defs = self._resolve_label_definitions(payload)
                except Exception as exc:
                    # 没有显式白名单时可以回退默认标签；显式白名单会在下方
                    # fail-closed，避免把未知标签误当成默认动作。
                    logger.warning("标签定义解析失败，未提供显式白名单时回退默认标签: %s", exc)
                    label_defs = None

                if (isinstance(allowed_labels, list) or has_label_snapshot) and not label_defs:
                    # 任务明确给了白名单或标签快照，但其中没有一个有语义定义的标签。
                    # 继续推理只能产生不可解释的空结果或幻觉标签，直接在唤醒 GPU
                    # 之前终止，并把问题暴露给 platform/任务生产方。
                    raise JobFailure(
                        "标签契约",
                        "任务标签中没有可用的已定义标签；请同时提交 labels/label_definitions/label_schema.labels 的 description",
                    )

                # 模型只在「确定要推理」时唤醒：契约不合法、租约校验不过、下载失败
                # 都不该先吃掉 17GB 显存。
                self._ensure_model_loaded()
                heartbeat.raise_if_lost("加载模型后")

                started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                start_time = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                clean_segments, total_windows, duration_ms = self.infer_video(
                    local_video,
                    label_defs=label_defs,
                    allowed_labels=allowed_labels,
                    heartbeat=heartbeat,
                )
                inference_duration_ms = max(0, int((time.time() - start_time) * 1000))
                completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

                # 严格对齐 Core API 的 ActionCandidateResultRequest
                req_body = {
                    "message_id": message_id,
                    "model_name": self.model_name,
                    "prompt_version": ACTION_LABEL_SCHEMA_VERSION,
                    "total_windows": total_windows,
                    "duration_ms": duration_ms,
                    "inference_duration_ms": inference_duration_ms,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "segments": clean_segments,
                }

                heartbeat.raise_if_lost("提交结果前")
                self._submit_result(job_id, attempt, lease_token, req_body)
                logger.info(
                    "[%s] 任务 %s 结果回传成功 (windows=%d, duration=%dms, inference=%dms, segments=%d)",
                    self.worker_id, job_id, total_windows, duration_ms, inference_duration_ms,
                    len(clean_segments),
                )
            finally:
                _remove_quietly(tmp_download)
                # 释放推理过程中申请的瞬时动态显存与碎片
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _download_media(self, lease_grant: dict[str, Any], target: str,
                        heartbeat: LeaseHeartbeat) -> str:
        """按租约下载媒体并逐字节校验摘要；任何不一致都在推理之前中止。

        边下边算 sha256 并设字节上限：既不完全信任 Content-Length，也不会为了一个
        已经不可能通过校验的文件把磁盘写满。
        """
        input_url = lease_grant["input_url"]
        expected_size = int(lease_grant["input_size_bytes"])
        expected_sha256 = str(lease_grant["input_sha256"])

        logger.info("[%s] 正在下载租约媒体至 %s (size=%s, sha256=%s...)...",
                    self.worker_id, target, expected_size, expected_sha256[:12])
        digest = hashlib.sha256()
        downloaded = 0
        try:
            with httpx.stream("GET", input_url, timeout=120.0, trust_env=self.trust_env) as resp:
                if resp.status_code != 200:
                    # 这里的 403 来自对象存储（预签名 URL 过期 / SigV4 host 不匹配），
                    # 不是 Core API 的认证失败 —— 不传 core_api=True：丢弃这条消息、
                    # 让平台派新 attempt 换一个新 URL 才是正确处置。
                    raise JobFailure(
                        "媒体下载", _response_excerpt(resp), status=resp.status_code,
                        hint=_download_error_hint(resp.status_code),
                    )
                with open(target, "wb") as f:
                    for chunk in resp.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise JobFailure(
                                "媒体下载",
                                f"实际字节数已超过租约声明的 {expected_size}，提前中止以免占满磁盘",
                            )
                        digest.update(chunk)
                        f.write(chunk)
            if downloaded != expected_size:
                raise JobFailure("媒体下载", f"大小与租约不符: 期望 {expected_size}，实际 {downloaded}",
                                 retryable=True)
            if digest.hexdigest() != expected_sha256:
                raise JobFailure(
                    "媒体下载",
                    f"内容 sha256 与租约不符: 期望 {expected_sha256[:12]}… 实际 {digest.hexdigest()[:12]}…",
                    retryable=True,
                )
        except JobFailure:
            _remove_quietly(target)
            raise
        except (httpx.HTTPError, OSError) as exc:
            _remove_quietly(target)
            raise JobFailure("媒体下载", f"下载或落盘失败: {exc}", retryable=True) from exc
        heartbeat.raise_if_lost("下载媒体后")
        return target

    def _resolve_label_definitions(self, payload: dict[str, Any]) -> dict[str, dict[str, str]]:
        """解析本次任务的标签定义，并对没有语义定义的新标签 fail-closed。

        ``allowed_labels`` 只有 ID，不能让模型理解一个刚由 platform 新增的标签。
        如果任务同时携带 ``labels``/``label_definitions``/``label_schema.labels``，优先使用其中的名称、
        描述和正反例；未知且没有描述的 ID 不进入 prompt，避免模型按标签名字猜测。
        """
        allowed = payload.get("allowed_labels")
        allowed_present = isinstance(allowed, list)
        allowed_keys = {_canonical_label_key(item) for item in allowed} if allowed_present else set()

        label_entries: list[Any] = []
        label_schema = payload.get("label_schema")
        if isinstance(label_schema, dict) and isinstance(label_schema.get("labels"), list):
            label_entries.extend(label_schema["labels"])
        if isinstance(payload.get("labels"), list):
            label_entries.extend(payload["labels"])
        if isinstance(payload.get("label_definitions"), list):
            label_entries.extend(payload["label_definitions"])

        supplied: dict[str, dict[str, str]] = {}

        def evidence_text(value: Any) -> str:
            if isinstance(value, (list, tuple)):
                return "；".join(str(part).strip() for part in value if str(part).strip())
            return str(value or "").strip()

        def from_item(item: Any) -> tuple[str, dict[str, str]] | None:
            if isinstance(item, str):
                key = _canonical_label_key(item)
                if not key:
                    return None
                if key in IGNORED_LABELS:
                    logger.info("忽略泛化或非物理配药标签: %s", item)
                    return None
                definition = ALL_LABEL_DEFINITIONS.get(key)
                if definition is None:
                    logger.warning("标签 %r 没有可供模型使用的定义，跳过该标签。", item)
                    return None
                return key, dict(definition)

            if not isinstance(item, dict):
                logger.warning("忽略无法识别的标签定义条目: %r", item)
                return None
            key = _canonical_label_key(item.get("id", item.get("label_id", item.get("label", ""))))
            if not key:
                return None
            if key in IGNORED_LABELS:
                logger.info("忽略泛化或非物理配药标签: %s", key)
                return None
            base = ALL_LABEL_DEFINITIONS.get(key, {})
            name = evidence_text(item.get("name", item.get("display_name", base.get("name", ""))))
            desc = evidence_text(item.get("criteria", item.get("description", base.get("desc", ""))))
            positive = evidence_text(item.get("positive_evidence", item.get("positive_examples", "")))
            negative = evidence_text(item.get("negative_evidence", item.get("negative_examples", "")))
            exclusion = evidence_text(item.get("exclusion_criteria", item.get("exclusions", "")))
            if exclusion:
                negative = f"{negative}；排除条件：{exclusion}" if negative else f"排除条件：{exclusion}"
            if positive:
                desc = f"{desc}；正面证据：{positive}" if desc else f"正面证据：{positive}"
            if negative:
                desc = f"{desc}；反面证据：{negative}" if desc else f"反面证据：{negative}"
            if not desc:
                logger.warning("标签 %r 缺少 description/criteria，跳过该标签。", key)
                return None
            return key, {"name": name or key, "desc": desc}

        for entry in label_entries:
            parsed = from_item(entry)
            if parsed is not None:
                supplied[parsed[0]] = parsed[1]

        def fetch_remote_definitions() -> dict[str, dict[str, str]]:
            project_id = payload.get("project_id")
            if not project_id or not self.api_base_url:
                return {}
            try:
                headers = {"Authorization": f"Bearer {self._read_bearer_token()}"}
                base = self.api_base_url.rstrip("/")
                if not base.endswith("/api/v1"):
                    base = f"{base}/api/v1"
                url = f"{base}/projects/{project_id}/annotation-label-schema"
                resp = httpx.get(url, headers=headers, timeout=5.0, trust_env=self.trust_env)
                if resp.status_code != 200:
                    logger.warning("Could not fetch remote label schema: HTTP %s", resp.status_code)
                    return {}
                data = resp.json()
                labels_list = data.get("labels", []) if isinstance(data, dict) else []
                resolved: dict[str, dict[str, str]] = {}
                if isinstance(labels_list, list):
                    for item in labels_list:
                        parsed = from_item(item)
                        if parsed is not None:
                            resolved[parsed[0]] = parsed[1]
                if resolved:
                    logger.info("Dynamically loaded %d labels from Core API schema", len(resolved))
                return resolved
            except Exception as exc:
                logger.warning("Could not fetch remote label schema: %s. Using local definitions.", exc)
                return {}

        # 1. 明确的 allowed_labels 是本任务的白名单；即使是旧消息里的泛化 ID，
        # 也不再把它放进 prompt。未知 ID 必须同时提供定义，否则直接跳过。
        if allowed_present:
            resolved: dict[str, dict[str, str]] = {}
            remote: dict[str, dict[str, str]] = {}
            remote_loaded = False
            for item in allowed:
                key = _canonical_label_key(item)
                if not key or key in IGNORED_LABELS:
                    if key in IGNORED_LABELS:
                        logger.info("忽略 allowed_labels 中的泛化标签: %s", item)
                    continue
                definition = supplied.get(key)
                if definition is None:
                    parsed = from_item(item)
                    definition = parsed[1] if parsed is not None else None
                if definition is None:
                    if not remote_loaded:
                        remote = fetch_remote_definitions()
                        remote_loaded = True
                    definition = remote.get(key)
                if definition is not None:
                    resolved[key] = definition
            logger.info(
                "本任务标签白名单: requested=%d, usable=%d, ignored_or_undefined=%d",
                len(allowed), len(resolved), max(0, len(allowed_keys) - len(resolved)),
            )
            return resolved

        # 2. 兼容 labels / label_schema.labels 列表（dict 或 str）。
        if label_entries:
            return supplied

        # 3. 兼容从 Core API 远程获取标签架构。
        remote = fetch_remote_definitions()
        if remote:
            return remote

        return dict(DEFAULT_LABEL_DEFINITIONS)

    def _submit_result(self, job_id: str, attempt: int, lease_token: str, body: dict[str, Any]) -> None:
        base = self.api_base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{job_id}/attempts/{attempt}/action-result"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": lease_token,
        }
        headers["Authorization"] = f"Bearer {self._read_bearer_token()}"

        logger.info("[%s] 正在向 Core API 提交推理结果: %s", self.worker_id, url)
        with httpx.Client(timeout=30.0, trust_env=self.trust_env) as client:
            try:
                resp = client.put(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                raise JobFailure("结果回传", f"网络错误: {exc}", retryable=True) from exc
            if resp.status_code not in (200, 204):
                raise JobFailure(
                    "结果回传", resp.text[:300].strip(),
                    status=resp.status_code, hint=_result_error_hint(resp.status_code),
                    core_api=True,
                )

    def infer_video(
        self,
        video_file: str,
        label_defs: dict[str, dict[str, str]] | None = None,
        allowed_labels: list[str] | None = None,
        heartbeat: LeaseHeartbeat | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        # ``{}`` 代表本次任务没有可用的已定义标签，不能偷偷回退默认词表；
        # 否则 platform 新增但未提供语义定义的标签会被误判成默认动作。
        active_labels = DEFAULT_LABEL_DEFINITIONS if label_defs is None else label_defs
        total_duration = self._get_video_duration(video_file)
        duration_ms = max(1, int(round(total_duration * 1000)))
        chunks_plan = self._plan_chunks(total_duration)
        total_windows = max(1, len(chunks_plan))

        all_actions = []
        temp_chunk = os.path.join(_scratch_dir(), f"_tmp_chunk_{os.getpid()}.mp4")

        try:
            for cp in chunks_plan:
                # 每个窗口之间检查一次租约：单窗口要跑 ffmpeg 切片 + 一次完整 generate，
                # 是整条链路上最自然的检查点。
                if heartbeat is not None:
                    heartbeat.raise_if_lost(f"窗口 {cp['window_idx']} 开始前")

                w_idx = cp["window_idx"]
                s_start = cp["slice_start"]
                s_dur = cp["slice_dur"]
                l_core_start = cp["local_core_start"]
                l_core_end = cp["local_core_end"]

                if not self._extract_chunk(video_file, s_start, s_dur, temp_chunk):
                    # 切片失败是输入/环境问题，重投一次结果一样，不该白占一次消息
                    raise JobFailure(
                        "窗口切片",
                        f"ffmpeg 未能切出窗口 {w_idx} (slice_start={s_start:.3f}s, slice_dur={s_dur:.3f}s)",
                    )

                chunk_actions = self._infer_single_window(temp_chunk, s_dur, active_labels)
                boundary = int(round(cp["core_abs_start"]))

                for item in chunk_actions:
                    local_s = int(item["start_sec"])
                    local_e = int(item["end_sec"])
                    label = str(item["label"]).strip()
                    conf = float(item["confidence"])

                    g_s = max(0, min(int(round(s_start + local_s)), int(round(total_duration))))
                    g_e = max(g_s + 1, min(int(round(s_start + local_e)), int(round(total_duration))))

                    seg = {
                        "start_sec": g_s,
                        "end_sec": g_e,
                        "label": label,
                        "confidence": round(conf, 2),
                        "source_window": w_idx,
                        "stitched_windows": [w_idx],
                    }
                    if item.get("item"):
                        seg["item"] = item["item"]

                    # 续段判定：pad 区里检出的动作，若「已收集动作」中有一条同标签、时间相邻，
                    # 且那条一直延伸到本窗口边界，说明它是跨窗续段——接回去，而不是新起一段。
                    candidates = [
                        act for act in all_actions
                        if act["end_sec"] >= boundary - CONTINUATION_GAP_SEC and _can_continue(act, seg)
                    ]
                    if candidates:
                        _merge_same_label(max(candidates, key=lambda a: a["end_sec"]), seg)
                        continue

                    if l_core_start <= local_s < l_core_end:
                        all_actions.append(seg)
                    elif local_s < l_core_start and local_e > l_core_start:
                        # 动作横跨边界且未续接到前一窗口：截取落入本窗口核心区的部分，避免接缝盲区漏检
                        seg["start_sec"] = boundary
                        if seg["end_sec"] > seg["start_sec"]:
                            all_actions.append(seg)
        finally:
            if os.path.exists(temp_chunk):
                try:
                    os.remove(temp_chunk)
                except Exception:
                    pass

        # 全局排序与单层不重叠整理
        all_actions.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        validated = []
        for act in all_actions:
            if not validated:
                validated.append(act)
                continue
            prev = validated[-1]
            if _merge_same_label(prev, act):
                continue
            if act["start_sec"] < prev["end_sec"]:
                if act["end_sec"] > prev["end_sec"]:
                    mid = (prev["end_sec"] + act["start_sec"]) // 2
                    prev["end_sec"] = max(prev["start_sec"] + 1, mid)
                    act["start_sec"] = max(prev["end_sec"], mid)
                else:
                    # 包含关系：若内部碎片置信度显著更高则截断外层，否则跳过碎片保护原完整区间
                    if act["confidence"] > prev["confidence"] + 0.15:
                        prev["end_sec"] = act["start_sec"]
                    else:
                        continue
            if act["end_sec"] > act["start_sec"]:
                validated.append(act)

        # 严格对齐 Core API 的 ActionCandidateSegmentItem 契约：
        # 1. 仅保留契约字段：start_ms, end_ms, label, confidence, source_window, stitched_windows
        # 2. 标签大小写与平台 allowed_labels 完全对齐（统一使用小写 ID）
        # 3. 严格区间约束：0 <= start_ms < end_ms <= duration_ms 且非重叠 (start_ms >= prev_end_ms)
        allowed_map = {str(lbl).strip().lower(): str(lbl).strip() for lbl in allowed_labels if str(lbl).strip()} if allowed_labels else {}

        clean_segments: list[dict[str, Any]] = []
        prev_end_ms = 0

        for act in validated:
            raw_label = str(act["label"]).strip()
            norm_label = allowed_map.get(raw_label.lower(), raw_label.lower())
            if allowed_map and raw_label.lower() not in allowed_map:
                logger.warning("[%s] 丢弃不属于平台 allowed_labels 的动作片段: %s", self.worker_id, raw_label)
                continue

            s_ms = max(0, int(act["start_sec"] * 1000))
            e_ms = min(duration_ms, int(act["end_sec"] * 1000))

            if s_ms < prev_end_ms:
                s_ms = prev_end_ms
            if e_ms <= s_ms:
                continue
            if e_ms > duration_ms:
                e_ms = duration_ms
                if e_ms <= s_ms:
                    continue

            conf = max(0.0, min(1.0, float(act.get("confidence", 0.9))))
            src_win = int(act.get("source_window", 1))
            stitched = [int(w) for w in act.get("stitched_windows", [src_win])]

            clean_segments.append({
                "start_ms": s_ms,
                "end_ms": e_ms,
                "label": norm_label,
                "confidence": round(conf, 4),
                "source_window": src_win,
                "stitched_windows": stitched,
            })
            prev_end_ms = e_ms

        return clean_segments, total_windows, duration_ms

    def _infer_single_window(
        self, chunk_path: str, dur_sec: float, label_defs: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        chunk_max = int(round(dur_sec))
        if chunk_max <= 0:
            # 亚秒级切片：0~0 之间不存在满足 start_sec < end_sec 的整数秒，
            # 发这种 prompt 只会得到垃圾输出，直接跳过。
            logger.warning("切片时长 %.3fs 不足 1s，跳过该窗口。", dur_sec)
            return []
        prompt = self._build_prompt(label_defs, chunk_max)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": chunk_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kwargs: dict[str, Any] = {}
        if self.model_family == "qwen3_5":
            # Qwen3.5 默认开启 thinking；动作 Worker 需要干净 JSON，且不应为
            # 每个窗口额外生成思考内容。Qwen3-VL baseline 保持原有模板行为。
            # 这里是直接调用 Transformers Processor，使用模板参数本身；
            # chat_template_kwargs 是 vLLM/OpenAI 兼容 API 的外层写法。
            template_kwargs["enable_thinking"] = False
        video_kwargs: dict[str, Any] = (
            {"processor_kwargs": {
                "fps": self.sampling_fps,
                "do_sample_frames": True,
                # Qwen3.5 的 Transformers 处理器默认不启用逐帧像素上限；
                # 采用 qwen-vl-utils 参考行为，控制视觉 token 和显存波动。
                "cap_pixels_per_frame": True,
            }}
            if self.model_family == "qwen3_5"
            else {"fps": self.sampling_fps}
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **video_kwargs,
            **template_kwargs,
        ).to("cuda")

        token_limits = _token_limits(self.max_new_tokens)

        for attempt, token_limit in enumerate(token_limits, 1):
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=token_limit,
                    do_sample=False,
                    use_cache=True,
                )
            output_text = self.processor.decode(
                output_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
            ).strip()
            del output_ids

            try:
                return self._parse_output(output_text, chunk_max, label_defs)
            except JobFailure:
                if attempt >= len(token_limits) or not _output_needs_token_retry(output_text):
                    if attempt >= len(token_limits) and _output_needs_token_retry(output_text):
                        salvaged = self._salvage_truncated_output(output_text, chunk_max, label_defs)
                        if salvaged:
                            logger.warning(
                                "窗口推理输出达到生成上限 %s 截断，保底抢救出 %d 条完整动作片段。",
                                token_limit, len(salvaged),
                            )
                            return salvaged
                    raise
                logger.warning(
                    "窗口输出疑似被 max_new_tokens=%s 截断，重试上限为 %s。",
                    token_limit, token_limits[attempt],
                )

        # token_limits 至少包含一次尝试；此处只为静态分析保留明确出口。
        raise JobFailure("模型输出解析", "模型窗口推理未返回结果", retryable=True)

    def _build_prompt(self, label_defs: dict[str, dict[str, str]], chunk_max: int) -> str:
        lines = [
            "你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。",
            "",
            "动作标签定义与判据：",
        ]
        for key, info in label_defs.items():
            desc = f"（{info['desc']}）" if info.get("desc") else ""
            lines.append(f"- {key}: {info.get('name', key)}{desc}")

        lines.extend([
            "",
            "目标物品类别（用于 acquire_item 等动作指明受体）：",
            "- vial: 西林瓶/粉针瓶",
            "- ampoule: 安瓿瓶",
            "- syringe: 注射器",
            "- infusion_bottle: 输液瓶/输液袋",
            "- needle: 针头/穿刺针",
            "- cotton_swab: 棉签",
            "- packaging: 包装耗材",
            "- other_tool: 砂轮/开瓶器等工具",
            "",
            "【判据与输出硬性约束】",
            "1. 实事求是：依据画面中护士手部实际操作的物理动作准确分类，仅标注画面中有明确视觉动作的区间；",
            "2. 严禁流程脑补：严禁脱离画面强行背诵或补全整套流程！若当前片段无明确动作、无手在操作、处于间歇或等待状态，必须直接输出空数组 []；",
            "3. 动作与受体区分：",
            "   - acquire_item: 必须是手伸向原静止物品并将其拿起进入操作区的【瞬态短动作】（通常仅 1~3 秒），输出时必须携带 item 字段注明拿取的物品；",
            "   - 严禁误报 acquire_item: 手在空中移动、单纯握持物体不动、调整握持姿势、拆包装、未使物体发生位移、或空操作台时，绝不可标为 acquire_item；",
            "   - 动作严格区分: 只有看到针头实际进入容器才标 insert_needle；只有看到活塞回抽且药液进入注射器才标 draw_liquid；只有向目标容器推注才标 transfer_liquid；看到掰开安瓿或取下盖子标 open_container；",
            "4. 真实动作时长：起止时间严格以手部实际开始与结束操作的时间为准，严禁默认从 0 秒开始标记；",
            f"5. 起止时间必须为 0 到 {chunk_max} 之间的整数秒，start_sec < end_sec 且动作区间不可重叠；",
            "6. 格式严格为紧凑的单个 JSON 数组（严禁任何 Markdown 围栏、额外换行缩进或解释文字，所有动作紧凑排在同一数组中）：",
            '示例 1 (有动作): [{"start_sec": 1, "end_sec": 3, "label": "open_container", "confidence": 0.95}, {"start_sec": 5, "end_sec": 7, "label": "acquire_item", "item": "syringe", "confidence": 0.92}]',
            "示例 2 (无动作/空镜/等待): []",
        ])
        return "\n".join(lines)

    @staticmethod
    def _collect_raw_items(cleaned: str) -> list[Any]:
        """从完整 JSON 中收集候选条目；不从截断数组中捡取前缀。"""
        items: list[Any] = []

        def take(obj: Any) -> None:
            if isinstance(obj, list):
                for elem in obj:
                    if isinstance(elem, dict) or (isinstance(elem, (list, tuple)) and len(elem) >= 3):
                        items.append(elem)
            elif isinstance(obj, dict):
                items.append(obj)

        # 1) 配对完整的数组（含 [[...]] 嵌套、markdown 围栏、前后夹带说明文字）
        array_groups = _scan_balanced(cleaned, "[", "]")
        for group in array_groups:
            take(_loads_lenient(group))
        if items:
            return items

        # 有未闭合数组时，下面的对象扫描可能捡到前面已经完整的对象，
        # 这会把 token 截断伪装成成功结果。完整数组优先，未闭合数组直接失败。
        if "[" in cleaned:
            return items
        if "{" in cleaned and not _scan_balanced(cleaned, "{", "}"):
            return items

        # 2) 配对完整的对象；{"actions": [...]} 这类外层再包一层的情况取内层数组
        for group in _scan_balanced(cleaned, "{", "}"):
            obj = _loads_lenient(group)
            if isinstance(obj, dict):
                nested = next((v for v in obj.values() if isinstance(v, list)), None)
                take(nested if nested is not None else obj)
        if items:
            return items

        # 3) 逐行兜底（模型把每个动作单独放在一行时）
        for line in cleaned.splitlines():
            for group in _scan_balanced(line, "[", "]") + _scan_balanced(line, "{", "}"):
                take(_loads_lenient(group))
            if items:
                return items

        return items

    @staticmethod
    def _normalize_confidence(raw: Any) -> float:
        """置信度归一到 [0,1]：95 -> 0.95；轻微越界 1.5 -> 1.0；非法 -> 0.9。"""
        try:
            conf = float(raw)
        except (ValueError, TypeError):
            return 0.9
        if not math.isfinite(conf) or conf < 0.0:
            return 0.9
        if conf > 1.0:
            # >= 2.0 认定是百分制（95 -> 0.95）；1~2 之间视为笔误，夹到 1.0
            return min(1.0, round(conf / 100.0, 2)) if conf >= 2.0 else 1.0
        return round(conf, 2)

    def _parse_output(
        self, text: str, chunk_max: int, label_defs: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        cleaned = text.strip()
        if cleaned == "[]" or chunk_max <= 0:
            return []

        raw_items = self._collect_raw_items(cleaned)
        if not raw_items:
            if _is_explicit_empty_output(cleaned):
                return []
            # 非空且不含 [] 的输出不是“无动作”，而是协议失败。返回空数组会把
            # 模型解析错误伪装成成功任务，平台随后无法区分漏检和坏输出。
            raise JobFailure(
                "模型输出解析",
                f"未能解析为动作 JSON 数组，原文片段: {cleaned[:200]!r}",
                retryable=True,
            )

        actions: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                item = {
                    "start_sec": item[0],
                    "end_sec": item[1],
                    "label": item[2],
                    "confidence": item[3] if len(item) > 3 else 0.9,
                }
            if not isinstance(item, dict):
                continue
            if "start_sec" not in item or "end_sec" not in item or "label" not in item:
                continue
            try:
                raw_s = float(item["start_sec"])
                raw_e = float(item["end_sec"])
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(raw_s) and math.isfinite(raw_e)):
                continue

            # 越界一律钳制而不是丢弃：end=99（窗口只有 27s）表达的是「动作持续到窗口末」，
            # 丢掉等于把模型真实检出的动作整条抹除；而 start>=end 是退化区间，
            # 它没有携带任何时长信息，凭空补 1s 等于编数据，所以那种情况才丢弃。
            s = max(0, min(int(round(raw_s)), chunk_max))
            e = max(0, min(int(round(raw_e)), chunk_max))
            if s >= e:
                logger.warning("丢弃退化区间 start=%s end=%s (chunk_max=%s)", raw_s, raw_e, chunk_max)
                continue

            raw_lbl = str(item["label"]).strip()
            matched_key = next((k for k in label_defs if k.upper() == raw_lbl.upper()), None)
            if not matched_key:
                # 静默丢弃会让「模型确实检出了动作、只是词表对不上」完全不可见 ——
                # 现象是漏标，却查不出原因。
                logger.warning("丢弃词表外的动作标签 label=%r（当前词表: %s）", raw_lbl,
                               ", ".join(label_defs) or "<空>")
                continue

            # 针对 acquire_item 施加单段时长约束：瞬态拿取超过 5 秒的异常长段进行夹紧
            if matched_key == "acquire_item":
                max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
                if (e - s) > max_dur:
                    logger.info("对单段超长 acquire_item 进行瞬态时长夹紧: [%s, %s] -> [%s, %s]", s, e, s, s + max_dur)
                    e = s + max_dur

            act_item: dict[str, Any] = {
                "start_sec": s,
                "end_sec": e,
                "label": matched_key,
                "confidence": self._normalize_confidence(item.get("confidence", 0.9)),
            }
            raw_item_type = str(item.get("item", "")).strip().lower()
            if raw_item_type:
                act_item["item"] = raw_item_type
            actions.append(act_item)
        if raw_items and not actions:
            raise JobFailure(
                "模型输出校验",
                "模型返回了候选条目，但没有一条符合本次任务的标签和时间 schema",
                retryable=True,
            )
        actions.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["label"]))
        return actions

    def _salvage_truncated_output(
        self, text: str, chunk_max: int, label_defs: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        """当单窗口达到最大 token 上限发生截断时，保底提取截断前已完整闭合的动作对象。"""
        cleaned = text.strip()
        if "[" not in cleaned:
            return []

        raw_items: list[Any] = []
        for group in _scan_balanced(cleaned, "{", "}"):
            obj = _loads_lenient(group)
            if isinstance(obj, dict):
                raw_items.append(obj)

        if not raw_items:
            return []

        actions: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if "start_sec" not in item or "end_sec" not in item or "label" not in item:
                continue
            try:
                raw_s = float(item["start_sec"])
                raw_e = float(item["end_sec"])
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(raw_s) and math.isfinite(raw_e)):
                continue

            s = max(0, min(int(round(raw_s)), chunk_max))
            e = max(0, min(int(round(raw_e)), chunk_max))
            if s >= e:
                continue

            raw_lbl = str(item["label"]).strip()
            matched_key = next((k for k in label_defs if k.upper() == raw_lbl.upper()), None)
            if not matched_key:
                continue

            if matched_key == "acquire_item":
                max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
                if (e - s) > max_dur:
                    e = s + max_dur

            act_item: dict[str, Any] = {
                "start_sec": s,
                "end_sec": e,
                "label": matched_key,
                "confidence": self._normalize_confidence(item.get("confidence", 0.9)),
            }
            raw_item_type = str(item.get("item", "")).strip().lower()
            if raw_item_type:
                act_item["item"] = raw_item_type
            actions.append(act_item)

        actions.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["label"]))
        return actions

    def _plan_chunks(self, total_duration: float) -> list[dict[str, Any]]:
        plan = []
        c_start = 0.0
        w_idx = 1
        while c_start < total_duration:
            if (total_duration - c_start) <= (CORE_CHUNK_SEC + 2.0):
                c_end = total_duration
            else:
                c_end = c_start + CORE_CHUNK_SEC

            slice_start = max(0.0, c_start - CONTEXT_PAD_SEC)
            slice_end = min(total_duration, c_end + CONTEXT_PAD_SEC)
            slice_dur = slice_end - slice_start

            local_core_start = int(round(c_start - slice_start))
            local_core_end = int(round(c_end - slice_start))

            plan.append({
                "window_idx": w_idx,
                "slice_start": slice_start,
                "slice_dur": slice_dur,
                "core_abs_start": c_start,
                "core_abs_end": c_end,
                "local_core_start": local_core_start,
                "local_core_end": local_core_end,
            })
            c_start = c_end
            w_idx += 1
        return plan

    def _get_video_duration(self, video_file: str) -> float:
        """读取视频物理时长。探测失败仍回落 30s，但必须留痕：
        静默兜底会让一个 10 分钟视频被当成 30 秒处理，且结果表面完全正常。"""
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
            logger.warning(
                "ffprobe 未返回有效时长 (rc=%s)，回落 30s 兜底；若源视频长于 30s，结果将只覆盖前 30s。stderr=%s",
                res.returncode, res.stderr.strip()[:200],
            )
        except Exception as exc:
            logger.warning("ffprobe 调用异常 (%s)，回落 30s 兜底；若源视频长于 30s，结果将只覆盖前 30s。", exc)
        return 30.0

    def _extract_chunk(self, src: str, s_sec: float, dur_sec: float, out: str) -> bool:
        cmd = ["ffmpeg", "-y", "-ss", f"{s_sec:.3f}", "-t", f"{dur_sec:.3f}", "-i", src, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Candidate 4090 Worker")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.5-4B"))
    parser.add_argument("--api-url", default=os.getenv("PLATFORM_API_BASE_URL") or os.getenv("PLATFORM_API_URL", "http://localhost:8080"))
    parser.add_argument("--rabbitmq-url", default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
    parser.add_argument("--queue", default="platform.ai.action-candidate.v1")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--heartbeat", type=int, default=DEFAULT_HEARTBEAT_SEC,
        help="AMQP 心跳秒数；长阻塞推理场景建议保持 0 关闭心跳 (默认: 0，可用 RABBITMQ_HEARTBEAT 覆盖)",
    )
    parser.add_argument(
        "--blocked-timeout", type=float, default=DEFAULT_BLOCKED_TIMEOUT_SEC,
        help="broker 触发 connection.blocked 后的最长等待秒数 (默认: 300；仅当 --heartbeat > 0 时生效)",
    )
    parser.add_argument(
        "--worker-id", default=os.getenv("AI_WORKER_ID") or os.getenv("WORKER_ID", ""),
        help="Worker 实例标识 ID。留空则自动推导：先取令牌 sub（生产 JWT 模式），"
             "否则用平台本地默认值 ai-worker-local-1。仅在平台侧改了 AI_WORKER_ID 时才需要显式指定",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("PLATFORM_WORKER_BEARER_TOKEN") or os.getenv("AI_WORKER_BEARER_TOKEN") or "",
        help="访问 Core API 的 Bearer Token（生产环境建议使用 --bearer-token-file）",
    )
    parser.add_argument(
        "--bearer-token-file",
        default=os.getenv("PLATFORM_WORKER_BEARER_TOKEN_FILE") or os.getenv("AI_WORKER_BEARER_TOKEN_FILE") or "",
        help="读取 Core API Bearer Token 的文件路径；每次请求重新读取以支持轮换",
    )
    parser.add_argument(
        "--trust-env", dest="trust_env", action="store_true", default=DEFAULT_TRUST_ENV,
        help="让 httpx 读取 HTTP_PROXY/HTTPS_PROXY 等环境代理 (默认关闭，可用 PLATFORM_TRUST_ENV=1 覆盖)",
    )
    parser.add_argument(
        "--no-trust-env", dest="trust_env", action="store_false",
        help="显式忽略环境代理变量 (默认行为)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="单窗口生成上限 (默认: 1024，可用 MAX_NEW_TOKENS 覆盖)。疑似截断时自动重试 1024",
    )
    parser.add_argument(
        "--use-hub-kernels", dest="use_hub_kernels", action="store_true",
        default=DEFAULT_USE_HUB_KERNELS,
        help="实验性启用 Qwen3.5 Transformers Hub kernels（默认关闭；需独立环境回归）",
    )
    parser.add_argument(
        "--idle-offload-seconds", type=int, default=DEFAULT_IDLE_OFFLOAD_SEC,
        help="显存空闲自动卸载超时秒数: -1=永久常驻(17GB); 0=任务完成立即释放; >0=空闲N秒自动释放 (默认: 300，可用 IDLE_OFFLOAD_SECONDS 覆盖)",
    )
    args = parser.parse_args()

    _warn_on_env_proxy()
    token_source = "file" if args.bearer_token_file else ("inline" if args.bearer_token else "missing")
    logger.info(
        "配置: worker_id=%s queue=%s api=%s token_source=%s fps=%s heartbeat=%ss idle_offload=%ss trust_env=%s max_new_tokens=%s hub_kernels=%s",
        args.worker_id or "<自动推导>", args.queue, args.api_url, token_source, args.fps, args.heartbeat, args.idle_offload_seconds, args.trust_env, args.max_new_tokens, args.use_hub_kernels,
    )

    try:
        worker = ActionWorker(
            model_path=args.model,
            api_base_url=args.api_url,
            rabbitmq_url=args.rabbitmq_url,
            queue_name=args.queue,
            worker_id=args.worker_id,
            bearer_token=args.bearer_token,
            bearer_token_file=args.bearer_token_file,
            sampling_fps=args.fps,
            heartbeat_sec=args.heartbeat,
            blocked_connection_timeout=args.blocked_timeout,
            trust_env=args.trust_env,
            max_new_tokens=args.max_new_tokens,
            idle_offload_sec=args.idle_offload_seconds,
            use_hub_kernels=args.use_hub_kernels,
        )
        worker.run()
    except KeyboardInterrupt:
        pass
