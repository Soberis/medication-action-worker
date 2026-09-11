"""Action Candidate Worker Daemon for Qwen3-VL on RTX 4090.

1. Model resident in VRAM (zero cold start).
2. RabbitMQ consumer with prefetch_count=1 (strict serial execution, no concurrent OOM).
3. Frame-accurate chunked inference with cross-window continuation stitching.
4. Result submission to Core API.

注意：本实现**不做租约续期**。回传时携带 `X-AI-Lease-Token`，但单次推理可能耗时数分钟，
若平台租约在此期间过期，回传会被拒（表现为 4xx），任务将按 nack 语义重投一次。
"""

from __future__ import annotations

import os

# 必须在 import torch 之前设置：CUDA 缓存分配器在首次分配时就固化参数，
# 放到 import torch 之后（哪怕只是 import 之后、还没用到 CUDA）也不保险。
# 用 setdefault 保证 run.sh / .env 里的显式配置优先。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
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
from typing import Any
import uuid

import httpx
import pika
import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("action_worker")

DEFAULT_LABEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "OPEN_CAP": {"name": "操作瓶盖", "desc": "接触药瓶顶端掰开安瓿瓶、拧开盖子或拔下针帽"},
    "DRAW_LIQUID": {"name": "抽取药液", "desc": "针头刺入药液，向后拉动注射器活塞吸入液体"},
    "INJECT_MIX": {"name": "注入混匀", "desc": "向前推动活塞注入药液，或在瓶内反复推拉活塞、摇晃药瓶混匀"},
    "CHECK_INFO": {"name": "核对信息", "desc": "视线聚焦药瓶标签、核对处方单或刻度"},
    "LABEL_SEAL": {"name": "贴签封包", "desc": "取标签纸粘贴到药瓶或装入密封袋打包"},
}

CORE_CHUNK_SEC = 24.0
CONTEXT_PAD_SEC = 3.0

# 断线重连退避：避免网络不通时以固定 5s 频率刷屏（截图里就是这个现象）
RECONNECT_BACKOFF_MIN_SEC = 5.0
RECONNECT_BACKOFF_MAX_SEC = 60.0


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

# 单个窗口的生成上限。256 对「4~5 个动作 + markdown 围栏」偏紧，
# 一旦被截断会丢掉窗口内最后一个动作；模型正常收尾时并不会真的用满，
# 抬高上限只在病态输出上多花时间，不影响常态延迟。
DEFAULT_MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 512)

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


def _repair_truncated(snippet: str) -> Any:
    """max_new_tokens 截断导致 JSON 未闭合时，把尾部补齐再解析。

    例：'[{"start_sec": 2, "label": "OPEN_CAP", "confidence": 0.98'
        -> '[{"start_sec": 2, "label": "OPEN_CAP", "confidence": 0.98}]'
    """
    s = snippet.rstrip().rstrip(",").rstrip()
    if not s:
        return None
    if s.count('"') % 2 == 1:                      # 字符串字面量被截断
        s += '"'
    s += "}" * max(0, s.count("{") - s.count("}"))  # 补齐未闭合对象与数组
    s += "]" * max(0, s.count("[") - s.count("]"))
    return _loads_lenient(s)


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


# 同标签两段之间允许的最大空档（秒）。超出即视为两个独立动作，不合并。
CONTINUATION_GAP_SEC = 1


def _can_continue(acc: dict[str, Any], seg: dict[str, Any]) -> bool:
    """seg 能否作为 acc 的同标签续段（同标签且时间相邻或相交）。"""
    return acc["label"] == seg["label"] and seg["start_sec"] <= acc["end_sec"] + CONTINUATION_GAP_SEC


def _merge_same_label(acc: dict[str, Any], seg: dict[str, Any]) -> bool:
    """同标签且相邻时把 seg 并入 acc，返回是否发生合并。

    这是**唯一**的「跨窗续段」合并入口：窗口内续接与全局收尾都调用它。
    判据只写在 `_can_continue` 一处，避免同一规则在两处各写一遍、日后悄悄漂移。
    """
    if not _can_continue(acc, seg):
        return False
    acc["end_sec"] = max(acc["end_sec"], seg["end_sec"])
    acc["confidence"] = round(max(acc["confidence"], seg["confidence"]), 2)
    windows = acc.setdefault("stitched_windows", [acc["source_window"]])
    for w in seg.get("stitched_windows", []):
        if w not in windows:
            windows.append(w)
    return True


class LeaseHeartbeat:
    """在后台线程中定期发送租约保活心跳，防止长时间推理导致 Core API 租约过期。"""

    def __init__(
        self,
        api_base_url: str,
        bearer_token: str,
        worker_id: str,
        job_id: str,
        attempt: int,
        lease_token: str,
        interval_sec: float,
        trust_env: bool = False,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.worker_id = worker_id
        self.job_id = job_id
        self.attempt = attempt
        self.lease_token = lease_token
        self.interval_sec = max(5.0, interval_sec) if interval_sec > 0 else 30.0
        self.trust_env = trust_env
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

    def _run(self) -> None:
        base = self.api_base_url
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{self.job_id}/attempts/{self.attempt}/heartbeat"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": self.lease_token,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        body = {"worker_id": self.worker_id}

        while not self._stop_event.wait(self.interval_sec):
            try:
                with httpx.Client(timeout=10.0, trust_env=self.trust_env) as client:
                    resp = client.post(url, json=body, headers=headers)
                    if resp.status_code == 200:
                        logger.debug("[%s] 租约心跳成功 (job: %s)", self.worker_id, self.job_id)
                    else:
                        logger.warning(
                            "[%s] 租约心跳警告 status=%s: %s",
                            self.worker_id, resp.status_code, resp.text[:200]
                        )
            except Exception as exc:
                logger.warning("[%s] 租约心跳请求异常: %s", self.worker_id, exc)


class ActionWorker:
    def __init__(
        self,
        model_path: str,
        api_base_url: str,
        rabbitmq_url: str,
        queue_name: str = "platform.ai.action-candidate.v1",
        worker_id: str = "action-worker-4090-01",
        bearer_token: str = "",
        sampling_fps: float = 1.0,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        blocked_connection_timeout: float = DEFAULT_BLOCKED_TIMEOUT_SEC,
        trust_env: bool = DEFAULT_TRUST_ENV,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        idle_offload_sec: int = DEFAULT_IDLE_OFFLOAD_SEC,
    ) -> None:
        self.model_path = model_path
        self.api_base_url = api_base_url.rstrip("/")
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.worker_id = worker_id
        self.bearer_token = bearer_token
        self.heartbeat_sec = heartbeat_sec
        self.blocked_connection_timeout = blocked_connection_timeout
        self.trust_env = trust_env
        self.idle_offload_sec = idle_offload_sec
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
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

        # 动态探测与自适应注意力内核 (FlashAttention-2 / SDPA)
        attn_impl = "sdpa"
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Detected flash-attn, enabled FlashAttention-2 hardware acceleration.")
        except ImportError:
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
        self.model = ModelClass.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_impl,
            device_map="cuda",
        )
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
                self._ensure_model_loaded()
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
        try:
            payload = json.loads(body.decode("utf-8"))
            job_id = payload.get("job_id", job_id)
            logger.info("[%s] 收到任务: %s", self.worker_id, job_id)
            self.process_job(payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("[%s] 任务 %s 处理完成并 ACK。", self.worker_id, job_id)
        except Exception as exc:
            logger.error("[%s] 任务 %s 处理失败: %s", self.worker_id, job_id, exc, exc_info=True)
            if method.redelivered:
                # 已经重投过一次仍然失败 -> 停止 requeue，避免毒药消息无限死循环
                logger.error("[%s] 任务 %s 重投后仍失败，nack(requeue=False)，交由 DLX 或丢弃。", self.worker_id, job_id)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            else:
                logger.warning("[%s] 任务 %s 将重投一次 (requeue=True)。", self.worker_id, job_id)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _acquire_lease(self, job_id: str, attempt: int, lease_token: str) -> dict[str, Any]:
        """向 Core API 申请任务租约并获取输入媒体授权 URL。"""
        base = self.api_base_url
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{job_id}/attempts/{attempt}/lease"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": lease_token,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        body = {"worker_id": self.worker_id}

        logger.info("[%s] 正在向 Core API 申请租约: %s", self.worker_id, url)
        with httpx.Client(timeout=30.0, trust_env=self.trust_env) as client:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Core API 租约获取失败 status={resp.status_code} - {resp.text}"
                )
            data = resp.json()
            if not isinstance(data, dict) or not data.get("input_url"):
                raise RuntimeError(f"Core API 返回的租约缺少有效 input_url: {data}")
            return data

    def process_job(self, payload: dict[str, Any]) -> None:
        self._ensure_model_loaded()
        job_id = payload["job_id"]
        attempt = payload.get("attempt", 1)
        lease_token = payload.get("lease_token", "")
        message_id = payload.get("message_id") or str(uuid.uuid4())
        allowed_labels = payload.get("allowed_labels")

        video_url = payload.get("video_url", "")
        video_path = payload.get("video_path", "")
        hb_interval = 60.0

        local_video = video_path
        tmp_download = None
        if not local_video or not os.path.exists(local_video):
            if not video_url:
                # 生产标准流程：通过 Core API 申请租约获取 input_url
                if not lease_token or not self.api_base_url:
                    raise ValueError(
                        f"任务 {job_id} 未提供 local video_path / video_url，且缺少 lease_token 或 api_base_url 无法向 Core API 申请租约。"
                    )
                lease_grant = self._acquire_lease(job_id, attempt, lease_token)
                video_url = lease_grant["input_url"]
                hb_interval = float(lease_grant.get("heartbeat_interval_seconds", 60.0))

            tmp_download = os.path.join(_scratch_dir(), f"_download_{_safe_job_id(job_id)}.mp4")
            logger.info("[%s] 正在从 %s 下载视频至 %s...", self.worker_id, video_url, tmp_download)
            with httpx.stream("GET", video_url, timeout=120.0, trust_env=self.trust_env) as resp:
                resp.raise_for_status()
                with open(tmp_download, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            local_video = tmp_download

        try:
            label_defs = self._resolve_label_definitions(payload)
        except Exception as exc:
            # 该调用点在下方 try 之外时，异常会冒到 _handle_message 把任务 nack 掉，
            # 白白付一次完整推理。标签解析失败只应降级到默认标签，不应让任务失败。
            logger.warning("标签定义解析失败，回退默认标签: %s", exc)
            label_defs = {}

        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            # 推理期间开启租约保活心跳，防止超长视频处理导致租约过期
            with LeaseHeartbeat(
                api_base_url=self.api_base_url,
                bearer_token=self.bearer_token,
                worker_id=self.worker_id,
                job_id=job_id,
                attempt=attempt,
                lease_token=lease_token,
                interval_sec=hb_interval,
                trust_env=self.trust_env,
            ):
                clean_segments, total_windows, duration_ms = self.infer_video(
                    local_video,
                    label_defs=label_defs,
                    allowed_labels=allowed_labels,
                )

            inference_duration_ms = max(0, int((time.time() - start_time) * 1000))
            completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

            # 严格对齐 Core API 的 ActionCandidateResultRequest
            req_body = {
                "message_id": message_id,
                "model_name": payload.get("model_sha256") or "Qwen/Qwen3-VL-8B-Instruct",
                "prompt_version": "v1",
                "total_windows": total_windows,
                "duration_ms": duration_ms,
                "inference_duration_ms": inference_duration_ms,
                "started_at": started_at,
                "completed_at": completed_at,
                "segments": clean_segments,
            }

            self._submit_result(job_id, attempt, lease_token, req_body)
            logger.info(
                "[%s] 任务 %s 结果回传成功 (windows=%d, duration=%dms, inference=%dms, segments=%d)",
                self.worker_id, job_id, total_windows, duration_ms, inference_duration_ms, len(clean_segments),
            )
        finally:
            if tmp_download and os.path.exists(tmp_download):
                try:
                    os.remove(tmp_download)
                except Exception:
                    pass
            # 释放推理过程中申请的瞬时动态显存与碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_label_definitions(self, payload: dict[str, Any]) -> dict[str, dict[str, str]]:
        # 1. 优先读取平台 RabbitMQ 契约中的 allowed_labels (小写字符串列表)
        allowed = payload.get("allowed_labels")
        if isinstance(allowed, list) and allowed:
            resolved: dict[str, dict[str, str]] = {}
            for item in allowed:
                if isinstance(item, str):
                    key_upper = item.upper().strip()
                    if not key_upper:
                        continue
                    if key_upper in DEFAULT_LABEL_DEFINITIONS:
                        resolved[key_upper] = DEFAULT_LABEL_DEFINITIONS[key_upper]
                    else:
                        resolved[key_upper] = {"name": item, "desc": ""}
            if resolved:
                return resolved

        # 2. 兼容 labels 列表字段 (dict 或 str)
        if isinstance(payload.get("labels"), list) and payload["labels"]:
            resolved = {}
            for item in payload["labels"]:
                if isinstance(item, str):
                    key = item.upper().strip()
                    if key:
                        resolved.setdefault(key, DEFAULT_LABEL_DEFINITIONS.get(key, {"name": key, "desc": ""}))
                    continue
                if not isinstance(item, dict):
                    logger.warning("忽略无法识别的标签定义条目: %r", item)
                    continue
                key = str(item.get("id", item.get("label_id", ""))).upper().strip()
                name = str(item.get("name", item.get("display_name", ""))).strip()
                desc = str(item.get("criteria", item.get("description", ""))).strip()
                if key:
                    resolved[key] = {"name": name or key, "desc": desc}
            if resolved:
                return resolved

        # 3. 兼容从 Core API 远程获取标签架构
        project_id = payload.get("project_id")
        if project_id and self.api_base_url:
            try:
                headers = {}
                if self.bearer_token:
                    headers["Authorization"] = f"Bearer {self.bearer_token}"
                base = self.api_base_url
                if not base.endswith("/api/v1"):
                    base = f"{base}/api/v1"
                url = f"{base}/projects/{project_id}/annotation-label-schema"
                resp = httpx.get(url, headers=headers, timeout=5.0, trust_env=self.trust_env)
                if resp.status_code == 200:
                    data = resp.json()
                    labels_list = data.get("labels", [])
                    resolved = {}
                    for item in labels_list:
                        key = str(item.get("label_id", "")).upper().strip()
                        name = str(item.get("display_name", "")).strip()
                        desc = str(item.get("description", "")).strip()
                        if key:
                            resolved[key] = {"name": name, "desc": desc}
                    if resolved:
                        logger.info("Dynamically loaded %d labels from Core API schema", len(resolved))
                        return resolved
            except Exception as exc:
                logger.warning("Could not fetch remote label schema: %s. Using default labels.", exc)

        return DEFAULT_LABEL_DEFINITIONS

    def _submit_result(self, job_id: str, attempt: int, lease_token: str, body: dict[str, Any]) -> None:
        base = self.api_base_url
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{job_id}/attempts/{attempt}/action-result"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": lease_token,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        with httpx.Client(timeout=30.0, trust_env=self.trust_env) as client:
            resp = client.put(url, json=body, headers=headers)
            if resp.status_code not in (200, 204):
                raise RuntimeError(f"Failed to submit result to {url}: {resp.status_code} - {resp.text}")

    def infer_video(
        self,
        video_file: str,
        label_defs: dict[str, dict[str, str]] | None = None,
        allowed_labels: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        active_labels = label_defs if label_defs else DEFAULT_LABEL_DEFINITIONS
        total_duration = self._get_video_duration(video_file)
        duration_ms = max(1, int(round(total_duration * 1000)))
        chunks_plan = self._plan_chunks(total_duration)
        total_windows = max(1, len(chunks_plan))

        all_actions = []
        temp_chunk = os.path.join(_scratch_dir(), f"_tmp_chunk_{os.getpid()}.mp4")

        try:
            for cp in chunks_plan:
                w_idx = cp["window_idx"]
                s_start = cp["slice_start"]
                s_dur = cp["slice_dur"]
                l_core_start = cp["local_core_start"]
                l_core_end = cp["local_core_end"]

                if not self._extract_chunk(video_file, s_start, s_dur, temp_chunk):
                    raise RuntimeError(f"Window {w_idx} slice extraction failed.")

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
        # 1. 仅保留 6 个合法字段：start_ms, end_ms, label, confidence, source_window, stitched_windows
        # 2. 标签大小写与平台 allowed_labels 完全对齐 (如 "open_cap")
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

        if clean_segments:
            duration_ms = max(duration_ms, max(s["end_ms"] for s in clean_segments))

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
                    {"type": "video", "video": chunk_path, "fps": self.sampling_fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        output_text = self.processor.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        return self._parse_output(output_text, chunk_max, label_defs)

    def _build_prompt(self, label_defs: dict[str, dict[str, str]], chunk_max: int) -> str:
        lines = [
            "你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。",
            "",
            "标签定义与判据：",
        ]
        for key, info in label_defs.items():
            desc = f"（{info['desc']}）" if info.get("desc") else ""
            lines.append(f"- {key}: {info.get('name', key)}{desc}")
        first_key = next(iter(label_defs.keys())) if label_defs else "OPEN_CAP"
        lines.extend([
            "",
            "【判据与输出硬性约束】",
            "1. 实事求是：依据画面中护士手部实际操作的物理动作准确分类，仅标注画面中有明确视觉动作的区间；",
            "2. 严禁流程脑补：严禁脱离画面强行背诵或补全整套流程！若当前片段无明确动作、处于间歇或准备状态，必须直接输出空数组 []；",
            "3. 真实动作时长：起止时间严格以手部实际开始与结束操作的时间为准，严禁默认从 0 秒开始标记；",
            f"4. 起止时间必须为 0 到 {chunk_max} 之间的整数秒，start_sec < end_sec 且动作区间不可重叠；",
            "5. 格式严格为单个 JSON 数组（严禁任何 Markdown 或解释文字）：",
            f'示例 1 (有动作): [{{"start_sec": 2, "end_sec": 5, "label": "{first_key}", "confidence": 0.98}}]',
            "示例 2 (无动作): []",
        ])
        return "\n".join(lines)

    @staticmethod
    def _collect_raw_items(cleaned: str) -> list[Any]:
        """从模型输出里尽最大努力捞出候选条目，四种策略依次兜底。"""
        items: list[Any] = []

        def take(obj: Any) -> None:
            if isinstance(obj, list):
                for elem in obj:
                    if isinstance(elem, dict) or (isinstance(elem, (list, tuple)) and len(elem) >= 3):
                        items.append(elem)
            elif isinstance(obj, dict):
                items.append(obj)

        # 1) 配对完整的数组（含 [[...]] 嵌套、markdown 围栏、前后夹带说明文字）
        for group in _scan_balanced(cleaned, "[", "]"):
            take(_loads_lenient(group))
        if items:
            return items

        # 2) 配对完整的对象；{"actions": [...]} 这类外层再包一层的情况取内层数组
        for group in _scan_balanced(cleaned, "{", "}"):
            obj = _loads_lenient(group)
            if isinstance(obj, dict):
                nested = next((v for v in obj.values() if isinstance(v, list)), None)
                take(nested if nested is not None else obj)
        if items:
            return items

        # 3) 输出被 max_new_tokens 截断：补全未闭合的括号后重试
        for open_ch in ("[", "{"):
            idx = cleaned.find(open_ch)
            if idx >= 0:
                take(_repair_truncated(cleaned[idx:]))
                if items:
                    return items

        # 4) 逐行兜底（模型把每个动作单独放在一行时）
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
            if "[]" in cleaned:
                return []
            logger.warning("未能从模型输出中解析出任何动作区间。模型原文片段: %s", cleaned[:200])
            return []

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
                continue

            actions.append({
                "start_sec": s,
                "end_sec": e,
                "label": matched_key,
                "confidence": self._normalize_confidence(item.get("confidence", 0.9)),
            })
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
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-VL-8B-Instruct"))
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
        "--worker-id", default=os.getenv("AI_WORKER_ID") or os.getenv("WORKER_ID", "action-worker-4090-01"),
        help="Worker 实例标识 ID (默认: action-worker-4090-01，可用 AI_WORKER_ID / WORKER_ID 覆盖)",
    )
    parser.add_argument(
        "--bearer-token", default=os.getenv("AI_WORKER_BEARER_TOKEN") or os.getenv("PLATFORM_WORKER_BEARER_TOKEN", ""),
        help="访问 Core API 的 Bearer Token (默认从 AI_WORKER_BEARER_TOKEN / PLATFORM_WORKER_BEARER_TOKEN 读取)",
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
        help="单窗口生成上限 (默认: 512，可用 MAX_NEW_TOKENS 覆盖)。过小会截断输出并丢掉最后一个动作",
    )
    parser.add_argument(
        "--idle-offload-seconds", type=int, default=DEFAULT_IDLE_OFFLOAD_SEC,
        help="显存空闲自动卸载超时秒数: -1=永久常驻(17GB); 0=任务完成立即释放; >0=空闲N秒自动释放 (默认: 300，可用 IDLE_OFFLOAD_SECONDS 覆盖)",
    )
    args = parser.parse_args()

    _warn_on_env_proxy()
    logger.info(
        "配置: worker_id=%s queue=%s fps=%s heartbeat=%ss idle_offload=%ss trust_env=%s max_new_tokens=%s",
        args.worker_id, args.queue, args.fps, args.heartbeat, args.idle_offload_seconds, args.trust_env, args.max_new_tokens,
    )

    try:
        worker = ActionWorker(
            model_path=args.model,
            api_base_url=args.api_url,
            rabbitmq_url=args.rabbitmq_url,
            queue_name=args.queue,
            worker_id=args.worker_id,
            bearer_token=args.bearer_token,
            sampling_fps=args.fps,
            heartbeat_sec=args.heartbeat,
            blocked_connection_timeout=args.blocked_timeout,
            trust_env=args.trust_env,
            max_new_tokens=args.max_new_tokens,
            idle_offload_sec=args.idle_offload_seconds,
        )
        worker.run()
    except KeyboardInterrupt:
        pass
