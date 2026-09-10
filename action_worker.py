"""Action Candidate Worker Daemon for Qwen3-VL on RTX 4090.

1. Model resident in VRAM (zero cold start).
2. RabbitMQ consumer with prefetch_count=1 (strict serial execution, no concurrent OOM).
3. Frame-accurate chunked inference with cross-window continuation stitching.
4. Auto-lease renewal & result submission to Core API.
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
import shutil
import subprocess
import sys
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
from transformers import AutoProcessor, TextStreamer

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


# BlockingConnection 在回调里跑 ffmpeg + 多窗口推理时会阻塞数分钟，
# 期间无法 pump 事件循环，因此默认关闭 AMQP 心跳（由 TCP keepalive 兜底）。
DEFAULT_HEARTBEAT_SEC = _env_int("RABBITMQ_HEARTBEAT", 0)
DEFAULT_BLOCKED_TIMEOUT_SEC = _env_float("RABBITMQ_BLOCKED_TIMEOUT", 300.0)


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
    ) -> None:
        self.model_path = model_path
        self.api_base_url = api_base_url.rstrip("/")
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.worker_id = worker_id
        self.bearer_token = bearer_token
        self.sampling_fps = sampling_fps
        self.heartbeat_sec = heartbeat_sec
        self.blocked_connection_timeout = blocked_connection_timeout
        self._reconnect_backoff = RECONNECT_BACKOFF_MIN_SEC

        logger.info("Initializing Action Worker with model: %s", model_path)
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

        self.model = ModelClass.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="cuda",
        )
        self.model.eval()
        logger.info("Model loaded successfully and resident in VRAM (attn=%s).", attn_impl)

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
        try:
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            channel.basic_qos(prefetch_count=1)

            logger.info(
                "已连接 %s，等待队列 '%s' 的任务 (prefetch=1, heartbeat=%ss)...",
                connection.params.host, self.queue_name, self.heartbeat_sec,
            )

            for method, _properties, body in channel.consume(self.queue_name):
                self._handle_message(channel, method, body)
        finally:
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
            logger.info("收到任务: %s", job_id)
            self.process_job(payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("任务 %s 处理完成并 ACK。", job_id)
        except Exception as exc:
            logger.error("任务 %s 处理失败: %s", job_id, exc, exc_info=True)
            if method.redelivered:
                # 已经重投过一次仍然失败 -> 停止 requeue，避免毒药消息无限死循环
                logger.error("任务 %s 重投后仍失败，nack(requeue=False)，交由 DLX 或丢弃。", job_id)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            else:
                logger.warning("任务 %s 将重投一次 (requeue=True)。", job_id)
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def process_job(self, payload: dict[str, Any]) -> None:
        job_id = payload["job_id"]
        attempt = payload.get("attempt", 1)
        lease_token = payload.get("lease_token", "")
        video_url = payload.get("video_url", "")
        video_path = payload.get("video_path", "")

        local_video = video_path
        tmp_download = None
        if not local_video or not os.path.exists(local_video):
            if not video_url:
                raise ValueError("Neither valid local video_path nor video_url provided.")
            tmp_download = f"_download_{job_id}.mp4"
            logger.info("Downloading video from %s to %s", video_url, tmp_download)
            with httpx.stream("GET", video_url, timeout=120.0) as resp:
                resp.raise_for_status()
                with open(tmp_download, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            local_video = tmp_download

        label_defs = self._resolve_label_definitions(payload)
        started_at = datetime.now(UTC).isoformat()
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        try:
            segments = self.infer_video(local_video, label_defs=label_defs)
            duration_ms = int((time.time() - start_time) * 1000)
            peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

            req_body = {
                "message_id": str(uuid.uuid4()),
                "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                "model_revision": "b9687d6",
                "sampling_fps": int(self.sampling_fps),
                "peak_memory_mb": peak_mb,
                "inference_time_ms": duration_ms,
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "segments": segments,
            }

            self._submit_result(job_id, attempt, lease_token, req_body)
        finally:
            if tmp_download and os.path.exists(tmp_download):
                try:
                    os.remove(tmp_download)
                except Exception:
                    pass
            # 释放推理过程中申请的瞬时动态显存与碎片
            torch.cuda.empty_cache()

    def _resolve_label_definitions(self, payload: dict[str, Any]) -> dict[str, dict[str, str]]:
        if "labels" in payload and isinstance(payload["labels"], list) and payload["labels"]:
            resolved: dict[str, dict[str, str]] = {}
            for item in payload["labels"]:
                key = str(item.get("id", item.get("label_id", ""))).upper().strip()
                name = str(item.get("name", item.get("display_name", ""))).strip()
                desc = str(item.get("criteria", item.get("description", ""))).strip()
                if key:
                    resolved[key] = {"name": name or key, "desc": desc}
            if resolved:
                return resolved

        project_id = payload.get("project_id")
        if project_id and self.api_base_url:
            try:
                headers = {}
                if self.bearer_token:
                    headers["Authorization"] = f"Bearer {self.bearer_token}"
                base = self.api_base_url.rstrip("/")
                if not base.endswith("/api/v1"):
                    base = f"{base}/api/v1"
                url = f"{base}/projects/{project_id}/annotation-label-schema"
                resp = httpx.get(url, headers=headers, timeout=5.0)
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
        base = self.api_base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        url = f"{base}/internal/ai-jobs/{job_id}/attempts/{attempt}/action-result"
        headers = {
            "Content-Type": "application/json",
            "X-AI-Lease-Token": lease_token,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        with httpx.Client(timeout=30.0) as client:
            resp = client.put(url, json=body, headers=headers)
            if resp.status_code not in (200, 204):
                raise RuntimeError(f"Failed to submit result to {url}: {resp.status_code} - {resp.text}")

    def infer_video(
        self, video_file: str, label_defs: dict[str, dict[str, str]] | None = None
    ) -> list[dict[str, Any]]:
        active_labels = label_defs if label_defs else DEFAULT_LABEL_DEFINITIONS
        total_duration = self._get_video_duration(video_file)
        chunks_plan = self._plan_chunks(total_duration)

        all_actions = []
        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else "."
        temp_chunk = os.path.join(shm_dir, f"_tmp_chunk_{os.getpid()}.mp4")

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
                prev_local_end = 0

                for item in chunk_actions:
                    local_s = int(item["start_sec"])
                    local_e = int(item["end_sec"])
                    label = str(item["label"]).strip()
                    conf = float(item["confidence"])
                    prev_local_end = local_e

                    g_s = max(0, min(int(round(s_start + local_s)), int(round(total_duration))))
                    g_e = max(g_s + 1, min(int(round(s_start + local_e)), int(round(total_duration))))

                    stitched = False
                    if all_actions:
                        last_act = all_actions[-1]
                        if (
                            last_act["label"] == label
                            and last_act["end_sec"] >= int(round(cp["core_abs_start"])) - 1
                            and g_s <= last_act["end_sec"] + 1
                        ):
                            last_act["end_sec"] = max(last_act["end_sec"], g_e)
                            last_act["confidence"] = round(max(last_act["confidence"], conf), 2)
                            if w_idx not in last_act["stitched_windows"]:
                                last_act["stitched_windows"].append(w_idx)
                            stitched = True

                    if stitched:
                        continue

                    if l_core_start <= local_s < l_core_end:
                        all_actions.append({
                            "start_sec": g_s,
                            "end_sec": g_e,
                            "label": label,
                            "confidence": round(conf, 2),
                            "source_window": w_idx,
                            "stitched_windows": [w_idx],
                        })
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
            if act["label"] == prev["label"] and act["start_sec"] <= prev["end_sec"] + 1:
                prev["end_sec"] = max(prev["end_sec"], act["end_sec"])
                prev["confidence"] = round(max(prev["confidence"], act["confidence"]), 2)
                continue
            if act["start_sec"] < prev["end_sec"]:
                mid = (prev["end_sec"] + act["start_sec"]) // 2
                prev["end_sec"] = max(prev["start_sec"] + 1, mid)
                act["start_sec"] = max(prev["end_sec"], mid)
            if act["end_sec"] > act["start_sec"]:
                validated.append(act)

        # 毫秒与秒双契约导出
        result_segments = []
        for act in validated:
            s_s = int(act["start_sec"])
            e_s = int(act["end_sec"])
            result_segments.append({
                "start_ms": s_s * 1000,
                "end_ms": e_s * 1000,
                "label": act["label"],
                "confidence": act["confidence"],
                "source_window": act["source_window"],
                "stitched_windows": act.get("stitched_windows", [act["source_window"]]),
            })
        return result_segments

    def _infer_single_window(
        self, chunk_path: str, dur_sec: float, label_defs: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        chunk_max = int(round(dur_sec))
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
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
        output_text = self.processor.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        return self._parse_output(output_text, chunk_max, label_defs)

    def _build_prompt(self, label_defs: dict[str, dict[str, str]], chunk_max: int) -> str:
        lines = [
            "你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。",
            "标签定义与判据：",
        ]
        for key, info in label_defs.items():
            desc = f"（{info['desc']}）" if info.get("desc") else ""
            lines.append(f"- {key}: {info.get('name', key)}{desc}")
        first_key = next(iter(label_defs.keys())) if label_defs else "OPEN_CAP"
        lines.append(f"【约束】严禁脑补；无明确动作直接输出 []；起止时间为 0 到 {chunk_max} 之间的整数秒；start_sec < end_sec；格式为 JSON 数组：")
        lines.append(f'[{{"start_sec": 2, "end_sec": 5, "label": "{first_key}", "confidence": 0.98}}]')
        return "\n".join(lines)

    def _parse_output(
        self, text: str, chunk_max: int, label_defs: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        if text.strip() == "[]":
            return []
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    actions = []
                    for item in data:
                        if isinstance(item, dict) and "start_sec" in item and "label" in item:
                            s = int(item["start_sec"])
                            e = int(item["end_sec"])
                            raw_lbl = str(item["label"]).strip()
                            matched_key = None
                            for k in label_defs:
                                if k.upper() == raw_lbl.upper():
                                    matched_key = k
                                    break
                            if matched_key and 0 <= s < e <= chunk_max:
                                actions.append({
                                    "start_sec": s,
                                    "end_sec": e,
                                    "label": matched_key,
                                    "confidence": float(item.get("confidence", 0.9)),
                                })
                    return actions
            except Exception:
                pass
        return []

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
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass
        return 30.0

    def _extract_chunk(self, src: str, s_sec: float, dur_sec: float, out: str) -> bool:
        cmd = ["ffmpeg", "-y", "-ss", f"{s_sec:.3f}", "-t", f"{dur_sec:.3f}", "-i", src, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Candidate 4090 Worker")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--api-url", default=os.getenv("PLATFORM_API_URL", "http://localhost:8080"))
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
    args = parser.parse_args()

    try:
        worker = ActionWorker(
            model_path=args.model,
            api_base_url=args.api_url,
            rabbitmq_url=args.rabbitmq_url,
            queue_name=args.queue,
            sampling_fps=args.fps,
            heartbeat_sec=args.heartbeat,
            blocked_connection_timeout=args.blocked_timeout,
        )
        worker.run()
    except KeyboardInterrupt:
        pass
