"""Qwen3.5-4B full-video action benchmark with bounded overlapping windows.

This script is intentionally offline: it does not connect to RabbitMQ or Core API.
It uses the same 24s core + 3s context windowing policy as the production worker,
so a long video can be evaluated without putting the whole video into one prompt.
If a window remains truncated after one bounded retry, its core interval is split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from label_schema import (
    DEFAULT_LABEL_DEFINITIONS,
    MAX_ACQUIRE_DURATION_SEC,
    RECOGNIZED_ITEM_TYPES,
)


CORE_CHUNK_SEC = 24.0
CONTEXT_PAD_SEC = 3.0
MIN_SPLIT_CORE_SEC = 6.0
CONTINUATION_GAP_SEC = 0
MAX_ACQUIRE_DURATION_SEC = 5
PIXELS_PER_UNIT = 28 * 28
RETRY_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = RETRY_MAX_NEW_TOKENS


def _token_limits(initial: int) -> list[int]:
    """返回有限的生成上限序列：初始值失败后最多扩大到 1024。"""
    limits = [initial]
    if initial < RETRY_MAX_NEW_TOKENS:
        limits.append(RETRY_MAX_NEW_TOKENS)
    return limits

LABEL_DEFINITIONS = DEFAULT_LABEL_DEFINITIONS


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[:300]}")
    return float(result.stdout.strip())


def extract_chunk(src: str, start_sec: float, duration_sec: float, output_path: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
            "-i", src, "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18", output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not Path(output_path).is_file() or Path(output_path).stat().st_size <= 1000:
        raise RuntimeError(f"ffmpeg chunk failed: {result.stderr.strip()[-500:]}")


def _make_chunk_plan(
    window_idx: int, core_start: float, core_end: float, total_duration: float,
) -> dict[str, float | int]:
    slice_start = max(0.0, core_start - CONTEXT_PAD_SEC)
    slice_end = min(total_duration, core_end + CONTEXT_PAD_SEC)
    slice_duration = slice_end - slice_start
    if slice_duration > 30.001:
        raise RuntimeError(f"window {window_idx} is {slice_duration:.3f}s, over the 30s limit")
    return {
        "window_idx": window_idx,
        "slice_start": slice_start,
        "slice_duration": slice_duration,
        "core_start": core_start,
        "core_end": core_end,
        "local_core_start": int(round(core_start - slice_start)),
        "local_core_end": int(round(core_end - slice_start)),
    }


def plan_chunks(total_duration: float) -> list[dict[str, float | int]]:
    plans: list[dict[str, float | int]] = []
    core_start = 0.0
    window_idx = 1
    while core_start < total_duration:
        remaining = total_duration - core_start
        core_end = total_duration if remaining <= CORE_CHUNK_SEC + 2.0 else core_start + CORE_CHUNK_SEC
        plans.append(_make_chunk_plan(window_idx, core_start, core_end, total_duration))
        core_start = core_end
        window_idx += 1
    return plans


def split_chunk_plan(
    plan: dict[str, float | int], total_duration: float, first_window_idx: int,
) -> list[dict[str, float | int]]:
    """将输出过长的核心区间二分，避免单纯增大生成上限。"""
    core_start = float(plan["core_start"])
    core_end = float(plan["core_end"])
    if core_end - core_start <= MIN_SPLIT_CORE_SEC:
        return []

    midpoint = int(round(core_start + (core_end - core_start) / 2.0))
    if not core_start < midpoint < core_end:
        return []
    return [
        _make_chunk_plan(first_window_idx, core_start, float(midpoint), total_duration),
        _make_chunk_plan(first_window_idx + 1, float(midpoint), core_end, total_duration),
    ]


def build_prompt(chunk_max: int) -> str:
    lines = [
        "你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。",
        "",
        f"【视频切片有效时间范围：0 到 {chunk_max} 秒】",
        f"重要：本视频切片总时长仅为 {chunk_max} 秒！所有 start_sec 和 end_sec 必须在 [0, {chunk_max}] 范围内，绝对严禁输出任何大于 {chunk_max} 的秒数！",
        "",
        "动作标签定义与判据：",
    ]
    for key, info in LABEL_DEFINITIONS.items():
        lines.append(f"- {key}: {info['name']}（{info['desc']}）")

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
        "1. 物理连续性与密度约束（核心铁律）：",
        "   - 配药操作是连贯的物理过程，单个动作通常持续 2 到 8 秒；",
        "   - 严禁以 1 秒为周期机械交替输出（绝对严禁每秒交替 acquire_item -> discard_sharps 等高频循环）；",
        "   - 宁缺毋滥：手部若仅在固定持物、悬停、调整握姿或空闲等待，不要进行标记；单个 20~30 秒切片通常仅含 1~4 个主要动作。若无明确操作直接输出空数组 []；",
        "2. 动作与受体区分：",
        "   - acquire_item: 必须是手伸向原静止物品并将其拿起进入操作区的【瞬态短动作】（通常仅 1~3 秒），输出时必须携带 item 字段注明拿取的物品；手持续握持不动、拆包装或未使物体发生位移时绝不可标为 acquire_item；",
        "   - 动作严格区分: 只有看到针头实际进入容器才标 insert_needle；只有看到活塞回抽且药液进入注射器才标 draw_liquid；只有向目标容器推注才标 transfer_liquid；看到掰开安瓿或取下盖子标 open_container；必须伴随锐器投入利器盒才标 discard_sharps；",
        f"3. 起止时间必须为 0 到 {chunk_max} 之间的整数秒，start_sec < end_sec 且动作区间不可重叠；",
        "4. 格式严格为紧凑的单个 JSON 数组（严禁任何 Markdown 围栏、额外换行缩进或解释文字，所有动作紧凑排在同一数组中）：",
        '示例 1 (有动作): [{"start_sec": 2, "end_sec": 5, "label": "open_container", "confidence": 0.95}, {"start_sec": 6, "end_sec": 9, "label": "acquire_item", "item": "syringe", "confidence": 0.92}]',
        "示例 2 (无动作/空镜/等待): []",
    ])
    return "\n".join(lines)


_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> list[str]:
    groups: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            if depth == 0:
                start = index
            depth += 1
        elif char == close_ch and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(text[start:index + 1])
                start = -1
    return groups


def _loads_lenient(snippet: str) -> Any | None:
    """Load JSON while tolerating a trailing comma from model output."""
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", snippet)
        if repaired != snippet:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None


def _coerce_action_value(value: Any) -> list[dict[str, Any]] | None:
    """Normalize arrays, compact rows, and an optional actions wrapper."""
    if isinstance(value, dict):
        if {"start_sec", "end_sec", "label"}.issubset(value):
            item = dict(value)
            item.setdefault("confidence", 0.9)
            return [item]
        nested = next(
            (candidate for key, candidate in value.items()
             if key.lower() in {"actions", "segments", "results", "data"}
             and isinstance(candidate, list)),
            None,
        )
        if nested is None:
            nested = next((candidate for candidate in value.values() if isinstance(candidate, list)), None)
        return _coerce_action_value(nested) if nested is not None else None

    if not isinstance(value, list):
        return None
    if not value:
        return []
    if all(isinstance(item, dict) for item in value):
        normalized = []
        for item in value:
            copied = dict(item)
            copied.setdefault("confidence", 0.9)
            normalized.append(copied)
        return normalized
    if all(isinstance(item, (list, tuple)) and len(item) >= 3 for item in value):
        return [
            {
                "start_sec": item[0],
                "end_sec": item[1],
                "label": item[2],
                "confidence": item[3] if len(item) > 3 else 0.9,
            }
            for item in value
        ]
    return None


def parse_model_output(raw_output: str) -> list[dict[str, Any]] | None:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    for candidate in _scan_balanced(cleaned, "[", "]"):
        parsed = _coerce_action_value(_loads_lenient(candidate))
        if parsed is not None:
            # Empty output is accepted only as the complete response (or as
            # a complete wrapper handled below); prose containing "[]" is
            # not enough to claim that no action occurred.
            if parsed == [] and candidate.strip() != cleaned:
                continue
            return parsed

    whole_value = _coerce_action_value(_loads_lenient(cleaned))
    if whole_value is not None:
        return whole_value

    # If an opening array remains unmatched, do not harvest complete inner
    # objects: that would silently accept only the prefix before truncation.
    if "[" in cleaned:
        return None
    if "{" in cleaned and not _scan_balanced(cleaned, "{", "}"):
        return None

    # Some generations return one complete action object instead of a
    # one-element array.  A complete {"actions": [...]} wrapper is already
    # handled by the array pass above.
    for candidate in _scan_balanced(cleaned, "{", "}"):
        parsed = _coerce_action_value(_loads_lenient(candidate))
        if parsed is not None:
            return parsed

    return None


def write_raw_snapshot(raw_path: Path, raw_sections: list[str]) -> None:
    """Persist completed windows so a failed long-video run remains inspectable."""
    try:
        raw_path.write_text("\n".join(raw_sections), encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] unable to save raw outputs to {raw_path}: {exc}", file=sys.stderr)


def generate_output_text(model: Any, processor: Any, inputs: Any, max_new_tokens: int) -> str:
    """Generate one response and decode only newly generated tokens."""
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    output_text = processor.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()
    del output_ids
    return output_text


def validate_local_actions(items: list[dict[str, Any]], chunk_max: int) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        required = ("start_sec", "end_sec", "label", "confidence")
        if not all(field in item for field in required):
            raise RuntimeError(f"action {index} is missing a required field: {item}")
        try:
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            confidence = float(item["confidence"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"action {index} contains a non-numeric field: {item}") from exc
        label = str(item["label"]).strip()
        if not all(math.isfinite(value) for value in (start, end, confidence)):
            raise RuntimeError(f"action {index} contains a non-finite value: {item}")
        if not start.is_integer() or not end.is_integer():
            raise RuntimeError(f"action {index} is not aligned to integer seconds: {item}")
        start_i, end_i = int(start), int(end)
        if not 0 <= start_i < end_i <= chunk_max:
            raise RuntimeError(f"action {index} is outside the window: {item}")
        if label not in LABEL_DEFINITIONS:
            raise RuntimeError(f"action {index} has an unknown label: {label}")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"action {index} has invalid confidence: {confidence}")
        # 针对 acquire_item 瞬态动作进行时长上限夹紧
        if label == "acquire_item":
            max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
            if (end_i - start_i) > max_dur:
                end_i = start_i + max_dur

        validated_item: dict[str, Any] = {
            "start_sec": start_i,
            "end_sec": end_i,
            "label": label,
            "confidence": round(confidence, 2),
        }
        raw_item = str(item.get("item", "")).strip().lower()
        if raw_item:
            validated_item["item"] = raw_item
        validated.append(validated_item)
    # 模型可能把相邻动作报成轻微重叠或乱序；这是候选结果的常见边界误差，
    # 由后面的 merge_actions 统一排序并按确定规则整理，不在这里终止整段视频。
    return sorted(validated, key=lambda item: (item["start_sec"], item["end_sec"], item["label"]))


def _suppress_micro_oscillations(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """消除高频交替的 <=1 秒微脉冲噪声（自回归震荡退化产生的密集交替碎片）。"""
    if len(actions) < 3:
        return actions

    is_micro = [int(a["end_sec"]) - int(a["start_sec"]) <= 1 for a in actions]
    suppressed = []
    i = 0
    while i < len(actions):
        if is_micro[i]:
            j = i
            while j < len(actions) and is_micro[j]:
                # 严格时序连续性：震荡链相邻动作之间的时间间隔必须 <= 1 秒；
                # 彼此独立的远距离 1 秒短动作严禁跨越数十秒时间段被强行串联成巨段
                if j > i and int(actions[j]["start_sec"]) > int(actions[j - 1]["end_sec"]) + 1:
                    break
                j += 1
            chain_len = j - i
            unique_labels = {a["label"] for a in actions[i:j]}
            if chain_len >= 3 and len(unique_labels) > 1:
                best = max(actions[i:j], key=lambda x: float(x.get("confidence", 0.0)))
                best_copy = dict(best)
                best_copy["start_sec"] = actions[i]["start_sec"]
                best_copy["end_sec"] = actions[j - 1]["end_sec"]
                if best_copy["label"] == "acquire_item":
                    max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
                    if best_copy["end_sec"] - best_copy["start_sec"] > max_dur:
                        best_copy["end_sec"] = best_copy["start_sec"] + max_dur
                windows = best_copy.setdefault("stitched_windows", [best_copy.get("source_window", 1)])
                for act_k in actions[i:j]:
                    for w in act_k.get("stitched_windows", [act_k.get("source_window", 1)]):
                        if w not in windows:
                            windows.append(w)
                suppressed.append(best_copy)
                i = j
                continue
        suppressed.append(actions[i])
        i += 1
    return suppressed


def merge_actions(actions: list[dict[str, Any]], total_duration: float) -> list[dict[str, Any]]:
    actions.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged: list[dict[str, Any]] = []
    for action in actions:
        action["start_sec"] = max(0, min(int(action["start_sec"]), int(round(total_duration))))
        action["end_sec"] = max(action["start_sec"] + 1, min(int(action["end_sec"]), int(round(total_duration))))
        if action["end_sec"] <= action["start_sec"]:
            continue
        if not merged:
            merged.append(action)
            continue
        previous = merged[-1]

        can_merge_acquire = True
        if action["label"] == "acquire_item":
            max_dur = globals().get("MAX_ACQUIRE_DURATION_SEC", 5)
            if max(previous["end_sec"], action["end_sec"]) - min(previous["start_sec"], action["start_sec"]) > max_dur:
                can_merge_acquire = False
            if previous.get("item") and action.get("item") and previous["item"] != action["item"]:
                can_merge_acquire = False

        if (
            action["label"] == previous["label"]
            and action["start_sec"] <= previous["end_sec"] + CONTINUATION_GAP_SEC
            and can_merge_acquire
        ):
            previous["end_sec"] = max(previous["end_sec"], action["end_sec"])
            previous["confidence"] = round(max(previous["confidence"], action["confidence"]), 2)
            if not previous.get("item") and action.get("item"):
                previous["item"] = action["item"]
            prev_windows = previous.setdefault("stitched_windows", [previous.get("source_window", 1)])
            for window in action.get("stitched_windows", [action.get("source_window", 1)]):
                if window not in prev_windows:
                    prev_windows.append(window)
            continue
        if action["start_sec"] == previous["start_sec"]:
            # 同起点冲突候选：二者在同一时刻触发，保留置信度更高者，避免切出虚假中点
            if action["confidence"] > previous["confidence"]:
                merged[-1] = action
            continue
        if action["start_sec"] < previous["end_sec"]:
            if action["end_sec"] > previous["end_sec"]:
                # 延伸型重叠：两条候选都携带时间证据，按中点切分，避免
                # 因模型边界误差直接丢掉后一个动作。
                midpoint = (previous["end_sec"] + action["start_sec"]) // 2
                previous["end_sec"] = max(previous["start_sec"] + 1, midpoint)
                action["start_sec"] = max(previous["end_sec"], midpoint)
            elif action["confidence"] > previous["confidence"] + 0.15:
                # 包含型重叠：只有新候选明显更可信时，才让它取代旧片段的尾部。
                previous["end_sec"] = action["start_sec"]
            else:
                continue
        if action["end_sec"] > action["start_sec"]:
            merged.append(action)
    return _suppress_micro_oscillations(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 完整视频动作离线切片评测")
    parser.add_argument("video", nargs="?", default="video1.mp4")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.5-4B"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="单窗口生成上限；疑似截断时重试 1024，仍失败则二分核心窗口",
    )
    parser.add_argument(
        "--pixel-scale", type=float, default=1.0,
        help="视觉像素预算倍率（默认 1.0；可用 0.5/1/2 做效果-速度对照）",
    )
    parser.add_argument(
        "--use-hub-kernels", action="store_true",
        help="实验性启用 Qwen3.5 Hub kernels；需在独立环境安装并回归",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser().resolve())
    video_path = str(Path(args.video).expanduser().resolve())
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.max_new_tokens < 64:
        raise ValueError("--max-new-tokens must be at least 64")
    if args.pixel_scale <= 0:
        raise ValueError("--pixel-scale must be greater than zero")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    total_duration = get_video_duration(video_path)
    plans = plan_chunks(total_duration)
    output_path = Path(args.output or f"actions_qwen35_{Path(video_path).stem}_chunked.json").expanduser().resolve()
    raw_path = output_path.with_suffix(".raw.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
    print(f"model: {model_path}")
    print(f"video: {video_path} | duration: {total_duration:.2f}s | fps: {args.fps} | pixel_scale: {args.pixel_scale}")
    print(f"labels: {', '.join(LABEL_DEFINITIONS)}")
    print(f"windows: {len(plans)} (core={CORE_CHUNK_SEC:.0f}s, context=+/-{CONTEXT_PAD_SEC:.0f}s)")
    for plan in plans:
        print(
            f"  window {plan['window_idx']}: "
            f"slice={plan['slice_start']:.1f}-{float(plan['slice_start']) + float(plan['slice_duration']):.1f}s, "
            f"core={plan['core_start']:.1f}-{plan['core_end']:.1f}s"
        )

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=int(128 * PIXELS_PER_UNIT * args.pixel_scale),
        max_pixels=int(256 * PIXELS_PER_UNIT * args.pixel_scale),
    )
    model_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.use_hub_kernels:
        model_kwargs.update(use_kernels=True, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()
    load_sec = time.perf_counter() - load_start
    print(f"model load: {load_sec:.2f}s")

    temp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else "."
    temp_chunk = os.path.join(temp_dir, f"_tmp_qwen35_chunk_{os.getpid()}.mp4")
    collected: list[dict[str, Any]] = []
    raw_sections: list[str] = []
    pending_plans = deque(plans)
    next_window_idx = len(plans) + 1
    inference_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    try:
        while pending_plans:
            plan = pending_plans.popleft()
            window_idx = int(plan["window_idx"])
            slice_start = float(plan["slice_start"])
            slice_duration = float(plan["slice_duration"])
            local_core_start = int(plan["local_core_start"])
            local_core_end = int(plan["local_core_end"])
            chunk_max = max(1, int(round(slice_duration)))
            print(
                f"\n--- window {window_idx}: "
                f"slice={slice_start:.1f}-{slice_start + slice_duration:.1f}s ---"
            )
            extract_chunk(video_path, slice_start, slice_duration, temp_chunk)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": temp_chunk},
                    {"type": "text", "text": build_prompt(chunk_max)},
                ],
            }]
            prepare_start = time.perf_counter()
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "fps": args.fps,
                    "do_sample_frames": True,
                    "cap_pixels_per_frame": True,
                },
                enable_thinking=False,
            ).to("cuda")
            prepare_sec = time.perf_counter() - prepare_start

            token_limits = _token_limits(args.max_new_tokens)
            infer_sec = 0.0
            parsed: list[dict[str, Any]] | None = None
            output_text = ""
            for attempt, token_limit in enumerate(token_limits, 1):
                attempt_start = time.perf_counter()
                output_text = generate_output_text(model, processor, inputs, token_limit)
                infer_sec += time.perf_counter() - attempt_start
                raw_sections.append(
                    f"===== window {window_idx} attempt {attempt} "
                    f"max_new_tokens={token_limit} =====\n{output_text}\n"
                )
                write_raw_snapshot(raw_path, raw_sections)
                parsed = parse_model_output(output_text)
                if parsed is not None:
                    break
                if attempt < len(token_limits):
                    print(
                        f"window {window_idx}: output is incomplete or not JSON; "
                        f"retrying with max_new_tokens={token_limits[attempt]}"
                    )
            if parsed is None:
                children = split_chunk_plan(plan, total_duration, next_window_idx)
                if children:
                    next_window_idx += len(children)
                    pending_plans.extendleft(reversed(children))
                    del inputs
                    torch.cuda.empty_cache()
                    print(
                        f"window {window_idx}: output is still incomplete after "
                        f"{len(token_limits)} attempt(s); split core "
                        f"{float(plan['core_start']):.1f}-{float(plan['core_end']):.1f}s"
                    )
                    continue
                del inputs
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"window {window_idx} output is not a complete JSON array "
                    f"after {len(token_limits)} attempt(s), and its core interval "
                    f"is already <= {MIN_SPLIT_CORE_SEC:.0f}s"
                )
            local_actions = validate_local_actions(parsed, chunk_max)
            accepted = 0
            for item in local_actions:
                local_start = item["start_sec"]
                local_end = item["end_sec"]
                global_start = int(round(slice_start + local_start))
                global_end = int(round(slice_start + local_end))
                if local_core_start <= local_start < local_core_end:
                    accepted_start = global_start
                    accepted_end = global_end
                elif local_start < local_core_start < local_end:
                    accepted_start = int(round(float(plan["core_start"])))
                    accepted_end = global_end
                else:
                    continue
                if accepted_end <= accepted_start:
                    continue
                coll_entry: dict[str, Any] = {
                    "start_sec": accepted_start,
                    "end_sec": accepted_end,
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "source_window": window_idx,
                    "stitched_windows": [window_idx],
                }
                if "item" in item and item["item"]:
                    coll_entry["item"] = item["item"]
                collected.append(coll_entry)
                accepted += 1
            print(
                f"window {window_idx}: actions={len(local_actions)}, accepted={accepted}, "
                f"prepare={prepare_sec:.2f}s, inference={infer_sec:.2f}s"
            )
            del inputs
            torch.cuda.empty_cache()
    except Exception:
        write_raw_snapshot(raw_path, raw_sections)
        print(f"partial raw outputs saved: {raw_path}", file=sys.stderr)
        raise
    finally:
        if os.path.exists(temp_chunk):
            os.remove(temp_chunk)

    actions = merge_actions(collected, total_duration)
    export = []
    for item in actions:
        start_sec = int(item["start_sec"])
        end_sec = int(item["end_sec"])
        exp_entry: dict[str, Any] = {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_ms": start_sec * 1000,
            "end_ms": end_sec * 1000,
            "duration_ms": (end_sec - start_sec) * 1000,
            "label": item["label"],
            "confidence": item["confidence"],
            "source_window": item["source_window"],
            "stitched_windows": item["stitched_windows"],
        }
        if "item" in item and item["item"]:
            exp_entry["item"] = item["item"]
            exp_entry["item_name"] = RECOGNIZED_ITEM_TYPES.get(item["item"], item["item"])
        export.append(exp_entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_path.write_text("\n".join(raw_sections), encoding="utf-8")

    inference_sec = time.perf_counter() - inference_start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print("\n========== Qwen3.5 full-video result ==========")
    print(f"actions: {len(export)}")
    print(f"inference total (excluding model load): {inference_sec:.2f}s")
    print(f"real-time factor: {total_duration / inference_sec:.2f}x")
    print(f"peak VRAM after reset: {peak_vram:.2f} GB")
    print(f"saved JSON: {output_path}")
    print(f"saved raw outputs: {raw_path}")
    print(json.dumps(export, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
