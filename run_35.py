"""Qwen3.5-4B standalone video/action smoke test.

This script deliberately bypasses RabbitMQ, Core API and action_worker.py.  It
only verifies the Qwen3.5 Transformers path, local video decoding, the action
prompt, JSON output, latency and GPU memory before production integration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from label_schema import DEFAULT_LABEL_DEFINITIONS


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


def build_prompt(chunk_max: int) -> str:
    lines = [
        "你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。",
        "",
        "标签定义与判据：",
    ]
    for key, info in LABEL_DEFINITIONS.items():
        lines.append(f"- {key}: {info['name']}（{info['desc']}）")
    first_key = next(iter(LABEL_DEFINITIONS), "open_container")
    lines.extend([
        "",
        "【判据与输出硬性约束】",
        "1. 仅标注画面中有明确视觉动作的区间；无法确认时输出 []。",
        "2. 严禁脱离画面背诵或补全配药流程。",
        "3. 只有看到针头实际进入容器才标 insert_needle；只有看到活塞回抽且药液进入注射器才标 draw_liquid；只有看到向另一容器推注才标 transfer_liquid。",
        "4. 起止时间以手部实际开始和结束操作为准，严禁默认从 0 秒开始。",
        f"5. 起止时间必须为 0 到 {chunk_max} 之间的整数秒，start_sec < end_sec，区间不可重叠。",
        "6. 只能输出单个合法 JSON 数组，严禁 Markdown、解释文字或思考过程。",
        f'- 有动作示例：[{{"start_sec":2,"end_sec":5,"label":"{first_key}","confidence":0.98}}]',
        "- 无动作示例：[]",
    ])
    return "\n".join(lines)


def parse_action_json(raw_output: str) -> tuple[object | None, str]:
    """解析模型动作 JSON，并兼容模型偶尔返回的 Markdown 代码围栏。"""
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 兼容模型在 JSON 前后添加一句解释；生产 worker 自身还有更宽松的
        # 区间扫描器，这里只做冒烟测试需要的最小容错。
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            return None, "FAIL (no JSON array found)"
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            return None, f"FAIL ({exc})"

    if not isinstance(parsed, list):
        return parsed, "FAIL (root is not a list)"
    return parsed, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 视频动作离线冒烟测试")
    parser.add_argument("video", nargs="?", default="test_30s.mp4")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.5-4B"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--pixel-scale", type=float, default=1.0,
        help="视觉像素预算倍率（默认 1.0；可用 0.5/1/2 做效果-速度对照）",
    )
    parser.add_argument(
        "--use-hub-kernels", action="store_true",
        help="实验性启用 Qwen3.5 Hub kernels；需在独立环境安装并回归",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="解析成功后保存动作数组的 JSON 路径（默认不落盘）",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model))
    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.max_new_tokens < 64:
        raise ValueError("--max-new-tokens must be at least 64")
    if args.pixel_scale <= 0:
        raise ValueError("--pixel-scale must be greater than zero")

    duration = get_video_duration(video_path)
    chunk_max = max(1, int(round(duration)))
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
    print(f"model: {model_path}")
    print(f"video: {video_path} | duration: {duration:.2f}s | fps: {args.fps} | pixel_scale: {args.pixel_scale}")
    print(f"labels: {', '.join(LABEL_DEFINITIONS)}")

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path,
        # 与 Qwen3-VL-8B 离线基线保持相同的视觉像素预算，避免比较时
        # 把输入分辨率差异误判成模型能力或速度差异。
        min_pixels=int(128 * 28 * 28 * args.pixel_scale),
        max_pixels=int(256 * 28 * 28 * args.pixel_scale),
    )
    model_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.use_hub_kernels:
        model_kwargs.update(use_kernels=True, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()
    load_sec = time.perf_counter() - load_start
    print(f"model load: {load_sec:.2f}s")

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": build_prompt(chunk_max)},
        ],
    }]

    torch.cuda.reset_peak_memory_stats()
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
            # 采用 qwen-vl-utils 的逐帧像素上限，避免高分辨率视频产生过多视觉 token。
            "cap_pixels_per_frame": True,
        },
        enable_thinking=False,
    ).to("cuda")
    prepare_sec = time.perf_counter() - prepare_start

    infer_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    infer_sec = time.perf_counter() - infer_start

    prompt_len = inputs["input_ids"].shape[-1]
    raw_output = processor.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()
    parsed, json_status = parse_action_json(raw_output)

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print("\n========== Qwen3.5 video/action result ==========")
    print(f"JSON status: {json_status}")
    print("raw output:")
    print(raw_output)
    if parsed is not None:
        print("parsed output:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        if isinstance(parsed, list) and args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"saved JSON: {output_path}")
    print(f"prepare: {prepare_sec:.2f}s")
    print(f"inference: {infer_sec:.2f}s")
    print(f"total (excluding model load): {prepare_sec + infer_sec:.2f}s")
    print(f"peak VRAM after model load: {peak_vram:.2f} GB")
    return 0 if json_status == "OK" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
