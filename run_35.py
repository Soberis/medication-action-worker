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


LABEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "acquire_item": {"name": "拿取容器/工具", "desc": "手明确到达并取得将参与配药的容器或工具，物体离开原稳定位置并进入操作区；不标持续持有、调整位置或重复抓握"},
    "open_container": {"name": "开启容器", "desc": "明确打开药品主容器，如掰开安瓿或移除西林瓶/输液容器的保护部件；不包括拆注射器包装"},
    "disinfect_access": {"name": "消毒容器接口", "desc": "看到棉签或其他消毒用品实际擦拭瓶塞、安瓿颈或容器接口；仅出现用品不构成该动作"},
    "insert_needle": {"name": "针头进入容器", "desc": "清楚看到针头穿过瓶塞或接口并进入目标药品容器；仅靠近或被遮挡时不标注"},
    "draw_liquid": {"name": "抽取药液", "desc": "针头已进入容器且活塞向后移动，能看到药液进入注射器；仅进针没有抽液证据时不标注"},
    "transfer_liquid": {"name": "转移药液", "desc": "使用注射器把药液从一个容器推入另一个目标容器；要求能确认来源、目标和推注方向，不表示面向患者给药"},
    "agitate_solution": {"name": "混匀药液", "desc": "明确摇晃、翻转或振荡装有药液的容器以混匀；仅拿着或移动容器不标注"},
    "discard_sharps": {"name": "处置锐器", "desc": "明确把针头、安瓿或其他锐器放入锐器盒；仅出现锐器盒或丢弃普通耗材不标注"},
    "label_prepared_medication": {"name": "给成品贴签", "desc": "看到标签实际粘贴到注射器、药瓶或成品输液容器；仅查看、拿着标签或成品包装不标注"},
}


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
    lines.extend([
        "",
        "【判据与输出硬性约束】",
        "1. 仅标注画面中有明确视觉动作的区间；无法确认时输出 []。",
        "2. 严禁脱离画面背诵或补全配药流程。",
        "3. 起止时间以手部实际开始和结束操作为准，严禁默认从 0 秒开始。",
        f"4. 起止时间必须为 0 到 {chunk_max} 之间的整数秒，start_sec < end_sec，区间不可重叠。",
        "5. 只能输出单个合法 JSON 数组，严禁 Markdown、解释文字或思考过程。",
        '- 有动作示例：[{"start_sec":2,"end_sec":5,"label":"acquire_item","confidence":0.98}]',
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
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

    duration = get_video_duration(video_path)
    chunk_max = max(1, int(round(duration)))
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
    print(f"model: {model_path}")
    print(f"video: {video_path} | duration: {duration:.2f}s | fps: {args.fps}")

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path,
        # 与 Qwen3-VL-8B 离线基线保持相同的视觉像素预算，避免比较时
        # 把输入分辨率差异误判成模型能力或速度差异。
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
        cap_pixels_per_frame=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
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
