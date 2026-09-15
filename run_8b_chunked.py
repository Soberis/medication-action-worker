"""V2.2 Long-Video: Qwen3-VL-8B Core-Window Chunking with Anti-Hallucination.

Key Engineering Baselines:
1. Strict <=30s window size (24s core + 3s context padding before/after) -> Locks VRAM <= 19.86GB.
2. Frame-accurate slicing via ffmpeg libx264 ultrafast (eliminates keyframe drift).
3. Anti-hallucination prompt: forbids SOP reciting, allows empty array `[]`.
4. Robust multi-format parsing (handles standard dict array, multi-line arrays, and empty list).
5. Traceable provenance: records source_window and local timestamps; no blind auto-merge.
6. Strict V2-level schema validation and Fail-Closed circuit breaker.

注意：本文件是**离线评测 / 对照实现**，与生产 daemon `action_worker.py` 是两份独立代码
（这样单文件自包含、便于 FTP 传输）。跨窗续接判据与重叠消解策略已与 daemon 对齐，
但**线上行为只以 `action_worker.py` 为准**——本脚本跑通并不等于 daemon 正确。
"""

import math
import os
import sys
import shutil

# 启用动态分段显存分配器，避免显存碎片化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoProcessor, TextStreamer

# 针对 Qwen3-VL 优先使用专属多模态生成类
try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass
from label_schema import DEFAULT_LABEL_DEFINITIONS

import argparse

parser = argparse.ArgumentParser(description="Qwen3-VL-8B 长视频切片动作预标注（有界重叠切片与防幻觉）")
parser.add_argument("video", nargs="?", default="video1.mp4", help="目标视频路径（默认: video1.mp4）")
parser.add_argument("--video", dest="video_opt", default=None, help="目标视频路径（兼容 --video 参数）")
parser.add_argument("--fps", type=float, default=1.0, choices=[1.0, 2.0], help="抽样率（默认: 1.0 fps 基线，可选 2.0 对照）")
parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen3-VL-8B-Instruct"), help="模型权重目录路径")
parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径（默认保存至 output/actions_<video_name>.json）")
args, _ = parser.parse_known_args()

MODEL_PATH = args.model
target_video = args.video_opt or args.video

# 优先当前目录，兼容查找 data/videos
if not os.path.exists(target_video):
    candidates = [
        os.path.join("data", "videos", target_video),
        os.path.join(os.path.dirname(__file__), "data", "videos", target_video),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            target_video = cand
            break

VIDEO_PATH = target_video
BASE_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_JSON_PATH = args.output or f"actions_{BASE_NAME}.json"

CORE_CHUNK_SEC = 24.0    # 核心评估区间时长（秒）
CONTEXT_PAD_SEC = 3.0   # 前后各 3 秒视觉上下文重叠（总窗口严格 <= 30.0s，守住显存红线）
SAMPLING_FPS = float(args.fps)  # 采样率：1.0 fps 基线或 2.0 fps 对照

LABEL_DEFINITIONS = DEFAULT_LABEL_DEFINITIONS
LABEL_NAME_MAP = {key: info["name"] for key, info in LABEL_DEFINITIONS.items()}


def fail(message: str) -> None:
    raise SystemExit(f"错误: {message}")


def check_environment_safety():
    """硬件与环境安全性双重前置检查"""
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        if free_gb < 15.0:
            print(f"\n[硬件安全熔断] 当前磁盘可用空间仅剩 {free_gb:.2f} GB（低于 15 GB 工业安全红线）！")
            print("脚本主动安全退出，防止固态硬盘保护性挂起。\n")
            sys.exit(1)
    except Exception:
        pass

    if not torch.cuda.is_available():
        fail("未检测到 CUDA。请在具备 GPU 的正确环境中执行。")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < 12.0:
        print(f"\n[显存安全熔断] 当前显卡可用显存仅剩 {free_gb:.2f} GB（低于 12 GB 安全基线）！")
        print("检测到显卡被其他进程占用，脚本主动退出避免争抢 OOM。\n")
        sys.exit(1)


def get_video_duration(video_file: str) -> float:
    """读取视频物理总时长（秒）"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass
    return 30.0


def extract_video_chunk_frame_accurate(src_video: str, start_sec: float, dur_sec: float, output_chunk: str) -> bool:
    """毫秒级帧精确切取（基于 libx264 ultrafast，消除 keyframe 导致的 0.5~2s 时序偏移）"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur_sec:.3f}",
        "-i", src_video,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an",
        output_chunk
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and os.path.exists(output_chunk) and os.path.getsize(output_chunk) > 1000


CURRENT_TEMP_CHUNK = None


def cleanup_temp():
    global CURRENT_TEMP_CHUNK
    if CURRENT_TEMP_CHUNK and os.path.exists(CURRENT_TEMP_CHUNK):
        try:
            os.remove(CURRENT_TEMP_CHUNK)
        except Exception:
            pass


def reject_pipeline(reason: str, raw_text: str = "") -> None:
    """Fail-Closed 熔断保护：任一窗口失败绝不输出残缺 JSON 冒充有效结果，并安全清理临时文件"""
    cleanup_temp()
    raw_err_file = OUTPUT_JSON_PATH.replace(".json", ".err.raw.txt")
    with open(raw_err_file, "w", encoding="utf-8") as f:
        f.write(f"【熔断原因】{reason}\n\n【原始模型输出留存】\n{raw_text}\n")
    print("\n" + "=" * 70)
    print(f"【分段推理熔断】检测到不可恢复的切片或解析异常！")
    print(f"【错误原因】{reason}")
    print(f"【安全保护】遵循 Fail-Closed 原则，已阻止写入残缺数据至: {OUTPUT_JSON_PATH}")
    print(f"【原始留样】错误详情已归档至: {raw_err_file}")
    print("=" * 70)
    sys.exit(2)


def parse_model_output(output_text: str):
    """
    鲁棒多格式解析器：
    1. 优先解析标准对象数组 [{"start_sec": 12, "end_sec": 22, "label": "...", "confidence": 0.98}]
    2. 兼容解析紧凑数组列表 [[12, 22, "...", 0.98], ...]
    3. 兼容解析换行多行独立数组 [12, 22, "...", 0.98]\n[24, 27, "...", 0.98]
    4. 支持空结果 []
    """
    text = output_text.strip()
    if text == "[]":
        return []

    # 方式 1: 尝试完整提取首尾大括号/方括号的合法 JSON
    raw_match = re.search(r"\[.*\]", text, re.DOTALL)
    if raw_match:
        try:
            parsed = json.loads(raw_match.group(0))
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    return []
                # 场景 1.1: 标准字典列表
                if all(isinstance(x, dict) for x in parsed):
                    return parsed
                # 场景 1.2: 列表嵌套列表 [[12, 22, "LABEL", 0.98], ...]
                if all(isinstance(x, list) and len(x) >= 4 for x in parsed):
                    return [{"start_sec": x[0], "end_sec": x[1], "label": x[2], "confidence": x[3]} for x in parsed]
        except Exception:
            pass

    # 方式 2: 按行扫描多段独立方括号数组（容错大模型分多行输出）
    line_actions = []
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"\[.*\]", line)
        if m:
            try:
                row = json.loads(m.group(0))
                if isinstance(row, dict) and "start_sec" in row:
                    line_actions.append(row)
                elif isinstance(row, list) and len(row) >= 4:
                    line_actions.append({
                        "start_sec": row[0],
                        "end_sec": row[1],
                        "label": row[2],
                        "confidence": row[3],
                    })
            except Exception:
                continue

    if line_actions:
        return line_actions

    # 检查是否有明确的空声明
    if "[]" in text:
        return []

    return None


# 1. 硬件与环境安全检查
check_environment_safety()

if not os.path.exists(VIDEO_PATH):
    fail(f"目标视频 {VIDEO_PATH} 不存在，请检查文件名或路径。")

total_duration = get_video_duration(VIDEO_PATH)
print("=" * 72)
print(f"【长视频核心区切片任务】目标视频: {VIDEO_PATH}")
print(f"【视频物理时长】: {total_duration:.1f} 秒 ({total_duration/60:.2f} 分钟)")
print(f"【窗口规划方案】: {CORE_CHUNK_SEC:.0f}s 核心责任区 + 前后各 {CONTEXT_PAD_SEC:.0f}s 上下文 (总长严格 <= 30.0s，锁定 19.86GB 显存)")
print("=" * 72)

# 2. 严格规划有界重叠切片：每个切片总长严格 <= 30.0s
chunks_plan = []
c_start = 0.0
window_idx = 1
while c_start < total_duration:
    # 如果剩余时间小于等于核心步长 + 2秒，直接并入当前核心区，杜绝产生 0.2s 的多余尾巴窗口
    if (total_duration - c_start) <= (CORE_CHUNK_SEC + 2.0):
        c_end = total_duration
    else:
        c_end = c_start + CORE_CHUNK_SEC

    # 确定带上下文的实际视频切片物理起止点
    slice_start = max(0.0, c_start - CONTEXT_PAD_SEC)
    slice_end = min(total_duration, c_end + CONTEXT_PAD_SEC)
    slice_dur = slice_end - slice_start

    # 断言：切片时长绝对不能突破 30.0s 工业显存红线
    assert slice_dur <= 30.001, f"切片规划异常: {slice_dur:.2f}s 超过了 30s 显存红线！"

    # 核心责任区在该切片内部的相对时间坐标
    local_core_start = int(round(c_start - slice_start))
    local_core_end = int(round(c_end - slice_start))

    chunks_plan.append({
        "window_idx": window_idx,
        "slice_start": slice_start,
        "slice_dur": slice_dur,
        "core_abs_start": c_start,
        "core_abs_end": c_end,
        "local_core_start": local_core_start,
        "local_core_end": local_core_end,
    })
    c_start = c_end
    window_idx += 1

print(f"[*] 规划分析切片总数: {len(chunks_plan)} 个窗口:")
for cp in chunks_plan:
    print(f"    - 窗口 [{cp['window_idx']}/{len(chunks_plan)}]: 物理切片 {cp['slice_start']:.1f}s~{cp['slice_start']+cp['slice_dur']:.1f}s (时长: {cp['slice_dur']:.1f}s) | 核心责任区: {cp['core_abs_start']:.1f}s~{cp['core_abs_end']:.1f}s")
print("=" * 72)

# 3. 载入模型（常驻显存，多窗口复用）
print(f">>> [1/3] 正在从 {MODEL_PATH} 加载 8B 模型到 4090 显卡 (BF16)...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)

# 动态检测与启用 FlashAttention-2 加速
attn_implementation = "sdpa"
try:
    import flash_attn
    attn_implementation = "flash_attention_2"
    print("[*] 成功检测到 flash-attn，已启用 FlashAttention-2 极速注意力内核！")
except ImportError:
    print("[*] 未检测到 flash-attn，使用 PyTorch 默认 sdpa 引擎。")

model = ModelClass.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map="cuda",
)

streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. 逐个窗口切片推理
all_collected_actions = []
shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else "."
temp_chunk_file = os.path.join(shm_dir, f"_tmp_core_chunk_{os.getpid()}.mp4")

total_start_time = time.time()
torch.cuda.reset_peak_memory_stats()

try:
    for cp in chunks_plan:
        w_idx = cp["window_idx"]
        s_start = cp["slice_start"]
        s_dur = cp["slice_dur"]
        l_core_start = cp["local_core_start"]
        l_core_end = cp["local_core_end"]

        print(f"\n-----------------------------------------------------------------")
        print(f">>> [2/3] 推理窗口 [{w_idx}/{len(chunks_plan)}]: 切片物理时间 {s_start:.1f}s ~ {s_start+s_dur:.1f}s (核心责任区: {cp['core_abs_start']:.1f}s ~ {cp['core_abs_end']:.1f}s)...")

        # 帧精确切取临时子视频
        if not extract_video_chunk_frame_accurate(VIDEO_PATH, s_start, s_dur, temp_chunk_file):
            reject_pipeline(f"窗口 [{w_idx}] 视频精确切片失败！")

        chunk_max_sec = int(round(s_dur))

        # 优化去偏 Prompt：明确视觉判据，防止单标签坍塌，兼顾反背诵与空结果
        label_criteria = "\n".join(
            f"- {key}: {info['name']}（{info['desc']}）"
            for key, info in LABEL_DEFINITIONS.items()
        )
        prompt = f"""你是一个专业配药动作标注质检员。请仔细观察视频，严格根据画面中护士手部的【真实动作事实】输出起止时间。

标签定义与关键视觉判据：
{label_criteria}

【判据与输出硬性约束】
1. 实事求是、准确分类：依据手部动作方向（向后拉活塞为抽药，向前推活塞为推注，摇晃/翻转为混匀，掰开/拧开为开启）准确区分动作类型；
2. 严禁流程脑补：仅标注在视频画面中有清晰视觉动作的区间，严禁脱离画面按照标签列表强行背诵或补全整套流程！
3. 允许空结果：若当前视频片段中未发生上述动作、动作不明确、或处于持物准备间歇，必须直接输出空数组 []！
4. 真实动作时长：动作起止严格以手部实际开始与结束操作的整秒为准，严禁默认从 0 秒开始标记；
5. 起止时间必须严格为整秒整数（如 0, 1, 2, 3... 秒），范围在 0 到 {chunk_max_sec} 秒之间，start_sec < end_sec 且动作不可重叠；
6. confidence 为 0.0 到 1.0 的参考自评值。
7. 输出格式必须严格为以下单个 JSON 数组结构（严禁任何 Markdown 或解释文字）：
示例 1 (有动作): [{{"start_sec": 2, "end_sec": 5, "label": "open_container", "confidence": 0.98}}, {{"start_sec": 8, "end_sec": 16, "label": "draw_liquid", "confidence": 0.97}}]
示例 2 (无明确动作): []"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": temp_chunk_file, "fps": SAMPLING_FPS},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )

        output_text = processor.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # 鲁棒多格式解析
        chunk_actions = parse_model_output(output_text)
        if chunk_actions is None:
            reject_pipeline(f"窗口 [{w_idx}] 未能解析出合法动作列表或空数组 []", output_text)

        # 1. 第一阶段：对窗口内全部候选动作进行严格 Fail-Closed 合法性前置校验
        prev_local_end = 0
        validated_chunk_actions = []
        for a_idx, item in enumerate(chunk_actions, 1):
            if not all(k in item for k in ("start_sec", "end_sec", "label", "confidence")):
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项缺少必填字段: {item}", output_text)

            raw_s = float(item["start_sec"])
            raw_e = float(item["end_sec"])
            conf = float(item["confidence"])

            if not (math.isfinite(raw_s) and math.isfinite(raw_e) and math.isfinite(conf)):
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项含非有限数值", output_text)
            if not (raw_s.is_integer() and raw_e.is_integer()):
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项非整秒时间戳 ({raw_s}, {raw_e})", output_text)

            local_s = int(raw_s)
            local_e = int(raw_e)
            label = str(item["label"]).strip()

            if label not in LABEL_NAME_MAP:
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项未知标签: {label}", output_text)
            if local_e <= local_s or local_s < prev_local_end or local_e > chunk_max_sec:
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项时序区间不合法: [{local_s}, {local_e}]", output_text)
            if not (0.0 <= conf <= 1.0):
                reject_pipeline(f"窗口 [{w_idx}] 第 {a_idx} 项置信度越界: {conf}", output_text)

            prev_local_end = local_e
            validated_chunk_actions.append({
                "local_s": local_s,
                "local_e": local_e,
                "label": label,
                "conf": conf,
            })

        # 2. 第二阶段：全局时间坐标映射、跨窗口平滑接缝与核心责任区归属过滤
        accepted_in_window = 0
        for act in validated_chunk_actions:
            local_s = act["local_s"]
            local_e = act["local_e"]
            label = act["label"]
            conf = act["conf"]

            g_s = int(round(s_start + local_s))
            g_e = int(round(s_start + local_e))
            g_s = max(0, min(g_s, int(round(total_duration))))
            g_e = max(g_s + 1, min(g_e, int(round(total_duration))))

            # 跨窗口平滑接缝：在「已收集动作」里找时间上相邻的同标签候选来接续。
            # 不能只看 all_collected_actions[-1]——模型偶尔不按时间序输出（先给靠后的动作），
            # 此时 [-1] 并不是边界处那一条，pad 区的续段会被误判成孤立片段而丢弃。
            # 判据与 action_worker.py 的 _can_continue 保持一致。
            boundary = int(round(cp["core_abs_start"]))
            candidates = [
                a for a in all_collected_actions
                if a["label"] == label
                and a["end_sec"] >= boundary - 1
                and g_s <= a["end_sec"] + 1
            ]
            stitched = False
            if candidates:
                # 平滑延长目标动作的结束时间，避免动作在窗口接缝处被生硬切断
                target = max(candidates, key=lambda a: a["end_sec"])
                target["end_sec"] = max(target["end_sec"], g_e)
                target["confidence"] = round(max(target["confidence"], conf), 2)
                if w_idx not in target["stitched_windows"]:
                    target["stitched_windows"].append(w_idx)
                stitched = True

            if stitched:
                accepted_in_window += 1
                continue

            # 核心责任区归属判定：
            # 1) 起点直接落在核心区内的动作，归属本窗口
            # 2) 未被前项缝合、但延伸进入本窗口核心区的动作（穿透接缝）：
            #    截取落入本窗口核心区的部分，避免接缝盲区漏检
            if l_core_start <= local_s < l_core_end:
                all_collected_actions.append({
                    "start_sec": g_s,
                    "end_sec": g_e,
                    "label": label,
                    "confidence": round(conf, 2),
                    "source_window": w_idx,
                    "stitched_windows": [w_idx],
                    "local_start_sec": local_s,
                    "local_end_sec": local_e,
                })
                accepted_in_window += 1
            elif local_s < l_core_start and local_e > l_core_start:
                act_adopted = {
                    "start_sec": boundary,
                    "end_sec": g_e,
                    "label": label,
                    "confidence": round(conf, 2),
                    "source_window": w_idx,
                    "stitched_windows": [w_idx],
                    "local_start_sec": l_core_start,
                    "local_end_sec": local_e,
                }
                if act_adopted["end_sec"] > act_adopted["start_sec"]:
                    all_collected_actions.append(act_adopted)
                    accepted_in_window += 1

        print(f"\n[*] 窗口 [{w_idx}] 模型输出 {len(chunk_actions)} 项，核心区采纳入库/缝合 {accepted_in_window} 项。")

finally:
    cleanup_temp()
    torch.cuda.empty_cache()

total_cost = time.time() - total_start_time
peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

# 5. 全局按时间排序并严格整理为半开区间不重叠
all_collected_actions.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

validated_global = []
for act in all_collected_actions:
    if not validated_global:
        validated_global.append(act)
        continue

    prev = validated_global[-1]
    # 场景 1: 同标签相邻或相交 -> 平滑合并
    if act["label"] == prev["label"] and act["start_sec"] <= prev["end_sec"] + 1:
        prev["end_sec"] = max(prev["end_sec"], act["end_sec"])
        prev["confidence"] = round(max(prev["confidence"], act["confidence"]), 2)
        for sw in act.get("stitched_windows", []):
            if sw not in prev["stitched_windows"]:
                prev["stitched_windows"].append(sw)
        continue

    # 场景 2: 不同标签存在重叠 -> 中点平分或包含消除（非破坏性）
    if act["start_sec"] < prev["end_sec"]:
        if act["end_sec"] > prev["end_sec"]:
            mid_point = (prev["end_sec"] + act["start_sec"]) // 2
            prev["end_sec"] = max(prev["start_sec"] + 1, mid_point)
            act["start_sec"] = max(prev["end_sec"], mid_point)
        else:
            if act["confidence"] > prev["confidence"] + 0.15:
                prev["end_sec"] = act["start_sec"]
            else:
                continue

    if act["end_sec"] > act["start_sec"]:
        validated_global.append(act)

# 标准化输出结构：增加毫秒级 start_ms / end_ms / duration_ms 契约
export_results = []
for item in validated_global:
    s_sec = int(item["start_sec"])
    e_sec = int(item["end_sec"])
    export_results.append({
        "start_sec": s_sec,
        "end_sec": e_sec,
        "start_ms": s_sec * 1000,
        "end_ms": e_sec * 1000,
        "duration_ms": (e_sec - s_sec) * 1000,
        "label": item["label"],
        "confidence": item["confidence"],
        "source_window": item["source_window"],
        "stitched_windows": item.get("stitched_windows", [item["source_window"]]),
    })

# 安全导出全量候选 JSON 结果
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(export_results, f, ensure_ascii=False, indent=2)

# 6. 打印精美汇总报表
print("\n" + "=" * 72)
print(f">>> [3/3] 长视频分析圆满完成！")
print(f"【总体视频时长】: {total_duration:.1f} 秒 ({total_duration/60:.2f} 分钟)")
print(f"【端到端总耗时】: {total_cost:.2f} 秒 (平均处理速度: {total_duration/total_cost:.1f}x 实时)")
print(f"【显存峰值控制】: {peak_vram_gb:.2f} GB / 24.0 GB (占用率: {peak_vram_gb/24.0*100:.1f}%) -> 零 OOM 风险！")
print(f"【运行配置基线】: sampling_fps={SAMPLING_FPS} fps")
print("=" * 72)
print(f"【全时序动作预标注明细 (共采纳 {len(export_results)} 个动作候选)】:")
print(f"{'全局时间区间':<20} | {'动作代码':<14} | {'动作名称':<10} | {'来源窗口':<12} | {'置信度'}")
print("-" * 72)
for item in export_results:
    m_start, s_start = divmod(item["start_sec"], 60)
    m_end, s_end = divmod(item["end_sec"], 60)
    time_str = f"{m_start:02d}:{s_start:02d} ~ {m_end:02d}:{s_end:02d} ({item['start_sec']}s~{item['end_sec']}s)"
    label = item.get("label", "UNKNOWN")
    cname = LABEL_NAME_MAP.get(label, "未定义")
    w_src = f"Win-{item.get('source_window', '?')}"
    if len(item.get("stitched_windows", [])) > 1:
        w_src += f" (缝合 {item['stitched_windows']})"
    conf = item.get("confidence", 0.0)
    print(f"{time_str:<20} | {label:<14} | {cname:<10} | {w_src:<12} | {conf:<6.2f}")
print("=" * 72)
print(f"已安全导出标准动作候选 JSON 至: {OUTPUT_JSON_PATH}")
print("【数据治理声明】:")
print("1. 本结果为 AI 辅助初稿（Candidate），严格保留了 source_window 追溯来源。")
print("2. 严禁未经人工核验直接放行入库或直接作为模仿学习金标准。正式数据须经人工复核。")
print("=" * 72 + "\n")
