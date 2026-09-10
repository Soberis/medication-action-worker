"""V2: Qwen3-VL-8B medication-action pre-annotation smoke test."""

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

MODEL_PATH = os.path.expanduser("~/models/Qwen3-VL-8B-Instruct")
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "test_30s.mp4"
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_JSON_PATH = "actions_8b.json" if VIDEO_PATH == "test_30s.mp4" else f"actions_{base_name}.json"
SAMPLING_FPS = 1.0  # 当前采样率：每秒 1 帧（对应整秒视觉时间刻度）

LABEL_NAME_MAP = {
    "CHECK_INFO": "核对信息",
    "OPEN_CAP": "操作瓶盖",
    "DRAW_LIQUID": "抽取药液",
    "INJECT_MIX": "注入混匀",
    "LABEL_SEAL": "贴签封包",
}


def fail(message: str) -> None:
    raise SystemExit(f"错误: {message}")


def check_environment_safety():
    """硬件与环境安全性双重前置检查，绝对杜绝满盘或显存争抢导致系统挂起"""
    # 1. 磁盘空间安全红线检查（避免满盘导致固态硬盘主控自我保护挂起）
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        if free_gb < 15.0:
            print(f"\n[硬件安全熔断] 当前磁盘可用空间仅剩 {free_gb:.2f} GB（低于 15 GB 工业安全红线）！")
            print("为了绝对保护固态硬盘不触发物理掉盘，脚本已主动安全退出。")
            print("建议：请先删除无用大文件，释放至少 50GB 以上空间后再启动。\n")
            sys.exit(1)
    except Exception:
        pass

    # 2. GPU 显存可用空间安全前置检查
    if not torch.cuda.is_available():
        fail("未检测到 CUDA。请确认在目标 GPU 主机的正确 Python 环境中执行。")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < 12.0:
        print(f"\n[显存安全熔断] 当前显卡可用显存仅剩 {free_gb:.2f} GB（低于 12 GB 安全基线）！")
        print("检测到显卡当前被其他进程大量占用，脚本主动退出，避免显存争抢或 OOM。\n")
        sys.exit(1)

def get_video_duration(video_file: str) -> float:
    """安全读取视频物理总时长（秒）"""
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

# 执行环境安全熔断检查
check_environment_safety()

if not os.path.exists(VIDEO_PATH):
    fail(f"目标视频 {VIDEO_PATH} 不存在，已安全中止。")

video_duration = get_video_duration(VIDEO_PATH)
max_sec = int(round(video_duration))
print(f"当前分析视频: {VIDEO_PATH} (物理时长: {video_duration:.1f} 秒 | 采样率: {SAMPLING_FPS} fps)")

print(f">>> [1/3] 正在从 {MODEL_PATH} 加载 8B 模型到 4090 显卡 (BF16)...")
try:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
        cap_pixels_per_frame=True,
    )
except Exception:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

model = ModelClass.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)

prompt = f"""你是一个专业配药动作标注质检员。请分析视频，严格按照以下标签输出各操作的起止时间：
- CHECK_INFO: 核对信息
- OPEN_CAP: 操作瓶盖
- DRAW_LIQUID: 抽取药液
- INJECT_MIX: 注入混匀
- LABEL_SEAL: 贴签封包

约束要求：
1. 仅输出合法 JSON 数组，严禁包含任何前缀、解释、思考过程或 Markdown 标记；
2. 每个对象包含 start_sec, end_sec, label, confidence 四个字段；
3. 【时序精度对齐】输入视频按每秒 1 帧（fps={SAMPLING_FPS}）采样，起止时间必须严格对齐到整秒整数（如 0, 1, 2, 3... 秒，严禁输出未经采样的 0.5s 等半秒分数）；
4. 时间范围在 0 到 {max_sec} 秒之间，start_sec < end_sec 且各动作区间不可重叠；
5. confidence 为 0.0 到 1.0 的参考自评值。"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH, "fps": SAMPLING_FPS},
            {"type": "text", "text": prompt},
        ],
    }
]

print(">>> [2/3] 正在准备视频帧采样与输入张量...")
torch.cuda.reset_peak_memory_stats()
start_t = time.time()

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

torch.cuda.empty_cache()

print(">>> [2/3] 输入数据准备完成（帧采样与输入张量就绪）。")
print(">>> [2/3] 正在调用 4090 显卡执行多模态特征编码与模型推理 (前几秒为显卡 Prefill 运算，请勿中断)...")

# 实时流式输出，逐字实时可见
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        streamer=streamer,
    )

output_text = processor.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

cost_sec = time.time() - start_t
peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

# 严格解析与校验
raw_json_match = re.search(r"\[.*\]", output_text, re.DOTALL)
actions = []
parse_error = None

if raw_json_match:
    try:
        parsed = json.loads(raw_json_match.group(0))
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            actions = parsed
        else:
            parse_error = "JSON 解析成功但顶层结构不是对象数组"
    except Exception as exc:
        parse_error = f"JSON 语法解析失败: {exc}"
else:
    parse_error = "未在模型输出中提取到有效的 JSON 数组 [...]"


def reject_output(reason: str) -> None:
    """留存模型原文，且绝不以失败结果覆盖最后一次有效输出。"""
    raw_error_file = OUTPUT_JSON_PATH.replace(".json", ".err.raw.txt")
    with open(raw_error_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print("\n" + "=" * 65)
    print(f"【推理失败】模型输出未能解析为合法动作数组！")
    print(f"【错误原因】{reason}")
    print(f"【失败保护】已阻止写入空数据覆盖现有有效结果: {OUTPUT_JSON_PATH}")
    print(f"【原始留样】模型原文已完整归档至: {raw_error_file}")
    print("=" * 65)
    raise SystemExit(2)


if not actions:
    reject_output(parse_error or "输出为空")

# 保证时序精度与采样率严格一致（整秒约束对齐），校验区间合法性。
# 不擅自修正模型给出的时间；不合法输出应由人工或下一次推理处理。
validated_actions = []
previous_end_sec = 0
validation_errors = []
for index, item in enumerate(actions, start=1):
    try:
        if not all(field in item for field in ("start_sec", "end_sec", "label", "confidence")):
            raise ValueError("缺少必填字段")

        raw_start = float(item["start_sec"])
        raw_end = float(item["end_sec"])
        confidence = float(item["confidence"])
        if not all(math.isfinite(value) for value in (raw_start, raw_end, confidence)):
            raise ValueError("时间或置信度不是有限数值")
        if not raw_start.is_integer() or not raw_end.is_integer():
            raise ValueError("1 fps 采样下时间必须为整秒")

        start_sec = int(raw_start)
        end_sec = int(raw_end)
        if end_sec <= start_sec:
            raise ValueError("end_sec 必须大于 start_sec")
        if not 0 <= start_sec < end_sec <= max_sec:
            raise ValueError(f"时间必须位于 0 至 {max_sec} 秒")
        if start_sec < previous_end_sec:
            raise ValueError("动作区间与前一项重叠或未按时间升序排列")

        label = str(item["label"]).strip()
        if label not in LABEL_NAME_MAP:
            raise ValueError(f"未知标签: {label}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须位于 0.0 至 1.0")

        validated_actions.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
            "confidence": round(confidence, 2),
        })
        previous_end_sec = end_sec
    except (TypeError, ValueError) as exc:
        validation_errors.append(f"第 {index} 项: {exc}")

if validation_errors:
    reject_output("; ".join(validation_errors))

# 保存标准 JSON 文件（安全落盘）
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(validated_actions, f, ensure_ascii=False, indent=2)

print("\n>>> [3/3] 测试完成！")
print("=================================================================")
print(f"【Qwen3-VL-8B】视频时长: {video_duration:.1f} 秒 | 推理耗时: {cost_sec:.2f} 秒")
print(f"【Qwen3-VL-8B】显存峰值: {peak_vram_gb:.2f} GB / 24.0 GB (占用率: {peak_vram_gb/24.0*100:.1f}%)")
print("=================================================================")
print("【动作时序预标注明细 (采样粒度对齐: 整秒)】:")
print(f"{'时间区间':<16} | {'动作代码':<14} | {'动作名称':<10} | {'置信度(自报参考值)':<16}")
print("-" * 65)
for item in validated_actions:
    time_range = f"{item['start_sec']:>2d}s ~ {item['end_sec']:>2d}s"
    label = item.get("label", "UNKNOWN")
    cname = LABEL_NAME_MAP.get(label, "未定义")
    conf = item.get("confidence", 0.0)
    print(f"{time_range:<16} | {label:<14} | {cname:<10} | {conf:<16.2f}")
print("=================================================================")
print(f"已安全写入标准结果文件: {OUTPUT_JSON_PATH}")
print("提示：上述置信度仅为大模型自评数值，不可作为自动放行依据，正式入库须经人工核验。")
