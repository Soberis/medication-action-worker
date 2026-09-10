# Medication Action Worker (Qwen3-VL-8B)

专为 **RTX 4090 (24GB 显存)** 深度优化的多模态视频动作候选标注 Worker 服务。作为 [medication-video-platform](https://github.com/) 生态的异步 AI 推理节点，监听 RabbitMQ 消息队列，利用视觉大语言模型（VLM）对调配过程视频进行高精度的动作片段识别与边界提取。

---

## 核心设计与工程基线

1. **显存常驻（Zero Cold-Start）**：服务启动后模型常驻 24GB 显存，避免每次推理重复加载数十 GB 权重的冷启动延迟。
2. **严格单任务串行（Anti-OOM）**：RabbitMQ 消费者配置 `prefetch_count=1`，确保单卡严格串行推理，从根本上防止多任务并发引发显存溢出。
3. **动态显存优化**：强制启用 PyTorch `expandable_segments:True` 动态分段内存分配器，消除长时运行下的显存碎片化。
4. **长视频有界重叠切片（Bounded Core-Window Chunking）**：
   - 核心评估区间（Core Window）：24 秒；
   - 前后上下文冗余（Context Padding）：前后各 3 秒；
   - 总推理窗口严格锁定在 `<= 30.0s`，实测显存锁定在 `<= 19.86GB` 安全红线以内。
5. **帧级硬切片（Frame-Accurate Slicing）**：利用 `ffmpeg libx264 ultrafast` 在本地提取硬切片，彻底消除 OpenCV/解码器关键帧漂移（Keyframe Drift）。
6. **防幻觉策略（Anti-Hallucination）**：专用 System Prompt 约束模型杜绝背诵 SOP，允许输出空列表 `[]`，并进行严格 Schema 校验。
7. **租约自愈与链路回传**：自动进行 Core API 任务租约续期（Auto-Lease Renewal），完成推理后结构化组装结果并回传至平台。

---

## 识别动作分类规范

| 动作标识 (`label`) | 中文释义 | 说明 |
| :--- | :--- | :--- |
| `CHECK_INFO` | 核对信息 | 拿取、比对处方单、药瓶标签、患者信息条码等 |
| `OPEN_CAP` | 操作瓶盖 | 启开药瓶铝塑盖、拔下注射器保护帽或针头帽 |
| `DRAW_LIQUID` | 抽取药液 | 注射器穿刺溶媒/原液瓶并回抽抽取液体 |
| `INJECT_MIX` | 注入混匀 | 向目标输液袋/瓶注入药液，翻转摇匀或震荡混匀 |
| `LABEL_SEAL` | 贴签封包 | 粘贴成品标签、套袋封口或放入转运框 |

---

## 目录结构（单层平铺，开箱即用）

```text
medication-action-worker/
├── run.sh                # 运行与生命周期管理脚本 (./run.sh 前台直接跑，支持 start/stop/log)
├── action_worker.py      # 核心生产常驻 Daemon（RabbitMQ 消费、切片推理、平台回传）
├── run_8b_chunked.py     # 长视频核心区切片推理与防幻觉评测（单文件完整自包含，方便 FTP 传输）
├── run_8b.py             # 30s 单窗口快速冒烟测试（单文件完整自包含）
├── requirements.txt      # Python 依赖清单
├── .env.example          # 环境变量配置模板
├── .gitignore            # Git 忽略规则（过滤视频、图片与模型输出）
└── README.md             # 项目文档
```

---

## 环境准备

### 1. 硬件与底层依赖
- **GPU**：NVIDIA GeForce RTX 4090 (24GB VRAM) 或同等计算卡
- **系统环境**：Linux (Ubuntu 22.04+) 或 Windows 11 (CUDA 12.1+)
- **多媒体工具**：系统必须安装 `ffmpeg` 与 `ffprobe` 并加入 `PATH`

### 2. 安装 Python 依赖
推荐在虚拟环境中安装：
```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

*(可选加速)* 如环境支持编译，可安装 `flash-attn` 以进一步提升长视觉 Token 的推理速度：
```bash
pip install flash-attn --no-build-isolation
```

### 3. 模型下载
将 `Qwen/Qwen3-VL-8B-Instruct` 权重下载至本地目录（如 `~/models/Qwen3-VL-8B-Instruct`）。

---

## 快速使用

### 1. 生产模式：启动常驻 Worker Daemon

```bash
python action_worker.py \
  --model ~/models/Qwen3-VL-8B-Instruct \
  --api-url http://localhost:8080 \
  --rabbitmq-url amqp://guest:guest@localhost:5672/ \
  --queue platform.ai.action-candidate.v1 \
  --fps 1.0
```

也可以通过环境变量配置（参考 `.env.example`）：
```bash
export PLATFORM_API_URL=http://192.168.5.100:8080
export RABBITMQ_URL=amqp://guest:guest@192.168.5.100:5672/
python action_worker.py
```

### 2. 离线工具：长视频有界切片推理评估

用于脱离 RabbitMQ 和平台服务，直接测试完整长视频的动作识别与切片拼接效果：

```bash
python run_8b_chunked.py video1.mp4 --fps 1.0
```

### 3. 离线工具：30s 短视频快速冒烟测试

用于快速排查 GPU 驱动、模型加载、Transformers 多模态管道是否正常：

```bash
python run_8b.py test_30s.mp4
```

---

## 输出数据格式示例

推理输出为标准的动作候选数组：

```json
[
  {
    "start_sec": 12,
    "end_sec": 22,
    "label": "DRAW_LIQUID",
    "confidence": 0.98,
    "source_window": 1,
    "local_start_sec": 12,
    "local_end_sec": 22
  },
  {
    "start_sec": 47,
    "end_sec": 50,
    "label": "DRAW_LIQUID",
    "confidence": 0.96,
    "source_window": 2,
    "local_start_sec": 23,
    "local_end_sec": 26
  }
]
```
