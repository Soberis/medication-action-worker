# Medication Action Worker (Qwen3.5-4B)

配药视觉质检体系中的 **GPU AI 动作时序推理节点**。作为 [medication-video-platform](https://github.com/Soberis/medication-video-platform/tree/dev) 生态的异步推理服务，专为 **RTX 4090 (24GB 显存)** 优化，采用 `Qwen3.5-4B` 视觉大语言模型（VLM）；监听 RabbitMQ 消息队列，对配药过程视频进行动作片段识别、目标物品受体提取与时间边界确定。

---

## 关联项目生态

本项目是静脉用药调配（PIVAS）质检体系中的 GPU AI 动作推理节点：
- **云端中心平台**：[medication-video-platform](https://github.com/Soberis/medication-video-platform/tree/dev) — Java Core API + React 标注管理端，负责任务调度、租约控制、时间轴动作候选审核与数据集导出。
- **边缘采集端**：[medication-camera](https://github.com/Soberis/medication-camera/tree/dev) — RK3588 边缘工控机多机位音视频同步录制、本地缓冲池维护与批量直传流水线。
- **动作推理节点**：**medication-action-worker**（本项目）— 基于 Qwen3.5-4B 的有界切片动作时序推理、受体物品解析与租约安全回传。

---

## 核心设计与工程基线

1. **显存常驻与弹性卸载（Zero Cold-Start / Elastic Offload）**：
   - 默认服务启动后模型常驻显存，避免每次推理重复加载 4B 权重的冷启动延迟（Qwen3.5-4B 权重常驻仅约 **8.5 GB**，30s 窗口推理峰值约 **11.9 GB**，在 24GB 显存的 4090 上余量充足）；
   - 支持通过 `IDLE_OFFLOAD_SECONDS` 控制空闲超时自动卸载（如空闲 300 秒释放显存，新任务到达时动态唤醒）。
2. **严格单任务串行（Anti-OOM）**：RabbitMQ 消费者配置 `prefetch_count=1`，确保单卡严格串行处理任务，从根本上杜绝多并发导致的显存激增溢出。
3. **动态显存防碎片优化**：强制启用 PyTorch `expandable_segments:True` 动态分段内存分配器，消除长时运行下的显存碎片化。
4. **长视频有界重叠切片（Bounded Core-Window Chunking）**：
   - 核心评估区间（Core Window）：24 秒；
   - 前后上下文冗余（Context Padding）：前后各 3 秒；
   - 单次模型推理窗口严格锁定在 `<= 30.0s`，杜绝超长上下文导致显存无限膨胀。
5. **帧级硬切片（Frame-Accurate Slicing）**：利用 `ffmpeg libx264 ultrafast` 在本地提取关键帧对齐的硬切片，消除视频关键帧漂移（Keyframe Drift）。
6. **防幻觉策略与受体物品实体绑定（Anti-Hallucination & Object Grounding）**：
   - 专用 System Prompt 严禁模型背诵或脑补标准操作规程（SOP），空操作台或无有效动作时必须直接输出 `[]`；
   - `acquire_item` 必须明确标注所拿取的受体物品（如西林瓶、注射器等），严禁将手部悬停、持物等待、调整握姿泛化为拿取。
7. **链路回传与租约安全控制**：
   - 推理完成后将结构化候选动作（含 `label`、`start_ms`、`end_ms`、`target_object`、`confidence` 等）回传 Core API；
   - 全程由独立线程自动续期租约（Lease Heartbeat），任务结束或租约失效时自动提前安全退出，不空耗 GPU 算力。
8. **启动前自检 + 断线退避 + 可控重投**：
   - `./run.sh check` 在加载模型**之前**自动检测模型路径 / ffmpeg / GPU / RabbitMQ / Core API 连通性；
   - 运行期断线采用 5s → 60s 指数退避重连，避免频繁重试刷屏；
   - AMQP 心跳设为 0 关闭，适配切片与生成期间的长时间阻塞，防止被 Broker 误判掉线。

---

## 配药动作与目标物品规范 (Medication Action Taxonomy)

在静脉用药调配中心（PIVAS）质检业务中，单纯输出粗粒度的动作（例如“拿取”）无法满足质检与复溶审核要求。系统采用 **动作类型 (Action Label) + 目标受体 (Target Object)** 联合表达机制。

### 1. 核心物理动作 (9 Core Action Labels)

当前统一采用平台 `p0-atomic-action.v4` 词表，所有标签均对应视频中有明确视觉证据的物理操作：

| 动作标识 (`label`) | 中文释义 | 正向视觉证据 | 严禁误报判据（不应标注） |
| :--- | :--- | :--- | :--- |
| `acquire_item` | 拿取容器/工具 | 手伸向原静止物品并将其拿起离开台面的瞬间（通常仅 1~3 秒），物体进入操作区 | 持续持有、悬停等待、调整握持姿势、拆包装、未使物体发生位移 |
| `open_container` | 开启容器 | 明确打开药品主容器或包装密封，如掰开安瓿瓶、取下西林瓶铝塑料保护盖、拔下注射器针帽 | 仅拿起药瓶、触碰瓶口或把针头靠近容器 |
| `disinfect_access` | 消毒容器接口 | 看到棉签或其他消毒用品实际擦拭瓶塞、安瓿颈或容器接口（持续擦拭 2 秒以上） | 仅手持棉签、棉签仅在空中移动或轻微靠近 |
| `insert_needle` | 针头进入容器 | 清楚看到针头穿过瓶塞或接口并进入目标药品容器且保持进针状态 | 针头悬停在瓶口上方、视线被遮挡、仅靠近未刺入 |
| `draw_liquid` | 抽取药液 | 针头已在容器内，活塞向后拉动，清楚看到药液进入注射器针筒 | 仅针头插入容器但活塞未回抽、或无液体进入证据 |
| `transfer_liquid` | 转移药液 | 使用注射器把抽取到的药液推入另一目标药瓶、输液袋或容器（活塞前推） | 无法确认目标容器、或面向患者的给药注射 |
| `agitate_solution` | 混匀药液 | 明确上下翻转输液袋或反复摇晃/振荡药瓶以混匀药液 | 水剂直接抽吸后无混匀动作、仅手持或微调容器 |
| `discard_sharps` | 处置锐器 | 明确看到将针头、安瓿瓶等锐器投入专用硬质锐器盒（Sharps Container） | 手部离开台面、放下普通物品、或将普通垃圾扔进垃圾袋 |
| `label_prepared_medication` | 给成品贴签 | 明确看到配药成品标签实际粘贴到注射器、药瓶或成品输液容器上 | 仅查看核对标签、手持标签纸或拿取外包装 |

> **业务边界说明**：
> - 标签严格描述**可观察的物理事实**，不推断意图或正确性。
> - `RETRIEVE_MEDICINE`（泛化取物）、`CHECK_INFO`（核对信息）等无法从画面可靠验证的认知性标签已从词表淘汰。

### 2. 目标受体物品规范 (Target Objects / `item`)

为彻底解决 `acquire_item` 泛化且不知道拿取何物的问题，模型与平台端同步支持 `target_object`（`item`）属性字段：

| 受体标识 (`target_object`) | 中文名称 | 业务场景与实体说明 |
| :--- | :--- | :--- |
| `vial` | 西林瓶/粉针瓶 | 抗生素、冻干粉等需要注入溶媒复溶的玻璃/塑料瓶 |
| `ampoule` | 安瓿瓶 | 盛装注射用水、电解质或小容量注射液的易折玻璃容器 |
| `syringe` | 注射器 | 用于抽取药液、推注溶媒或定容的无菌注射器筒身与推杆 |
| `infusion_bottle` | 输液瓶/输液袋 | 盛装大容量静脉滴注药液（如生理盐水袋、葡萄糖瓶）的主容器 |
| `needle` | 针头/配药针 | 独立配药针头、穿刺加药针或注射器针头部件 |
| `cotton_swab` | 棉签/消毒棒 | 用于消毒瓶塞、安瓿折断处或输液袋加药口的消毒耗材 |
| `packaging` | 包装耗材 | 注射器纸塑包装袋、药品外包装盒或耗材托盘 |
| `other_tool` | 其他工具 | 砂轮、开瓶器、止血钳等辅助器械 |

### 3. 防泛化与时序合并硬规则

1. **瞬态时限截断（Max Duration Clamp）**：
   - 物理拿取是瞬时动作，Worker 强制限制单段 `acquire_item` 最大时长为 **5 秒**（`MAX_ACQUIRE_DURATION_SEC = 5`）；超过 5 秒的输出自动截断为 5 秒，防止模型将持续持物阶段误并入。
2. **跨窗口同物合并与异物互斥（Strict Merge Policy）**：
   - 跨切片窗口拼接时，只有相邻且 **`item` 完全相同** 的 `acquire_item` 且合并后总长 `<= 5s` 才允许续接合并；
   - 拿取不同物品（例如前段为 `vial`，后段为 `syringe`）**绝对禁止合并**，必须拆分为两个独立候选。
3. **全链路展示对齐**：
   - Worker 提取并回传 `target_object`；
   - 平台 Core API 写入 `action_candidate` 数据表；
   - Web 前端标注面板以 `<Tag color="cyan">西林瓶/粉针瓶</Tag>` 高亮标签直观展示，辅助审核员一键采纳。

---

## 目录结构

```text
medication-action-worker/
├── run.sh                       # 进程管理与生命周期脚本 (check/start/stop/status/log)
├── action_worker.py             # 核心生产常驻 Daemon（RabbitMQ 消费、Qwen3.5 推理、租约回传）
├── label_schema.py              # 配药动作与受体物品统一标准定义
├── run_35_chunked.py            # Qwen3.5-4B 长视频切片离线评测工具
├── run_35.py                    # Qwen3.5-4B 30s 单窗口冒烟测试工具
├── tests/                       # 单元测试套件
│   ├── test_worker_output.py    # 动作解析与受体提取单测
│   ├── test_chunked_output.py   # 切片合并与时长钳位单测
│   ├── test_label_resolution.py # 标签契约解析单测
│   └── test_message_handling.py # 消息处理与租约通信单测
├── requirements.txt             # Python 依赖清单
├── .env.example                 # 环境变量配置模板
└── README.md                    # 项目文档
```

---

## 部署到 4090 主机

目标：Linux GPU 主机（如 `192.168.1.27`，用户 `chenx2`），建议部署目录 `~/medication-action-worker/`。

### 文件传输清单

| 文件 | 是否必传 | 说明 |
| :--- | :--- | :--- |
| `action_worker.py` | **必传** | 生产 Daemon 本体 |
| `label_schema.py` | **必传** | 生产与离线脚本共用的标签规范，不可漏传 |
| `run.sh` | **必传** | 生命周期管理脚本 |
| `requirements.txt` | 仅首次 | 依赖安装清单 |
| `.env` | **必传（手动创建）** | 含 broker 密码和 API 地址，由部署人员基于 `.env.example` 本地配置 |
| `run_35_chunked.py`、`run_35.py` | 可选 | 离线评测与冒烟工具 |
| `tests/` | 可选 | 测试脚本 |

上传后在目标机执行：

```bash
cd ~/medication-action-worker
sed -i 's/\r$//' run.sh     # 消除 Windows CRLF 换行符
chmod +x run.sh
./run.sh check              # 启动前自检：模型 / ffmpeg / GPU / RabbitMQ / Core API
./run.sh start              # 后台启动服务（写日志至 worker.log，PID 写至 worker.pid）
```

---

## 环境准备与依赖安装

### 1. 系统要求
- **GPU**：NVIDIA GeForce RTX 4090 (24GB VRAM)
- **操作系统**：Linux (Ubuntu 22.04+) 或 Windows 11 (CUDA 12.1+)
- **多媒体**：系统中已安装 `ffmpeg` 与 `ffprobe` 并加入 `PATH`

### 2. Python 依赖
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

*(可选加速)* RTX 4090 采用 Ada 架构，可按需安装 `flash-attn`（FlashAttention-2）：
```bash
pip install flash-attn --no-build-isolation
```

### 3. 模型准备
推荐将 `Qwen/Qwen3.5-4B` 权重放置在 `~/models/Qwen3.5-4B`。

---

## 运行模式与常用命令

### 1. 生产模式：使用 `run.sh` 管理后台守护进程

```bash
cp docs/.env.example .env    # 首次使用时按需修改配置
./run.sh check          # 启动前自检环境与网络
./run.sh                # 前台启动，实时查看日志
./run.sh start          # 后台启动守护进程
./run.sh status         # 查看进程运行状态与 GPU 显存占用
./run.sh log            # 跟踪实时日志 (tail -f)
./run.sh stop           # 停止服务并释放 GPU 显存
```

原生命令行启动方式：
```bash
python action_worker.py \
  --model ~/models/Qwen3.5-4B \
  --api-url http://localhost:8080 \
  --rabbitmq-url amqp://guest:guest@localhost:5672/ \
  --queue platform.ai.action-candidate.v1 \
  --fps 1.0 \
  --bearer-token-file /etc/medication/platform-worker.token
```

### 2. 离线工具：长视频有界切片评测

脱离 RabbitMQ 和平台服务，直接对本地视频进行全流程推理评测，生成 JSON 标注结果：
```bash
python run_35_chunked.py data/videos/video1.mp4 --model ~/models/Qwen3.5-4B --fps 1.0
```

### 3. 离线工具：30s 短视频冒烟测试

快速验证驱动、模型权重加载与多模态 Pipeline：
```bash
python run_35.py data/videos/test_30s.mp4 --model ~/models/Qwen3.5-4B --fps 1.0
```

---

## 输出数据格式示例

推理输出为标准的动作候选结构，包含目标物品受体字段：

```json
[
  {
    "start_sec": 1,
    "end_sec": 3,
    "label": "open_container",
    "confidence": 0.95,
    "source_window": 1,
    "local_start_sec": 1,
    "local_end_sec": 3
  },
  {
    "start_sec": 5,
    "end_sec": 7,
    "label": "acquire_item",
    "item": "syringe",
    "item_name": "注射器",
    "confidence": 0.92,
    "source_window": 1,
    "local_start_sec": 5,
    "local_end_sec": 7
  },
  {
    "start_sec": 12,
    "end_sec": 22,
    "label": "draw_liquid",
    "confidence": 0.98,
    "source_window": 1,
    "local_start_sec": 12,
    "local_end_sec": 22
  }
]
```

---

## 常见问题排查 (FAQ)

### 1. 显存一直占着约 8.5 GB，是不是没释放？
不是。对 Qwen3.5-4B 来说，约 8.5 GB 是 bf16 模型常驻权重，属于“零冷启动延迟”的设计特性；推理时峰值约为 11.9 GB。若业务场景需要在空闲时完全释放显存，可在 `.env` 中设置 `IDLE_OFFLOAD_SECONDS=300`，空闲 5 分钟后模型将自动卸载并清空 VRAM。

### 2. 日志提示 `AMQPConnectorSocketConnectError: timeout`
- 请先在 Worker 端执行 `./run.sh check`，定位网络阻断点；
- 检查 Windows PortProxy 端口转发规则（5673 -> 5672）与防火墙入站规则；
- 检查第三方杀毒/主机防护软件（如火绒等）是否拦截了外部网段的端口访问。

### 3. 为什么任务推理时间较长时连接被断开？
- 密集视频解码与多切片推理期间会阻塞单线程事件循环数分钟；
- Worker 默认将 `RABBITMQ_HEARTBEAT` 设为 `0`（关闭 Broker 心跳检测），请勿随意改回正数。

### 4. 任务租约返回 403（`AI_HAND_POSE_WORKER_IDENTITY_MISMATCH`）
- 平台的身份匹配要求报送的 `worker_id` 与凭证推导出的身份一致；
- 请清空 `.env` 中的 `WORKER_ID=` 配置（留空），让 Worker 自动提取 Token 中的 `sub` 或回退到默认本地标识，不要手动填写自造名称。

### 5. 错误消息丢弃与重投策略
Worker 遵循严格的分级处理机制：
- **Core API 401/403**：不 ACK、不丢弃，休眠退避后重试（防止因短暂配置错误导致任务永久在队列中丢失）；
- **对象存储 401/403**：丢弃该消息，等待平台派发新 attempt 重新生成预签名 URL；
- **400 / 404 / 409**：非法消息或过期租约，直接丢弃；
- **5xx / 429 / 网络抖动**：重投一次，仍失败则丢弃让平台重新调度。
