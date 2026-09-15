# Medication Action Worker (Qwen3.5-4B / Qwen3-VL-8B baseline)

专为 **RTX 4090 (24GB 显存)** 优化的多模态视频动作候选标注 Worker 服务。默认使用 `Qwen3.5-4B`，保留 `Qwen3-VL-8B` 作为同口径离线对照；监听 RabbitMQ 消息队列，对配药过程视频进行动作片段识别与边界提取。

---

## 核心设计与工程基线

1. **显存常驻（Zero Cold-Start）**：服务启动后模型常驻显存，避免每次推理重复加载权重的冷启动延迟；默认 Qwen3.5-4B 的实测峰值约 11.9 GB，Qwen3-VL-8B 仍作为独立对照模型运行。
   - 空闲显存是否归还由 `IDLE_OFFLOAD_SECONDS` 控制；这属于显存策略，不是模型精度优化。
   - 当前 4090 显存足够，暂不默认量化；量化必须重新验证动作边界和短动作召回率。
2. **严格单任务串行（Anti-OOM）**：RabbitMQ 消费者配置 `prefetch_count=1`，确保单卡严格串行推理，从根本上防止多任务并发引发显存溢出。
3. **动态显存优化**：强制启用 PyTorch `expandable_segments:True` 动态分段内存分配器，消除长时运行下的显存碎片化。
4. **长视频有界重叠切片（Bounded Core-Window Chunking）**：
   - 核心评估区间（Core Window）：24 秒；
   - 前后上下文冗余（Context Padding）：前后各 3 秒；
   - 总推理窗口严格锁定在 `<= 30.0s`，**推理峰值**（含视觉激活与 KV cache）实测锁定在 `<= 19.86GB` 安全红线以内。
   - 注意区分：该红线是**峰值**而非常驻。空闲常驻由权重决定（≈17.1 GB），切片策略保证的是峰值不随视频时长增长。
5. **帧级硬切片（Frame-Accurate Slicing）**：利用 `ffmpeg libx264 ultrafast` 在本地提取硬切片，彻底消除 OpenCV/解码器关键帧漂移（Keyframe Drift）。
6. **防幻觉策略（Anti-Hallucination）**：专用 System Prompt 约束模型杜绝背诵 SOP，允许输出空列表 `[]`，并进行严格 Schema 校验。
7. **链路回传**：完成推理后结构化组装结果并 PUT 回传 Core API，请求携带 `X-AI-Lease-Token`；从下载、装载模型到回传的**全过程**由独立线程续期租约。生产任务下载后会校验租约声明的字节数与 SHA-256，避免处理被替换或截断的媒体。租约一旦被平台回收（心跳 404/409，或已越过 `lease_until`），会在当前窗口结束后**提前中止**，不再白烧 GPU。
8. **启动前自检 + 断线退避 + 可控重投**：
   - `./run.sh check` 在加载模型**之前**探测模型目录 / ffmpeg / GPU / RabbitMQ / Core API；broker 或 API 不可达时直接中止启动，避免「模型先吃掉 17GB 显存再空转」。可用 `SKIP_PRECHECK=1` 强制跳过。
   - 运行期断线采用 5s → 60s 指数退避重连，不再固定 5s 刷屏。
   - AMQP 心跳默认关闭（`RABBITMQ_HEARTBEAT=0`），以适配回调内长达数分钟的 ffmpeg 切片 + 多窗口 `generate()`。
   - 任务失败改为 `basic_nack`，并按**错误来源 + 状态码**分档（见第 6 节排查表）：Core API 的 401/403 既不确认也不丢弃、退避后重连；对象存储的 403 与其他永久错误丢弃；5xx / 429 / 网络抖动重投一次。不再静默 ACK 丢任务。

动作预标注是独立的可选作业；由标注工作流在需要时提交 `ai-action-job.v1`。Worker 不按项目配置筛选，直接按消息中的媒体与租约处理，因此同一实例可以服务所有项目的视频。MediaPipe 手部检测使用独立的 `platform.ai.hand-pose.v1` Worker，不依赖本 Worker。

---

## 识别动作分类规范

| 动作标识 (`label`) | 中文释义 | 说明 |
| :--- | :--- | :--- |
| `acquire_item` | 拿取容器/工具 | 手明确到达并取得将参与配药的容器或工具，物体离开原稳定位置并进入操作区；不标持续持有、调整位置或重复抓握 |
| `open_container` | 开启容器 | 明确打开药品主容器，如掰开安瓿或移除西林瓶/输液容器的保护部件；不包括拆注射器包装 |
| `disinfect_access` | 消毒容器接口 | 看到棉签或其他消毒用品实际擦拭瓶塞、安瓿颈或容器接口；仅出现用品不构成该动作 |
| `insert_needle` | 针头进入容器 | 清楚看到针头穿过瓶塞或接口并进入目标药品容器；仅靠近或被遮挡时不标注 |
| `draw_liquid` | 抽取药液 | 针头已进入容器且活塞向后移动，能看到药液进入注射器 |
| `transfer_liquid` | 转移药液 | 使用注射器把药液从一个容器推入另一个目标容器；不表示面向患者给药 |
| `agitate_solution` | 混匀药液 | 明确摇晃、翻转或振荡装有药液的容器以混匀；仅拿着或移动容器不标注 |
| `discard_sharps` | 处置锐器 | 明确把针头、安瓿或其他锐器放入锐器盒；仅出现锐器盒或丢弃普通耗材不标注 |
| `label_prepared_medication` | 给成品贴签 | 看到标签实际粘贴到注射器、药瓶或成品输液容器；仅查看、拿着标签或成品包装不标注 |

默认 prompt 包含上述 9 个可由视频直接证明的物理动作。任务若携带 `labels`、`label_definitions` 或 `label_schema.labels` 快照，worker 会优先使用其中的名称、description/criteria、正反例和排除条件；无法解释的新增标签会被跳过或使任务 fail-closed。

`RETRIEVE_MEDICINE`、`CHECK_INFO` 等泛化或无法由画面证明的标签不作为 Qwen 动作候选。历史任务中的旧 ID 只按其冻结 schema 兼容处理。`move_to`、`grasp`、`release` 等机器人执行原语属于控制系统数据，不作为 Qwen 视频标签。标签定义的唯一来源是 [label_schema.py](label_schema.py)；平台新增标签必须同时提交 description/criteria，否则 worker 会 fail-closed。

---

## 目录结构（单层平铺，开箱即用）

```text
medication-action-worker/
├── run.sh                # 运行与生命周期管理脚本 (./run.sh 前台直接跑，支持 start/stop/log)
├── action_worker.py      # 核心生产常驻 Daemon（RabbitMQ 消费、切片推理、平台回传）
├── run_35_chunked.py     # Qwen3.5-4B 长视频切片评测
├── run_35.py             # Qwen3.5-4B 30s 单窗口快速冒烟测试
├── run_8b_chunked.py     # Qwen3-VL-8B 长视频离线对照评测
├── run_8b.py             # Qwen3-VL-8B 30s 离线对照测试
├── label_schema.py       # Qwen3.5/3-VL 共用的标签定义与版本
├── requirements.txt      # Python 依赖清单
├── .env.example          # 环境变量配置模板
├── .gitignore            # Git 忽略规则（过滤视频、图片与模型输出）
└── README.md             # 项目文档
```

---

## 部署到 4090 主机（FTP 清单）

目标：Linux GPU 主机（当前部署 `192.168.1.27`，用户 `chenx2`），建议部署目录 `~/medication-action-worker/`。

| 文件 | 是否必传 | 说明 |
| :--- | :--- | :--- |
| `action_worker.py` | **必传** | 生产 Daemon 本体 |
| `label_schema.py` | **必传** | 生产与离线脚本共用的标签 schema；不可漏传 |
| `run.sh` | **必传** | 生命周期管理（启动前自检 / start / stop / status / log） |
| `requirements.txt` | 仅首次 | 安装依赖用，之后不必重复传 |
| `.env` | **必传，但不要从仓库取** | 含真实 broker 密码，已在 `.gitignore` 中、不会随仓库分发。请在目标机按 `.env.example` 手工创建，或单独私密传输 |
| `run_35_chunked.py`、`run_35.py`、`run_8b_chunked.py`、`run_8b.py` | 可选 | 离线评测与冒烟测试，不参与生产 |
| `README.md` | 可选 | 文档 |

**不要传**：`data/`（本地测试视频）、`output/`（本地推理结果）、`.workbuddy/`（本地 Agent 工作区）、`.git/`。

上传后必须执行：

```bash
cd ~/medication-action-worker
sed -i 's/\r$//' run.sh     # Windows 侧检出 / FTP 直传通常是 CRLF，Linux 会报 bad interpreter: /usr/bin/env bash^M
chmod +x run.sh
./run.sh check              # 自检：模型 / ffmpeg / GPU / RabbitMQ / Core API
./run.sh start              # 日志写入 worker.log，PID 写入 worker.pid
```

> 走 `git clone` 则不需要 `sed` 与 `chmod` 这两步：`.gitattributes` 已声明 `* text=auto eol=lf`，可执行位也随仓库保留。
> 但 **FTP 是二进制直传、不经过 git 的换行符归一化**，所以走 FTP 就必须手动修。

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

*(可选加速)* `flash-attn`、`causal-conv1d`、`flash-linear-attention` 都必须按当前 PyTorch/CUDA 版本在独立环境验证；不要直接覆盖生产环境。RTX 4090 可测试 FlashAttention-2，但 Qwen3.5 当前 warning 主要来自线性注意力 kernel：
```bash
# 仅示例：先查兼容矩阵，再安装并跑回归
pip install flash-attn --no-build-isolation
```

### 3. 模型下载
默认下载 `Qwen/Qwen3.5-4B` 至 `~/models/Qwen3.5-4B`；`Qwen3-VL-8B-Instruct` 仅用于离线对照。

---

## 快速使用

### 1. 生产模式：启动常驻 Worker Daemon

推荐直接用 `run.sh`（自动读取 `.env`、带启动前自检、带进程管理）：

```bash
cp .env.example .env   # 首次使用，按实际环境改好
./run.sh check         # 先自检：模型 / ffmpeg / GPU / RabbitMQ / Core API 连通性
./run.sh               # 前台运行，实时看日志，Ctrl+C 安全退出
./run.sh start         # 后台运行（日志写入 worker.log，PID 写入 worker.pid）
./run.sh status        # 查看运行状态 + 显存占用 + 各进程占用
./run.sh log           # 实时跟踪后台日志
./run.sh stop          # 优雅停止并释放显存
```

> `run` / `start` 会先自动执行一次 `check`；若 RabbitMQ 或 Core API 不可达则**直接中止启动**，
> 避免模型先占用约 17GB 显存却一直空转。确需强制启动：`SKIP_PRECHECK=1 ./run.sh start`

等价的原生命令行方式：

```bash
python action_worker.py \
  --model ~/models/Qwen3.5-4B \
  --api-url http://localhost:8080 \
  --rabbitmq-url amqp://guest:guest@localhost:5672/ \
  --queue platform.ai.action-candidate.v1 \
  --fps 1.0 \
  --heartbeat 0 \
  --bearer-token-file /etc/medication/platform-worker.token
```

也可以通过环境变量配置（参考 `.env.example`）：
```bash
export PLATFORM_API_URL=http://192.168.5.100:8080
export RABBITMQ_URL=amqp://guest:guest@192.168.5.100:5672/
export RABBITMQ_HEARTBEAT=0
# 令牌必须由部署环境提供；生产环境建议使用权限受限的文件并支持轮换
export PLATFORM_WORKER_BEARER_TOKEN_FILE=/etc/medication/platform-worker.token
python action_worker.py
```

### 2. 离线工具：长视频有界切片推理评估

用于脱离 RabbitMQ 和平台服务，直接测试完整长视频的动作识别与切片拼接效果：

```bash
python run_35_chunked.py video1.mp4 --model ~/models/Qwen3.5-4B --fps 1.0
```

### 3. 离线工具：30s 短视频快速冒烟测试

用于快速排查 GPU 驱动、模型加载、Transformers 多模态管道是否正常：

```bash
python run_35.py test_30s.mp4 --model ~/models/Qwen3.5-4B --fps 1.0
```

Qwen3-VL-8B 对照仍使用 `run_8b_chunked.py`；不要把 30 秒以上视频直接一次性送入 8B 模型，先使用分块脚本。

---

## 输出数据格式示例

推理输出为标准的动作候选数组：

```json
[
  {
    "start_sec": 12,
    "end_sec": 22,
    "label": "draw_liquid",
    "confidence": 0.98,
    "source_window": 1,
    "local_start_sec": 12,
    "local_end_sec": 22
  },
  {
    "start_sec": 47,
    "end_sec": 50,
    "label": "draw_liquid",
    "confidence": 0.96,
    "source_window": 2,
    "local_start_sec": 23,
    "local_end_sec": 26
  }
]
```

---

## 常见问题排查

### 1. Qwen3-VL-8B 显存一直占着 ~17GB，是不是没优化？

不是。对 Qwen3-VL-8B 来说，那 17GB 主要是 bf16 权重，属于设计上的「显存常驻」；Qwen3.5-4B 的常驻/峰值应以当前模型和日志为准。
`expandable_segments` 只抗碎片、24s 核心切片只压推理峰值，都不会降低这个常驻基线。

判断是否真的「卡住空转」看利用率：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

占着 17GB 但 `utilization.gpu` 长期为 0%，说明模型加载完却没在干活，几乎都是下游链路不通。

### 2. 日志刷 `AMQPConnectorSocketConnectError: timeout(... 5673)` 连不上 RabbitMQ

模型会在 3 秒内加载完并占住显存，然后一直重连。**先在 worker 侧执行 `./run.sh check`**，它会直接指出卡在哪一环，不必等模型吃掉 17GB 再报错。

本项目 broker 的部署形态是 **Windows 主机 + 端口转发**（`192.168.1.7`）：RabbitMQ 本体只监听 `127.0.0.1:5672`，对外通过 Windows portproxy 暴露：

```
netsh interface portproxy show all
# 0.0.0.0  5673  ->  127.0.0.1  5672
# 0.0.0.0 19001  ->  127.0.0.1 19000
```

所以排查必须按这个形态来（在 `192.168.1.7` 上，需管理员）：

```powershell
netstat -ano | findstr :5673              # 应看到 svchost.exe(iphlpsvc) 在 LISTENING
netsh interface portproxy show all        # 确认 5673 -> 127.0.0.1:5672 转发存在
sc query iphlpsvc                         # IP Helper 服务必须在跑，否则转发失效

# 关键一步：防火墙必须单独放行 5673
netsh advfirewall firewall add rule name="RabbitMQ-5673" dir=in action=allow protocol=TCP localport=5673 remoteip=192.168.1.0/24
```

> 两个已经踩过的坑，别再绕：
>
> 1. **`8080` 通不代表防火墙没问题。** Core API 由 `java.exe` 直接监听 `0.0.0.0:8080`，且有一条 `java` 程序规则放行**任意端口**，所以它天然可达；而 5673 由系统服务转发，**必须有对应的端口放行规则**，默认入站策略是拒绝。用一条链路通去推断「网络没问题」会得出完全错误的结论。
> 2. **在 `192.168.1.7` 本机探测 5673 一定会成功**——本机连自身 IP 不受该入站规则约束。只有从 worker 侧（`192.168.1.27`）探测才能暴露问题。
>
> 若放行后仍不通，检查第三方安全软件（如火绒 HipsDaemon 等 HIPS）的网络防护，它独立于 Windows 防火墙单独拦截端口。

在 worker 主机上确认：

```bash
nc -zv 192.168.1.7 5673
./run.sh check          # 期望 [√] RabbitMQ TCP 可达
```

### 3. 任务跑到一半连接被 broker 断开

回调里跑 ffmpeg + 多窗口 `generate()` 会阻塞事件循环数分钟，默认 60s AMQP 心跳必然超时。
本项目默认已设 `RABBITMQ_HEARTBEAT=0`，请勿改回正数；若确需心跳，必须把消费循环改成独立线程。

### 4. 确认运行中的进程真的吃到了显存优化开关

```bash
tr '\0' '\n' < /proc/<worker_pid>/environ | grep -i ALLOC
# 期望看到 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 5. `/lease` 返回 403（`AI_HAND_POSE_WORKER_IDENTITY_MISMATCH`）

平台已经认出你是谁了，只是请求体里自报的 `worker_id` 与它认定的身份不相等 ——
**401 是没通过，403 是通过了但名字对不上**。所以看到 403 说明 Token 完全正确，差的只是一个字符串。

不要手写这个名字。身份是凭证的推论：worker 先取令牌的 `sub`（生产 JWT 模式），取不到才回落到
平台 local 替身的默认值 `ai-worker-local-1`。`.env` 里的 `WORKER_ID` 留空即可；填一个自造的名字
（例如 `action-worker-4090-01`）与平台身份必然不等，默认部署就会 403。

```bash
grep -n WORKER_ID .env      # 期望: WORKER_ID=  （等号后面什么都没有）
```

启动日志里的 `worker_id_source` 会说明身份是从哪来的：`jwt-sub` / `local-default` / `configured`。

> worker 遇到 **Core API 的** 401/403 时**既不确认也不丢弃**该消息。平台只回收「已认领且租约过期」的
> 尝试，从未被认领的 QUEUED 作业不会自愈 —— 消息一丢，这个作业就永久卡住。此时进程按最大退避
> 重连，修好配置后会自动继续处理那条消息。
>
> 注意区分：**对象存储拒绝下载时返回的 403 完全不同**（预签名 URL 过期 / SigV4 host 不匹配）。
> 它不是凭证问题，改配置也不会让旧 URL 复活，所以按「丢弃」处理 —— 若也判成凭证问题，
> 这条消息会被无限重投，worker 会永久卡死在它上面。

### 6. 日志出现「不可恢复，丢弃该消息」

这是有意为之，不是 bug。worker 按**错误来源 + 状态码**分档处理：

| 失败来源 | 状态码 | 处理 | 原因 |
|---|---|---|---|
| Core API（lease / heartbeat / result） | 401 / 403 | 不 ACK、不丢弃，退避后重连 | 凭证 / 身份问题，丢消息会让作业永久卡住（见上一条）|
| 对象存储（预签名下载） | 401 / 403 | 丢弃 | URL 过期或签名不匹配，重投不会自愈；丢弃后平台会派新 attempt 换新 URL |
| 任意 | 400 / 404 / 409 | 丢弃 | 这条消息对应的 attempt 已过期或不符契约，没有价值 |
| 任意 | 5xx / 429 / 网络 | 重投一次，仍失败则丢弃 | 平台暂时不可用 |

丢弃旧消息不等于放弃这个作业：**平台的租约回收会把它重新派发成一个新的 attempt**。
旧实现把 401/403 无条件当成凭证问题，既让对象存储的 403 变成无限重投，又让真正的
凭证失败「重投一次后丢弃」而永久卡住作业 —— 两个方向的错都修在这张表里。
