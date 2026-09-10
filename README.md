# Medication Action Worker (Qwen3-VL-8B)

专为 **RTX 4090 (24GB 显存)** 深度优化的多模态视频动作候选标注 Worker 服务。作为 [medication-video-platform](https://github.com/) 生态的异步 AI 推理节点，监听 RabbitMQ 消息队列，利用视觉大语言模型（VLM）对调配过程视频进行高精度的动作片段识别与边界提取。

---

## 核心设计与工程基线

1. **显存常驻（Zero Cold-Start）**：服务启动后模型常驻显存，避免每次推理重复加载数十 GB 权重的冷启动延迟。
   - **常驻基线 ≈ 17.1 GB**：其中约 16.4 GiB 是 Qwen3-VL-8B 的 bf16 权重本身，任何配置都压不动；其余约 0.3 GiB 为 CUDA 上下文与 cuBLAS 工作区。
   - 服务空闲时 `nvidia-smi` 显示 **~17.6 GB 占用、GPU 利用率 0%**，这属于设计预期，不是显存泄漏。
   - 若要进一步降低常驻占用，只能走量化（int8 ≈ 9.5 GB / nf4 ≈ 6.2 GB）或放弃常驻改为按需加载，二者都必须重新做动作时序与标签稳定性回归。
2. **严格单任务串行（Anti-OOM）**：RabbitMQ 消费者配置 `prefetch_count=1`，确保单卡严格串行推理，从根本上防止多任务并发引发显存溢出。
3. **动态显存优化**：强制启用 PyTorch `expandable_segments:True` 动态分段内存分配器，消除长时运行下的显存碎片化。
4. **长视频有界重叠切片（Bounded Core-Window Chunking）**：
   - 核心评估区间（Core Window）：24 秒；
   - 前后上下文冗余（Context Padding）：前后各 3 秒；
   - 总推理窗口严格锁定在 `<= 30.0s`，**推理峰值**（含视觉激活与 KV cache）实测锁定在 `<= 19.86GB` 安全红线以内。
   - 注意区分：该红线是**峰值**而非常驻。空闲常驻由权重决定（≈17.1 GB），切片策略保证的是峰值不随视频时长增长。
5. **帧级硬切片（Frame-Accurate Slicing）**：利用 `ffmpeg libx264 ultrafast` 在本地提取硬切片，彻底消除 OpenCV/解码器关键帧漂移（Keyframe Drift）。
6. **防幻觉策略（Anti-Hallucination）**：专用 System Prompt 约束模型杜绝背诵 SOP，允许输出空列表 `[]`，并进行严格 Schema 校验。
7. **链路回传**：完成推理后结构化组装结果并 PUT 回传 Core API，请求携带 `X-AI-Lease-Token`。
   > **当前实现不做租约续期（Auto-Lease Renewal）。** 单次推理可能耗时数分钟，若平台租约在此期间过期，回传会被拒（表现为 4xx），任务随即按 nack 语义重投一次。若要支持更长任务，正确做法是在 `process_job` 内起独立线程续期；把续期塞进被 ffmpeg/`generate()` 阻塞的消费回调里是无效的。
8. **启动前自检 + 断线退避 + 可控重投**：
   - `./run.sh check` 在加载模型**之前**探测模型目录 / ffmpeg / GPU / RabbitMQ / Core API；broker 或 API 不可达时直接中止启动，避免「模型先吃掉 17GB 显存再空转」。可用 `SKIP_PRECHECK=1` 强制跳过。
   - 运行期断线采用 5s → 60s 指数退避重连，不再固定 5s 刷屏。
   - AMQP 心跳默认关闭（`RABBITMQ_HEARTBEAT=0`），以适配回调内长达数分钟的 ffmpeg 切片 + 多窗口 `generate()`。
   - 任务失败改为 `basic_nack`：首次失败重投一次，重投后仍失败则不再 requeue，避免毒药消息无限死循环；不再静默 ACK 丢任务。

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
├── run_8b_chunked.py     # 长视频切片推理评测（**离线对照实现**；单文件自包含便于 FTP，线上行为以 action_worker.py 为准）
├── run_8b.py             # 30s 单窗口快速冒烟测试（单文件完整自包含）
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
| `run.sh` | **必传** | 生命周期管理（启动前自检 / start / stop / status / log） |
| `requirements.txt` | 仅首次 | 安装依赖用，之后不必重复传 |
| `.env` | **必传，但不要从仓库取** | 含真实 broker 密码，已在 `.gitignore` 中、不会随仓库分发。请在目标机按 `.env.example` 手工创建，或单独私密传输 |
| `run_8b_chunked.py`、`run_8b.py` | 可选 | 离线评测与冒烟测试，不参与生产 |
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

*(可选加速)* 如环境支持编译，可安装 `flash-attn` 以进一步提升长视觉 Token 的推理速度：
```bash
pip install flash-attn --no-build-isolation
```

### 3. 模型下载
将 `Qwen/Qwen3-VL-8B-Instruct` 权重下载至本地目录（如 `~/models/Qwen3-VL-8B-Instruct`）。

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
  --model ~/models/Qwen3-VL-8B-Instruct \
  --api-url http://localhost:8080 \
  --rabbitmq-url amqp://guest:guest@localhost:5672/ \
  --queue platform.ai.action-candidate.v1 \
  --fps 1.0 \
  --heartbeat 0
```

也可以通过环境变量配置（参考 `.env.example`）：
```bash
export PLATFORM_API_URL=http://192.168.5.100:8080
export RABBITMQ_URL=amqp://guest:guest@192.168.5.100:5672/
export RABBITMQ_HEARTBEAT=0
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

---

## 常见问题排查

### 1. 显存一直占着 ~17GB，是不是没优化？

不是。那 17GB 里约 16.4 GiB 就是 Qwen3-VL-8B 的 bf16 权重，属于设计上的「显存常驻」。
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
