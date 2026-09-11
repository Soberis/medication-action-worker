#!/usr/bin/env bash
# ==============================================================================
# Medication Action Worker 4090 运行与管理脚本
# 用法:
#   ./run.sh            - (默认) 前台直接运行，实时看日志，按 Ctrl+C 停止
#   ./run.sh start      - 后台启动守护进程 (写入 worker.log 和 worker.pid)
#   ./run.sh stop       - 停止后台进程并释放 4090 显存
#   ./run.sh restart    - 重启后台服务
#   ./run.sh status     - 查看运行状态与显存占用
#   ./run.sh log        - 实时查看后台运行日志 (tail -f)
#   ./run.sh check      - 启动前自检 (模型/ffmpeg/GPU/RabbitMQ/Core API 连通性)
#
# 说明:
#   run / start 会先自动执行 check；若 RabbitMQ 或 Core API 不可达则直接退出，
#   避免「模型徒占 17GB 显存却空转」的情况。确需跳过时: SKIP_PRECHECK=1 ./run.sh start
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 进程环境中的显式身份优先于 .env，方便临时纠正部署配置。
# 之前 .env 里的空 WORKER_ID 会覆盖 `export WORKER_ID=...`，导致修正看似生效、实际仍用默认值。
PROCESS_WORKER_ID="${WORKER_ID:-${AI_WORKER_ID:-}}"

# 加载 .env 环境变量 (如果存在)
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    source ".env"
    set +a
fi

# 默认配置 (可被环境变量或 .env 覆盖)
MODEL_PATH="${MODEL_PATH:-/home/chenx2/models/Qwen3-VL-8B-Instruct}"
PLATFORM_API_URL="${PLATFORM_API_URL:-http://192.168.1.7:8080/api/v1}"
RABBITMQ_URL="${RABBITMQ_URL:-amqp://medication_local:change-me-local-only@192.168.1.7:5673/medication}"
RABBITMQ_QUEUE="${RABBITMQ_QUEUE:-platform.ai.action-candidate.v1}"
SAMPLING_FPS="${SAMPLING_FPS:-1.0}"
# 长阻塞推理(ffmpeg 切片 + 多窗口 generate)期间无法 pump 事件循环，
# 心跳保持 0 关闭，否则默认 60s 心跳会被 broker 判定失联而断开连接。
RABBITMQ_HEARTBEAT="${RABBITMQ_HEARTBEAT:-0}"
SKIP_PRECHECK="${SKIP_PRECHECK:-0}"
# 身份默认留空：由 action_worker.py 自动推导（令牌 sub，否则 ai-worker-local-1）。
# 不再默认一个自造的名字 —— 那会让默认部署与平台替身身份必然不一致而返回 403。
WORKER_ID="${PROCESS_WORKER_ID:-${WORKER_ID:-${AI_WORKER_ID:-}}}"
unset PROCESS_WORKER_ID
PLATFORM_WORKER_BEARER_TOKEN="${PLATFORM_WORKER_BEARER_TOKEN:-${AI_WORKER_BEARER_TOKEN:-}}"
PLATFORM_WORKER_BEARER_TOKEN_FILE="${PLATFORM_WORKER_BEARER_TOKEN_FILE:-${AI_WORKER_BEARER_TOKEN_FILE:-}}"
IDLE_OFFLOAD_SECONDS="${IDLE_OFFLOAD_SECONDS:-300}"

PID_FILE="worker.pid"
LOG_FILE="worker.log"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export RABBITMQ_HEARTBEAT

# timeout 在部分环境(如 macOS)不存在，缺失时降级为无超时探测，
# 否则会因 command not found 把可达的服务误判为不可达。
if command -v timeout >/dev/null 2>&1; then
    PROBE_TIMEOUT="timeout 4"
else
    PROBE_TIMEOUT=""
fi

tcp_probe() {
    local host="$1"
    local port="$2"
    if [ -z "$host" ] || [ -z "$port" ]; then
        return 1
    fi
    if command -v nc >/dev/null 2>&1; then
        if ${PROBE_TIMEOUT} nc -z "$host" "$port" >/dev/null 2>&1; then
            return 0
        fi
        return 1
    fi
    if ${PROBE_TIMEOUT} bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

split_hostport() {
    local url="$1"
    local default_port="$2"
    local rest="${url#*://}"
    rest="${rest#*@}"
    rest="${rest%%/*}"
    local h="${rest%%:*}"
    local p="${rest##*:}"
    if [ "$p" = "$rest" ]; then
        p="$default_port"
    fi
    printf '%s %s\n' "$h" "$p"
}

check_environment() {
    local failed=0
    echo "========================================================================"
    echo "【Action Worker 启动前自检】"
    echo "------------------------------------------------------------------------"

    if [ -d "$MODEL_PATH" ]; then
        local shards
        # 结尾的 || true: 配合 set -o pipefail，避免 find 失败时赋值语句非零导致脚本直接退出
        shards="$(find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l || true)"
        echo "[√] 模型目录: $MODEL_PATH (safetensors 分片: ${shards})"
    else
        echo "[X] 模型目录不存在: $MODEL_PATH"
        failed=1
    fi

    local bin
    for bin in ffmpeg ffprobe; do
        if command -v "$bin" >/dev/null 2>&1; then
            echo "[√] $bin: $(command -v "$bin")"
        else
            echo "[X] 缺少可执行文件: $bin"
            failed=1
        fi
    done

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[√] GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
    else
        echo "[X] 未检测到 nvidia-smi"
        failed=1
    fi

    if [ "$IDLE_OFFLOAD_SECONDS" -lt 0 ]; then
        echo "[*] 显存策略: 永久常驻模式 (IDLE_OFFLOAD_SECONDS=${IDLE_OFFLOAD_SECONDS}, 启动即占 17GB)"
    elif [ "$IDLE_OFFLOAD_SECONDS" -eq 0 ]; then
        echo "[*] 显存策略: 严格即用即卸模式 (IDLE_OFFLOAD_SECONDS=0, 平时 0MB, 单任务完结立即释放)"
    else
        echo "[*] 显存策略: 弹性显存引擎 (IDLE_OFFLOAD_SECONDS=${IDLE_OFFLOAD_SECONDS}s, 平时 0MB, 首发 1.6s 唤醒, 空闲 ${IDLE_OFFLOAD_SECONDS}s 归还显存)"
    fi

    echo "------------------------------------------------------------------------"
    local rmq_host rmq_port
    read -r rmq_host rmq_port <<< "$(split_hostport "$RABBITMQ_URL" 5672)"
    echo "[*] RabbitMQ: ${rmq_host}:${rmq_port}  (queue=${RABBITMQ_QUEUE}, heartbeat=${RABBITMQ_HEARTBEAT}s)"
    if tcp_probe "$rmq_host" "$rmq_port"; then
        echo "[√] RabbitMQ TCP 可达"
    else
        echo "[X] RabbitMQ TCP 不可达 —— 请先排查修复网络连接再启动！"
        echo "    请在 broker 主机 (${rmq_host}) 上依次确认:"
        echo "      1) ss -lntp | grep ${rmq_port}                # 是否真的在监听"
        echo "      2) 监听地址不是仅 127.0.0.1 (rabbitmq.conf: listeners.tcp.default = 0.0.0.0:${rmq_port})"
        echo "      3) sudo firewall-cmd --add-port=${rmq_port}/tcp   或   sudo ufw allow ${rmq_port}/tcp"
        echo "      4) sudo rabbitmqctl list_vhosts              # 确认 vhost 已创建"
        failed=1
    fi

    echo "------------------------------------------------------------------------"
    local api_host api_port
    read -r api_host api_port <<< "$(split_hostport "$PLATFORM_API_URL" 8080)"
    if tcp_probe "$api_host" "$api_port"; then
        echo "[√] Core API TCP 可达: ${api_host}:${api_port}"
        if command -v curl >/dev/null 2>&1; then
            local code
            # --noproxy '*'：避免本机 HTTP_PROXY/HTTPS_PROXY 环境变量把探测引到代理上，
            # 导致拿到代理自己的 502 而误判目标服务健康状态。
            code="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "$PLATFORM_API_URL" 2>/dev/null || true)"
            if [ -z "$code" ] || [ "$code" = "000" ]; then
                echo "    [!] HTTP 探测无响应（TCP 通但无 HTTP 应答），请确认服务已就绪。"
            elif [ "${code:0:1}" = "5" ]; then
                echo "    [!] HTTP ${code}：网关有响应但后端不可用，结果回传时大概率会失败。"
            else
                echo "    [√] HTTP ${code}"
            fi
        fi
    else
        echo "[X] Core API TCP 不可达: ${api_host}:${api_port}"
        echo "    请确认 ${api_host}:${api_port} 上的服务已启动，且本机出站可访问该端口。"
        failed=1
    fi

    echo "------------------------------------------------------------------------"
    if [ -n "$PLATFORM_WORKER_BEARER_TOKEN" ] && [ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ]; then
        echo "[X] Bearer Token 配置冲突：PLATFORM_WORKER_BEARER_TOKEN 与 *_FILE 只能设置一个"
        failed=1
    elif [ -n "$PLATFORM_WORKER_BEARER_TOKEN" ] || [ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ]; then
        if [ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ]; then
            if [ -r "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ] && [ -n "$(tr -d '[:space:]' < "$PLATFORM_WORKER_BEARER_TOKEN_FILE")" ]; then
                echo "[√] Bearer Token: 已配置 (file)"
            else
                echo "[X] Bearer Token 文件不可读或为空: $PLATFORM_WORKER_BEARER_TOKEN_FILE"
                failed=1
            fi
        else
            echo "[√] Bearer Token: 已配置 (inline)"
        fi
    else
        echo "[X] Bearer Token: 未配置 (Core API 鉴权将返回 401)！"
        failed=1
    fi

    echo "========================================================================"
    return $failed
}

precheck_or_exit() {
    local mode="$1"
    if [ "$SKIP_PRECHECK" = "1" ]; then
        echo "[*] SKIP_PRECHECK=1，已跳过启动前自检。"
        return 0
    fi
    if check_environment; then
        return 0
    fi
    echo ""
    echo "[X] 启动前自检未通过，已中止启动（避免白占 17GB 显存空转）。"
    echo "    修复上述问题后重试；确需强制启动请执行: SKIP_PRECHECK=1 $0 $mode"
    exit 1
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_foreground() {
    precheck_or_exit run
    echo "========================================================================"
    echo "【前台运行模式】Action Worker (按 Ctrl+C 随时安全退出)"
    echo "  - 模型路径: $MODEL_PATH"
    echo "  - Core API: $PLATFORM_API_URL"
    echo "  - 消息队列: $RABBITMQ_URL"
    echo "  - 采样帧率: $SAMPLING_FPS fps"
    echo "  - 显存策略: 空闲超时 ${IDLE_OFFLOAD_SECONDS}s 自动卸载 (弹性保活)"
    echo "  - 鉴权令牌: $([ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ] && echo file || echo inline)"
    echo "========================================================================"
    local auth_args=()
    if [ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ]; then auth_args+=(--bearer-token-file "$PLATFORM_WORKER_BEARER_TOKEN_FILE"); else auth_args+=(--bearer-token "$PLATFORM_WORKER_BEARER_TOKEN"); fi
    python action_worker.py \
        --model "$MODEL_PATH" \
        --api-url "$PLATFORM_API_URL" \
        --rabbitmq-url "$RABBITMQ_URL" \
        --queue "$RABBITMQ_QUEUE" \
        --fps "$SAMPLING_FPS" \
        --heartbeat "$RABBITMQ_HEARTBEAT" \
        --worker-id "$WORKER_ID" \
        "${auth_args[@]}" \
        --idle-offload-seconds "$IDLE_OFFLOAD_SECONDS"
}

start_background() {
    precheck_or_exit start
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[!] Action Worker 已在运行中 (PID: $pid)。如需重启请运行: ./run.sh restart"
        return 0
    fi

    echo "========================================================================"
    echo "【后台启动模式】正在启动 Action Worker 守护进程..."
    echo "  - 模型路径: $MODEL_PATH"
    echo "  - Core API: $PLATFORM_API_URL"
    echo "  - 消息队列: $RABBITMQ_URL"
    echo "  - 显存策略: 空闲超时 ${IDLE_OFFLOAD_SECONDS}s 自动卸载 (弹性保活)"
    echo "  - 鉴权令牌: $([ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ] && echo file || echo inline)"
    echo "  - 日志文件: $LOG_FILE"
    echo "========================================================================"

    local auth_args=()
    if [ -n "$PLATFORM_WORKER_BEARER_TOKEN_FILE" ]; then auth_args+=(--bearer-token-file "$PLATFORM_WORKER_BEARER_TOKEN_FILE"); else auth_args+=(--bearer-token "$PLATFORM_WORKER_BEARER_TOKEN"); fi
    nohup python action_worker.py \
        --model "$MODEL_PATH" \
        --api-url "$PLATFORM_API_URL" \
        --rabbitmq-url "$RABBITMQ_URL" \
        --queue "$RABBITMQ_QUEUE" \
        --fps "$SAMPLING_FPS" \
        --heartbeat "$RABBITMQ_HEARTBEAT" \
        --worker-id "$WORKER_ID" \
        "${auth_args[@]}" \
        --idle-offload-seconds "$IDLE_OFFLOAD_SECONDS" > "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        echo "[√] Action Worker 成功启动 (PID: $pid)"
        echo "[*] 查看实时日志请执行: ./run.sh log (或 tail -f $LOG_FILE)"
        echo "[*] 停止请执行: ./run.sh stop"
    else
        echo "[X] 启动失败，请检查日志:"
        tail -n 20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_worker() {
    if ! is_running; then
        echo "[*] Action Worker 未在运行。"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "[*] 正在停止 Action Worker (PID: $pid)..."

    kill -15 "$pid" 2>/dev/null || true

    local count=0
    while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 15 ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""

    if kill -0 "$pid" 2>/dev/null; then
        echo "[!] 优雅退出超时，强制终止进程 (SIGKILL)..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "[√] Action Worker 已完全停止，显存已释放。"
}

status_worker() {
    echo "========================================================================"
    echo "【Action Worker 状态检查】"
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[√] 运行状态: 正在运行 (PID: $pid)"
        echo "[*] 进程资源:"
        ps -p "$pid" -o pid,user,%cpu,%mem,etime,cmd 2>/dev/null || true
    else
        echo "[*] 运行状态: 未运行"
    fi
    echo "------------------------------------------------------------------------"
    echo "【当前 4090 GPU 显存】"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
        echo "[*] 各进程占用 (弹性模式下平时 0MB / 唤醒推理时约 17GB，常驻模式下约 17GB):"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    else
        echo "nvidia-smi 未检测到"
    fi
    echo "========================================================================"
}

show_log() {
    if [ ! -f "$LOG_FILE" ]; then
        touch "$LOG_FILE"
    fi
    echo "[*] 正在跟踪日志: $LOG_FILE (按 Ctrl+C 退出日志查看)"
    tail -f "$LOG_FILE"
}

case "${1:-run}" in
    run|"")
        start_foreground
        ;;
    start)
        start_background
        ;;
    stop)
        stop_worker
        ;;
    restart)
        stop_worker
        sleep 2
        start_background
        ;;
    status)
        status_worker
        ;;
    log)
        show_log
        ;;
    check)
        if check_environment; then
            echo "[√] 自检通过，可以启动: ./run.sh start"
            exit 0
        fi
        echo "[X] 自检未通过，请先修复上述问题。"
        exit 1
        ;;
    -h|--help|help)
        echo "使用方法: $0 [run|start|stop|restart|status|log|check]"
        echo ""
        echo "  ./run.sh         - (默认) 前台直接运行，实时看日志，按 Ctrl+C 停止"
        echo "  ./run.sh start   - 后台守护进程启动 (输出到 worker.log)"
        echo "  ./run.sh stop    - 优雅停止后台进程并释放 4090 显存"
        echo "  ./run.sh restart - 重启后台服务"
        echo "  ./run.sh status  - 查看运行状态与显存占用"
        echo "  ./run.sh log     - 实时查看后台运行日志"
        echo "  ./run.sh check   - 启动前自检 (模型/ffmpeg/GPU/RabbitMQ/Core API)"
        echo ""
        echo "环境变量: MODEL_PATH / PLATFORM_API_URL / RABBITMQ_URL / RABBITMQ_QUEUE"
        echo "          SAMPLING_FPS / RABBITMQ_HEARTBEAT / SKIP_PRECHECK"
        echo "          WORKER_ID (or AI_WORKER_ID，留空自动推导) / PLATFORM_WORKER_BEARER_TOKEN(_FILE) / IDLE_OFFLOAD_SECONDS"
        exit 0
        ;;
    *)
        echo "未知指令: $1"
        echo "执行 $0 --help 查看用法"
        exit 1
        ;;
esac
