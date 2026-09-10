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
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

PID_FILE="worker.pid"
LOG_FILE="worker.log"

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
    echo "========================================================================"
    echo "【前台运行模式】Action Worker (按 Ctrl+C 随时安全退出)"
    echo "  - 模型路径: $MODEL_PATH"
    echo "  - Core API: $PLATFORM_API_URL"
    echo "  - 消息队列: $RABBITMQ_URL"
    echo "  - 采样帧率: $SAMPLING_FPS fps"
    echo "========================================================================"
    python action_worker.py \
        --model "$MODEL_PATH" \
        --api-url "$PLATFORM_API_URL" \
        --rabbitmq-url "$RABBITMQ_URL" \
        --queue "$RABBITMQ_QUEUE" \
        --fps "$SAMPLING_FPS"
}

start_background() {
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
    echo "  - 日志文件: $LOG_FILE"
    echo "========================================================================"

    nohup python action_worker.py \
        --model "$MODEL_PATH" \
        --api-url "$PLATFORM_API_URL" \
        --rabbitmq-url "$RABBITMQ_URL" \
        --queue "$RABBITMQ_QUEUE" \
        --fps "$SAMPLING_FPS" > "$LOG_FILE" 2>&1 &

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
    -h|--help|help)
        echo "使用方法: $0 [run|start|stop|restart|status|log]"
        echo ""
        echo "  ./run.sh         - (默认) 前台直接运行，实时看日志，按 Ctrl+C 停止"
        echo "  ./run.sh start   - 后台守护进程启动 (输出到 worker.log)"
        echo "  ./run.sh stop    - 优雅停止后台进程并释放 4090 显存"
        echo "  ./run.sh restart - 重启后台服务"
        echo "  ./run.sh status  - 查看运行状态与显存占用"
        echo "  ./run.sh log     - 实时查看后台运行日志"
        exit 0
        ;;
    *)
        echo "未知指令: $1"
        echo "执行 $0 --help 查看用法"
        exit 1
        ;;
esac
