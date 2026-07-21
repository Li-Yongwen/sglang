#!/bin/bash
#
# Local performance test script for Qwen3.5-35B-A3B on Ascend NPU
# Scenario: 1 node, TP=2, input 128K (90% shared prefix), output 1K, TPOT target 50ms
#
# Usage:
#   1. Set MODEL_PATH to your local model path
#   2. Set ASCEND_VISIBLE_DEVICES to the NPU devices you want to use
#   3. Run: bash run_qwen3_5_35b_a3b_prefix90_perf.sh
#

set -e

# ==================== Configurable Parameters ====================

# Model path - modify to your local path
MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B}"

# NPU device IDs (2 devices for TP=2)
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1}"

# Server host and port
HOST="127.0.0.1"
PORT="${PORT:-30066}"

# Benchmark parameters - generated-shared-prefix dataset
MAX_CONCURRENCY=103
NUM_PROMPTS=412
INPUT_LEN=128000
OUTPUT_LEN=1000
REPEAT_RATE=0.9
REQUEST_RATE="inf"
SEED=1

# ==================== Environment Variables ====================

export ASCEND_VISIBLE_DEVICES

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export ASCEND_USE_FIA=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=30

# ==================== Launch Server ====================

echo "=========================================="
echo "Launching sglang server for Qwen3.5-35B-A3B (prefix90)"
echo "  Model:    ${MODEL_PATH}"
echo "  Devices:  ${ASCEND_VISIBLE_DEVICES}"
echo "  TP size:  2"
echo "  Host:     ${HOST}:${PORT}"
echo "=========================================="

sglang serve \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size 2 \
    --nnodes 1 \
    --attention-backend ascend \
    --device npu \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 65536 \
    --trust-remote-code \
    --enable-prefill-delayer \
    --mamba-scheduler-strategy extra_buffer \
    --max-running-requests 103 \
    --max-mamba-cache-size 85 \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 2 4 8 16 32 48 64 80 96 103 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    &

SERVER_PID=$!

# ==================== Wait for Server Ready ====================

echo "Waiting for server to be ready..."
MAX_WAIT=600
ELAPSED=0
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is ready after ${ELAPSED}s"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... waiting (${ELAPSED}s/${MAX_WAIT}s)"
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "ERROR: Server failed to start within ${MAX_WAIT}s"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

# ==================== Run Benchmark ====================

echo ""
echo "=========================================="
echo "Running bench-serving benchmark (generated-shared-prefix)"
echo "  Input length:       ${INPUT_LEN}"
echo "  Output length:      ${OUTPUT_LEN}"
echo "  Shared prefix:      $(echo "${REPEAT_RATE} * 100" | bc)%"
echo "  Max concurrency:    ${MAX_CONCURRENCY}"
echo "  Num prompts:        ${NUM_PROMPTS}"
echo "  Request rate:       ${REQUEST_RATE}"
echo "=========================================="

python3 -m sglang.bench_serving \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --backend sglang \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group "${NUM_PROMPTS}" \
    --gsp-system-prompt-len "$(echo "${INPUT_LEN} * ${REPEAT_RATE} / 1" | bc)" \
    --gsp-question-len "$(echo "${INPUT_LEN} * (1 - ${REPEAT_RATE}) / 1" | bc)" \
    --gsp-output-len "${OUTPUT_LEN}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --seed "${SEED}"

# ==================== Cleanup ====================

echo ""
echo "Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

echo "Done."
