#!/bin/bash
#
# Local performance test script for Qwen3.5-35B-A3B on Ascend NPU
# Scenario: 1 node, TP=2, input 128K, output 1K, TPOT target 50ms
#
# Usage:
#   1. Set MODEL_PATH to your local model path (or modify the default below)
#   2. Set ASCEND_VISIBLE_DEVICES to the NPU devices you want to use
#   3. Run: bash run_qwen3_5_35b_a3b_perf.sh
#

set -e

# ==================== Configurable Parameters ====================

# Model path - modify to your local path
export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B}"

# NPU device IDs (2 devices for TP=2)
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1}"

# ==================== Environment Variables ====================

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export DEEPEP_HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export ASCEND_USE_FIA=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=20
export GDN_ATTN_BACKEND_TRITON=1

# ==================== Run Test ====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/test/registered/ascend/performance/qwen3_5_35b_a3b/test_npu_qwen3_5_35b_a3b_1p_in128k_out1k_50ms.py"

echo "=========================================="
echo "Running Qwen3.5-35B-A3B performance test"
echo "  Model:   ${MODEL_PATH}"
echo "  Devices: ${ASCEND_VISIBLE_DEVICES}"
echo "  Test:    ${TEST_FILE}"
echo "=========================================="

python3 "${TEST_FILE}"
