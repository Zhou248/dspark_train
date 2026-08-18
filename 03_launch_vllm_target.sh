#!/bin/bash
# Step 3: NPU 上启动 vllm 服务 target 模型，在线抽取 thinker 隐状态
# 使用 4,5,6,7 四卡 TP=4（对齐 scripts/vllm_start.sh 的配置）
# 注意: --allowed-local-media-path 必须覆盖图像目录，否则 file:// 图像请求会被拒绝
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

cd "${MSPEC_ROOT}"
ASCEND_RT_VISIBLE_DEVICES="${QD_NPUS}" python3 scripts/launch_vllm.py "${TARGET_MODEL}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --hidden-states-path "${HIDDEN_STATES_DIR}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${QD_TP}" \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 4 \
    --max-model-len $((SEQ_LENGTH + 1024)) \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --allowed-local-media-path / \
    --enforce-eager \
    --additional-config '{"enable_cpu_binding":true}' \
    --async-scheduling
