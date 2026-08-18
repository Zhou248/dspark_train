#!/bin/bash
# Step 5: 直接使用 msModelSpec 产出的 Qwen3 DSpark checkpoint 启动投机推理
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

BEST_CKPT="${CKPT_DIR}/checkpoint_best"
if [ ! -e "${BEST_CKPT}" ]; then
    echo "checkpoint_best 不存在: ${BEST_CKPT}" >&2
    echo "请确认训练使用了 --save-best 并完成至少一次验证。" >&2
    exit 1
fi

# vllm-ascend 启动（draft + target 投机推理）
# block_size 必须 == num_speculative_tokens；markov_rank 等已在 ckpt config 中
ASCEND_RT_VISIBLE_DEVICES="${QD_NPUS}" python3 -m vllm.entrypoints.cli.main serve "${TARGET_MODEL}" \
    --port 8100 \
    --tensor-parallel-size "${QD_TP}" \
    --enable-expert-parallel \
    --trust-remote-code \
    --max-model-len $((SEQ_LENGTH + 1024)) \
    --gpu-memory-utilization 0.85 \
    --allowed-local-media-path /home \
    --limit-mm-per-prompt '{"image":1}' \
    --enforce-eager \
    --no-enable-prefix-caching \
    --speculative-config "{
        \"method\": \"dspark\",
        \"model\": \"${BEST_CKPT}\",
        \"num_speculative_tokens\": ${BLOCK_SIZE},
        \"draft_sample_method\": \"greedy\"
    }"

# 验证（多模态请求示例）:
# curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" -d '{
#   "model": "/home/z00909726/weights/Qwen3.6-35B-A3B",
#   "messages": [{"role":"user","content":[
#       {"type":"image_url","image_url":{"url":"file:///path/to/img.png"}},
#       {"type":"text","text":"请识别图中的手写内容，并用 LaTeX 表示。"}]}]
# }'
