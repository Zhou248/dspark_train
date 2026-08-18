#!/bin/bash
# Step 5: 转换 ckpt 为 Qwen3-Omni DSpark 格式，并用 vllm-ascend 投机推理启动服务
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

# 5.1 转换: 补齐 Qwen3OmniDSparkModel 所需的 config 字段与权重命名
python3 "${QD_VLLM_ROOT}/tools/convert_qwen3_dspark_to_qwen3_omni.py" \
    --draft-model "${CKPT_DIR}" \
    --target-model "${TARGET_MODEL}" \
    --output-dir "${CONVERTED_CKPT_DIR}" \
    --local-files-only

# 5.2 vllm-ascend 启动（draft + target 投机推理）
# block_size 必须 == num_speculative_tokens；markov_rank 等已在 ckpt config 中
ASCEND_RT_VISIBLE_DEVICES="${QD_NPUS}" python3 -m vllm.entrypoints.cli.main serve "${TARGET_MODEL}" \
    --port 8100 \
    --trust-remote-code \
    --max-model-len $((SEQ_LENGTH + 1024)) \
    --gpu-memory-utilization 0.85 \
    --allowed-local-media-path / \
    --speculative-config "{
        \"method\": \"dspark\",
        \"model\": \"${CONVERTED_CKPT_DIR}\",
        \"num_speculative_tokens\": ${BLOCK_SIZE}
    }"

# 验证（多模态请求示例）:
# curl http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" -d '{
#   "model": "qwen3-omni",
#   "messages": [{"role":"user","content":[
#       {"type":"image_url","image_url":{"url":"file:///path/to/img.png"}},
#       {"type":"text","text":"请识别图中的手写内容，并用 LaTeX 表示。"}]}]
# }'
