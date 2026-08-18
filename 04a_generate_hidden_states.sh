#!/bin/bash
# Step 4a: 离线批量生成隐状态（依赖 Step 3 的 vllm 服务）
# 背景: vllm 多模态批量 prefill 在并发 >8 时触发 AIV 索引越界崩溃
#       (vllm-ascend 对 qwen3.5_moe hybrid + extract_hidden_states 的 bug)，
#       因此用 --concurrency 4 限制客户端并发，慢速稳定地生成全部隐状态。
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

until curl -s "${QD_ENDPOINT}/models" > /dev/null 2>&1; do
    echo "waiting for vllm at ${QD_ENDPOINT} ..."; sleep 10
done

cd "${MSPEC_ROOT}"
python3 scripts/data_generation_offline.py \
    --model "${TARGET_MODEL}" \
    --endpoint "${QD_ENDPOINT}" \
    --preprocessed-data "${PREPARED_DATA_DIR}" \
    --output "${PREPARED_DATA_DIR}/hidden_states" \
    --max-samples "${MAX_SAMPLES}" \
    --concurrency 4 \
    --validate-outputs

echo "隐状态生成完成: ${PREPARED_DATA_DIR}/hidden_states"
