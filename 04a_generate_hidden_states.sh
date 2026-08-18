#!/bin/bash
# Step 4a: 离线批量生成隐状态（依赖 Step 3 的 vllm 服务）
# 背景: Qwen3.6 multimodal + hybrid/Mamba + extract_hidden_states 先按
#       concurrency=1 运行。确认无 NaN 后可外部设置 HS_CONCURRENCY=2/4。
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

until curl -s "${QD_ENDPOINT}/models" > /dev/null 2>&1; do
    echo "waiting for vllm at ${QD_ENDPOINT} ..."; sleep 10
done

# 旧版生成器在“移动文件后校验”，NaN 文件可能已残留在目录里并被
# resume 当成已完成。默认在续跑前校验并只删除确认无效的 hs_*.safetensors。
if [ "${REPAIR_EXISTING_HS}" = "1" ]; then
    python3 repair_hidden_states.py \
        --prepared-data "${PREPARED_DATA_DIR}" \
        --hidden-states "${PREPARED_DATA_DIR}/hidden_states" \
        --delete-invalid
fi

cd "${MSPEC_ROOT}"
python3 scripts/data_generation_offline.py \
    --model "${TARGET_MODEL}" \
    --endpoint "${QD_ENDPOINT}" \
    --preprocessed-data "${PREPARED_DATA_DIR}" \
    --output "${PREPARED_DATA_DIR}/hidden_states" \
    --max-samples "${MAX_SAMPLES}" \
    --concurrency "${HS_CONCURRENCY}" \
    --request-timeout 600 \
    --max-retries 3 \
    --max-consecutive-errors "${HS_MAX_CONSECUTIVE_ERRORS}" \
    --validate-outputs

echo "隐状态生成完成: ${PREPARED_DATA_DIR}/hidden_states"
