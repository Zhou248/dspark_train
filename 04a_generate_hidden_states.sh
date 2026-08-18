#!/bin/bash
# Step 4a: 离线批量生成隐状态（依赖 Step 3 的 vllm 服务）
# 背景: Qwen3.6 multimodal + hybrid/Mamba + extract_hidden_states 先按
#       concurrency=1 运行。确认无 NaN 后可外部设置 HS_CONCURRENCY=2/4。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source ./common_env.sh

until curl -s "${QD_ENDPOINT}/models" > /dev/null 2>&1; do
    echo "waiting for vllm at ${QD_ENDPOINT} ..."; sleep 10
done

# 旧版生成器在“移动文件后校验”，NaN 文件可能已残留在目录里并被
# resume 当成已完成。在生成前以及正常/异常退出时都扫描一次，
# 从而无需修改 msModelSpec-Dev 内部实现。
repair_invalid_hidden_states() {
    if [ "${REPAIR_EXISTING_HS}" != "1" ]; then
        return
    fi
    python3 "${SCRIPT_DIR}/repair_hidden_states.py" \
        --prepared-data "${PREPARED_DATA_DIR}" \
        --hidden-states "${PREPARED_DATA_DIR}/hidden_states" \
        --delete-invalid
}

repair_invalid_hidden_states
trap repair_invalid_hidden_states EXIT

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

trap - EXIT

echo "隐状态生成完成: ${PREPARED_DATA_DIR}/hidden_states"
