#!/bin/bash
# Step 2: 构建数据集 + 用 target processor 编码（生成 input_ids/loss_mask/messages）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source ./common_env.sh

# 2.1 数据归一化 -> conversations.jsonl
#     - ALLaVA .mm.json（json 数组, value 类型混杂 str/list）需归一化为 jsonl，
#       否则 Arrow 报 "cannot mix list and non-list, non-null values"
#     - parquet 源（如图像内嵌数据集）同样由 01 脚本转换
if [ ! -f "${DATASET_DIR}/conversations.jsonl" ]; then
    if [ -f "${RAW_DATA_DIR}/ALLaVA-Instruct-LAION-4V.mm.json" ]; then
        python3 01_build_multimodal_dataset.py \
            --input "${RAW_DATA_DIR}/ALLaVA-Instruct-LAION-4V.mm.json" \
            --output-dir "${DATASET_DIR}"
    elif [ -f "${RAW_DATA_DIR}/train-00000-of-00001.parquet" ]; then
        python3 01_build_multimodal_dataset.py \
            --input "${RAW_DATA_DIR}/train-00000-of-00001.parquet" \
            --output-dir "${DATASET_DIR}" \
            --limit "${MAX_SAMPLES}"
    fi
fi

# 2.2 msModelSpec-Dev 预处理：用 Qwen3-Omni 的 AutoProcessor 编码
#     多模态消息会保留在输出中（后续通过 chat completions API 发给 vllm）
cd "${MSPEC_ROOT}"
python3 scripts/prepare_data.py \
    --model "${TARGET_MODEL}" \
    --data "${DATASET_DIR}/conversations.jsonl" \
    --output "${PREPARED_DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --overwrite

# 2.3 生成 DSpark 的纯文本 Qwen3 decoder config。Qwen3.6 target 是 MRoPE，
#     但多模态 DSpark draft 必须使用不含 mrope_section 的 1-D RoPE。
python3 "${SCRIPT_DIR}/02a_build_draft_config.py" \
    --target-model "${TARGET_MODEL}" \
    --output-dir "${DRAFT_CONFIG_DIR}" \
    --num-layers "${NUM_LAYERS}"

echo "数据准备完成: ${PREPARED_DATA_DIR}"
