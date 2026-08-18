#!/bin/bash
# Step 4: NPU 上在线训练 Qwen3-Omni DSpark 草稿模型（依赖 Step 3 的 vllm 服务）
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

# 等待 vllm 就绪
until curl -s "${QD_ENDPOINT}/models" > /dev/null 2>&1; do
    echo "waiting for vllm at ${QD_ENDPOINT} ..."; sleep 10
done

cd "${MSPEC_ROOT}"
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS}" torchrun \
    --standalone --nproc_per_node "${TRAIN_NPROC}" \
    scripts/train.py \
    --verifier-name-or-path "${TARGET_MODEL}" \
    --draft-arch "${DRAFT_ARCH}" \
    --data-path "${PREPARED_DATA_DIR}" \
    --vllm-endpoint "${QD_ENDPOINT}" \
    --hidden-states-backend file \
    --hidden-states-path "${HIDDEN_STATES_DIR}" \
    --save-path "${CKPT_DIR}" \
    --speculator-type dspark \
    --num-layers "${NUM_LAYERS}" \
    --block-size "${BLOCK_SIZE}" \
    --max-anchors "${MAX_ANCHORS}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --draft-vocab-size "${DRAFT_VOCAB_SIZE}" \
    --markov-rank "${MARKOV_RANK}" \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --total-seq-len "${SEQ_LENGTH}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --on-missing generate \
    --on-generate delete

echo "训练完成，checkpoint: ${CKPT_DIR}"
