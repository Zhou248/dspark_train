#!/bin/bash
# Qwen3.6-35B-A3B DSpark 训练公共环境变量（NPU 环境）
# source 本文件后再执行其他脚本

# ============ 路径配置 ============
export MSPEC_ROOT="/home/z00909726/msModelSpec-Dev"          # 训练框架仓库
export QD_VLLM_ROOT="/home/z00909726/vllm"                      # vllm 仓库（releases/v0.26.0, 含 DSpark 适配）
export WORK_DIR="/home/z00909726/scripts/qwen36_dspark/work" # 工作目录
export TARGET_MODEL="/home/z00909726/weights/Qwen3.6-35B-A3B"

# 多模态数据（手写公式: 1200 条 图像+LaTeX）
export RAW_DATA_DIR="/home/w00608002/models/ALLaVA-4V/allava_laion"
export DATASET_DIR="${WORK_DIR}/dataset"          # conversations 格式 jsonl + 导出图像
export PREPARED_DATA_DIR="${WORK_DIR}/prepared"   # prepare_data.py 输出
export CKPT_DIR="${WORK_DIR}/checkpoints"         # 训练产出
export DRAFT_CONFIG_DIR="${WORK_DIR}/draft_config" # 1-D Qwen3 draft decoder config
export HIDDEN_STATES_DIR="${WORK_DIR}/hidden_states"

mkdir -p "${WORK_DIR}" "${DATASET_DIR}" "${PREPARED_DATA_DIR}" "${CKPT_DIR}" "${HIDDEN_STATES_DIR}" "${DRAFT_CONFIG_DIR}"

# ============ 模型结构参数（必须与 target text_config 一致）============
# Qwen3.6-35B-A3B: num_hidden_layers=40, hidden_size=2048, num_attention_heads=16,
#          num_key_value_heads=2, head_dim=256, vocab_size=248320 (hybrid mamba+attn)
export SEQ_LENGTH=4096
export TARGET_LAYER_IDS="2 20 37"   # 训练与 vllm 启动必须一致；last layer(40) 自动附加
export NUM_LAYERS=3                 # 草稿层数
export BLOCK_SIZE=7                 # 必须 == vllm num_speculative_tokens
export DRAFT_VOCAB_SIZE=32000       # 缩减词表；设为 248320 可用全词表
export MAX_ANCHORS=3072
export MARKOV_RANK=32               # 与 vllm 侧 config 的 markov_rank 一致
export EPOCHS=5
export LR=3e-4
export MAX_SAMPLES="${MAX_SAMPLES:-1000}"
export VLLM_PORT=8000
# Qwen3.6 是 hybrid/Mamba 多模态模型。hidden-state 抽取先以单请求稳定配置
# 跑通，再逐级调到 2/4；不要一开始就叠加 async scheduling + AIV。
export QD_MAX_NUM_SEQS="${QD_MAX_NUM_SEQS:-1}"
export HS_CONCURRENCY="${HS_CONCURRENCY:-1}"
# 单条坏样本由 04a 记录并跳过；只有连续多条失败才视为服务异常并停止。
export HS_MAX_CONSECUTIVE_ERRORS="${HS_MAX_CONSECUTIVE_ERRORS:-20}"
export REPAIR_EXISTING_HS="${REPAIR_EXISTING_HS:-1}"
export ENABLE_AIV="${ENABLE_AIV:-0}"
export QD_ENABLE_EP="${QD_ENABLE_EP:-0}"

# ============ NPU 配置 ============
# vllm 抽隐状态使用 4 卡 TP（对齐 scripts/vllm_start.sh）
export QD_NPUS="4,5,6,7"   # vllm 抽取隐状态的卡
export QD_TP=4
export TRAIN_NPUS="8,9,10,11,12,13,14,15"   # 训练用后 8 卡
export TRAIN_NPROC=8                            # torchrun 进程数
export QD_ENDPOINT="http://localhost:${VLLM_PORT}/v1"

# NPU 通用环境
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/runtime/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=${PYTHONPATH}:${MSPEC_ROOT}/src
export QD_USE_V1=1
# vllm 四卡 serving 参数。AIV 对普通 serving 可能有性能收益，但在
# Qwen3.6 multimodal + extract_hidden_states 链路上先默认关闭以避免数值/索引异常。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
if [ "${ENABLE_AIV}" = "1" ]; then
    export HCCL_OP_EXPANSION_MODE="AIV"
else
    unset HCCL_OP_EXPANSION_MODE || true
fi
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
