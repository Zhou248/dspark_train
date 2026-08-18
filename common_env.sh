#!/bin/bash
# Qwen3-Omni (qwen3.6) DSpark 训练公共环境变量（NPU 环境）
# source 本文件后再执行其他脚本

# ============ 路径配置 ============
export MSPEC_ROOT="/home/z00909726/msModelSpec-Dev"          # 训练框架仓库
export QD_VLLM_ROOT="/home/z00909726/vllm"                      # vllm 仓库（releases/v0.26.0, 含 qwen3-omni dspark 适配）
export WORK_DIR="/home/z00909726/scripts/qwen36_dspark/work" # 工作目录
export TARGET_MODEL="/home/z00909726/weights/Qwen3.6-35B-A3B"

# 多模态数据（手写公式: 1200 条 图像+LaTeX）
export RAW_DATA_DIR="/home/w00608002/models/ALLaVA-4V/allava_laion"
export DATASET_DIR="${WORK_DIR}/dataset"          # conversations 格式 jsonl + 导出图像
export PREPARED_DATA_DIR="${WORK_DIR}/prepared"   # prepare_data.py 输出
export CKPT_DIR="${WORK_DIR}/checkpoints"         # 训练产出
export CONVERTED_CKPT_DIR="${WORK_DIR}/qwen3_omni_dspark_ckpt"  # 转换后可部署 ckpt
export HIDDEN_STATES_DIR="${WORK_DIR}/hidden_states"

mkdir -p "${WORK_DIR}" "${DATASET_DIR}" "${PREPARED_DATA_DIR}" "${CKPT_DIR}" "${HIDDEN_STATES_DIR}"

# ============ 模型结构参数（必须与 target text_config 一致）============
# Qwen3.6-35B-A3B: num_hidden_layers=40, hidden_size=2048, num_attention_heads=16,
#          num_key_value_heads=2, head_dim=256, vocab_size=248320 (hybrid mamba+attn)
export SEQ_LENGTH=4096
export TARGET_LAYER_IDS="2 20 37"   # 训练与 vllm 启动必须一致；last layer(40) 自动附加
export DRAFT_ARCH="qwen3"
export NUM_LAYERS=3                 # 草稿层数
export BLOCK_SIZE=7                 # 必须 == vllm num_speculative_tokens
export DRAFT_VOCAB_SIZE=32000       # 缩减词表；设为 248320 可用全词表
export MAX_ANCHORS=3072
export MARKOV_RANK=32               # 与 vllm 侧 config 的 markov_rank 一致
export EPOCHS=5
export LR=3e-4
export MAX_SAMPLES=1000
export VLLM_PORT=8000

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
# vllm 四卡 serving 性能参数（对齐 scripts/vllm_start.sh）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
