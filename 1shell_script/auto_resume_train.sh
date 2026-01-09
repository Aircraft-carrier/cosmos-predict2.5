#!/bin/bash

# ================== 配置区 ==================
MAX_RETRIES=100               # 最大重试次数
RETRY_DELAY=10                # 每次失败后等待 N 秒再重启（可选）
LOG_FILE="train_auto_restart.log"
CONFIG_PATH="cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py"
EXPERIMENT_NAME="ac_gen_reason_embeddings_rectified_flow_2b_256_320"
OVERRIDE_ARGS="~dataloader_train.dataloaders"  # Hydra overrides
WANDB_RUN_DIR="/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs/cosmos_predict2_action_generation/libero_cosmos_predict_v2p5/2b_libero_action_congeneratioin/wandb"

# 环境设置
export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs
cd /gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5 || exit 1
export PYTHONPATH=$PWD

# ================== 执行区 ==================
echo "[$(date)] Starting auto-restart training script..." | tee -a "$LOG_FILE"

retry_count=0
while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "[$(date)] Attempt $((retry_count + 1)) of $MAX_RETRIES" | tee -a "$LOG_FILE"
    
    # 运行训练命令（关键：不加 &，要等它退出才能捕获状态）
    # CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node=2 \
    CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node=2 \
        --master_port=12341 \
        -m scripts.train \
        --config="$CONFIG_PATH" -- \
        experiment="$EXPERIMENT_NAME" $OVERRIDE_ARGS

    exit_code=$?
    
    # === 新增：无论成功失败，都尝试同步 WandB ===
    if [ -d "$WANDB_RUN_DIR/latest-run" ]; then
        echo "[$(date)] Syncing WandB run..." | tee -a "$LOG_FILE"
        wandb sync "$WANDB_RUN_DIR/latest-run" >> "$LOG_FILE" 2>&1
    else
        echo "[$(date)] No latest-run found, skipping sync." | tee -a "$LOG_FILE"
    fi
    # ==========================================

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Training finished successfully!" | tee -a "$LOG_FILE"
        break
    else
        retry_count=$((retry_count + 1))
        echo "[$(date)] Training failed with exit code $exit_code. Retry $retry_count/$MAX_RETRIES." | tee -a "$LOG_FILE"
        
        if [ $retry_count -lt $MAX_RETRIES ]; then
            echo "[$(date)] Waiting ${RETRY_DELAY} seconds before retry..." | tee -a "$LOG_FILE"
            sleep $RETRY_DELAY
        else
            echo "[$(date)] Max retries reached. Exiting." | tee -a "$LOG_FILE"
        fi
    fi
done
# cd WANDB_RUN_DIR
# for d in offline-run-*; do wandb sync "$d"; done