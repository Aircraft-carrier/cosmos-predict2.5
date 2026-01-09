#!/bin/bash

# 记录当前工作目录
ORIGINAL_DIR=$(pwd)

# 目标工作目录
TARGET_DIR="/gemini/space/telemen/zsh"

# 源路径前缀
SRC_BASE="/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5"

echo "切换到目标目录: $TARGET_DIR"
cd "$TARGET_DIR" || { echo "无法进入目标目录 $TARGET_DIR"; exit 1; }

echo "开始复制数据..."

# 执行复制命令
cp -r "$SRC_BASE/cosmos_predict2/_src/predict2/action_parallel_predict" ./
cp -r "$SRC_BASE/1shell_script" ./
cp -r "$SRC_BASE/scripts/pre_tokenizer" ./
cp -r "$SRC_BASE/cosmos_predict2/experiments/base/action_gen.py" ./

echo "数据迁移完成。"

# 返回原始目录
cd "$ORIGINAL_DIR"
echo "已返回原始目录: $(pwd)"