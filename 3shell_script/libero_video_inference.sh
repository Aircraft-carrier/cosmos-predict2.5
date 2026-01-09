export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5
export PYTHONPATH=$PWD  

CONFIG_PATH="cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py"
EXPERIMENT_NAME="reason_embeddings_rectified_flow_libero"
CHECKPOINT_PATH="/gemini/space/cosmos-predict/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

# /gemini/space/cosmos-predict/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt

CUDA_VISIBLE_DEVICES=0 torchrun  --nproc_per_node=1 \
  examples/inference.py \
  -i libero_asset/libero_video2world.json \
  -o outputs/cosmos_libero/ \
  --config-file "$CONFIG_PATH" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --experiment "$EXPERIMENT_NAME" \
  --disable-guardrails \
  --offload-text-encoder
  



