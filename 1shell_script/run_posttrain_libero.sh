export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5
export PYTHONPATH=$PWD  

CONFIG_PATH="cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py"
EXPERIMENT_NAME="reason_embeddings_rectified_flow_libero"
OVERRIDE_ARGS="~dataloader_train.dataloaders"

CUDA_VISIBLE_DEVICES=0 torchrun  --nproc_per_node=1 \
    --master_port=12341 \
    -m scripts.train \
    --config="$CONFIG_PATH" -- \
    experiment="$EXPERIMENT_NAME" $OVERRIDE_ARGS



