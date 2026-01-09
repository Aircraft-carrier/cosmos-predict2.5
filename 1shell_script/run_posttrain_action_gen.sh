export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5
export PYTHONPATH=$PWD 

# CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12340 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py  -- \
        experiment=ac_gen_reason_embeddings_rectified_flow_2b_256_320 ~dataloader_train.dataloaders
