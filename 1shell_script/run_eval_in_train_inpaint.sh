export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5
export PYTHONPATH=$PWD 

# 纯eval，需要注释掉 optimizer.step(324-326) LINK cosmos_predict2/_src/imaginaire/trainer.py:324
torchrun --nproc_per_node=1 --master_port=29600 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/config.py  \
        -- experiment=inpainting_libero_eval ~dataloader_train.dataloaders
