export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5
export PYTHONPATH=$PWD 

torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/config.py  \
        -- experiment=inpainting_libero_49frame_240_320_stride1 ~dataloader_train.dataloaders
