export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5
export PYTHONPATH=$PWD 

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
        --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  \
        -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320 #~dataloader_train.dataloaders
