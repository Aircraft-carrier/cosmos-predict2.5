export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs

cd /gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5
export PYTHONPATH=$PWD 

torchrun --nproc_per_node=1 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets