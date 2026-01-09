# Get path to the latest checkpoint
CHECKPOINTS_DIR=/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/logs/cosmos_predict2_action_generation/libero_cosmos_predict_v2p5/2b_libero_action_congeneratioin/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR