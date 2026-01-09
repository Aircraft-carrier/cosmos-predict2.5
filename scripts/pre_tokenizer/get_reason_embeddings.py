import os
import numpy as np
from pathlib import Path

from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import torch
"""
PYTHONPATH=. python scripts/pre_tokenizer/get_reason_embeddings.py
"""

# 配置路径
dataset_root = "/gemini/platform/public/embodiedAI/users/zsh/dataset/Lerobot/libero_spatialv3"
# "/gemini/platform/public/embodiedAI/users/zsh/dataset/Lerobot/hflibero"
text_embeddings_dir = os.path.join(dataset_root, "text_embeddings")
os.makedirs(text_embeddings_dir, exist_ok=True)

# 初始化 TextEncoder
text_encoder_class="reason1p1_7B"                
text_encoder_config = TextEncoderConfig(
    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
    compute_online=True,
    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
)


# 使用 LeRobotDatasetMetadata 获取任务列表
metadata = LeRobotDatasetMetadata(repo_id="", root=dataset_root)
task_names = metadata.tasks  
text_encoder = TextEncoder(text_encoder_config)
print(f"Found {len(task_names)} tasks:\n {task_names}")

def process_instruction(instruction: str, dataset_name: str) -> str:
    """
    Process the instruction based on dataset_name.
    For libero: capitalize first letter and add period at the end.
    """
    if dataset_name == "libero":
        # Capitalize first letter and add period if not present
        processed = instruction[0].upper() + instruction[1:]
        if not processed.endswith("."):
            processed += "."
        return processed
    elif dataset_name == "robotwin":
        # Add similar processing for robotwin if needed
        processed = instruction[0].upper() + instruction[1:]
        if not processed.endswith("."):
            processed += "."
        return processed
    return instruction

for task_desc, row in task_names.iterrows():
    idx = int(row["task_index"])
    print(f"Encoding task {idx}: '{task_desc}'")
    processed_instruction = process_instruction(task_desc, "robotwin")
    embedding = text_encoder.compute_text_embeddings_online(
        {
            "text": [processed_instruction],
        },
        "text",
    )
    embedding_np = embedding.cpu().float().numpy()    # shape: (1, seq_len, dim)
    
    save_path = os.path.join(text_embeddings_dir, f"{idx}.npy")
    np.save(save_path, embedding_np)
