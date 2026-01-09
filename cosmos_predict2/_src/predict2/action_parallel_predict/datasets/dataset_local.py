import os
import time
import numpy as np
import torch
from torchvision import transforms as T
from typing import List, Dict, Any
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action_parallel_predict/datasets/dataset_local.py
TODO:(zsh 2025-12-18)
- [x] Step 1: Debug the dataset_3D code.
- [x] Step 2: Implement Action_Dataset to support loading data from Libero in LeRobot format.
- [x] Step 3: Preprocess text inputs into embeddings.
- [x] Step 4: Enable retrieval and utilization of precomputed embeddings.
- Fix delta_indices by converting frame indices to time (in seconds) using the frame rate (fps)
- Fix add action normalization, Lerobot not doing normalization yet.
- Note: libero action is delta action.
"""

class ActionDataset(LeRobotDataset):
    def __init__(
        self, 
        *args, 
        number_frames: int = 17, 
        video_size: tuple[int, int],
        camera_views: List[str] = [], 
        load_t5_embeddings = False,
        normalization_type: str = "mean_std",
        **kwargs
    ):
        kwargs['repo_id'] = "HuggingFaceVLA/libero"
        super().__init__(*args, **kwargs)
        self.number_frames = number_frames
        self.camera_views = camera_views
        self.load_t5_embeddings = load_t5_embeddings
        self.delta_indices = {
            "action": [1, 4, 7, 10, 13, 16], #  [i for i in range((number_frames - 1) // 4 + 1)]
            **{str(view): [i for i in range(number_frames)] for view in camera_views}
        }
        self.normalization_type = normalization_type
        first_image_key = self.camera_views[0]
        img_shape = self.features[first_image_key]["shape"] 
        self.C, self.H, self.W = img_shape
        
        if self.load_t5_embeddings:
            self.text_embeddings_dir = os.path.join(self.root, "text_embeddings")            

        self.preprocess = T.Compose(
            [
                ResizePreprocess(tuple(video_size))
            ]
        )

        # default action normaliation 6-DOF + gripper
        self.action_dim = 7
        self.continuous_dims = list(range(6))      # [0,1,2,3,4,5]
        self.binary_dim = 6                        # gripper
        # load action normalization stats
        if normalization_type == "min_max":
            self.action_stats = {
                'min': torch.tensor(self.meta.stats['action']['min'], dtype=torch.float32)[self.continuous_dims],
                'max': torch.tensor(self.meta.stats['action']['max'], dtype=torch.float32)[self.continuous_dims]
            }
        elif normalization_type == "mean_std":
            self.action_stats = {
                'mean': torch.tensor(self.meta.stats['action']['mean'], dtype=torch.float32)[self.continuous_dims],
                'std': torch.tensor(self.meta.stats['action']['std'], dtype=torch.float32)[self.continuous_dims]
            }
        else:
            raise ValueError(f"Unsupported normalization_type: {normalization_type}")

        # If pre_load_emb is True, load all embeddings into memory
        self.pre_load_emb = False
        if self.load_t5_embeddings and self.pre_load_emb:
            self.text_embeddings = {}
            emb_dir = os.path.join(self.root, "text_embeddings")
            for f in os.listdir(emb_dir):
                if f.endswith(".npy"):
                    idx = int(f.split(".")[0])
                    self.text_embeddings[idx] = torch.from_numpy(np.load(os.path.join(emb_dir, f))).float()
          
        # self.valid_tasks = {0, 1, 2, 4, 5, 6}
        # self.cache_path = os.path.join(self.root, "task_index.txt")          
        # # Try to load from cache first
        # if os.path.exists(self.cache_path):
        #     print(f"Loading valid indices from cache: {self.cache_path}")
        #     with open(self.cache_path, 'r') as f:
        #         self.valid_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        # else:
        #     print("Cache not found. Scanning dataset to find valid indices...")
        #     self.valid_indices = [
        #         i for i in range(len(self.hf_dataset))
        #         if self.hf_dataset[i]['task_index'].item() in self.valid_tasks
        #     ]
        #     # Save to cache
        #     os.makedirs(self.root, exist_ok=True)
        #     with open(self.cache_path, 'w') as f:
        #         for idx in self.valid_indices:
        #             f.write(f"{idx}\n")
        #     print(f"Saved {len(self.valid_indices)} valid indices to {self.cache_path}")    
        
    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize the continuous part of the action using min-max or mean-std normalization.
        The binary part (e.g., gripper open/close) is left unchanged."""
        normalized = action.clone()
        cont_part = action[:, self.continuous_dims]  # [T, 6]

        if self.normalization_type == "min_max":
            mins = self.action_stats["min"].to(action.device)   # [6]
            maxs = self.action_stats["max"].to(action.device)   # [6]
            cont_norm = (cont_part - mins) / (maxs - mins + 1e-8)
            cont_norm = cont_norm * 2 - 1                       # map to [-1, 1] choice
        elif self.normalization_type == "mean_std":
            mean = self.action_stats["mean"].to(action.device)  # [6]
            std = self.action_stats["std"].to(action.device)    # [6]
            cont_norm = (cont_part - mean) / std
        else:
            raise ValueError

        normalized[:, self.continuous_dims] = cont_norm
        return normalized
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        # real_idx = self.valid_indices[idx % len(self.valid_indices)]
        item = super().__getitem__(idx)
        
        video = item[self.camera_views[0]]   # torch.float32 [0-1] [T, C, H, W]
        video = self.preprocess(video)       # reshape
        video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
        gt_action = item['action']              
        ai_caption = item['task']
        action = self._normalize_action(gt_action)
        sample = {
            'video': video.permute(1, 0, 2, 3),  # uint8 [0-256] [C, T, H, W]
            'ai_caption': ai_caption,
            'fps': self.fps,
            'image_size': [self.H, self.W, self.H, self.W],
            'num_frames': self.number_frames,
            'padding_mask': torch.ones(1, self.H, self.W, dtype=torch.bool),
            'action': action,                    # torch.float32 [0-1] [T, action_dim]
            'stats': self.action_stats,
            'normalization_type': self.normalization_type,
            'gt_action': gt_action,              # unnormalized action
        }

        if self.load_t5_embeddings:
            task_idx = item['task_index']
            if self.pre_load_emb and int(task_idx) in self.text_embeddings.keys():
                emb = self.text_embeddings[int(task_idx)]
            else:
                emb_path = os.path.join(self.text_embeddings_dir, f"{task_idx}.npy")
                emb = np.load(emb_path)  
                emb = torch.from_numpy(emb).float() 
            emb = torch.squeeze(emb)    
            mask = torch.ones(emb.shape[0], dtype=torch.bool)
            sample["t5_text_embeddings"] = emb     
            sample["t5_text_mask"] = mask    
            
        return sample


import os
import torch
import numpy as np
import imageio.v3 as iio  # 推荐使用 v3 API

def save_data_as_mp4(sample, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video = sample['video']  # [C, T, H, W], torch.uint8
    video_t_h_w_c = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C], uint8 ndarray

    C = video_t_h_w_c.shape[-1]
    if C == 1:
        video_t_h_w_c = np.repeat(video_t_h_w_c, 3, axis=-1)
    elif C != 3:
        print(f"[Warning] Video has {C} channels. Saving raw tensor instead.")
        torch.save(video.cpu(), os.path.join(save_dir, 'video.pt'))
        # 依然保存 caption 和 embeddings
    else:
        raw_fps = sample.get('fps', 30)
        if torch.is_tensor(raw_fps):
            fps = raw_fps.item()
        elif isinstance(raw_fps, (np.ndarray, np.generic)):
            fps = raw_fps.item()
        else:
            fps = raw_fps
        fps = float(fps)  # 确保是 Python float

        output_path = os.path.join(save_dir, 'video.mp4')

        # 使用 imageio 保存视频
        iio.imwrite(
            output_path,
            video_t_h_w_c,
            fps=fps,
            codec='h264',
            quality=10  # 注意：imageio 的 quality 范围通常是 0-10，10 最高质量（类似 crf=10）
        )

    # Save caption
    with open(os.path.join(save_dir, 'ai_caption.txt'), 'w') as f:
        f.write(sample['ai_caption'])

    # Save embeddings
    if "t5_text_embeddings" in sample:
        torch.save(sample["t5_text_embeddings"].cpu(), os.path.join(save_dir, 't5_text_embeddings.pt'))
        torch.save(sample["t5_text_mask"].cpu(), os.path.join(save_dir, 't5_text_mask.pt'))
        
if __name__ == "__main__":
    """
    PYTHONPATH=. python cosmos_predict2/_src/predict2/action_parallel_predict/datasets/dataset_local.py
    """
    import torchvision.transforms.functional as TF
    dataset_root = "/gemini/platform/public/embodiedAI/users/zsh/dataset/Lerobot/hflibero"
    dataset = ActionDataset(
        root=dataset_root,
        number_frames=17,
        video_size=(240, 320),
        camera_views=["observation.images.image"],
        load_t5_embeddings=True,
        normalization_type="min_max",
    )
    save_dir = "./debug_video_frames"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in [10, 16]:
        start_time = time.time()
        data = dataset[idx]
        # import ipdb;ipdb.set_trace()
        print(
            (
                f"{idx} : {data.keys()}\n"
                f"{data['video'].shape}\n"
                f"{data['action'].shape}\n"
                "---"
            )
        )
        save_data_as_mp4(data, f"{save_dir}/{idx}")

        end_time = time.time()
        print(f"Time: {end_time - start_time:.3f}s\n")
        
