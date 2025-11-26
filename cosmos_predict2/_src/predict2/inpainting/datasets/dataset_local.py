# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic render video dataset loader for multi-view video generation.

Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/inpainting/datasets/dataset_local.py
"""

import os
import traceback
from typing import Any, Optional
import re

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class Render_Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        sample_stride: int = 1,
        include_view_names: Optional[list[str]] = None,
        exclude_view_names: Optional[list[str]] = None,
        include_categories: Optional[list[str]] = None,
        exclude_categories: Optional[list[str]] = None,
    ) -> None:
        """Base dataset class for loading render videos with ground truth and masks.
        
        The dataset follows this directory structure:
        {dataset_dir}/{category}/{gt_videos/render_masks/render_videos}/{view_name}/{video_name}.mp4
        
        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            sample_stride (int): Stride for sampling frames (default: 1)
            include_view_names (Optional[list[str]]): List of view names to include. If specified,
                only these views will be used. Default: None (use all)
            exclude_view_names (Optional[list[str]]): List of view names to exclude. If specified,
                these views will be excluded from the selection. Default: None (exclude none)
            include_categories (Optional[list[str]]): List of categories to include. 
                For Libero: task suites (e.g., 'libero_10', 'libero_goal'). 
                For Robotwin: task names (e.g., 'blocks_ranking_rgb', 'blocks_ranking_size'). 
                If specified, only these categories will be used. Default: None (use all)
            exclude_categories (Optional[list[str]]): List of categories to exclude. If specified,
                these categories will be excluded from the selection. Default: None (exclude none)
            
        Returns dict with:
            - video: Ground truth RGB frames tensor [C,T,H,W]
            - rendered_video: Rendered RGB frames tensor [C,T,H,W]
            - rendered_mask: Rendered mask frames tensor [C,T,H,W]
            - ai_caption: Empty string (placeholder)
            - fps: Video frame rate
            - image_size: Tensor [h, w, h, w]
            - num_frames: Number of frames in sequence
            - padding_mask: Zero tensor [1,H,W]
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.sample_stride = sample_stride
        
        # Collect all valid video paths
        self.video_items = []
        
        # Filter categories based on include/exclude
        all_categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        if include_categories is not None:
            # Start with only included categories
            categories = [c for c in include_categories if c in all_categories]
        else:
            # Start with all categories
            categories = all_categories
        
        if exclude_categories is not None:
            # Remove excluded categories
            categories = [c for c in categories if c not in exclude_categories]
        
        log.info(f"Selected {len(categories)} categories: {categories}")
        
        for category in categories:
            category_path = os.path.join(dataset_dir, category)
            
            # NOTE: 这里以render_videos为准
            render_videos_dir = os.path.join(category_path, "render_videos")
            if not os.path.isdir(render_videos_dir):
                log.warning(f"gt_videos directory not found: {render_videos_dir}")
                continue
            
            # Filter view names based on include/exclude
            available_views = [d for d in os.listdir(render_videos_dir) if os.path.isdir(os.path.join(render_videos_dir, d))]
            
            if include_view_names is not None:
                # Start with only included views
                selected_views = [v for v in include_view_names if v in available_views]
            else:
                # Start with all available views
                selected_views = available_views
            
            if exclude_view_names is not None:
                # Remove excluded views
                selected_views = [v for v in selected_views if v not in exclude_view_names]
            
            for view_name in selected_views:
                view_dir = os.path.join(render_videos_dir, view_name)
                video_files = [f for f in os.listdir(view_dir) if f.endswith(".mp4")]
                
                for video_file in video_files:
                    video_name = os.path.splitext(video_file)[0]
                    
                    # Build paths for all three video types
                    gt_video_path = os.path.join(category_path, "gt_videos", view_name, video_file)
                    render_video_path = os.path.join(category_path, "render_videos", view_name, video_file)
                    render_mask_path = os.path.join(category_path, "render_masks", view_name, video_file)
                    
                    # Check if all required files exist
                    if os.path.exists(gt_video_path) and os.path.exists(render_video_path) and os.path.exists(render_mask_path):
                        identifier = f"{category}_{view_name}_{video_name}"
                        self.video_items.append({
                            'identifier': identifier,
                            'category': category,
                            'view_name': view_name,
                            'video_name': video_name,
                            'gt_video_path': gt_video_path,
                            'render_video_path': render_video_path,
                            'render_mask_path': render_mask_path,
                        })
                    else:
                        missing = []
                        if not os.path.exists(gt_video_path):
                            missing.append("gt_videos")
                        if not os.path.exists(render_video_path):
                            missing.append("render_videos")
                        if not os.path.exists(render_mask_path):
                            missing.append("render_masks")
                        log.warning(f"Missing files for {video_file} in {category}/{view_name}: {', '.join(missing)}")
        
        log.info(f"{len(self.video_items)} video items in total from {dataset_dir}")
        
        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])
    
    def __str__(self) -> str:
        return f"{len(self.video_items)} samples from {self.dataset_dir}"
    
    def __len__(self) -> int:
        return len(self.video_items)
    
    def _load_video(self, video_path: str, start_frame: Optional[int] = None) -> tuple[np.ndarray, float, int]:
        """Load video frames with stride-based sampling.
        
        Args:
            video_path (str): Path to video file
            start_frame (Optional[int]): Starting frame index. If None, randomly select one.
            
        Returns:
            tuple[np.ndarray, float, int]: Frame data, FPS, and the start_frame used
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        
        # Calculate required frames considering stride
        required_frames = (self.sequence_length - 1) * self.sample_stride + 1
        
        if total_frames < required_frames:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {required_frames} frames are required for {self.sequence_length} frames with stride {self.sample_stride}."
            )
        
        # Randomly sample a sequence of frames with stride (or use provided start_frame)
        max_start_idx = total_frames - required_frames
        if start_frame is None:
            start_frame = np.random.randint(0, max_start_idx + 1)
        else:
            # Validate provided start_frame
            if start_frame < 0 or start_frame > max_start_idx:
                raise ValueError(
                    f"start_frame {start_frame} is out of valid range [0, {max_start_idx}] for video {video_path}"
                )
        
        frame_ids = np.arange(start_frame, start_frame + required_frames, self.sample_stride).tolist()
        
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache
        
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, start_frame
    
    def _get_frames(self, video_path: str, start_frame: Optional[int] = None) -> tuple[torch.Tensor, float, int]:
        """Load and preprocess video frames.
        
        Args:
            video_path (str): Path to video file
            start_frame (Optional[int]): Starting frame index. If None, randomly select one.
            
        Returns:
            tuple[torch.Tensor, float, int]: Processed frames [T,C,H,W], FPS, and start_frame used
        """
        frames, fps, start_frame = self._load_video(video_path, start_frame)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, start_frame
    
    def __getitem__(self, index: int) -> dict | Any:
        """Get a video item with ground truth, render, and mask videos.
        
        Args:
            index (int): Index of the video item
            
        Returns:
            dict: Dictionary containing all video data and metadata
        """
        try:
            data = dict()
            item = self.video_items[index]
            
            # Load all three types of videos using the same start_frame
            # First load gt_video and get the start_frame
            gt_video, fps, start_frame = self._get_frames(item['gt_video_path'])
            # Use the same start_frame for render_video and render_mask
            render_video, _, _ = self._get_frames(item['render_video_path'], start_frame=start_frame)
            render_mask, _, _ = self._get_frames(item['render_mask_path'], start_frame=start_frame)
            
            # Rearrange from [T, C, H, W] to [C, T, H, W]
            gt_video = gt_video.permute(1, 0, 2, 3)
            render_video = render_video.permute(1, 0, 2, 3)
            render_mask = render_mask.permute(1, 0, 2, 3)
            
            data["video"] = gt_video
            data["rendered_video"] = render_video
            data["rendered_mask"] = render_mask

            raw_video_name = item['video_name']
            temp_name = re.sub(r'_demo_\d+$', '', raw_video_name)
            temp_name = re.sub(r'^[A-Z][A-Z0-9_]*_', '', temp_name)
            data["ai_caption"] = temp_name.replace('_', ' ')
            
            _, _, h, w = gt_video.shape
            
            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)
            
            # Store metadata for debugging
            data["video_name"] = item['identifier']
            
            return data
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video item {self.video_items[index]['identifier']} "
                f"(total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_items))]



if __name__ == "__main__":
    from torchvision.io import write_video
    """
    PYTHONPATH=. python cosmos_predict2/_src/predict2/inpainting/datasets/dataset_local.py
    """
    task_name = 'libero'    # robotwin
    dataset_dir = f'/gemini/platform/public/embodiedAI/users/fanchenyou/dataset/{task_name}_novel_view'
    num_frames = 49
    video_size = (240, 320)
    sample_stride = 1

    if task_name == 'libero':
        include_categories = ['libero_10', 'libero_goal', 'libero_object']
        include_view_names = None
        exclude_categories = None
        exclude_view_names = ['agentview', 'robot0_eye_in_hand', 'robot0_eye_in_hand1', 'robot0_eye_in_hand2','robot0_eye_in_hand3', 
        'robot0_eye_in_hand4', 'robot0_eye_in_hand5', 'robot0_eye_in_hand6', 'robot0_eye_in_hand7', 'robot0_eye_in_hand8', 'robot0_eye_in_hand9']
    elif task_name == 'robotwin':
        include_categories = None
        include_view_names = None
        exclude_categories = None
        exclude_view_names = None

    dataset = Render_Dataset(
        dataset_dir=dataset_dir,
        num_frames=num_frames,
        video_size=video_size,
        sample_stride=sample_stride,
        include_categories=include_categories,
        include_view_names=include_view_names,
        exclude_categories=exclude_categories,
        exclude_view_names=exclude_view_names,
    )

    indices = [0, 1, 6708, 34, 990, 3456, -1]
    for index in indices:
        data = dataset[index]
        inner_data = dataset.video_items[index]

        gt_video = data['video']
        render_video = data['render_videos']
        render_mask = data['render_masks']
        sample_name = data['video_name']

        concat_video = torch.cat([gt_video, render_video, render_mask], dim=3)
        concat_video = concat_video.permute(1, 2, 3, 0)  # [T,H,W,C]
        
        debug_dir = os.path.join(os.path.dirname(__file__), f'debug_data/stride_{sample_stride}_{video_size[0]}-{video_size[1]}_{num_frames}frames')
        os.makedirs(debug_dir, exist_ok=True)

        output_path = os.path.join(debug_dir, f'{sample_name}.mp4')
        write_video(output_path, concat_video, fps=int(data['fps']), video_codec='h264')
        print(f"Saved concatenated video to: {output_path}")

