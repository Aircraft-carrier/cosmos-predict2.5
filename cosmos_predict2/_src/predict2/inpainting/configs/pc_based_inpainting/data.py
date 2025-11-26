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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.inpainting.datasets.dataset_local import Render_Dataset


# experiment for Libero novel view dataset
libero_dataset_dir = '/gemini/platform/public/embodiedAI/users/fanchenyou/dataset/libero_novel_view'
include_categories = ['libero_10', 'libero_goal', 'libero_object']
include_view_names = None
exclude_categories = None
exclude_view_names = ['agentview', 'robot0_eye_in_hand', 'robot0_eye_in_hand1', 'robot0_eye_in_hand2','robot0_eye_in_hand3', 
        'robot0_eye_in_hand4', 'robot0_eye_in_hand5', 'robot0_eye_in_hand6', 'robot0_eye_in_hand7', 'robot0_eye_in_hand8', 'robot0_eye_in_hand9']

libero_49frame_240_320_stride1_train_dataset = L(Render_Dataset)(
    dataset_dir=libero_dataset_dir,
    num_frames=49,
    video_size=(240, 320),
    sample_stride=1,
    include_categories=include_categories,
    include_view_names=include_view_names,
    exclude_categories=exclude_categories,
    exclude_view_names=exclude_view_names,
)


# ------------------------------------------------------------


# create dataloader for each dataset
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


libero_49frame_240_320_stride1_train_dataloader = L(DataLoader)(
    dataset=libero_49frame_240_320_stride1_train_dataset,
    sampler=L(get_sampler)(dataset=libero_49frame_240_320_stride1_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
)


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    # Libero 49 frame 240 320 stride 1
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="libero_49frame_240_320_stride1_train",
        node=libero_49frame_240_320_stride1_train_dataloader,
    )

