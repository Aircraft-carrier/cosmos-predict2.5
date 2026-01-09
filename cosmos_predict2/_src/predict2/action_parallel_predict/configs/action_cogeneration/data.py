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
from cosmos_predict2._src.predict2.action_parallel_predict.datasets.dataset_local import ActionDataset

libero_data_root = "/gemini/platform/public/embodiedAI/users/zsh/dataset/Lerobot/hflibero"

libero_spatial_240_320_train_dataset = L(ActionDataset)(
    root=libero_data_root,
    number_frames=17,
    video_size=(240, 320),
    camera_views=["observation.images.image"],
    load_t5_embeddings=True,
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


libero_spatial_240_320_train_dataloader = L(DataLoader)(
    dataset=libero_spatial_240_320_train_dataset,
    sampler=L(get_sampler)(dataset=libero_spatial_240_320_train_dataset),
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

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="libero_spatial_240_320_train_dataset",
        node=libero_spatial_240_320_train_dataloader,
    )

