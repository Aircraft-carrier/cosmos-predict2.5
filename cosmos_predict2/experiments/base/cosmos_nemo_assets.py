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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# TODO 应该是用pretrain的checkpoint，开源的post-train model是RL+merge之后的
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]
# DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]      # 用post-trained的checkpoint


# Cosmos-NeMo-Assets video2world dataset and dataloader
# example_video_dataset_cosmos_nemo_assets = L(VideoDataset)(
#     dataset_dir="datasets/cosmos_nemo_assets",
#     num_frames=93,
#     video_size=(704, 1280),
# )
example_video_dataset_cosmos_nemo_assets = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=17,
    video_size=(64, 64),
)

dataloader_train_cosmos_nemo_assets = L(get_generic_dataloader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

# Video2World post-training configuration for 2B model
# torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_cosmos_nemo_assets
# 
# Hydra 处理顺序:
# 1. 加载父实验配置 -> 2.应用数据覆盖 -> 3. 应用当前配置 (_self_)，这里显式指定了_self_，不指定的话默认是在defaults的第一个
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    defaults=[          # defaults 字段定义了继承关系
        # pretrian对应的是 Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2
        # LINK cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:434
        # 最终最上层的配置是 # LINK cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:262
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",     # 继承另一个实验配置
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",       # 表明当前配置是最高优先级
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_nemo_assets",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,
    checkpoint=dict(
        save_iter=200,
        # pyrefly: ignore  # missing-attribute
        # load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),     # LINK cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py:165
        load_path='/gemini/space/cosmos-predict/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt',
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[2_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=1000,
        callbacks=dict(                 # 具体的callback定义在父类配置中： LINK cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:274
            heart_beat=dict(            # 下面三个属于basic LINK cosmos_predict2/_src/predict2/configs/video2world/defaults/callbacks.py:28
                save_s3=False,          # heart_beat: LINK cosmos_predict2/_src/predict2/callbacks/heart_beat.py:29
            ),
            iter_speed=dict(            # LINK cosmos_predict2/_src/predict2/callbacks/iter_speed.py:30
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(        # LINK cosmos_predict2/_src/predict2/callbacks/device_monitor.py:80
                save_s3=False,
            ),
            every_n_sample_reg=dict(    # 下面两个属于viz_online_sampling  LINK cosmos_predict2/_src/predict2/configs/video2world/defaults/callbacks.py:33
                every_n=200,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
            ),
            wandb=dict(                 # LINK cosmos_predict2/_src/predict2/configs/common/defaults/callbacks.py:54
                save_s3=False,
            ),
            # wandb_10x=dict(           # LINK cosmos_predict2/_src/predict2/configs/common/defaults/callbacks.py:60
            #     save_s3=False,
            # ),
            dataloader_speed=dict(      # 这个属于cluster_speed 
                save_s3=False,          # LINK cosmos_predict2/_src/predict2/configs/common/defaults/callbacks.py:67
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            tokenizer=dict(
                vae_pth="/gemini/space/cosmos-predict/tokenizer.pth",
            ),
            # text_encoder_config=dict(
            #     compute_online=False, 
            # ),
        ),
    ),
)

cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    predict2_video2world_training_2b_cosmos_nemo_assets,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
