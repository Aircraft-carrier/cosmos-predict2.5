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

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint which has the correct experiment reference
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]  # This uses post_trained=True by default


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=inpainting_240_320
"""
inpainting_libero_49frame_240_320_stride1 = LazyDict(
    dict(
        # post-trained base 对应 LINK: cosmos_predict2/_src/predict2/configs/video2world/experiment/specialized_model/SFT_2B_RF.py:756
        # 其继承自 LINK: cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:262
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "inpainting_concat_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_inpainting_concat"},
            {"override /conditioner": "inpainting_conditioner"},
            {"override /data_train": "libero_49frame_240_320_stride1_train"},
            {"override /data_val": "mock"},         # TODO: current cosmos does not support validation (See https://github.com/nvidia-cosmos/cosmos-predict2.5/issues/30)
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2p5_inpainting",
            group="libero",
            name="49frame_240_320_stride1",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.001,
        ),
        checkpoint=dict(
            save_iter=1_000,
            # pyrefly: ignore  # missing-attribute
            load_path='/gemini/platform/public/embodiedAI/users/fanchenyou/models/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt',
            # load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=200_000,
                    save_videos=True,
                    save_images=False,
                    do_x0_prediction=False,
                    guidance=[3],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=200_000,
                    save_videos=True,
                    save_images=False,
                    do_x0_prediction=False,
                    guidance=[3],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=100,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                # wandb_10x=dict(
                #     save_s3=False,
                # ),
                dataloader_speed=dict(
                    save_s3=False,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                tokenizer=dict(
                    vae_pth="/gemini/platform/public/embodiedAI/users/fanchenyou/models/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
                ),
            ),
        ),        
        dataloader_train=dict(
            batch_size=16,
            sampler=dict(
                dataset=dict(num_frames=49, video_size=[240, 320]),
            ),
            dataset=dict(num_frames=49, video_size=[240, 320]),
            num_workers=16
        ),
    ),
    flags={"allow_objects": True},
)

CKPT_ITER = 27000
inpainting_libero_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/inpainting_libero_49frame_240_320_stride1",
        ],
        job=dict(
            project="cosmos_predict2p5_inpainting_eval",
            name=f"eval_{CKPT_ITER}_iter_no_drop_render",
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    save_videos=True,
                    save_images=False,
                    every_n=1,
                    n_viz_sample=4,
                    guidance=[0,3,7],
                ),
                every_n_sample_ema=dict(
                    save_videos=True,
                    save_images=False,
                    every_n=1,
                    n_viz_sample=4,
                    guidance=[0,3,7],
                ),
            )
        ),
        checkpoint=dict(
            load_path=f'/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs/cosmos_predict2p5_inpainting/libero/49frame_240_320_stride1/checkpoints/iter_{CKPT_ITER:09d}',
        ),
        dataloader_train=dict(
            batch_size=4
        )
    )
)

cs = ConfigStore.instance()

for _item in [inpainting_libero_49frame_240_320_stride1, inpainting_libero_eval]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
