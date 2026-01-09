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
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py  -- experiment=ac_gen_reason_embeddings_rectified_flow_2b_256_320
"""
# LINK cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/model_2B_reason_1p1_rectified_flow.py:262
ac_gen_reason_embeddings_rectified_flow_2b_256_320_action_time = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_generation_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_generation"},
            {"override /conditioner": "action_generation_conditioner"},
            {"override /data_train": "libero_spatial_240_320_train_dataset"},
            {"override /data_val": "mock"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_generation",
            group="libero_cosmos_predict_v2p5",
            name="2b_libero_action_time",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=500,
            load_path='/gemini/space/cosmos-predict/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt',
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            grad_accum_iter=4,
            max_iter=150000,
            logging_iter=100,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,  
                    save_videos=True,
                    save_images=False,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=10,
                    save_s3=False,
                ),
                # every_n_sample_ema=dict(
                #     every_n=5000,
                #     save_videos=True,
                #     save_images=False,
                #     do_x0_prediction=False,
                #     guidance=[0, 3, 7],
                #     fps=10,
                #     save_s3=False,
                # ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=10000,
                    save_s3=False,
                ), 
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=dict(
                    save_s3=False,
                ),
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
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                conditional_frame_timestep=-1.0,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7
                ),
                tokenizer=dict(
                    vae_pth="/gemini/space/cosmos-predict/tokenizer.pth",
                ),
                text_encoder_config=dict(
                    compute_online=False, 
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,  
            sampler=dict(
                dataset=dict(video_size=[240, 320]),  # 240, 320
            ),
            dataset=dict(video_size=[240, 320]),
            num_workers=4, 
            pin_memory=False,
        ),
        dataloader_val=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        pin_memory=False,
                    ),
                ),
                video_data=dict(
                    dataloader=dict(
                        pin_memory=False,
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item in [ac_gen_reason_embeddings_rectified_flow_2b_256_320_action_time]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
