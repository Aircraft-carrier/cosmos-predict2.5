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

import random
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.predict2.conditioner import (
    BooleanFlag,
    GeneralConditioner,
    ReMapkey,
    Text2WorldCondition,
    TextAttr,
)


@dataclass(frozen=True)
class Video2WorldCondition(Text2WorldCondition):
    use_video_condition: bool = False
    # the following two attributes are used to set the video condition; during training, inference
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldCondition":
        """
        Sets the video conditioning frames for video-to-video generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles both image batches (T=1) and video batches (T>1) differently.

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames
                H = height
                W = width

            random_min_num_conditional_frames: Minimum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames: Maximum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames: Optional; If provided, all examples in the batch will use
                exactly this many frames for conditioning. If None, a random number of frames
                between random_min_num_conditional_frames and random_max_num_conditional_frames
                will be selected for each example in the batch.

            conditional_frames_probs: Optional; Dictionary mapping number of frames to probabilities.
                If provided, overrides the random_min/max_num_conditional_frames with weighted sampling.
                Example: {0: 0.5, 1: 0.25, 2: 0.25} for 50% chance of 0 frames, 25% for 1, 25% for 2.

        Returns:
            A new Video2WorldCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask_B_C_T_H_W) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - For image batches (T=1), no conditioning frames are used (num_conditional_frames_B = 0).
            - For video batches:
                - If num_conditional_frames is provided, all examples use that fixed number of frames.
                - Otherwise, each example randomly uses between random_min_num_conditional_frames and
                random_max_num_conditional_frames frames.
            - The mask marks the first N frames as conditioning frames (set to 1) for each example.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if num_conditional_frames is not None:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            elif conditional_frames_probs is not None:
                # Use weighted sampling based on provided probabilities
                frames_options = list(conditional_frames_probs.keys())
                weights = list(conditional_frames_probs.values())
                num_conditional_frames_B = torch.tensor(
                    random.choices(frames_options, weights=weights, k=B), dtype=torch.int32
                )
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames, random_max_num_conditional_frames + 1, size=(B,)
                )
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[idx, :, : num_conditional_frames_B[idx], :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldCondition":
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Video2WorldCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)


class Video2WorldConditionV2(Video2WorldCondition):
    """
    compared to Video2WorldCondition, this class apply zero frames when use_video_condition is False~(unconditional generation in cfg)
    in the case, we do zero-out conditional frames in the video condition
    """

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
    ) -> "Video2WorldConditionV2":
        num_conditional_frames = 0 if not self.use_video_condition else num_conditional_frames
        return super().set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldConditionV2":
        del is_cfg_conditional
        _condition = super().set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        return _condition


class Video2WorldConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldCondition(**output)


class Video2WorldConditionerV2(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldConditionV2:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldConditionV2(**output)


@dataclass(frozen=True)
class InpaintingCondition(Video2WorldCondition):
    use_render_condition: bool = True
    inpainting_rendered_video_B_C_T_H_W: Optional[torch.Tensor] = None
    inpainting_rendered_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def drop_render_condition(self) -> "InpaintingCondition":
        if self.use_render_condition:
            return self
        else:
            kwargs = self.to_dict(skip_underscore=False)

            dropped_inpainting_rendered_video_B_C_T_H_W = torch.zeros_like(kwargs["inpainting_rendered_video_B_C_T_H_W"])
            dropped_inpainting_rendered_mask_B_C_T_H_W = torch.zeros_like(kwargs["inpainting_rendered_mask_B_C_T_H_W"])

            # FOR DEBUG:
            # import torchvision
            # videos = [kwargs["inpainting_rendered_video_B_C_T_H_W"].cpu(), kwargs["inpainting_rendered_mask_B_C_T_H_W"].cpu(), 
            #             dropped_inpainting_rendered_video_B_C_T_H_W.cpu(), dropped_inpainting_rendered_mask_B_C_T_H_W.cpu()]
            # row_tensor = torch.cat(videos, dim=4)
            # final_tensor = torch.cat(list(row_tensor.unbind(0)), dim=2)
            # video_to_save = final_tensor.permute(1, 2, 3, 0)
            # torchvision.io.write_video('debug_dropout_video.mp4', video_to_save, fps=int(self.fps[0]))

            kwargs["inpainting_rendered_video_B_C_T_H_W"] = dropped_inpainting_rendered_video_B_C_T_H_W
            kwargs["inpainting_rendered_mask_B_C_T_H_W"] = dropped_inpainting_rendered_mask_B_C_T_H_W

            return type(self)(**kwargs)

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "InpaintingCondition":
        if self.use_render_condition:
            # In this case, we maintain the rendered videos and masks as the input condition
            # Meanwhile, we make sure the gt frames are not used for conditioning
            num_conditional_frames = 0
            return super().set_video_condition(
                gt_frames=gt_frames,
                random_min_num_conditional_frames=random_min_num_conditional_frames,
                random_max_num_conditional_frames=random_max_num_conditional_frames,
                num_conditional_frames=num_conditional_frames,
            )

        else:
            # Otherwise, we make the rendered videos and masks are zeros (See drop_render_condition) and back to the original video frames condition
            return super().set_video_condition(
                gt_frames=gt_frames,
                random_min_num_conditional_frames=random_min_num_conditional_frames,
                random_max_num_conditional_frames=random_max_num_conditional_frames,
                num_conditional_frames=num_conditional_frames,
                conditional_frames_probs=conditional_frames_probs,
            )

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "InpaintingCondition":
        num_conditional_frames = 0 if self.use_render_condition else num_conditional_frames
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition


class InpaintingConditioner(Video2WorldConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> InpaintingCondition:
        # LINK cosmos_predict2/_src/predict2/conditioner.py:465
        output = super()._forward(batch, override_dropout_rate)
        # output： {
        # 'fps': [2], 
        # 'padding_mask': [2, 1, 240, 320], 
        # 'crossattn_emb' [2, 512, 100352], 
        # 'use_video_condition' [1]
        # 'inpainting_rendered_video_B_C_T_H_W': [2, 3, 17, 240, 320] torch.uint8, 
        # 'inpainting_rendered_mask_B_C_T_H_W': [2, 3, 17, 240, 320] torch.uint8, 
        # 'use_render_condition': [1]
        # }

        assert "inpainting_rendered_video_B_C_T_H_W" in output, "InpaintingConditioner requires 'inpainting_rendered_video_B_C_T_H_W' in output"
        assert "inpainting_rendered_mask_B_C_T_H_W" in output, "InpaintingConditioner requires 'inpainting_rendered_mask_B_C_T_H_W' in output"
        assert torch.equal(output["inpainting_rendered_video_B_C_T_H_W"], batch["rendered_video"]), "inpainting_rendered_video_B_C_T_H_W is not equal to rendered_video in batch"
        assert torch.equal(output["inpainting_rendered_mask_B_C_T_H_W"], batch["rendered_mask"]), "inpainting_rendered_mask_B_C_T_H_W is not equal to rendered_mask in batch"
        # For DEBUG:
        # torch.equal(output["fps"], batch["fps"])
        # torch.equal(output["padding_mask"], batch["padding_mask"])
        # torch.equal(output["inpainting_rendered_video_B_C_T_H_W"], batch["rendered_video"])
        # torch.equal(output["inpainting_rendered_mask_B_C_T_H_W"], batch["rendered_mask"])

        return InpaintingCondition(**output)

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        This is only used for evaluation, not for training
        Adapted from GeneralConditioner
        This function will call 'drop_render_condition' after the new dropout rates are set
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0      # 'text' and 'use_render_condition' is 1.0, others is 0.0

        # TODO: This can be used to test I2V w/o render cond
        # cond_dropout_rates["use_render_condition"] = 1.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)

        # TODO: for now, testing on Libero dataset, dropout the render condition in cfg inference will lead to worse results
        # condition = condition.drop_render_condition()
        # un_condition = un_condition.drop_render_condition()
        return condition, un_condition


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

VideoPredictionConditioner: LazyDict = L(Video2WorldConditioner)(
    **_SHARED_CONFIG,
)

VideoPredictionConditionerV2: LazyDict = L(Video2WorldConditionerV2)(
    **_SHARED_CONFIG,
)

InpaintingConditionerConfig: LazyDict = L(InpaintingConditioner)(
    **_SHARED_CONFIG,
    rendered_video=L(ReMapkey)(
        input_key="rendered_video",
        output_key="inpainting_rendered_video_B_C_T_H_W",
        dropout_rate=0.0,
        dtype=None,
    ),
    rendered_mask=L(ReMapkey)(
        input_key="rendered_mask",
        output_key="inpainting_rendered_mask_B_C_T_H_W",
        dropout_rate=0.0,
        dtype=None,
    ),
    use_render_condition=L(BooleanFlag)(
        input_key="rendered_video",
        output_key="use_render_condition",
        dropout_rate=0.2,
    ),
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="inpainting_conditioner",
        node=InpaintingConditionerConfig,
    )
