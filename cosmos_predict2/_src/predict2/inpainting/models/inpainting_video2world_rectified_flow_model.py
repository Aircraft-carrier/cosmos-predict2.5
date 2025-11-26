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

from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
import tqdm
from megatron.core import parallel_state
from torch import Tensor
import torch.nn.functional as F

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.inpainting.configs.pc_based_inpainting.conditioner import InpaintingCondition
from cosmos_predict2._src.predict2.models.text2world_model import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import Video2WorldModelRectifiedFlowConfig

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class InpaintingModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    add_input_layer_channel: bool = True


class InpaintingConcatVideo2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, InpaintingCondition]:
        # encode videos into latent, dropout conditions
        # NOTE: here only truly drop the text (i.e., turn into empty string), 
        # video_condition and render_videos/masks are not truly dropped, it only generates a flag (True/False)
        # LINK cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:699

        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)

        # Truly drop the video contion and renndered videos/masks
        # Drop video: num_conditional_frames = 0
        # Drop renndered videos/masks: renndered videos/masks are zeros
        # LINK cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/conditioner.py:226
        condition = condition.drop_render_condition()
        # LINK cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/conditioner.py:255
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    @torch.no_grad()
    def generate_samples_with_latents_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        query_steps=[0, 9, 18, 27, 34],
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps

        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
        latents = noise

        latent_to_save = {}
        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))

        for num_step, t in enumerate(timesteps_iter):
            if num_step in query_steps:
                latent_to_save[num_step] = latents
                print(f"Saving latent at step {num_step}, timestep {t}")

            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        latent_to_save[num_step] = latents

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents, latent_to_save

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner
            Note: For inpainting, there should be 'inpainting_rendered_video' and 'inpainting_rendered_mask' in the condition.

        Returns:
            velocity prediction
        """
        # First unfreeze the condition
        cond_dict = condition.to_dict()

        # FOR DEBUG
        # import torchvision
        # def process_video_tensor(tensor):
        #     tensor = tensor.permute(2, 0, 1, 3, 4) 
        #     T, B, C, H, W = tensor.shape
        #     tensor = tensor.permute(0, 2, 1, 3, 4)
        #     tensor = tensor.reshape(T, C, B * H, W)
        #     tensor = tensor.permute(0, 2, 3, 1)
        #     return tensor
        # raw_video_processed = process_video_tensor(cond_dict['inpainting_rendered_video_B_C_T_H_W'])
        # raw_mask_processed  = process_video_tensor(cond_dict['inpainting_rendered_mask_B_C_T_H_W'])
        # video_to_save = torch.cat([raw_video_processed, raw_mask_processed], dim=2)
        # torchvision.io.write_video('debug_denoise_func_cond.mp4', video_to_save, fps=30)
        # import ipdb; ipdb.set_trace()

        # encode inpainting_rendered_video_B_C_T_H_W to latent state
        self._normalize_render_video_condition_inplace(cond_dict)
        inpainting_rendered_video_latent = self.encode(cond_dict['inpainting_rendered_video_B_C_T_H_W']).contiguous().float()
        cond_dict['inpainting_rendered_video_B_C_T_H_W'] = inpainting_rendered_video_latent
        
        # resize inpainting_rendered_mask_B_C_T_H_W and make it binary
        self._process_render_mask_condition_inplace(cond_dict)
        
        # We donot need to modify following code, since when the render condition is used, the condition_video_input_mask is always all zeros
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

        # forward pass through the network
        # LINK cosmos_predict2/_src/predict2/inpainting/networks/inpainting_dit.py:49
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **cond_dict,
        ).float()

        # In inpaiting, we donot directly replace the gt frames, so here should not be replaced
        # We keep it since the rendered videos/masks may be dropped
        if not condition.use_render_condition and condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            # LINK cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/conditioner.py:325
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        # when using render condition, num_conditional_frames will be set internally to 0
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        # LINK cosmos_predict2/_src/predict2/inpainting/configs/pc_based_inpainting/conditioner.py:278
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        # ONLY FOR DEBUG
        def debug_split_condition(cond, batch_idx: int):
            unchanged_keys = {'use_render_condition', 'data_type', 'use_video_condition'}
            cond_dict = cond.to_dict()
            sub_cond_dict = {}
            for key, value in cond_dict.items():
                if key in unchanged_keys:
                    sub_cond_dict[key] = value
                elif isinstance(value, torch.Tensor):
                    sub_cond_dict[key] = value[batch_idx:batch_idx+1]
                else:
                    sub_cond_dict[key] = value
            
            return type(cond)(**sub_cond_dict)
        def debug_velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            batch_size = noise.shape[0]
            
            # Process each batch item separately for testing multi-batch functionality
            velocity_pred_list = []
            
            for i in range(batch_size):
                # Split inputs for single batch
                single_noise = noise[i:i+1]
                single_noise_x = noise_x[i:i+1]
                
                # Split conditions
                single_condition = debug_split_condition(condition, i)
                single_uncondition = debug_split_condition(uncondition, i)
                
                import torchvision
                single_cond_dict = single_condition.to_dict()
                single_uncond_dict = single_uncondition.to_dict()
                cond_video = single_cond_dict['inpainting_rendered_video_B_C_T_H_W'][0].permute(1,2,3,0).cpu()
                cond_mask = single_cond_dict['inpainting_rendered_mask_B_C_T_H_W'][0].permute(1,2,3,0).cpu()
                uncond_video = single_uncond_dict['inpainting_rendered_video_B_C_T_H_W'][0].permute(1,2,3,0).cpu()
                uncond_mask = single_uncond_dict['inpainting_rendered_mask_B_C_T_H_W'][0].permute(1,2,3,0).cpu()

                cond_gt_video_latent = single_cond_dict['gt_frames']
                uncond_gt_video_latent = single_uncond_dict['gt_frames']
                cond_gt_video = self.decode(cond_gt_video_latent)
                uncond_gt_video = self.decode(uncond_gt_video_latent)
                cond_gt_video = (1.0 + cond_gt_video.clamp(-1, 1)) / 2.0
                uncond_gt_video = (1.0 + uncond_gt_video.clamp(-1, 1)) / 2.0
                cond_gt_video = (cond_gt_video * 255).clamp(0, 255).to(torch.uint8)
                uncond_gt_video = (uncond_gt_video * 255).clamp(0, 255).to(torch.uint8)
                cond_gt_video = cond_gt_video[0].permute(1,2,3,0).cpu()
                uncond_gt_video = uncond_gt_video[0].permute(1,2,3,0).cpu()
                
                cond_combined = torch.cat([cond_video, cond_mask, cond_gt_video], dim=1)
                uncond_combined = torch.cat([uncond_video, uncond_mask, uncond_gt_video], dim=1)

                debug_visual = torch.cat([cond_combined, uncond_combined], dim=2)
                torchvision.io.write_video(f"debug_eval_cond_uncond_visual_batch_{i}.mp4", debug_visual, fps=30)

                # Compute denoise for single batch
                single_cond_v = self.denoise(single_noise, single_noise_x, timestep, single_condition)
                single_uncond_v = self.denoise(single_noise, single_noise_x, timestep, single_uncondition)
                single_velocity_pred = single_cond_v + guidance * (single_cond_v - single_uncond_v)
                velocity_pred_list.append(single_velocity_pred)
                
            # Concatenate results along batch dimension
            velocity_pred = torch.cat(velocity_pred_list, dim=0)
            return velocity_pred

        return velocity_fn


    def _normalize_render_video_condition_inplace(self, condition: dict[str, Tensor]) -> None:
        """
        Adapted from '_normalize_video_databatch_inplace' see LINK cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:717

        Here, we strictly limit the input_key to 'inpainting_rendered_video_B_C_T_H_W' since it should be here if previous steps are done correctly
        """
        
        assert condition['inpainting_rendered_video_B_C_T_H_W'].dtype == torch.uint8, (
            f"Condition render video data is not in uint8 format. Got {condition['inpainting_rendered_video_B_C_T_H_W'].dtype}"
        )
        condition['inpainting_rendered_video_B_C_T_H_W'] = condition['inpainting_rendered_video_B_C_T_H_W'].to(**self.tensor_kwargs) / 127.5 - 1.0

    def _process_render_mask_condition_inplace(self, condition: dict[str, Tensor]) -> None:
        """
        Process the render mask condition in-place
        It includes:
        - make the 3-channel mask to 1-channel binary mask
        - resize to the same size as the video latent
        """
        input_mask = condition['inpainting_rendered_mask_B_C_T_H_W']
        mask_sum = torch.sum(input_mask, dim=1, keepdim=True)
        binary_mask = (mask_sum > 0).float()
        latent_size = condition['inpainting_rendered_video_B_C_T_H_W'].shape
        resized_render_mask = self._resize_render_mask_condition(binary_mask, latent_size)
        condition['inpainting_rendered_mask_B_C_T_H_W'] = resized_render_mask

    def _resize_render_mask_condition(self, render_mask: Tensor, latent_size: Tuple[int, int, int, int, int]) -> None:
        """
        Resize the render mask condition to the same size as the video latent
        """
        # [2, 1, 17, 240, 320] -> [2, 1, 20, 240, 320]
        render_mask = torch.concat(
                        [
                            torch.repeat_interleave(render_mask[:, :, 0:1], repeats=4, dim=2), 
                            render_mask[:, :, 1:]
                        ], dim=2
                    )
        # [2, 1, 20, 240, 320] -> [2, 5, 4, 240, 320]
        render_mask = render_mask.view(render_mask.shape[0], render_mask.shape[2] // 4, 4, render_mask.shape[3], render_mask.shape[4])
        # [2, 5, 4, 240, 320] -> [2, 4, 5, 240, 320]
        render_mask = render_mask.transpose(1, 2)

        # [2, 4, 5, 240, 320] -> [2, 4, 5, 30, 40]
        resized_render_mask = self._resize_mask(render_mask, latent_size)
        return resized_render_mask

    # Adapted from https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1_fun/train.py#L93
    def _resize_mask(self, mask, latent_size, process_first_frame_only=True):
        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            
            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )
                resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
            else:
                resized_mask = first_frame_resized
        else:
            target_size = list(latent_size[2:])
            resized_mask = F.interpolate(
                mask,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
        return resized_mask
        