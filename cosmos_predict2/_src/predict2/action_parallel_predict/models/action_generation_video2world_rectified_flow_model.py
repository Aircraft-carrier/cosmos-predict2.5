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
from typing import Callable, Dict, Optional, Mapping, Tuple, Any
import collections
from cosmos_predict2._src.imaginaire.utils import log, misc
from torch.nn.modules.module import _IncompatibleKeys
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
import attrs
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch
import tqdm
from megatron.core import parallel_state
from torch import Tensor
from einops import rearrange
from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import  broadcast, broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.text2world_model import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow


NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"

class MultiModelRectifiedFlow(RectifiedFlow):
    
    def sample_train_time_dual(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time1 = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        time2 = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        return time1, time2
    
    def get_interpolation_dual(
        self,
        x0_video: torch.Tensor, x1_video: torch.Tensor, t_video: torch.Tensor,
        x0_action: torch.Tensor, x1_action: torch.Tensor, t_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t_video, dot_x_t_video = super().get_interpolation(x0_video, x1_video, t_video)
        x_t_action, dot_x_t_action = super().get_interpolation(x0_action, x1_action, t_action)
        return x_t_video, dot_x_t_video, x_t_action, dot_x_t_action

    def get_discrete_timestamps_dual(
        self,
        u_video: torch.Tensor,
        u_action: torch.Tensor,
        tensor_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_video = super().get_discrete_timestamp(u_video, tensor_kwargs)
        t_action = super().get_discrete_timestamp(u_action, tensor_kwargs)
        return t_video, t_action

    def get_sigmas_dual(
        self,
        timesteps_video: torch.Tensor,
        timesteps_action: torch.Tensor,
        tensor_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_video = super().get_sigmas(timesteps_video, tensor_kwargs)
        sigma_action = super().get_sigmas(timesteps_action, tensor_kwargs)
        return sigma_video, sigma_action
    
    
class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    num_conditional_frames: int = 1 # Force the conditioning frames to 1.
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]

class TemporalMode(Enum):
    VIDEO_FIXED = "video_fixed"      
    ACTION_FIXED = "action_fixed"    
    NO_FIXED = "no_fixed"

class ActionVideo2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When shift == 1, the transformation (shift * sigmas / (1 + (shift - 1) * sigmas)) is identity(恒等)
        # it only establishes the base sigma_min/sigma_max for use in set_timesteps.
        self.sample_scheduler4action = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        self.rectified_flow = MultiModelRectifiedFlow(
            velocity_field=self.net,
            train_time_distribution=self.config.train_time_distribution,
            use_dynamic_shift=self.config.use_dynamic_shift,
            shift=self.config.shift,
            train_time_weight_method=self.config.train_time_weight,
            device=torch.device("cuda"),
            dtype=self.tensor_kwargs_fp32["dtype"],
        )
        self.temporal_mode = TemporalMode.ACTION_FIXED

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.
        Modified: 
            - Extract action_B_T_D from data_batch and generate epsilon_B_T_D randomly using the same interpolation strategy as in image generation. 
            - Both xt_B_C_T_H_W and action_t_B_T_D are jointly passed into the velocity field prediction network to predict the velocity field.         
            - Add the action velocity field loss.
        """
        
        # 这是一个计数器，统计总共见过多少 data_sample
        self._update_train_stats(data_batch)
        # Obtain text embeddings online
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")
        # ['video', 'ai_caption', 'fps', 'image_size', 'num_frames', 'padding_mask', 'action', 't5_text_embeddings', 't5_text_mask']
        # video  :   [B, 3, 17, 256, 320] [b c t h w]
        # action :   [B, 17, 7]           [b t action_dim]
        # t5_text_embeddings: [B, 512, 100352]
        # t5_text_mask： [B, 512]
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        action_B_T_D = data_batch['action']  # [B, 17, 7]
        
        # Sample pertubation noise levels and N(0, 1) noises
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        epsilon_B_T_D = torch.randn(action_B_T_D.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        if self.temporal_mode == TemporalMode.VIDEO_FIXED:
            t_B = torch.full((batch_size, 1), 0.999, **self.tensor_kwargs_fp32) 
            t_B4action = rearrange(self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32), "b -> b 1")
        elif self.temporal_mode == TemporalMode.ACTION_FIXED:
            t_B4action = torch.full((batch_size, 1), 0.999, **self.tensor_kwargs_fp32) 
            t_B = rearrange(self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32), "b -> b 1")    
        else:
            t_B, t_B4action = map(
                lambda t: rearrange(t.to(**self.tensor_kwargs_fp32), "b -> b 1"),
                self.rectified_flow.sample_train_time_dual(batch_size)
            )    
            
        x0_B_C_T_H_W, action_B_T_D, condition, epsilon_B_C_T_H_W, epsilon_B_T_D, t_B, t_B4action = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, action_B_T_D, condition, epsilon_B_C_T_H_W, epsilon_B_T_D, t_B, t_B4action
        )  
        
        timesteps, timesteps4action = self.rectified_flow.get_discrete_timestamps_dual(
            t_B, t_B4action, self.tensor_kwargs_fp32
        ) # [0-1] -> [1-1000] long ->  [1-1000] time_schedule [B]
         
        if self.config.use_high_sigma_strategy:  # False
            # Use high sigma strategy
            mask = torch.rand(timesteps.shape, device=timesteps.device) < self.config.high_sigma_ratio

            candidate_timesteps = self.rectified_flow.noise_scheduler.timesteps.to(device=timesteps.device)
            candidate_timesteps = candidate_timesteps[
                (candidate_timesteps >= self.config.high_sigma_timesteps_min)
                & (candidate_timesteps <= self.config.high_sigma_timesteps_max)
            ]

            if len(candidate_timesteps) > 0:
                # Sample timesteps.shape values from candidate_timesteps with replacement
                new_timesteps = candidate_timesteps[torch.randint(0, len(candidate_timesteps), timesteps.shape)]
                timesteps = torch.where(mask, new_timesteps, timesteps)
            else:
                raise ValueError("No candidate timesteps found for high sigma strategy")
        
        sigmas, sigmas4action = self.rectified_flow.get_sigmas_dual(
            timesteps,
            timesteps4action,
            self.tensor_kwargs_fp32,
        ) # [B]

        timesteps, sigmas, timesteps4action, sigmas4action = map(
            lambda x: rearrange(x, "b -> b 1"),
            (timesteps, sigmas, timesteps4action, sigmas4action)
        )
        
        xt_B_C_T_H_W, vt_B_C_T_H_W , action_t_B_T_D, vt_B_T_D = self.rectified_flow.get_interpolation_dual(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas,
            epsilon_B_T_D, action_B_T_D, sigmas4action,
        )  # [B, 17, 7]

        vt_pred_B_C_T_H_W, vt_pred_B_T_D  = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            action_t_B_T_D=action_t_B_T_D.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps, # [B, 1]
            action_timesteps_B_T=timesteps4action, # [B, 1]
            condition=condition,
        )
        
        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        
        action_loss = torch.mean(
            (vt_pred_B_T_D - vt_B_T_D) ** 2, dim=list(range(1, vt_pred_B_T_D.dim()))
        )
        if self.temporal_mode == TemporalMode.VIDEO_FIXED:
            loss = torch.mean(action_loss)
        elif self.temporal_mode == TemporalMode.ACTION_FIXED:
            loss = torch.mean(time_weights_B * (per_instance_loss))
        else:
            loss = torch.mean(time_weights_B * (per_instance_loss)) + torch.mean(action_loss)
        
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": loss,
            "video_loss": torch.mean(per_instance_loss),
            "action_loss": torch.mean(action_loss),
            "action_0": action_B_T_D,
            "action_t": action_t_B_T_D,
            "action_model_pred": vt_pred_B_T_D
        }

        return output_batch, loss
        
    def denoise(
        self,
        noise: torch.Tensor,                   # [2, 16, 5, 32, 40]
        xt_B_C_T_H_W: torch.Tensor,            # [B, 16, 5, 32, 40]
        action_t_B_T_D: torch.Tensor,          # [B, 17, 7]
        timesteps_B_T: torch.Tensor,           # [B, 1]
        action_timesteps_B_T: torch.Tensor,    # [B, 1]
        condition: Text2WorldCondition,
    ) -> tuple[DenoisePrediction, DenoisePrediction]:
        """
        Modified: 
            action_t_B_T_D (Tensor): Action trajectory tensor of shape [B, T, D]
            used as an auxiliary input to the network alongside the primary inputs
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)  # condition_state_in_B_C_T_H_W : [B, 16, 5, 32, 40]
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            ) # condition_video_mask : [B, 1, 5, 32, 40]
            
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0: # pass
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)  # [B, 1, T, 1, 1]
                # condition_video_mask_B_1_T_1_1[0,0,:,0,0] = tensor([1., 0., 0., 0., 0.]  The first frame is the ground-truth frame.
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                ) # [B, 1, T, 1, 1] all 0.1001

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T[:, None, :, None, None] * (
                    1 - condition_video_mask_B_1_T_1_1
                ) # [0,0,:,0,0] = [0.1001, 732.6160, 732.6160, 732.6160, 732.6160]

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )  # add dimension for batch

        # LINK: cosmos_predict2/_src/predict2/action_parallel_predict/networks/action_dit.py:366
        net_output_B_C_T_H_W, net_output_B_T_D = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs), 
            timesteps_B_T=timesteps_B_T,  
            **condition.to_dict(),
            action=action_t_B_T_D.to(**self.tensor_kwargs),
            action_timesteps_B_T=action_timesteps_B_T, 
        )
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()
        net_output_B_T_D = net_output_B_T_D.float()
        
        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W, net_output_B_T_D
    
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        """
        Modified:
            Add a gt_frames attribute to condition.
        """
        # LINK: cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:701
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # raw_state: [B, 3, 17, 256, 320], latent_state: [B, 16, 5, 32, 40]
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition
    
    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, action_B_T_D, condition, epsilon_B_C_T_H_W, epsilon_B_T_D, sigma_B_T, sigma_B_T4action):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        Modified: 
            Added broadcasting support for action_B_T_D, epsilon_B_T_D.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            action_B_T_D = broadcast_split_tensor(action_B_T_D, seq_dim=1, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_T_D = broadcast_split_tensor(epsilon_B_T_D, seq_dim=1, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if sigma_B_T4action is not None:
                assert sigma_B_T4action.ndim == 2, "sigma_B_T4action should be 2D tensor"
                if sigma_B_T4action.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T4action = broadcast(sigma_B_T4action, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T4action = broadcast_split_tensor(sigma_B_T4action, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, action_B_T_D, condition, epsilon_B_C_T_H_W, epsilon_B_T_D, sigma_B_T, sigma_B_T4action


    # ------------------------ Sampling ------------------------

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Tuple[Callable, torch.Tensor]:
        """
        Generates a callable function `velocity_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `velocity_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `velocity_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)  # we need always process the data batch first.
        action_B_T_D = data_batch['action']  # Extract action data from batch
        # override condition with inference mode; num_conditional_frames used Here!
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
        
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )
        
        _, _, condition, _, _, _, _ = self.broadcast_split_for_model_parallelsim(x0, action_B_T_D, condition, None, None, None, None)
        _, _, uncondition, _, _, _, _ = self.broadcast_split_for_model_parallelsim(x0, action_B_T_D, uncondition, None, None, None, None)

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, noise_action: torch.Tensor, timestep: torch.Tensor, action_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            cond_v, cond_v_action  = self.denoise(noise, noise_x, noise_action, timestep, action_t, condition)
            # uncond_v, uncond_v_action = self.denoise(noise, noise_x, noise_action, timestep, action_t, uncondition)
            # velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            # velocity_pred_action = cond_v_action + guidance * (cond_v_action - uncond_v_action)
            return cond_v, cond_v_action  # Classifier-free guidance disabled for simplicity
            # return velocity_pred, velocity_pred_action # [B, C, T, H, W], [B, T, D]

        return velocity_fn, x0
    
    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
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
        # TODO: Implement fixed video time mode (`self.use_fixed_video_time`):
        # - [x] Initialize video latents as GT frames + minimal noise.
        # - [x] Use a single fixed video timestep (e.g., t=0.999) repeated to match the action denoising schedule length.
        # - [x] Do not update video latents during the sampling loop (keep them fixed).
        self._normalize_video_databatch_inplace(data_batch) #  scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].
        self._augment_image_dim_inplace(data_batch)         #  transform image from "b c h w" to  "b c 1 h w"
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,                               # latent channel
                self.tokenizer.get_latent_num_frames(_T),           # compressed temporal frames
                _H // self.tokenizer.spatial_compression_factor,    # compressed height
                _W // self.tokenizer.spatial_compression_factor,    # compressed width
            ]
        action_state_shape = [
            n_sample, 
            data_batch['action'].shape[1], 
            data_batch['action'].shape[2]
        ]                                         # [B, T, action_dim]

        noise = misc.arch_invariant_rand(         # Cross-architecture (CPU/GPU/TPU) consistent random number generator 
            (n_sample,) + tuple(state_shape),     # latent space noise
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )                                          # [B, C, T, H, W] : [2, 16, 5, 30, 40]
        
        action_noise = misc.arch_invariant_rand( 
            tuple(action_state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed + 1,
        )                                          # [B, T, D] : [2, 17, 7]

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        
        self.sample_scheduler4action.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps  # [36]

        velocity_fn, gt_frames = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
            action_noise = broadcast_split_tensor(tensor=action_noise, seq_dim=1, process_group=self.get_context_parallel_group())
        
        if self.temporal_mode == TemporalMode.VIDEO_FIXED:
            t_B = torch.full((n_sample, 1), 0.999, **self.tensor_kwargs_fp32)   
            fixed_timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
            sigmas = self.rectified_flow.get_sigmas(fixed_timesteps, self.tensor_kwargs_fp32)
            fixed_timesteps, sigmas = map(
                lambda x: rearrange(x, "b -> b 1"),
                (fixed_timesteps, sigmas)
            )
            latents, _ = self.rectified_flow.get_interpolation(noise, gt_frames, sigmas)
        else:
            latents = noise
        
        latents_action = action_noise

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))
        for _, t in enumerate(timesteps_iter):          # 36 step 
            latent_model_input = latents                # [b, c, t, h, w]   : [2, 16, 5, 30, 40]
            latent_model_input_action = latents_action  # [b, t, action_dim]: [2, 17, 7]
            timestep = [t]

            timestep = torch.stack(timestep)    # [1]
            timestep = timestep.unsqueeze(0)    # [1,1]
            velocity_pred, velocity_pred_action = velocity_fn(
                noise,                          # latent noise [b, c, t, h, w] for gt conditioned video
                latent_model_input,             # latent current state [b, c, t, h, w]
                latent_model_input_action,      # latent current action state [b, t, action_dim]
                timestep if self.temporal_mode != TemporalMode.VIDEO_FIXED else fixed_timesteps,                       
                timestep if self.temporal_mode != TemporalMode.VIDEO_FIXED else timestep.expand(n_sample, 1)
            )   # [b, c, t, h, w]
             
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            
            temp_x0_action = self.sample_scheduler4action.step(
                velocity_pred_action.unsqueeze(0), t, latents_action.unsqueeze(0), return_dict=False, generator=seed_g
            )[0]    
            # FlowUniPCMultistep scheduler caches internal state (e.g., step index, past outputs)
            # so separate instances are required for video and action to avoid state interference.
            if self.temporal_mode != TemporalMode.VIDEO_FIXED:
                latents = temp_x0.squeeze(0)    # [1, b, c, t, h, w] -> [b, c, t, h, w]
            latents_action = temp_x0_action.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())
            latents_action = cat_outputs_cp(latents_action, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents, latents_action
    
    # ------------------ Checkpointing ------------------

    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: F821
    #     """
    #     Loads a state dictionary into the model and optionally its EMA counterpart.
    #     Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

    #     Parameters:e
    #         state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
    #                                         potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
    #         strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
    #                                 those in the model and EMA model (if applicable). Defaults to True.
    #         assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
    #                                 matching keys one-by-one. This is typically used when loading parts of state dicts
    #                                 or using customized loading procedures. Defaults to False.
    #     """
    #     target_suffixes = [
    #         "adaln_modulation_self_attn.1.weight",
    #         "adaln_modulation_cross_attn.1.weight",
    #         "adaln_modulation_mlp.1.weight",
    #     ]
    #     def _expand_input_dim_weight(v: torch.Tensor) -> torch.Tensor:
    #         """Helper: convert [out_dim, in_dim] weight to [out_dim, 2 * in_dim]，back half init  0"""
    #         from torch.distributed._tensor import DTensor
            
    #         is_dtensor = isinstance(v, DTensor)
    #         if is_dtensor:
    #             placements = v.placements
    #             device_mesh = v.device_mesh
    #             local_v = v.to_local()  
    #         else:
    #             local_v = v

    #         *leading_dims, in_dim = local_v.shape
    #         if in_dim > 2048:
    #             return v
    #         new_local_weight = torch.randn(
    #             *leading_dims, 2 * in_dim,
    #             device=local_v.device,
    #             dtype=local_v.dtype
    #         ) * 1e-6
    #         new_local_weight[..., :in_dim] = local_v
    #         # zeros = torch.zeros_like(local_v)
    #         # new_local_weight = torch.cat([local_v, zeros], dim=-1)
            
    #         if is_dtensor:
    #             from torch.distributed._tensor import Replicate
    #             new_placements = [Replicate()] * len(placements)  
    #             new_weight = DTensor.from_local(
    #                 new_local_weight,
    #                 device_mesh=device_mesh,
    #                 placements=new_placements,
    #                 run_check=False
    #             )
    #         else:
    #             new_weight = new_local_weight

    #         return new_weight
        
    #     _reg_state_dict = collections.OrderedDict()
    #     _ema_state_dict = collections.OrderedDict()
        
    #     for k, v in state_dict.items():
    #         if k.startswith("net."):
    #             # Special mapping for t_embedder
    #             new_k = k.replace("net.", "")
    #             final_v = v
                
    #             # =====================   2 times expansion for time rely weights
    #             if new_k.startswith("t_embedder.1.linear_"):  # Map: t_embedder.1.linear_X.weight -> t_embedder.proj.linear_X.weight
    #                 new_k = new_k.replace("t_embedder.1.", "t_embedder.proj.", 1)
    #                 if new_k.endswith("linear_1.weight"):
    #                     final_v = _expand_input_dim_weight(v)          
    #             elif new_k.startswith("t_embedding_norm.weight") or any(new_k.endswith(suffix) for suffix in target_suffixes):
    #                 final_v = _expand_input_dim_weight(v) # [adaln_lora_dim, model_channels] -> [adaln_lora_dim, 2 * model_channels]
    #             # =====================
    #             _reg_state_dict[new_k] = final_v
                
    #         elif k.startswith("net_ema."):
    #             _ema_state_dict[k.replace("net_ema.", "")] = v

    #     state_dict = _reg_state_dict
    #     # list(state_dict.keys())[:10]
    #     # ['net.t_embedder.1.linear_1.weight', 'net.t_embedder.1.linear_2.weight']
    #     # p list(self.net.state_dict().keys())[:10]
    #     # [..., 't_embedder.mlp.weight', 't_embedder.proj.linear_1.weight', 't_embedder.proj.linear_2.weight' ,]
    #     if strict:
    #         reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)
    #         if self.config.ema.enabled:
    #             ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
    #                 _ema_state_dict, strict=strict, assign=assign
    #             )

    #         return _IncompatibleKeys(
    #             missing_keys=reg_results.missing_keys + (ema_results.missing_keys if self.config.ema.enabled else []),
    #             unexpected_keys=reg_results.unexpected_keys
    #             + (ema_results.unexpected_keys if self.config.ema.enabled else []),
    #         )
    #     else:
    #         # This fork
    #         log.critical("load model in non-strict mode")
    #         log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
    #         if self.config.ema.enabled:
    #             log.critical("load ema model in non-strict mode")
    #             log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
