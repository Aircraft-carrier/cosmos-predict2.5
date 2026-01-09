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

from typing import Callable,Tuple
import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = "uniform",
    ):
        # Map reweighting -> uniform to support inference for existing checkpoints.
        if weight == "reweighting":
            weight = "uniform"

        self.weight = weight
        self.noise_scheduler = noise_scheduler

        assert self.weight == "uniform", "Only uniform loss weight is supported in RF"

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")

        return t


class RectifiedFlow:
    def __init__(
        self,
        velocity_field: Callable,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight_method: str = "uniform",
        use_dynamic_shift: bool = False,
        shift: int = 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            velocity_field (`Callable`):
                A function that predicts the velocity given the current state and time.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
        """
        self.velocity_field = velocity_field
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )

        if use_dynamic_shift:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=use_dynamic_shift)
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        self.train_time_weight = TrainTimeWeight(self.noise_scheduler, train_time_weight_method)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int):
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled training times with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        time = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        return time

    def get_discrete_timestamp(self, u, tensor_kwargs):
        r"""This method map time from 0,1 to discrete steps"""

        indices = (u.squeeze() * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)[indices]
        return timesteps.unsqueeze(0) if timesteps.ndim == 0 else timesteps

    def get_sigmas(self, timesteps, tensor_kwargs):
        sigmas = self.noise_scheduler.sigmas.to(**tensor_kwargs)
        schedule_timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)
        step_indices = [(schedule_timesteps == t).nonzero().squeeze().tolist() for t in timesteps]
        assert len(step_indices) == timesteps.shape[0], "Number of indices do not match the given timesteps."
        sigma = sigmas[step_indices].flatten()

        return sigma

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        r"""
        This method computes interpolation `X_t` and their time derivatives `dotX_t` at the specified time points `t`.
        Note that `x_0` is the noise, and `x_1` is the clean data. This is aligned with the notation in the recified flow community,
        but different from the notation in the diffusion community.

        Args:
            x_0 (`torch.Tensor`):
                noise, shape `(B, D1, D2, ..., Dn)`, where `B` is the batch size, and `D1, D2, ..., Dn` are the data dimensions.
            x_1 (`torch.Tensor`):
                clean data, with the same shape as `x_0`
            t (`torch.Tensor`):
                A tensor of time steps, with shape `(B,)`, where each value is in `[0, 1]`.

        Returns:
            (x_t, dot_x_t) (`Tuple[torch.Tensor, torch.Tensor]`):
                - x_t (`torch.Tensor`): The interpolated state, with shape `(B, D1, D2, ..., Dn)`.
                - dot_x_t (torch.Tensor): The time derivative of the interpolated state, with the same shape as `x_t`.
        """
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        assert t.shape[0] == x_1.shape[0], "Batch size of t must match x_1."
        # Reshape t to match dimensions of x_1
        t = t.view(t.shape[0], *([1] * (len(x_1.shape) - 1)))
        x_t = x_0 * t + x_1 * (1 - t)
        dot_x_t = x_0 - x_1
        return x_t, dot_x_t

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
    