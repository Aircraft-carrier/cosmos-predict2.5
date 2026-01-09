# --- ADD THIS AT THE VERY TOP OF targets_shortcut.py ---
import sys
from unittest.mock import MagicMock

# Mock problematic GPU-only modules
# ---------------------------------------------------------
sys.modules["transformer_engine"] = MagicMock()
sys.modules["transformer_engine.pytorch"] = MagicMock()
sys.modules["transformer_engine.pytorch.module"] = MagicMock()
sys.modules["megatron"] = MagicMock()
sys.modules["megatron.core"] = MagicMock()
sys.modules["megatron.core.parallel_state"] = MagicMock()
# ---------------------------------------------------------
import torch
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from einops import rearrange
import torch.nn as nn
from typing import Tuple, Optional
from cosmos_predict2._src.predict2.action_parallel_predict.configs.action_cogeneration.conditioner import (
    ActionGenerationConditioner,
    ActionGenerationCondition
)    


"""
TODO:(zsh-2025/12/29)
- [x] Step 1. Initial code debugging and verification
- [ ] Step 2. Integrate timestep conditioning into DiT (Diffusion Transformer)
(Note: Short Cut Model uses the same timestepEmbedder for both t and dt Then directly adds them together)
- [ ] Step 3. Incorporate the modified DiT into the training pipeline
- [ ] Step 4. Train with large batch sizes using gradient accumulation
Question:
How should we effectively train with large batch sizes
How do we control video duration (or temporal length) during training?
"""
class MockModel1(nn.Module):
    def __init__(
        self, 
        video_channels = 3, 
        action_dim = 4,
    ):
        super().__init__()
        self.video_channels = video_channels
        self.action_dim = action_dim

    def forward(
        self,
        x_video: torch.Tensor,
        x_action: torch.Tensor,
        t_video: torch.Tensor,
        t_action: torch.Tensor,
        dt_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict velocity (v) for video and action.
        Returns:
            v_video : [B, C, T, H, W]
            v_action: [B, T, action_dim]
        """
        # Replace with real model logic
        v_video = torch.randn_like(x_video)
        v_action = torch.randn_like(x_action)
        return v_video, v_action

class MockModel2(nn.Module):
    def __init__(
        self, 
        rectified_flow: RectifiedFlow = None,
        conditioner: ActionGenerationConditioner = None,
        model: MockModel1 = None
    ):
        super().__init__()
        self.rectified_flow=rectified_flow          
        self.conditioner=conditioner
        self.model=model
    
    @torch.no_grad()
    def get_targets(
        self,
        videos: torch.Tensor,          # [B, C, T, H, W] 
        action: torch.Tensor,          # [B, T, action_dim]
        force_dt: float = -1.0,
        bootstrap_every: int = 8,
        denoise_timesteps: Optional[int] = None,
        chunk_size: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training targets for short cut Model with action and video data.
        Args:
            model: The rectified flow model used as the teacher for bootstrap targets.
            videos: Input video tensor of shape [B, C, T, H, W].
            action: Input action tensor of shape [B, T, action_dim].
            force_dt: If not -1, forces the dt_base to a specific value.
            rectified_flow: An instance of MultiModelRectifiedFlow for interpolation and time sampling.
            bootstrap_every: Frequency of bootstrap samples in the batch.
            denoise_timesteps: Number of denoising timesteps in the rectified flow.
        Returns:
            x_t: Noisy video samples at time t.
            xa_t: Noisy action samples at time t.
            v_t: Target velocities for video.
            va_t: Target velocities for action.
            dt_base_out: Time step size levels for each sample in the batch.
        """
        
        device = videos.device # 'cuda:0' or 'cpu'
        B = videos.shape[0]    # 256
        tensor_kwargs = {"device": device, "dtype": videos.dtype}
        # ======================================================================
        # 1)  Build dt_base (time step size level) for the bootstrap part
        # ======================================================================
        bootstrap_batchsize = B // bootstrap_every      # e.g., 256 // 8 = 32
        log2_sections = int(torch.log2(torch.tensor(denoise_timesteps)).item()) # denoise_timesteps = 128 -> log2_sections = 7
        dt_vals = torch.arange(log2_sections - 1, -1, -1, device=device) # [6, 5, 4, 3, 2, 1, 0]
        if bootstrap_batchsize >= log2_sections:
            repeats = bootstrap_batchsize // log2_sections # 32 // 7 = 4
            dt_base_boot = dt_vals.repeat_interleave(repeats)  
        else:
            weights =(dt_vals + 1).float()
            indices = torch.multinomial(weights, bootstrap_batchsize, replacement=True)
            dt_base_boot = dt_vals[indices]
            
        # [6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0] 
        # If the number of samples is insufficient, pad with 6 (corresponding to the finest granularity dt=1/128)
        if dt_base_boot.shape[0] < bootstrap_batchsize:
            pad = torch.ones(bootstrap_batchsize - dt_base_boot.shape[0], device=device, dtype=torch.long) * (log2_sections - 1)
            dt_base_boot = torch.cat([pad, dt_base_boot]) 

        if force_dt != -1:
            dt_base = torch.full_like(dt_base_boot, int(force_dt))

        dt = 1.0 / (2.0 ** dt_base_boot.float()) # [1/64, ..., 1/32, ..., 1]  # dt_base_boot 越大反而 dt 越小，为 0 是步长为 1
        dt_base_bootstrap = dt_base_boot + 1     # [7, ..., 6, ..., 1] 
        # For teacher model to integration # dt_base = 6 -> dt_base_bootstrap = 7
        dt_bootstrap = dt / 2.0            # [1/128, ..., 1/64, ..., 1/2]
        # Actual integration step size     # dt / 2 (each step takes half in the two-step Euler method)

        # ======================================================================
        # 2) Sample time t \in [0,1) for the bootstrap part
        # ======================================================================
        # 0-1 uniform samples for video and action, shape=[32, 1]
        # --- Sample t for video (using rectified_flow's standard method) ---
        
        t_v_cont = rearrange(self.rectified_flow.sample_train_time(bootstrap_batchsize), "b -> b 1")  # [32, 1]
        t_v_disc = self.rectified_flow.get_discrete_timestamp(t_v_cont, tensor_kwargs)   # [32]
        sigmas_v = self.rectified_flow.get_sigmas(t_v_disc, tensor_kwargs)               # [32]

        # --- Sample t for action (multi-resolution discrete grid) ---
        dt_sections = (2 ** dt_base_boot).long()  # [64, 32, ..., 1]
        t_int_list = (torch.rand(bootstrap_batchsize, device=device) * dt_sections).floor().long()
        t_a_cont = t_int_list.float() / dt_sections.float()  # [32]
        t_a_cont = rearrange(t_a_cont, "b -> b 1")    # [5/64, 17/64, ..., 2/32, ..., 7/16, ..., 0]
        t_a_disc = self.rectified_flow.get_discrete_timestamp(t_a_cont, tensor_kwargs) # [873.0398, 750.0000, ..., 1000]
        sigmas_a = self.rectified_flow.get_sigmas(t_a_disc, tensor_kwargs) # [0.8730, 0.7500, ..., 1.0000] # time reversed sigma
        # Add channel/time dims for broadcasting
        t_v_disc, sigmas_v, t_a_disc, sigmas_a = map(
            lambda x: rearrange(x, "b -> b 1"),
            (t_v_disc, sigmas_v, t_a_disc, sigmas_a)
        )
        
        # ======================================================================
        # 3) Generate bootstrap targets (using teacher model)
        # ======================================================================
        # Clean samples (x_1): take the first bootstrap_batchsize samples
        x_1 = videos[:bootstrap_batchsize]
        a_1 = action[:bootstrap_batchsize]

        x_0 = torch.randn_like(x_1)
        a_0 = torch.randn_like(a_1)
        # x_0 noise, x_1 clean video (t * x_0 + (1-t) * x_1) # x_0 ~ N(0, I)
        # t_a_cont: [5/64, 17/64, ..., 2/32, ..., 7/16, ..., 0]  # t -> 1 
        # t_a_disc：[873.0398, 750.0000, ..., 1000]              # x_t -> noise 
        # sigmas_a: [0.8730, 0.7500, ..., 1.0000]                # t_a_disc -> 0
        x_t_boot, v_t_boot = self.rectified_flow.get_interpolation(x_0, x_1, sigmas_v)
        a_t_boot, va_t_boot = self.rectified_flow.get_interpolation(a_0, a_1, sigmas_a)

        condition : ActionGenerationCondition = self.conditioner({})
        condition = condition.set_video_condition(
            gt_frames=videos,
            random_min_num_conditional_frames=1,
            random_max_num_conditional_frames=1,
            num_conditional_frames=None,
        )
        
        cond_state = condition.gt_frames[:bootstrap_batchsize].type_as(x_t_boot) 
        _, C, _, _, _ = x_t_boot.shape
        cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1)[:bootstrap_batchsize].type_as(x_t_boot)
        x_t_boot = cond_state * cond_mask + x_t_boot * (1 - cond_mask)
        
        # ======================================================================
        # 4) Generate bootstrap targets (using teacher model)
        # ======================================================================
        def _process_chunk(x_chunk, a_chunk, t_v_chunk, t_a_chunk, dt_base_chunk, dt_chunk, t_a_cont_chunk):
            # First velocity prediction # 
            _, va_t1 = self.model(x_chunk, a_chunk, t_v_chunk, t_a_chunk, dt_base_chunk)
            # Advance action by half-step
            t_a1_cont = t_a_cont_chunk.squeeze(-1) + dt_chunk  # [chunk_B]
            t_a1_cont = torch.clamp(t_a1_cont, 0.0, 1.0)
            t_a1_disc = self.rectified_flow.get_discrete_timestamp(
                rearrange(t_a1_cont, "b -> b 1"), tensor_kwargs
            )
            t_a1_disc = rearrange(t_a1_disc, "b -> b 1")
            a_t1 = a_chunk + dt_chunk.view(-1, *[1] * (a_chunk.dim() - 1)) * va_t1
            # Second velocity prediction # 
            _, va_t2 = self.model(x_chunk, a_t1, t_v_chunk, t_a1_disc, dt_base_chunk)
            # Trapezoidal average
            return (va_t1 + va_t2) / 2.0
    
        num_chunks = (bootstrap_batchsize + chunk_size - 1) // chunk_size
        va_t_targets = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, bootstrap_batchsize)
            x_chunk, a_chunk, t_v_chunk, t_a_chunk, dt_base_chunk, dt_chunk, t_a_cont_chunk = map(
                lambda x: x[start:end],
                (x_t_boot, a_t_boot, t_v_disc, t_a_disc, dt_base_bootstrap, dt_bootstrap, t_a_cont)
            )

            va_t_target_chunk = _process_chunk(
                x_chunk, a_chunk, t_v_chunk, t_a_chunk,
                dt_base_chunk, dt_chunk, t_a_cont_chunk
            )
            va_t_targets.append(va_t_target_chunk)

        va_t_target = torch.cat(va_t_targets, dim=0)

        # ======================================================================
        remaining_batchsize = B - bootstrap_batchsize
        if remaining_batchsize > 0:
            # [7, 7, 7, 7, 7, ...]
            finest_dt_base = torch.full((remaining_batchsize,), log2_sections, device=device, dtype=torch.long)
            
            # Sample continuous t ~ U(0,1)
            t_rem_cont_a = rearrange(self.rectified_flow.sample_train_time(remaining_batchsize), "b -> b 1")
            t_rem_disc_a = self.rectified_flow.get_discrete_timestamp(t_rem_cont_a, tensor_kwargs)
            sigmas_rem_a = self.rectified_flow.get_sigmas(t_rem_disc_a, tensor_kwargs)
            
            t_rem_cont_v = rearrange(self.rectified_flow.sample_train_time(remaining_batchsize), "b -> b 1")
            t_rem_disc_v = self.rectified_flow.get_discrete_timestamp(t_rem_cont_v, tensor_kwargs)
            sigmas_rem_v = self.rectified_flow.get_sigmas(t_rem_disc_v, tensor_kwargs)        

            t_rem_disc_a, sigmas_rem_a, t_rem_disc_v, sigmas_rem_v = map(
                lambda x: rearrange(x, "b -> b 1"),
                (t_rem_disc_a, sigmas_rem_a, t_rem_disc_v, sigmas_rem_v)
            )

            # Interpolation for remaining samples
            x_1_rem = videos[bootstrap_batchsize:]
            a_1_rem = action[bootstrap_batchsize:]

            x_0_rem = torch.randn_like(x_1_rem)
            a_0_rem = torch.randn_like(a_1_rem)

            x_t_rem, v_t_rem = self.rectified_flow.get_interpolation(x_0_rem, x_1_rem, sigmas_rem_v)
            a_t_rem, va_t_rem = self.rectified_flow.get_interpolation(a_0_rem, a_1_rem, sigmas_rem_a)

            # For non-bootstrap samples, target velocity = direct rectified flow velocity
            va_t_rem_target = va_t_rem

            # Concatenate bootstrap + remaining
            x_t = torch.cat([x_t_boot, x_t_rem], dim=0)
            a_t = torch.cat([a_t_boot, a_t_rem], dim=0)
            v_t = torch.cat([v_t_boot, v_t_rem], dim=0)
            va_t = torch.cat([va_t_target, va_t_rem_target], dim=0)
            dt_base_out = torch.cat([dt_base_boot, finest_dt_base], dim=0)
        else:
            # No remaining samples
            x_t = x_t_boot
            a_t = a_t_boot
            v_t = v_t_boot
            va_t = va_t_target
            dt_base_out = dt_base_boot

        return x_t, a_t, v_t, va_t, dt_base_out
    
    
"""
PYTHONPATH=. python cosmos_predict2/_src/predict2/action_parallel_predict/utils/targets_shortcut.py
"""
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 8
    C, T, H, W = 16, 5, 32, 40
    action_dim = 16

    # Build mock video and action data
    videos = torch.randn(B, C, T, H, W, device=device)
    action = torch.randn(B, T, action_dim, device=device)

    # Initialize mock model
    model1 = MockModel1(video_channels=C, action_dim=action_dim).to(device)

    # Initialize MultiModelRectifiedFlow 
    rectified_flow = RectifiedFlow(
        velocity_field=model1, 
        train_time_distribution="logitnormal",
        use_dynamic_shift=False,
        shift=5,
        train_time_weight_method="reweighting",
    )
    
    conditioner = ActionGenerationConditioner()
    
    model2 = MockModel2(rectified_flow=rectified_flow, conditioner=conditioner, model=model1)

    # Call get_targets
    x_t, xa_t, v_t, va_t, dt_base_out = model2.get_targets(
        videos=videos,
        action=action,
        bootstrap_every=2,
        denoise_timesteps=128,
    )

    # Debug output
    print("✅ Debug Output Shapes:")
    print(f"x_t (noisy video):      {x_t.shape}")
    print(f"xa_t (noisy action):    {xa_t.shape}")
    print(f"v_t (video velocity):   {v_t.shape}")
    print(f"va_t (action velocity): {va_t.shape}")
    print(f"dt_base_out:            {dt_base_out.shape} (values: min={dt_base_out.min().item()}, max={dt_base_out.max().item()})")

    print("\n✅ All tensors on device:", x_t.device)
    print("✅ Debug complete!")

if __name__ == "__main__":
    main()
