# import numpy as np
# import torch

# # Assume: num_train_timesteps (int), shift (float) are defined
# num_train_timesteps = 1000
# shift = 5
# num_inference_steps = 36
# shift_inference = 5
# # 1. Generate timesteps in [1, num_train_timesteps] and reverse to [num_train_timesteps, ..., 1]
# timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1]
# # [1000, 999, ..., 1]
# # 2. Convert to normalized sigmas in [1/num_train_timesteps, 1.0]
# sigmas = timesteps / num_train_timesteps  # shape: (num_train_timesteps,)
# # [1, 0.999, ..., 0.001]
# # 3. Apply shift transformation (element-wise on NumPy array)
# sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
# # [1, 0.9997998, ..., 0.00498008] 数值更接近 1 后期生成更精细
# # 4. Recompute timesteps from shifted sigmas (still in [0, num_train_timesteps])
# timesteps = sigmas * num_train_timesteps

# sigma_min = float(sigmas[-1])  # smallest sigma (final step)
# sigma_max = float(sigmas[0])   # largest sigma (initial step)

# print(f"sigma_max: {sigma_max:.6f}")  
# print(f"sigma_min: {sigma_min:.6f}")  
# # shift = 5 :  1.000000 -> 0.004980
# # shift = 1 :  1.000000 -> 0.001000
# sigma_max_inference = 1
# sigma_min_inference = 0.001

# sigmas_inference = np.linspace(sigma_max_inference, sigma_min_inference, num_inference_steps + 1).copy()[:-1]  
# sigmas_inference = shift * sigmas_inference / (1 + (shift - 1) * sigmas_inference)
# timesteps_inference = sigmas_inference * num_train_timesteps

# for i, timestep_infer in enumerate(timesteps_inference):
#     closest_idx = (np.abs(timesteps - timestep_infer)).argmin()
#     closest_timestep = timesteps[closest_idx]
#     diff = np.abs(closest_timestep - timestep_infer)
    
#     if diff < 0.5:  
#         print(f"Inference timestep {i} ({timestep_infer:.5f}) matches training timestep {closest_timestep:.5f}")
#     else:
#         print(f"Warning: Inference timestep {i} ({timestep_infer:.5f}) does not closely match any training timestep {closest_timestep:.5f}")

from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow

sample_scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
)

rectified_flow = RectifiedFlow(
    velocity_field=None,
    train_time_distribution="logitnormal",
    use_dynamic_shift=False,
    shift=5,
    train_time_weight_method="reweighting",
)

sample_scheduler.set_timesteps(
    36, shift=5, use_kerras_sigma=True
)

# print(rectified_flow.noise_scheduler.timesteps)
# [1000.0000,  999.7998,  999.5994,  999.3986,  999.1974,  998.9960,           # 900 段是很密集的, 清晰化的部分很多
#  ....
#  19.6850,   14.8221,    9.9206,    4.9801])
# print(sample_scheduler.timesteps)
# [995, 994, 993, 992, 990, 989, 987, 984, 982, 978, 974, 969, 963, 955,
# 945, 933, 918, 900, 877, 849, 814, 771, 721, 661, 593, 519, 440, 360,
# 284, 215, 157, 110,  74,  47,  29,  17,   9]
# PYTHONPATH=. python zsh/time_step.py
import torch
from einops import rearrange

n_sample = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_kwargs_fp32 = {"dtype": torch.float32, "device": device}
noise = torch.randn(n_sample, 3, 64, 64, **tensor_kwargs_fp32)      
gt_frames = torch.randn(n_sample, 3, 64, 64, **tensor_kwargs_fp32)  
t_B = torch.full((n_sample,), 0.999, **tensor_kwargs_fp32)  
import ipdb;ipdb.set_trace()
timesteps = rectified_flow.get_discrete_timestamp(t_B, tensor_kwargs=tensor_kwargs_fp32)  
sigmas = rectified_flow.get_sigmas(timesteps, tensor_kwargs=tensor_kwargs_fp32)  
sigmas = rearrange(sigmas, "b -> b 1")  
latents, _ = rectified_flow.get_interpolation(noise, gt_frames, sigmas)  