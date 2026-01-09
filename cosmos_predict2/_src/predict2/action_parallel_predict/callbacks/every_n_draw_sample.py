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

import json
import math
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from typing import List, Union, Dict, Any
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.predict2.action_parallel_predict.utils.utils import unnormalize_action

def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """
    Resize the image to the given size. This is done so that wandb can display the image correctly.
    """
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    """
    This callback sample condition inputs from training data, run inference and save the results to wandb and s3.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        n_viz_sample (int, optional): for each batch, min(n_viz_sample, batch_size) samples will be saved to wandb. Defaults to 3.
        n_sample_to_save (int, optional): number of samples to save. The actual number of samples to save is min(n_sample_to_save, data parallel instances). Defaults to 128.
        num_sampling_step (int, optional): number of sampling steps. Defaults to 35.
        guidance (List[float], optional): guidance scale. Defaults to [0.0, 3.0, 7.0].
        do_x0_prediction (bool, optional): whether to do x0 prediction. Defaults to True.
        n_sigmas_for_x0_prediction (int, optional): number of sigmas to use for x0 prediction. Defaults to 4.
        save_s3 (bool, optional): whether to save to s3. Defaults to False.
        is_ema (bool, optional): whether the callback is run for ema model. Defaults to False.
        use_negative_prompt (bool, optional): whether to use negative prompt. Defaults to False.
        fps (int, optional): frames per second when saving the video. Defaults to 16.
        save_videos (bool, optional): whether to save videos. Defaults to False.
        save_images (bool, optional): whether to save images (3-frame preview). Defaults to True.
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [0.0, 3.0, 7.0],
        do_x0_prediction: bool = True,
        n_sigmas_for_x0_prediction: int = 4,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        prompt_type: str = "t5_xxl",
        fps: int = 16,
        run_at_start: bool = False,
        save_videos: bool = False,
        save_images: bool = True,
    ):
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: 1 file, # per file: min(batch_size, n_viz_sample)
        super().__init__(every_n, step_size, run_at_start=run_at_start)

        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.do_x0_prediction = do_x0_prediction
        self.n_sigmas_for_x0_prediction = n_sigmas_for_x0_prediction
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.prompt_type = prompt_type
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.save_videos = save_videos
        self.save_images = save_images

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        self.image_dir = f"{self.local_dir}/image"
        self.video_dir = f"{self.local_dir}/video"
        self.action_dir = f"{self.local_dir}/action"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.video_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            if self.prompt_type == "t5_xxl":
                self.negative_prompt_data = easy_io.load(
                    "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/000000.pkl"
                )
            elif self.prompt_type == "umt5_xxl":
                self.negative_prompt_data = easy_io.load(
                    "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/umt5_neg.pt"
                )
            else:
                raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        action_0 = data_batch['action']
        x0, action_0, condition, _, _, _, _ = model.broadcast_split_for_model_parallelsim(x0, action_0, condition, None, None, None, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_sigmas_for_x0_prediction + 1
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        random_action_noise = torch.randn(*action_0.shape, generator=generator, **model.tensor_kwargs)
        
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        mse_action_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            x_sigma_action = sigma * random_action_noise + action_0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            # TODO: debug x0 prediction with action
            # sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            sample, action_sample = model.denoise(random_noise, x_sigma, x_sigma_action, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            action_loss = distributed.dist_reduce_tensor(F.mse_loss(action_sample, action_0))
            mse_loss_list.append(mse_loss)
            mse_action_loss_list.append(action_loss)

            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            if self.do_x0_prediction:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )

            log.debug("entering, sample", rank0_only=False)
            sample_img_fp, action_fp = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            log.debug("done, sample", rank0_only=False)
            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                x0_image_path = x0_img_fp["image"] if isinstance(x0_img_fp, dict) else x0_img_fp
                if self.save_images and x0_image_path:
                    info[f"{self.name}/{tag}_x0"] = wandb.Image(x0_image_path, caption=f"{sample_counter}")
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            # Log image (3-frame preview) if save_images is enabled
            if self.save_images and isinstance(sample_img_fp, dict) and sample_img_fp.get("image"):
                info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp["image"], caption=f"{sample_counter}")

            # Log video if save_videos is enabled
            if self.save_videos and isinstance(sample_img_fp, dict) and sample_img_fp.get("video"):
                info[f"{self.name}/{tag}_sample_video"] = wandb.Video(
                    sample_img_fp["video"], fps=self.fps, caption=f"{sample_counter}"
                ) # TODO: add format="mp4"
                
            if action_fp is not None and isinstance(action_fp, dict):
                losses_cont = action_fp["metadata"]["losses_action_cont"]
                avg_loss_cont = sum(losses_cont) / len(losses_cont) if losses_cont else 0.0
                info["train/N_sample/action_mse_loss"] = avg_loss_cont
                
                losses_disc = action_fp["metadata"]["losses_action_disc"]
                avg_loss_disc = sum(losses_disc) / len(losses_disc) if losses_disc else 0.0
                info["train/N_sample/action_bce_loss"] = avg_loss_disc
                
                losses_video = action_fp["metadata"]["video_mse_loss"]
                avg_loss_video = sum(losses_video) / len(losses_video) if losses_video else 0.0
                info["train/N_sample/video_mse_loss"] = avg_loss_video

            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            if self.negative_prompt_data["t5_text_embeddings"].shape != data_batch["t5_text_embeddings"].shape:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    repeat(
                        self.negative_prompt_data["t5_text_embeddings"],
                        "... -> b ...",
                        b=batch_size,
                    ),
                    **model.tensor_kwargs,
                )
            else:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    self.negative_prompt_data["t5_text_embeddings"],
                    **model.tensor_kwargs,
                )

            assert data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape, (
                f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]
        
        
        action_0 = data_batch['action']         # [B, T, D] : [1, 17, 7]
        action_gt = data_batch['gt_action']     # unnormalized action
        stats = data_batch['stats']
        normalization_type = data_batch['normalization_type']
        
        x0, action_0, condition, _, _, _, _ = model.broadcast_split_for_model_parallelsim(x0, action_0, condition, None, None, None, None)
        action_gt = broadcast_split_tensor(action_gt, 1, model.get_context_parallel_group())
        
        to_show = []
        all_action_samples = []
        # For continuous and discrete action loss logging
        action_cont_loss_list = []  # MSE
        action_disc_loss_list = []  # Binary Cross Entropy
        
        mse_loss_list = []
        for guidance in self.guidance: # [0, 3, 7]
            # Action Gen: # LINK cosmos_predict2/_src/predict2/action_parallel_predict/models/action_generation_video2world_rectified_flow_model.py:365
            sample, action_sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=True if self.use_negative_prompt else False,
            )
            # TODO: add action visualization later 
            # real_action = data_batch['action']
            # to_show_action = action_sample.float().cpu()
            
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            # ---- loss ----
            action_sample = unnormalize_action(action_sample, stats, normalization_type[0])  # [B, T, D] : [1, 17, 7]
            cont_pred = action_sample[..., :6]
            cont_gt   = action_gt[..., :6] 
            disc_pred_logits = action_sample[..., 6] 
            disc_gt_labels   = action_gt[..., 6].float()
            # Continuous part: MSE
            cont_loss = F.mse_loss(cont_pred, cont_gt)
            cont_loss_reduced = distributed.dist_reduce_tensor(cont_loss)
            action_cont_loss_list.append(cont_loss_reduced)
            # Discrete part: Binary Cross-Entropy with Logits
            bce_loss = F.binary_cross_entropy_with_logits(disc_pred_logits, disc_gt_labels)
            bce_loss_reduced = distributed.dist_reduce_tensor(bce_loss)
            action_disc_loss_list.append(bce_loss_reduced)    
            # Video-level MSE loss
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, raw_data))  # [B, C, T, H, W] : [1, 3, 17, 240, 320] -> tensor(0.0538)
            mse_loss_list.append(mse_loss)
            # --- save action samples to json ---
            all_action_samples.append(action_sample.float().cpu().numpy().tolist())   
            # ---- visualization ----
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())
        all_action_samples.append(action_gt.float().cpu().numpy().tolist())   

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"
        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            show_data = self.run_save(to_show, batch_size, base_fp_wo_ext)     # logs/task_name/EveryNDrawSample/image(video)/...
            action_data = self.save_actions_to_json(
                all_action_samples, 
                action_cont_loss_list, 
                action_disc_loss_list, 
                mse_loss_list,
                batch_size, 
                base_fp_wo_ext
            )
            return show_data, action_data
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        image_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        video_base_fp = f"{base_fp_wo_ext}_video.mp4"
        image_path = f"{self.image_dir}/{image_base_fp}"
        video_path = f"{self.video_dir}/{video_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                if self.save_images:
                    to_show = rearrange(
                        to_show[:, :n_viz_sample],
                        "n b c t h w -> t c (n h) (b w)",
                    )
                    image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                    # resize so that wandb can handle it
                    torchvision.utils.save_image(resize_image(image_grid, 1024), image_path, nrow=1, scale_each=True)
                return {"image": image_path if self.save_images else None, "video": None}
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, t, h, w]

                if self.save_videos:
                    # n categories concat in height, b batches concat in width
                    video_tensor = rearrange(to_show, "n b c t h w -> c t (n h) (b w)")
                    video_tensor = rearrange(video_tensor, "c t h w -> t h w c")
                    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
                    torchvision.io.write_video(video_path, video_tensor, fps=self.fps)

                if self.save_images:
                    # resize 3 frames frames so that we can display them on wandb
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show_frames = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                    to_show_frames = rearrange(
                        to_show_frames,
                        "n b c t h w -> 1 c (n h) (b t w)",
                    )

                    # resize so that wandb can handle it
                    image_grid = torchvision.utils.make_grid(to_show_frames, nrow=1, padding=0, normalize=False)
                    torchvision.utils.save_image(
                        resize_image(image_grid, log_image_size), image_path, nrow=1, scale_each=True
                    )

                return {"image": image_path if self.save_images else None, "video": video_path if self.save_videos else None}
        return None

    def save_actions_to_json(
        self,
        all_action_samples: List[List],  # list of [B, T, D] as nested lists
        action_cont_loss_list: List[Union[float, torch.Tensor]],
        action_disc_loss_list: List[Union[float, torch.Tensor]],
        video_mse_loss_list: Union[float, torch.Tensor],
        batch_size: int,
        base_fp_wo_ext: str,
    ) -> Dict[str, Any]:
        """
        Construct a structured dictionary containing action samples and their corresponding MSE losses.
        Does NOT write to disk — returns the data as a Python dict for flexible downstream use.

        Args:
            all_action_samples: List where each element is a list of shape [B, T, D] (already .tolist()).
                                The last element is the ground truth action.
            mse_action_loss_list: List of MSE losses for each guidance scale (one per sample in all_action_samples[:-1]).
            batch_size: Number of samples in the batch.
            base_fp_wo_ext: Base file path without extension (kept for metadata consistency, though not used for I/O).

        Returns:
            Dict[str, Any]: A structured dictionary containing:
                - metadata (batch size, guidance scales, losses)
                - ground_truth actions
                - generated_samples per guidance
        """
        # Convert losses to float (in case they are tensors or on GPU)
        def _tensor_to_float_rounded(loss_list, decimals=4):
            result = []
            for loss in loss_list:
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                result.append(round(float(loss), decimals))
            return result

        action_cont_loss = _tensor_to_float_rounded(action_cont_loss_list)
        action_disc_loss = _tensor_to_float_rounded(action_disc_loss_list)
        video_mse_loss = _tensor_to_float_rounded(video_mse_loss_list)

        # Separate generated samples and ground truth
        generated_samples = all_action_samples[:-1]  # All but last
        ground_truth = all_action_samples[-1]        # Last is GT

        # Build structured output
        output_data = {
            "metadata": {
                "batch_size": batch_size,
                "num_guidance_scales": len(generated_samples),
                "guidance_scales": list(getattr(self, 'guidance', list(range(len(generated_samples))))),
                "losses_action_cont": action_cont_loss,
                "losses_action_disc": action_disc_loss,
                "video_mse_loss": video_mse_loss,
            },
            "ground_truth": ground_truth,  # shape: [B, T, D]
            "generated_samples": []        # list of [B, T, D] for each guidance
        }

        for i, sample in enumerate(generated_samples):
            output_data["generated_samples"].append({
                "guidance_index": i,
                "guidance_scale": output_data["metadata"]["guidance_scales"][i] if i < len(output_data["metadata"]["guidance_scales"]) else None,
                "actions": sample  # already list of lists
            })

        # Ensure output directory exists
        os.makedirs(self.action_dir, exist_ok=True)
        action_base_fp = f"{base_fp_wo_ext}_action.json"
        json_path = f"{self.action_dir}/{action_base_fp}"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        return output_data
