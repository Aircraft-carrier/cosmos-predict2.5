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

"""Base model inference script."""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).
    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""


def main(
    args: Args,
):
    inference_samples = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    # name = 'libero_lift1'
    # prompt_path = '/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/libero_asset/libero_lift1.txt'  # Prompt 文件路径
    # prompt = 'put the white mug on the left plate and put the yellow and white mug on the right plate'  # 正向提示文本
    # negative_prompt = (
    #     "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, "
    #     "shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    #     "poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, "
    #     "unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
    #     "visual noise, and flickering. Overall, the video is of poor quality."
    # ) 
    # seed = 0                   
    # guidance = 0                    # Classifier-free guidance scale
    # inference_type = 'video2world'  # 推理模式：从视频生成世界状态（如动作、物体位置等）
    # input_path = '/gemini/platform/public/embodiedAI/users/zsh/code/cosmos-predict2.5/libero_asset/libero_lift1.mp4'  # 输入视频路径
    # resolution = 'none'             # 分辨率策略：不调整（由模型或配置决定）
    # num_output_frames = 77          # 生成 77 帧输出
    # num_steps = 35                  # 扩散采样步数
    # enable_autoregressive = False   # 禁用自回归生成（一次性生成全部帧）
    # chunk_size = 77                 # 视频处理分块大小（等于总帧数，即不分块）
    # chunk_overlap = 1               # 分块重叠帧数（虽设为1，但因 chunk_size == total，实际未分块）
    
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    # LINK: cosmos_predict2/inference.py:40
    from cosmos_predict2.inference import Inference
    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()
    try:
        # LINK: /root/miniconda3/envs/cosmos/lib/python3.10/site-packages/tyro/_cli.py:232
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )

    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()

# input_files = ['libero_asset/libero_video2world.json']  # 输入 JSON 文件路径
# output_dir = 'outputs/cosmos_libero'                    # 输出目录
# model = '2B/post-trained'                               # 模型标识：2B 规模，post-trained 版本
# checkpoint_path = '/gemini/space/cosmos-predict/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt'  # EMA 权重（BF16 精度）
# experiment = 'reason_embeddings_rectified_flow_libero'  # 实验名称
# config_file = 'cosmos_predict2/_src/predict2/action_parallel_predict/configs/action_cogeneration/config.py'  # 推理配置文件路径
# context_parallel_size = 1                               # 上下文并行大小（单卡模式）
# offload_diffusion_model = False                         # 不加载扩散模型到 CPU
# offload_tokenizer = False                               # 不加载 tokenizer
# offload_text_encoder = False                            # 不加载文本编码器
# disable_guardrails = False                              # 启用安全护栏（guardrails）
# offload_guardrail_models = True                         # 卸载 guardrail 模型以节省显存
# keep_going = True                                       # 出错时跳过并继续处理后续样本
# profile = False                                         # 不启用性能分析