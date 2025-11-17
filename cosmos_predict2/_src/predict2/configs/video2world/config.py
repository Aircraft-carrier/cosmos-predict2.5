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

import importlib
from typing import Any, List

import attrs

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_predict2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_predict2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_predict2._src.predict2.configs.common.defaults.dataloader import register_training_and_val_data
from cosmos_predict2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_predict2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_predict2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_predict2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_predict2._src.predict2.configs.video2world.defaults.callbacks import register_callbacks
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import register_conditioner
from cosmos_predict2._src.predict2.configs.video2world.defaults.model import register_model
from cosmos_predict2._src.predict2.configs.video2world.defaults.net import register_net


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "ddp"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "video_prediction_conditioner"},
            {"ema": "power"},
            {"tokenizer": "cosmos_tokenizer_causal_cv8x8x8_c16_res720_t121_it121_v1_0"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    # JobConfig -> LINK cosmos-predict2.5/cosmos_predict2/_src/imaginaire/config.py:181
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # TrainerConfig -> LINK cosmos-predict2.5/cosmos_predict2/_src/imaginaire/config.py:348
    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False           # 禁用掉队检测（用于多GPU训练中检测慢节点）
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    # * 将不同模块的配置注册到 Hydra ConfigStore ---> 这里其实就是在定义有哪些参数，类似 parser.add_argument，这里复杂一些，用hydra来管理
    # 以 optimizer 为例，注册过程就是告诉 Hydra：有一个名为 optimizer 的配置组，这个组有两个选项：fusedadamw 和 adamw，当用户选择 optimizer=fusedadamw 时，使用这个配置对象
    # 所有 register_*() 函数都遵循相同的模式：获取 Hydra ConfigStore 实例
    # group -> 配置组名；package -> 配置在 Config 对象中的路径；name -> 选项名称；node -> 配置对象/值
    register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_net()
    # Sets the video conditioning frames for video-to-video generation.
    register_conditioner()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    # 递归导入指定包下的所有 Python 模块，确保实验级配置被注册到 Hydra。
    # * 就是设置很多种可选的默认参数，根据experiment=xxx来选择
    # TODO 这段代码的逻辑就是导入experiment下所有文件，注册各种配置名称，这样在train.py中就可以通过experiment=xxx来选择配置 <-- 感觉不需要这个功能，能否删除？
    # NOTE 不能，因为需要继承原来的训练配置，详见 cosmos-predict2.5/cosmos_predict2/experiments/base/cosmos_nemo_assets.py
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    try:
        if importlib.util.find_spec("cosmos_predict2.experiments.internal") is not None:
            import_all_modules_from_package("cosmos_predict2.experiments.internal", reload=True)
    except ModuleNotFoundError:
        pass  # Module or parent package doesn't exist
    # 遍历cosmos_predict2.experiments下所有目录，现在就是base，遍历cosmos_predict2.experiments.base下所有文件
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)

    return c
