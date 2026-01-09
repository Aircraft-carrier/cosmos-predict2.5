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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.configs.common.defaults.callbacks import (
    BASIC_CALLBACKS,
    SPEED_CALLBACKS,
    # WANDB_CALLBACK,
)
from cosmos_predict2._src.predict2.action_parallel_predict.callbacks.wandb_log_action import Wandb4ActionCallback
from cosmos_predict2._src.predict2.action_parallel_predict.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_predict2._src.predict2.action_parallel_predict.callbacks.iter_speed import IterSpeed4Action


_basic_callback = copy.deepcopy(BASIC_CALLBACKS)
_basic_callback["iter_speed"] = L(IterSpeed4Action)(
    every_n="${trainer.logging_iter}",
    save_s3="${upload_reproducible_setup}",
    save_s3_every_log_n=5,
)
DEBUG_CALLBACKS = dict()
LONG_RUNNING_CALLBACKS = dict()

VIZ_ONLINE_SAMPLING_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample)(
        every_n=5000,
        save_s3=True,
        do_x0_prediction=False,
    ),
    every_n_sample_ema=L(EveryNDrawSample)(
        every_n=5000,
        is_ema=True,
        save_s3=True,
        do_x0_prediction=False,
    ),
)

WANDB4Action_CALLBACKS = dict(
    wandb=L(Wandb4ActionCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,     # 设置了save_s3为False，这个参数没用了
    ),
    # wandb_10x=L(WandbCallback)(           # TODO 拉长记录间隔，offline模式下会产生两个log文件，这里先注释掉
    #     logging_iter_multipler=10,
    #     save_logging_iter_multipler=1,
    #     save_s3="${upload_reproducible_setup}",
    # ),
)

def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    # cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="debug", node=DEBUG_CALLBACKS)
    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="long", node=LONG_RUNNING_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="cluster_speed", node=SPEED_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB4Action_CALLBACKS)
    