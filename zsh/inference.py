import math
import os
from typing import TYPE_CHECKING

import torch
import torchvision
from megatron.core import parallel_state
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

class Easyvideo2worldInference:
    
    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
    ):
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
        )
        if TYPE_CHECKING:
            from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
                Video2WorldModelRectifiedFlow,
            )

            model: Video2WorldModelRectifiedFlow = model

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None