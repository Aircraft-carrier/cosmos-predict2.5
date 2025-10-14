# Cosmos Predict2.5 训练流程函数调用分析

本文档详细说明运行 `run_posttrain_nemo.sh` 脚本后，整个训练流程中调用的所有关键函数及其作用。

---

## 一、脚本入口分析

### 1.1 Shell脚本：`run_posttrain_nemo.sh`

**位置**: `1shell_script/run_posttrain_nemo.sh`

**脚本内容**:
```bash
export IMAGINAIRE_OUTPUT_ROOT=/gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5/logs
cd /gemini/platform/public/embodiedAI/users/fanchenyou/code/cosmos/cosmos-predict2.5
export PYTHONPATH=$PWD 

torchrun --nproc_per_node=1 scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

**作用**:
- 设置输出根目录环境变量 `IMAGINAIRE_OUTPUT_ROOT`
- 设置Python路径为项目根目录
- 使用 `torchrun` 启动分布式训练（单GPU模式：`nproc_per_node=1`）
- 调用训练脚本 `scripts/train.py`
- 加载配置文件 `config.py`
- 选择实验配置 `predict2_video2world_training_2b_cosmos_nemo_assets`

---

## 二、主训练脚本：`scripts/train.py`

### 2.1 主函数流程

**位置**: `scripts/train.py`

#### 函数调用顺序：

1. **`if __name__ == "__main__"`**
   - 解析命令行参数（config路径、opts覆盖选项、dryrun、smoke等）
   
2. **`get_config_module(args.config)`**
   - **位置**: `cosmos_predict2/_src/imaginaire/utils/config_helper.py`
   - **作用**: 将配置文件路径转换为Python模块路径
   - **输入**: `cosmos_predict2/_src/predict2/configs/video2world/config.py`
   - **输出**: `cosmos_predict2._src.predict2.configs.video2world.config`

3. **`importlib.import_module(config_module).make_config()`**
   - 动态导入配置模块
   - 调用配置模块的 `make_config()` 函数创建配置对象

4. **`override(config, overrides)`**
   - **位置**: `cosmos_predict2/_src/imaginaire/utils/config_helper.py`
   - **作用**: 使用命令行参数覆盖配置
   - 应用实验配置：`experiment=predict2_video2world_training_2b_cosmos_nemo_assets`

5. **`launch(config, args)`**
   - 启动主训练函数

---

### 2.2 `launch()` 函数详解

**位置**: `scripts/train.py:34`

**函数签名**: `launch(config: Config, args: argparse.Namespace) -> None`

**执行流程**:

#### 2.2.1 初始化分布式环境
```python
distributed.init()
```
- **位置**: `cosmos_predict2/_src/imaginaire/utils/distributed.py`
- **作用**: 初始化PyTorch分布式训练环境
- 设置进程组、rank、world_size等

#### 2.2.2 验证和冻结配置
```python
config.validate()
config.freeze()
```
- **作用**: 验证配置的有效性并冻结，防止训练过程中被修改

#### 2.2.3 创建训练器
```python
trainer = config.trainer.type(config)
```
- **类型**: `ImaginaireTrainer`
- **位置**: `cosmos_predict2/_src/imaginaire/trainer.py`
- 创建训练器实例（详见第三章）

#### 2.2.4 创建或加载模型
```python
if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
    model = create_model_from_consolidated_checkpoint_with_fsdp(config)
else:
    model = instantiate(config.model)
```
- **位置**: `cosmos_predict2/_src/predict2/utils/model_loader.py`
- **作用**: 
  - 如果checkpoint是`.pt`格式，从合并的checkpoint创建FSDP模型
  - 否则使用lazy config实例化模型
- **模型类型**: `Video2WorldModel`（详见第四章）

#### 2.2.5 创建数据加载器
```python
dataloader_train = instantiate(config.dataloader_train)
dataloader_val = instantiate(config.dataloader_val)
```
- **作用**: 实例化训练和验证数据加载器（详见第五章）

#### 2.2.6 启动训练
```python
trainer.train(model, dataloader_train, dataloader_val)
```
- 进入主训练循环（详见第三章）

---

## 三、训练器：`ImaginaireTrainer`

### 3.1 训练器初始化：`__init__()`

**位置**: `cosmos_predict2/_src/imaginaire/trainer.py:58`

**关键初始化步骤**:

1. **初始化分布式环境**
   ```python
   with distributed_init():
       distributed.init()
   ```

2. **初始化Megatron并行状态**（如果使用）
   ```python
   parallel_state.initialize_model_parallel(
       pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
       tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
       context_parallel_size=config.model_parallel.context_parallel_size,
   )
   ```
   - **作用**: 设置模型并行、张量并行、上下文并行的大小

3. **创建本地作业目录**
   ```python
   os.makedirs(config.job.path_local, exist_ok=True)
   LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
   LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
   ```
   - 保存配置文件到本地

4. **初始化日志系统**
   ```python
   log.init_loguru_file(f"{config.job.path_local}/stdout.log")
   ```

5. **设置随机种子**
   ```python
   misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
   ```
   - 保证训练的可重复性

6. **初始化回调函数组**
   ```python
   self.callbacks = callback.CallBackGroup(config=config, trainer=self)
   ```
   - **作用**: 管理训练过程中的所有回调函数（详见第七章）

7. **初始化检查点管理器**
   ```python
   self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
   ```
   - **位置**: `cosmos_predict2/_src/imaginaire/utils/checkpointer.py`
   - **作用**: 管理模型检查点的保存和加载

8. **初始化训练计时器**
   ```python
   self.training_timer = misc.TrainingTimer()
   ```

9. **初始化掉队检测器**
   ```python
   self.straggler_detector = StragglerDetectorV2(...)
   ```
   - **作用**: 检测分布式训练中的慢节点

---

### 3.2 主训练函数：`train()`

**位置**: `cosmos_predict2/_src/imaginaire/trainer.py:150`

**函数签名**: 
```python
train(model: ImaginaireModel, 
      dataloader_train: DataLoader, 
      dataloader_val: DataLoader) -> None
```

#### 3.2.1 模型准备

```python
model = model.to("cuda", memory_format=self.config.trainer.memory_format)
model.on_train_start(self.config.trainer.memory_format)
```
- **作用**: 将模型移动到GPU，调用模型的训练前准备钩子

#### 3.2.2 初始化优化器和调度器

```python
self.callbacks.on_optimizer_init_start()
optimizer, scheduler = model.init_optimizer_scheduler(
    self.config.optimizer, 
    self.config.scheduler
)
grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
self.callbacks.on_optimizer_init_end()
```

**`model.init_optimizer_scheduler()` 详解**:
- **位置**: `cosmos_predict2/_src/imaginaire/model.py:38`
- **调用链**:
  1. 设置优化器参数为模型参数
  2. 实例化优化器配置（FusedAdamW）
  3. 实例化学习率调度器（LambdaLinear）
- **返回**: `(optimizer, scheduler)` 元组

#### 3.2.3 加载检查点

```python
iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
```

**`checkpointer.load()` 详解**:
- **位置**: `cosmos_predict2/_src/imaginaire/utils/checkpointer.py`
- **作用**: 
  - 从配置指定的路径加载checkpoint
  - 本案例中加载: `/gemini/platform/public/embodiedAI/users/fanchenyou/models/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt`
  - 恢复模型权重、优化器状态、调度器状态
  - 返回当前迭代次数

#### 3.2.4 创建DDP/FSDP包装器

```python
if self.config.trainer.distributed_parallelism == "ddp":
    model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
elif self.config.trainer.distributed_parallelism == "fsdp":
    model_ddp = model
```
- **作用**: 根据配置选择DDP或FSDP模式包装模型

#### 3.2.5 主训练循环

```python
self.callbacks.on_train_start(model, iteration=iteration)

while True:
    dataloader_train_iter = iter(dataloader_train)
    while True:
        # 加载数据批次
        self.callbacks.on_before_dataloading(iteration)
        data_batch = next(dataloader_train_iter)
        self.callbacks.on_after_dataloading(iteration)
        
        # 移动数据到GPU
        data_batch = misc.to(data_batch, device="cuda")
        
        # 训练步骤
        self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
        output_batch, loss, grad_accum_iter = self.training_step(
            model_ddp, optimizer, scheduler, grad_scaler, 
            data_batch, iteration=iteration, grad_accum_iter=grad_accum_iter
        )
        self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        
        # 保存检查点
        if iteration % self.config.checkpoint.save_iter == 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        
        # 验证
        if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        
        iteration += 1
```

---

### 3.3 训练步骤：`training_step()`

**位置**: `cosmos_predict2/_src/imaginaire/trainer.py:266`

**函数签名**:
```python
training_step(
    model_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler,
    data: dict[str, torch.Tensor],
    iteration: int = 0,
    grad_accum_iter: int = 0
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]
```

**执行流程**:

#### 3.3.1 前向传播
```python
self.callbacks.on_before_forward(iteration=iteration)
with self.training_timer("forward"):
    output_batch, loss = model_ddp.training_step(data, iteration)
self.callbacks.on_after_forward(iteration=iteration)
```

**`model_ddp.training_step()` 详解**（核心）:
- **位置**: `cosmos_predict2/_src/predict2/models/video2world_model.py` 和 `text2world_model.py`
- 详见第四章

#### 3.3.2 反向传播
```python
self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
with self.training_timer("backward"):
    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
    loss_scaled.backward()
    model_ddp.on_after_backward()
self.callbacks.on_after_backward(model_ddp, iteration=iteration)
```

#### 3.3.3 优化器步骤
```python
if grad_accum_iter == self.config.trainer.grad_accum_iter:
    self.callbacks.on_before_optimizer_step(model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scheduler.step()
    self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
    optimizer.zero_grad(set_to_none=True)
    grad_accum_iter = 0
```

---

### 3.4 验证函数：`validate()`

**位置**: `cosmos_predict2/_src/imaginaire/trainer.py:335`

**执行流程**:
```python
self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
model.eval()
with ema.ema_scope(model, enabled=model.config.ema.enabled):
    for val_iter, data_batch in enumerate(dataloader_val):
        data_batch = misc.to(data_batch, device="cuda")
        self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
        output_batch, loss = model.validation_step(data_batch, iteration)
        self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
self.callbacks.on_validation_end(model, iteration=iteration)
```

**关键点**:
- 使用EMA（指数移动平均）权重进行验证
- 调用模型的 `validation_step()` 方法

---

## 四、模型：`Video2WorldModel`

### 4.1 模型配置

**位置**: `cosmos_predict2/_src/predict2/models/video2world_model.py`

**类层次结构**:
```
ImaginaireModel (基类)
  └── Text2WorldModel (DiffusionModel)
        └── Video2WorldModel
```

**配置类**: `Video2WorldConfig`
- 继承自 `Text2WorldModelConfig`
- 关键参数:
  - `min_num_conditional_frames`: 最小条件帧数（默认1）
  - `max_num_conditional_frames`: 最大条件帧数（默认2）
  - `sigma_conditional`: 条件帧的噪声水平（默认0.0001）
  - `conditioning_strategy`: 条件策略（"frame_replace"）
  - `high_sigma_strategy`: 高sigma采样策略
  - `fsdp_shard_size`: FSDP分片大小

---

### 4.2 训练步骤：`training_step()`

**位置**: 继承自 `Text2WorldModel`（`cosmos_predict2/_src/predict2/models/text2world_model.py`）

**执行流程**:

#### 4.2.1 获取数据和条件
```python
raw_state, latent_state, condition = self.get_data_and_condition(data_batch)
```

**`get_data_and_condition()` 详解**:
- **位置**: `video2world_model.py:75`
- **作用**:
  1. 调用父类方法获取原始状态和潜在状态
  2. 设置视频条件（随机选择条件帧数）
  3. 返回 `(raw_state, latent_state, Video2WorldCondition)` 三元组

**条件生成过程**:
```python
condition = condition.set_video_condition(
    gt_frames=latent_state.to(**self.tensor_kwargs),
    random_min_num_conditional_frames=self.config.min_num_conditional_frames,
    random_max_num_conditional_frames=self.config.max_num_conditional_frames,
    num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
    conditional_frames_probs=self.config.conditional_frames_probs,
)
```

#### 4.2.2 采样噪声和sigma
```python
sigma_B_1, epsilon = self.draw_training_sigma_and_epsilon(latent_state.shape[0], condition)
```

**`draw_training_sigma_and_epsilon()` 详解**:
- **位置**: `video2world_model.py:89`
- **作用**:
  1. 从基础分布采样sigma（噪声水平）
  2. 如果是视频批次，根据高sigma策略调整sigma
  3. 生成高斯噪声epsilon
  4. 支持多种高sigma策略:
     - `UNIFORM80_2000`: 均匀分布 [80, 2000]
     - `LOGUNIFORM200_100000`: 对数均匀分布 [200, 100000]
     - `SHIFT24`: 时间shift策略
     - `BALANCED_TWO_HEADS_V1`: 平衡高低sigma
     - `HARDCODED_20steps`: 硬编码的20步sigma

#### 4.2.3 加噪
```python
xt_B_C_T_H_W = self.add_noise(latent_state, epsilon, sigma_B_1)
```
- **作用**: 使用SDE（随机微分方程）添加噪声到潜在状态

#### 4.2.4 去噪预测
```python
denoise_output: DenoisePrediction = self.denoise(xt_B_C_T_H_W, sigma_B_1, condition)
```

**`denoise()` 详解**（核心函数）:
- **位置**: `video2world_model.py:168`

**执行步骤**:

1. **准备输入**:
   ```python
   sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
   c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma_B_1_T_1_1)
   net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in
   ```
   - 计算预处理系数（基于Karras等人的扩散模型论文）

2. **处理条件帧**（视频模式）:
   ```python
   if condition.is_video:
       condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
       
       # 生成条件掩码
       condition_video_input_mask_B_1_T_1_1 = torch.zeros(...)
       condition_video_input_mask_B_1_T_1_1[:, :, :num_conditional_frames, :, :] = 1.0
       
       # 替换前N帧为条件帧
       if self.config.conditioning_strategy == "frame_replace":
           net_state_in_B_C_T_H_W[:, :, :num_conditional_frames, :, :] = condition_state_in_B_C_T_H_W[:, :, :num_conditional_frames, :, :]
   ```

3. **调用神经网络**（核心DiT网络）:
   ```python
   net_out_B_C_T_H_W = self.net(
       x_B_C_T_H_W=net_state_in_B_C_T_H_W,
       timesteps_B_T=c_noise.squeeze(),
       crossattn_emb=condition.crossattn_emb,
       condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
       fps=condition.fps,
       padding_mask=condition.padding_mask,
       data_type=condition.data_type,
       img_context_emb=condition.img_context_emb,
   )
   ```
   
   **`self.net` 详解**:
   - **类型**: `MinimalV1LVGDiT`（2B模型）
   - **位置**: `cosmos_predict2/_src/predict2/networks/minimal_v1_lvg_dit.py`
   - **架构**: Diffusion Transformer (DiT)
   - **关键参数**:
     - `model_channels`: 2048（通道数）
     - `num_heads`: 16（注意力头数）
     - `num_blocks`: 28（Transformer层数）
     - `patch_spatial`: 2（空间patch大小）
     - `patch_temporal`: 1（时间patch大小）
     - `pos_emb_cls`: "rope3d"（3D旋转位置编码）
     - `atten_backend`: "minimal_a2a"（all-to-all注意力后端）

4. **后处理输出**:
   ```python
   x0_pred_B_C_T_H_W = c_skip * xt_B_C_T_H_W + c_out * net_out_B_C_T_H_W
   eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1
   ```
   - 计算预测的干净数据 `x0_pred`
   - 计算预测的噪声 `eps_pred`

5. **返回预测结果**:
   ```python
   return DenoisePrediction(
       x0=x0_pred_B_C_T_H_W,
       eps=eps_pred_B_C_T_H_W,
   )
   ```

#### 4.2.5 计算损失
```python
loss, loss_dict_B = self.loss_func(
    data_batch=data_batch,
    raw_state=raw_state,
    latent_state=latent_state,
    noise_pred=denoise_output.eps,
    target=epsilon,
    sigma=sigma_B_1,
    condition=condition,
)
```

**损失函数详解**:
- **类型**: 通常是MSE损失（均方误差）
- **目标**: 最小化预测噪声和真实噪声的差异
- **可能包含额外损失项**:
  - 感知损失
  - 正则化损失
  - 条件一致性损失

#### 4.2.6 返回结果
```python
return output_batch, loss
```

---

### 4.3 神经网络：`MinimalV1LVGDiT`

**位置**: `cosmos_predict2/_src/predict2/networks/minimal_v1_lvg_dit.py`

**类层次结构**:
```
MinimalV1LVGDiT
  └── MiniTrainDIT
```

**前向传播**: `forward()`

**输入参数**:
- `x_B_C_T_H_W`: 噪声输入 [Batch, Channel, Time, Height, Width]
- `timesteps_B_T`: 时间步 [Batch, Time]
- `crossattn_emb`: 交叉注意力嵌入（文本、图像条件）
- `condition_video_input_mask_B_C_T_H_W`: 条件帧掩码
- `fps`: 帧率
- `padding_mask`: 填充掩码
- `data_type`: 数据类型（VIDEO或IMAGE）

**执行步骤**:

1. **拼接条件掩码**:
   ```python
   x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
   ```
   - 将条件掩码作为额外通道拼接到输入

2. **调用父类前向传播**:
   ```python
   return super().forward(
       x_B_C_T_H_W=x_B_C_T_H_W,
       timesteps_B_T=timesteps_B_T * self.timestep_scale,
       crossattn_emb=crossattn_emb,
       fps=fps,
       padding_mask=padding_mask,
       data_type=data_type,
   )
   ```

**`MiniTrainDIT` 网络结构**（简化说明）:
1. **Patch Embedding**: 将输入分割为时空patch
2. **位置编码**: 3D RoPE（旋转位置编码）
3. **Transformer Blocks** (28层):
   - Self-Attention (空间-时间注意力)
   - Cross-Attention (条件注意力)
   - Feed-Forward Network (前馈网络)
   - Layer Normalization
   - Adaptive Layer Normalization (AdaLN)
4. **输出投影**: 从隐藏维度投影回像素空间

---

## 五、数据加载

### 5.1 数据集配置

**实验配置**: `predict2_video2world_training_2b_cosmos_nemo_assets`
- **位置**: `cosmos_predict2/experiments/base/cosmos_nemo_assets.py`

**数据集定义**:
```python
example_video_dataset_cosmos_nemo_assets = L(VideoDataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)
```

**数据加载器定义**:
```python
dataloader_train_cosmos_nemo_assets = L(get_generic_dataloader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
```

---

### 5.2 `VideoDataset` 类

**位置**: `cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py`

**初始化**: `__init__()`
```python
def __init__(self, dataset_dir: str, num_frames: int, video_size: tuple[int, int]):
    self.dataset_dir = dataset_dir
    self.sequence_length = num_frames
    
    video_dir = os.path.join(self.dataset_dir, "videos")
    self.caption_dir = os.path.join(self.dataset_dir, "metas")
    
    self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    self.video_paths = sorted(self.video_paths)
    
    self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])
```

**作用**:
- 扫描数据集目录，收集所有视频文件路径
- 初始化预处理管道（转换为Tensor、调整大小）

---

**数据获取**: `__getitem__()`
```python
def __getitem__(self, index: int) -> dict:
    data = dict()
    
    # 1. 加载视频帧
    video, fps = self._get_frames(self.video_paths[index])
    video = video.permute(1, 0, 2, 3)  # [T,C,H,W] -> [C,T,H,W]
    
    # 2. 加载文本caption
    video_path = self.video_paths[index]
    caption_path = os.path.join(
        self.caption_dir,
        os.path.basename(video_path).replace(".mp4", ".txt"),
    )
    data["video"] = video
    data["ai_caption"] = self._load_text(Path(caption_path))
    
    # 3. 添加元数据
    _, _, h, w = video.shape
    data["fps"] = fps
    data["image_size"] = torch.tensor([h, w, h, w])
    data["num_frames"] = self.sequence_length
    data["padding_mask"] = torch.zeros(1, h, w)
    
    return data
```

**`_get_frames()` 详解**:
```python
def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
    frames, fps = self._load_video(video_path)
    frames = frames.astype(np.uint8)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = self.preprocess(frames)
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
    return frames, fps
```

**`_load_video()` 详解**:
```python
def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    
    # 随机采样序列起始位置
    max_start_idx = total_frames - self.sequence_length
    start_frame = np.random.randint(0, max_start_idx)
    end_frame = start_frame + self.sequence_length
    frame_ids = np.arange(start_frame, end_frame).tolist()
    
    # 读取帧
    frame_data = vr.get_batch(frame_ids).asnumpy()
    vr.seek(0)
    
    try:
        fps = vr.get_avg_fps()
    except Exception:
        fps = 16
    
    del vr
    return frame_data, fps
```

**作用**:
- 使用decord库读取视频
- 随机采样一个长度为93帧的序列
- 返回帧数据和帧率

---

### 5.3 数据预处理

**预处理管道**:
1. **`ToTensorVideo()`**: 将numpy数组转换为PyTorch Tensor
2. **`ResizePreprocess()`**: 调整视频尺寸到 (704, 1280)
3. **归一化**: 缩放到 [0, 255] 范围

---

### 5.4 数据加载器创建

**函数**: `get_generic_dataloader()`
- **位置**: `cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py:146`

```python
def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
```

**参数说明**:
- `batch_size=1`: 每批1个视频（由于视频数据量大）
- `num_workers=4`: 使用4个子进程加载数据
- `pin_memory=True`: 将数据固定在内存，加速GPU传输
- `drop_last=True`: 丢弃最后不完整的批次

---

## 六、优化器和调度器

### 6.1 优化器：FusedAdamW

**配置位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/optimizer.py`

**配置**:
```python
FusedAdamWConfig = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,  # 学习率
    weight_decay=0.1,  # 权重衰减
    betas=[0.9, 0.99],  # Adam beta参数
    optim_type="fusedadam",  # 优化器类型
    eps=1e-8,  # 数值稳定性参数
    master_weights=True,  # 使用master权重（混合精度训练）
    capturable=True,  # 支持CUDA图捕获
)
```

**实验覆盖**（cosmos_nemo_assets）:
```python
optimizer=dict(
    lr=2 ** (-14.5),  # 约 5.66e-5
    weight_decay=0.001,
)
```

**实例化函数**: `get_base_optimizer()`
- **位置**: `cosmos_predict2/_src/predict2/utils/optim_instantiate.py`
- **作用**: 根据配置创建优化器实例

---

### 6.2 学习率调度器：LambdaLinear

**配置位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/scheduler.py`

**配置**（推测基于config模式）:
```python
scheduler=dict(
    f_max=[0.5],  # 最大学习率因子
    f_min=[0.2],  # 最小学习率因子
    warm_up_steps=[2_000],  # 预热步数
    cycle_lengths=[100000],  # 周期长度
)
```

**调度策略**:
1. **预热阶段** (0-2000步): 学习率从0线性增加到 `lr * f_max`
2. **主训练阶段** (2000-100000步): 学习率从 `lr * f_max` 线性衰减到 `lr * f_min`
3. **稳定阶段** (100000步之后): 保持 `lr * f_min`

---

## 七、回调函数系统

### 7.1 回调配置

**基础回调**: `BASIC_CALLBACKS`
- **位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/callbacks.py`

**视频2世界回调**: 
- **位置**: `cosmos_predict2/_src/predict2/configs/video2world/defaults/callbacks.py`

**实验配置回调**（cosmos_nemo_assets）:
```python
trainer=dict(
    logging_iter=100,
    max_iter=1000,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=200, save_s3=False),
        every_n_sample_ema=dict(every_n=200, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
    ),
)
```

---

### 7.2 关键回调函数

#### 7.2.1 `EveryNDrawSample`
- **位置**: `cosmos_predict2/_src/predict2/callbacks/every_n_draw_sample.py`
- **作用**: 每N个迭代生成样本图像/视频
- **触发**: `on_training_step_end`
- **功能**:
  - 使用当前模型生成样本
  - 保存到本地和/或S3
  - 支持常规权重和EMA权重两种模式

#### 7.2.2 WandB回调
- **作用**: 记录训练指标到Weights & Biases
- **记录内容**:
  - 损失曲线
  - 学习率变化
  - 生成样本
  - 系统指标

#### 7.2.3 设备监控回调
- **作用**: 监控GPU使用情况
- **监控内容**:
  - GPU内存使用
  - GPU利用率
  - 温度

#### 7.2.4 速度回调
- **作用**: 监控训练速度
- **监控内容**:
  - 每秒迭代数
  - 每秒样本数
  - 数据加载时间

---

### 7.3 回调触发点

完整的回调钩子调用顺序：

```
训练开始:
  on_optimizer_init_start()
  on_optimizer_init_end()
  on_train_start()

每个迭代:
  on_before_dataloading()
  on_after_dataloading()
  on_training_step_start()
  on_training_step_batch_start()
    on_before_forward()
    on_after_forward()
    on_before_backward()
    on_after_backward()
    on_before_optimizer_step()
    on_before_zero_grad()
  on_training_step_batch_end()
  on_training_step_end()

检查点保存:
  on_save_checkpoint_start()
  on_save_checkpoint()
  on_save_checkpoint_end()

验证阶段:
  on_validation_start()
    on_validation_step_start()
    on_validation_step_end()
  on_validation_end()

训练结束:
  on_train_end()
  on_app_end()
```

---

## 八、配置系统

### 8.1 配置加载流程

**Hydra配置系统**:
1. **基础配置**: `config.py` 中的 `make_config()`
2. **默认组**: `defaults` 列表中定义的配置组
3. **实验配置**: 通过 `experiment=xxx` 指定
4. **命令行覆盖**: 通过 `opts` 参数覆盖

---

### 8.2 配置组注册

**`make_config()` 函数**:
- **位置**: `cosmos_predict2/_src/predict2/configs/video2world/config.py:62`

**执行步骤**:
```python
def make_config() -> Config:
    c = Config(model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None)
    
    # 设置基础配置
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    
    c.trainer.type = Trainer
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    
    # 注册配置组
    register_training_and_val_data()  # 数据配置
    register_optimizer()  # 优化器配置
    register_scheduler()  # 调度器配置
    register_model()  # 模型配置
    register_callbacks()  # 回调配置
    register_net()  # 网络配置
    register_conditioner()  # 条件器配置
    register_ema()  # EMA配置
    register_tokenizer()  # Tokenizer配置
    register_checkpoint()  # 检查点配置
    register_ckpt_type()  # 检查点类型配置
    
    # 导入实验配置
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    
    return c
```

---

### 8.3 实验配置：`predict2_video2world_training_2b_cosmos_nemo_assets`

**位置**: `cosmos_predict2/experiments/base/cosmos_nemo_assets.py:48`

**完整配置**:
```python
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",  # 继承基础实验配置
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_nemo_assets",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,  # 自定义数据加载器
    checkpoint=dict(
        save_iter=200,  # 每200步保存一次
        load_path='/gemini/platform/public/embodiedAI/users/fanchenyou/models/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt',
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),  # 5.66e-5
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[2_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,  # 每100步记录日志
        max_iter=1000,  # 最大迭代1000步
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=200, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=200, save_s3=False),
            every_n_sample_ema=dict(every_n=200, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,  # 上下文并行大小
    ),
)
```

---

## 九、关键技术细节

### 9.1 扩散模型训练

**训练目标**:
最小化预测噪声和真实噪声的差异：
```
L = E[||ε_pred - ε||^2]
```

**训练流程**:
1. 从数据集采样干净数据 x0
2. 随机采样噪声水平 σ
3. 生成噪声 ε ~ N(0, I)
4. 加噪: x_σ = x0 + σ * ε
5. 模型预测噪声: ε_pred = model(x_σ, σ, condition)
6. 计算损失: L = ||ε_pred - ε||^2
7. 反向传播并更新参数

---

### 9.2 条件视频生成

**条件策略**: Frame Replace
- 使用前1-2帧作为条件
- 这些条件帧被添加低噪声（σ=0.0001）
- 在网络输入中直接替换相应位置
- 使用掩码向网络指示哪些帧是条件帧

**条件信息**:
- **视频帧条件**: 前N帧的latent表示
- **文本条件**: 通过cross-attention嵌入注入
- **时间条件**: 通过FPS和timestep编码
- **空间条件**: 通过padding mask处理不规则分辨率

---

### 9.3 FSDP（Fully Sharded Data Parallel）

**作用**: 在多GPU间分片模型参数、梯度和优化器状态

**配置**:
```python
config.trainer.distributed_parallelism = "fsdp"
config.model.config.fsdp_shard_size = 8
```

**工作原理**:
1. 模型参数按层分片到不同GPU
2. 前向传播时：
   - 通过all-gather收集当前层的完整参数
   - 计算完成后释放其他分片
3. 反向传播时：
   - 重新收集参数
   - 计算梯度
   - reduce-scatter同步梯度
   - 释放参数

**优势**:
- 大幅减少每个GPU的内存占用
- 支持训练超大模型（2B、7B、14B参数）

---

### 9.4 混合精度训练

**使用**: `torch.amp.GradScaler`

**流程**:
1. 前向传播使用FP16/BF16
2. 损失使用FP32累积
3. 梯度缩放防止下溢
4. 优化器步骤使用master weights（FP32）

---

### 9.5 EMA（指数移动平均）

**配置**: `config.ema.enabled = True`

**作用**:
- 维护模型权重的指数移动平均
- 验证和推理时使用EMA权重
- 通常能获得更好的生成质量和稳定性

**更新公式**:
```
θ_ema = decay * θ_ema + (1 - decay) * θ
```

---

### 9.6 3D RoPE（旋转位置编码）

**配置**: `pos_emb_cls="rope3d"`

**作用**:
- 为时空patch提供位置信息
- 支持外推到更长序列和更高分辨率
- 时间维度外推比率：1.0（2B模型）

---

## 十、完整函数调用链总结

```
run_posttrain_nemo.sh
  └── torchrun scripts/train.py
        └── __main__
              ├── get_config_module()
              ├── importlib.import_module().make_config()
              │     ├── Config.__init__()
              │     ├── register_training_and_val_data()
              │     ├── register_optimizer()
              │     ├── register_scheduler()
              │     ├── register_model()
              │     ├── register_callbacks()
              │     ├── register_net()
              │     ├── register_conditioner()
              │     ├── register_ema()
              │     ├── register_tokenizer()
              │     ├── register_checkpoint()
              │     ├── register_ckpt_type()
              │     └── import_all_modules_from_package()  # 加载实验配置
              ├── override()  # 应用实验配置覆盖
              └── launch()
                    ├── distributed.init()
                    ├── config.validate()
                    ├── config.freeze()
                    ├── ImaginaireTrainer.__init__()
                    │     ├── distributed.init()
                    │     ├── parallel_state.initialize_model_parallel()
                    │     ├── LazyConfig.save_pkl()
                    │     ├── LazyConfig.save_yaml()
                    │     ├── log.init_loguru_file()
                    │     ├── misc.set_random_seed()
                    │     ├── callback.CallBackGroup.__init__()
                    │     ├── Checkpointer.__init__()
                    │     ├── misc.TrainingTimer.__init__()
                    │     └── StragglerDetectorV2.__init__()
                    ├── create_model_from_consolidated_checkpoint_with_fsdp()  或 instantiate(config.model)
                    │     └── Video2WorldModel.__init__()
                    │           ├── Text2WorldModel.__init__()
                    │           │     ├── ImaginaireModel.__init__()
                    │           │     ├── instantiate(config.net)
                    │           │     │     └── MinimalV1LVGDiT.__init__()
                    │           │     ├── instantiate(config.conditioner)
                    │           │     ├── instantiate(config.tokenizer)
                    │           │     └── wrap_with_fsdp()
                    │           └── ...
                    ├── instantiate(config.dataloader_train)
                    │     ├── VideoDataset.__init__()
                    │     └── get_generic_dataloader()
                    ├── instantiate(config.dataloader_val)
                    └── ImaginaireTrainer.train()
                          ├── model.to("cuda")
                          ├── model.on_train_start()
                          ├── callbacks.on_optimizer_init_start()
                          ├── model.init_optimizer_scheduler()
                          │     ├── instantiate(optimizer_config)
                          │     │     └── get_base_optimizer()  # FusedAdamW
                          │     └── instantiate(scheduler_config)  # LambdaLinear
                          ├── torch.amp.GradScaler.__init__()
                          ├── callbacks.on_optimizer_init_end()
                          ├── checkpointer.load()
                          │     ├── object_store.load()  或 torch.load()
                          │     ├── model.load_state_dict()
                          │     ├── optimizer.load_state_dict()
                          │     ├── scheduler.load_state_dict()
                          │     └── grad_scaler.load_state_dict()
                          ├── distributed.parallel_model_wrapper()  # DDP包装
                          ├── callbacks.on_train_start()
                          └── [主训练循环]
                                ├── iter(dataloader_train)
                                └── for each batch:
                                      ├── callbacks.on_before_dataloading()
                                      ├── next(dataloader_train_iter)
                                      │     └── VideoDataset.__getitem__()
                                      │           ├── _load_video()
                                      │           │     └── VideoReader.get_batch()
                                      │           ├── _load_text()
                                      │           └── preprocess()
                                      ├── callbacks.on_after_dataloading()
                                      ├── misc.to(data_batch, device="cuda")
                                      ├── callbacks.on_training_step_start()
                                      ├── callbacks.on_training_step_batch_start()
                                      ├── ImaginaireTrainer.training_step()
                                      │     ├── callbacks.on_before_forward()
                                      │     ├── model_ddp.training_step()
                                      │     │     ├── Video2WorldModel.get_data_and_condition()
                                      │     │     │     ├── tokenizer.encode()
                                      │     │     │     ├── conditioner()
                                      │     │     │     └── condition.set_video_condition()
                                      │     │     ├── Video2WorldModel.draw_training_sigma_and_epsilon()
                                      │     │     │     └── 高sigma策略采样
                                      │     │     ├── Video2WorldModel.add_noise()
                                      │     │     ├── Video2WorldModel.denoise()
                                      │     │     │     ├── scaling()  # 计算预处理系数
                                      │     │     │     ├── 构造条件输入和掩码
                                      │     │     │     ├── MinimalV1LVGDiT.forward()
                                      │     │     │     │     ├── torch.cat()  # 拼接条件掩码
                                      │     │     │     │     └── MiniTrainDIT.forward()
                                      │     │     │     │           ├── patchify()
                                      │     │     │     │           ├── pos_emb()  # 3D RoPE
                                      │     │     │     │           ├── [28 x DiT Block]
                                      │     │     │     │           │     ├── self_attention()
                                      │     │     │     │           │     ├── cross_attention()
                                      │     │     │     │           │     ├── feedforward()
                                      │     │     │     │           │     └── adaln()
                                      │     │     │     │           └── unpatchify()
                                      │     │     │     ├── 计算x0_pred和eps_pred
                                      │     │     │     └── return DenoisePrediction()
                                      │     │     └── loss_func()
                                      │     │           └── MSE(eps_pred, epsilon)
                                      │     ├── callbacks.on_after_forward()
                                      │     ├── callbacks.on_before_backward()
                                      │     ├── grad_scaler.scale(loss).backward()
                                      │     ├── model_ddp.on_after_backward()
                                      │     ├── callbacks.on_after_backward()
                                      │     └── [如果梯度累积完成]
                                      │           ├── callbacks.on_before_optimizer_step()
                                      │           ├── grad_scaler.step(optimizer)
                                      │           ├── grad_scaler.update()
                                      │           ├── scheduler.step()
                                      │           ├── callbacks.on_before_zero_grad()
                                      │           ├── model_ddp.on_before_zero_grad()
                                      │           └── optimizer.zero_grad()
                                      ├── callbacks.on_training_step_batch_end()
                                      ├── callbacks.on_training_step_end()
                                      ├── [每save_iter次迭代]
                                      │     └── checkpointer.save()
                                      │           ├── callbacks.on_save_checkpoint_start()
                                      │           ├── model.state_dict()
                                      │           ├── optimizer.state_dict()
                                      │           ├── scheduler.state_dict()
                                      │           ├── grad_scaler.state_dict()
                                      │           ├── callbacks.on_save_checkpoint()
                                      │           └── torch.save()  或 object_store.save()
                                      └── [每validation_iter次迭代]
                                            └── ImaginaireTrainer.validate()
                                                  ├── callbacks.on_validation_start()
                                                  ├── model.eval()
                                                  ├── ema.ema_scope()
                                                  └── for each val_batch:
                                                        ├── callbacks.on_validation_step_start()
                                                        ├── model.validation_step()
                                                        └── callbacks.on_validation_step_end()
                                                  └── callbacks.on_validation_end()
```

---

## 十一、配置参数总结

### 11.1 实验配置（cosmos_nemo_assets）

| 参数 | 值 | 说明 |
|------|-----|------|
| **项目设置** | | |
| project | cosmos_predict_v2p5 | WandB项目名 |
| group | video2world | 实验组 |
| name | 2b_cosmos_nemo_assets | 实验名称 |
| **训练设置** | | |
| max_iter | 1000 | 最大训练步数 |
| logging_iter | 100 | 日志记录间隔 |
| validation_iter | 100 | 验证间隔 |
| save_iter | 200 | 检查点保存间隔 |
| **数据设置** | | |
| dataset_dir | datasets/cosmos_nemo_assets | 数据集路径 |
| num_frames | 93 | 每个序列帧数 |
| video_size | (704, 1280) | 视频分辨率 |
| batch_size | 1 | 批次大小 |
| num_workers | 4 | 数据加载进程数 |
| **优化器设置** | | |
| optimizer_type | FusedAdamW | 优化器类型 |
| lr | 5.66e-5 | 学习率 |
| weight_decay | 0.001 | 权重衰减 |
| betas | [0.9, 0.99] | Adam beta参数 |
| **调度器设置** | | |
| scheduler_type | LambdaLinear | 调度器类型 |
| warm_up_steps | 2000 | 预热步数 |
| f_max | 0.5 | 最大学习率因子 |
| f_min | 0.2 | 最小学习率因子 |
| cycle_lengths | 100000 | 周期长度 |
| **模型设置** | | |
| model_type | Video2WorldModel | 模型类型 |
| net_type | MinimalV1LVGDiT | 网络架构 |
| model_channels | 2048 | 模型通道数 |
| num_heads | 16 | 注意力头数 |
| num_blocks | 28 | Transformer层数 |
| **条件设置** | | |
| min_num_conditional_frames | 1 | 最小条件帧数 |
| max_num_conditional_frames | 2 | 最大条件帧数 |
| sigma_conditional | 0.0001 | 条件帧噪声水平 |
| conditioning_strategy | frame_replace | 条件策略 |
| **并行设置** | | |
| distributed_parallelism | ddp | 并行模式（DDP/FSDP） |
| context_parallel_size | 1 | 上下文并行大小 |
| **检查点设置** | | |
| load_path | .../81edfebe...ema_bf16.pt | 预训练模型路径 |
| load_from_object_store | False | 从对象存储加载 |
| save_to_object_store | False | 保存到对象存储 |

---

## 十二、总结

运行 `run_posttrain_nemo.sh` 后，整个训练流程包含以下主要阶段：

1. **初始化阶段**:
   - 加载和验证配置
   - 初始化分布式环境
   - 创建训练器、模型、数据加载器

2. **准备阶段**:
   - 加载预训练检查点
   - 初始化优化器和调度器
   - 设置回调函数和监控

3. **训练循环**:
   - 加载数据批次
   - 前向传播（扩散模型去噪）
   - 计算损失
   - 反向传播和优化
   - 定期保存检查点和验证

4. **核心计算**:
   - 视频编码到latent空间
   - 添加噪声
   - 使用DiT网络预测噪声
   - 最小化预测噪声和真实噪声的差异

整个系统设计模块化、可扩展，支持大规模分布式训练和灵活的配置管理。

---

**文档生成时间**: 2025-10-11  
**项目**: Cosmos Predict2.5  
**版本**: 2B模型训练流程分析

