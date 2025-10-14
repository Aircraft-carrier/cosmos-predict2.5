# `make_config()` 函数深度分析文档

本文档详细分析 Cosmos Predict2.5 项目中的核心配置函数 `make_config()`，包括它的调用流程、每个子函数的作用、以及整个配置系统的工作原理。

---

## 目录

1. [函数概述](#1-函数概述)
2. [函数位置与签名](#2-函数位置与签名)
3. [执行流程详解](#3-执行流程详解)
4. [配置类体系](#4-配置类体系)
5. [注册函数详解](#5-注册函数详解)
6. [Hydra配置系统](#6-hydra配置系统)
7. [配置覆盖机制](#7-配置覆盖机制)
8. [完整调用链](#8-完整调用链)
9. [使用示例](#9-使用示例)
10. [总结](#10-总结)

---

## 1. 函数概述

### 1.1 什么是 `make_config()`？

`make_config()` 是 Cosmos Predict2.5 项目中负责创建和初始化训练配置的核心函数。它的主要职责是：

1. **创建配置对象**：实例化一个 `Config` 类对象，包含所有训练所需的配置参数
2. **设置默认值**：为作业、训练器等设置默认配置
3. **注册配置组**：将各个模块的配置选项注册到 Hydra 配置系统
4. **导入实验配置**：加载实验级别的配置模块

### 1.2 为什么需要 `make_config()`？

在深度学习训练中，需要管理大量配置参数（模型架构、优化器、数据加载器、检查点等）。`make_config()` 提供了一个统一的、模块化的配置管理系统：

- ✅ **模块化**：不同模块的配置分离，易于维护
- ✅ **可扩展**：通过注册机制轻松添加新配置
- ✅ **灵活性**：支持命令行动态覆盖任意配置
- ✅ **类型安全**：使用 attrs 定义的配置类提供类型检查
- ✅ **可重用**：配置可以继承和组合

---

## 2. 函数位置与签名

### 2.1 位置

**文件路径**: `cosmos_predict2/_src/predict2/configs/video2world/config.py`

**行号**: 62-102

### 2.2 函数签名

```python
def make_config() -> Config:
```

**返回值**: `Config` 类型的配置对象

### 2.3 在哪里被调用？

在 `scripts/train.py` 中被调用：

```python
# scripts/train.py, line 93
config_module = get_config_module(args.config)
config = importlib.import_module(config_module).make_config()
```

**调用流程**:
1. `get_config_module()` 将配置文件路径转换为 Python 模块路径
2. `importlib.import_module()` 动态导入配置模块
3. 调用模块的 `make_config()` 函数获取配置对象

---

## 3. 执行流程详解

### 3.1 完整源代码

```python
def make_config() -> Config:
    # 第1步：创建 Config 实例
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # 第2步：设置作业和训练器的默认值
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # 第3步：注册各模块配置组
    register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_net()
    register_conditioner()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    # 第4步：导入实验配置
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)

    return c
```

---

### 3.2 第1步：创建 Config 实例

```python
c = Config(
    model=None,
    optimizer=None,
    scheduler=None,
    dataloader_train=None,
    dataloader_val=None,
)
```

#### 作用

创建一个 `Config` 类的实例，这是整个配置系统的根对象。

#### Config 类定义

**位置**: `cosmos_predict2/_src/predict2/configs/video2world/config.py:36`

```python
@attrs.define(slots=False)
class Config(config.Config):
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
            {"experiment": None},
        ]
    )
```

#### defaults 字段的含义

`defaults` 是 Hydra 配置系统的核心概念，定义了配置组的默认值：

| 配置组 | 默认值 | 说明 |
|--------|--------|------|
| `_self_` | - | 表示当前配置文件本身 |
| `data_train` | `mock` | 训练数据配置 |
| `data_val` | `mock` | 验证数据配置 |
| `optimizer` | `fusedadamw` | 优化器类型 |
| `scheduler` | `lambdalinear` | 学习率调度器 |
| `model` | `ddp` | 模型并行策略 |
| `callbacks` | `basic` | 回调函数组 |
| `net` | `None` | 神经网络架构 |
| `conditioner` | `video_prediction_conditioner` | 条件生成器 |
| `ema` | `power` | EMA配置 |
| `tokenizer` | `cosmos_tokenizer_...` | Tokenizer配置 |
| `checkpoint` | `s3` | 检查点保存/加载 |
| `ckpt_type` | `dummy` | 检查点类型 |
| `experiment` | `None` | 实验级配置 |

#### 初始化的参数为什么是 None？

这5个参数（`model`, `optimizer`, `scheduler`, `dataloader_train`, `dataloader_val`）初始化为 `None` 是因为：

1. **延迟绑定**：这些配置的具体值将在后续通过 Hydra 的配置组机制设置
2. **类型占位**：告诉系统这些字段存在，但具体实例稍后创建
3. **灵活覆盖**：允许通过命令行或实验配置动态指定

---

### 3.3 第2步：设置默认值

```python
# 作业配置
c.job.project = "cosmos_diffusion_v2"
c.job.group = "debug"
c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

# 训练器配置
c.trainer.type = Trainer
c.trainer.straggler_detection.enabled = False
c.trainer.max_iter = 400_000
c.trainer.logging_iter = 10
c.trainer.validation_iter = 100
c.trainer.run_validation = False
c.trainer.callbacks = None
```

#### 3.3.1 作业配置 (c.job)

**`c.job` 类型**: `JobConfig`

**位置**: `cosmos_predict2/_src/imaginaire/config.py:180`

```python
@make_freezable
@attrs.define(slots=False)
class JobConfig:
    project: str = ""      # 项目名
    group: str = ""        # 实验组名
    name: str = ""         # 运行名称
    wandb_mode: str = "offline"  # W&B模式
    
    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"
    
    @property
    def path_local(self) -> str:
        local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        return f"{local_root}/{self.path}"
```

**设置的值**:

1. **`project = "cosmos_diffusion_v2"`**: 
   - W&B 项目名
   - 用于组织多个相关实验

2. **`group = "debug"`**: 
   - 实验组名
   - 可以将多个实验归为一组
   - 默认为 "debug"，通常在实验配置中覆盖

3. **`name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"`**: 
   - 运行的唯一标识符
   - 使用时间戳自动生成
   - `${now:...}` 是 Hydra 的变量插值语法
   - 示例: `delete_2025-10-11_14-30-45`

**路径属性**:
- `c.job.path`: 返回 `"cosmos_diffusion_v2/debug/delete_2025-10-11_14-30-45"`
- `c.job.path_local`: 返回完整的本地路径，例如 `/gemini/.../logs/cosmos_diffusion_v2/debug/delete_2025-10-11_14-30-45`

#### 3.3.2 训练器配置 (c.trainer)

**`c.trainer` 类型**: `TrainerConfig`

**位置**: `cosmos_predict2/_src/imaginaire/config.py:347`

```python
@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    type: Type[ImaginaireTrainer] = ImaginaireTrainer
    callbacks: LazyDict = LazyDict(...)
    distributed_parallelism: str = "ddp"
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    seed: int = 0
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    max_iter: int = 999999999
    max_val_iter: int | None = None
    logging_iter: int = 100
    run_validation: bool = True
    validation_iter: int = 999999999
    timeout_period: int = 999999999
    memory_format: torch.memory_format = torch.preserve_format
    grad_accum_iter: int = 1
    straggler_detection: StragglerDetectionConfig = attrs.field(factory=StragglerDetectionConfig)
    profiling: Profiling = attrs.field(factory=Profiling)
```

**设置的值**:

1. **`type = Trainer`**: 
   - 训练器类，指定使用 `ImaginaireTrainer`
   - 类型: `Type[ImaginaireTrainer]`

2. **`straggler_detection.enabled = False`**: 
   - 禁用掉队检测（用于多GPU训练中检测慢节点）
   - 单GPU训练不需要

3. **`max_iter = 400_000`**: 
   - 最大训练迭代次数
   - 40万次迭代

4. **`logging_iter = 10`**: 
   - 每10次迭代记录一次日志
   - 包括损失、学习率等指标

5. **`validation_iter = 100`**: 
   - 每100次迭代运行一次验证

6. **`run_validation = False`**: 
   - 禁用验证
   - 训练时不运行验证集评估

7. **`callbacks = None`**: 
   - 回调函数设为 None
   - 将在后续通过配置组设置

---

### 3.4 第3步：注册配置组

这是 `make_config()` 的核心步骤，调用一系列 `register_*()` 函数将不同模块的配置注册到 Hydra ConfigStore。

```python
register_training_and_val_data()
register_optimizer()
register_scheduler()
register_model()
register_callbacks()
register_net()
register_conditioner()
register_ema()
register_tokenizer()
register_checkpoint()
register_ckpt_type()
```

#### 为什么需要注册？

Hydra 配置系统需要预先知道所有可用的配置选项。注册过程就是告诉 Hydra：
- "我有一个名为 `optimizer` 的配置组"
- "这个组有两个选项：`fusedadamw` 和 `adamw`"
- "当用户选择 `optimizer=fusedadamw` 时，使用这个配置对象"

#### 注册函数的通用模式

所有 `register_*()` 函数都遵循相同的模式：

```python
def register_xxx():
    cs = ConfigStore.instance()  # 获取 Hydra ConfigStore 单例
    cs.store(
        group="xxx",           # 配置组名
        package="path.to.field",  # 配置在 Config 对象中的路径
        name="option_name",    # 选项名称
        node=ConfigObject,     # 配置对象/值
    )
```

详见[第5章：注册函数详解](#5-注册函数详解)

---

### 3.5 第4步：导入实验配置

```python
import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
```

#### 作用

递归导入指定包下的所有 Python 模块，确保实验级配置被注册到 Hydra。

#### `import_all_modules_from_package()` 详解

**位置**: `cosmos_predict2/_src/imaginaire/utils/config_helper.py:170`

**函数签名**:
```python
def import_all_modules_from_package(
    package_path: str, 
    reload: bool = False, 
    skip_underscore: bool = True
) -> None:
```

**参数**:
- `package_path`: 包的点分路径，如 `"cosmos_predict2.experiments"`
- `reload`: 是否重新加载已导入的模块
- `skip_underscore`: 是否跳过下划线开头的模块（私有模块）

**实现逻辑**:

```python
def import_all_modules_from_package(package_path: str, reload: bool = False, skip_underscore: bool = True) -> None:
    log.critical(f"{'Reloading' if reload else 'Importing'} all modules from package {package_path}")
    package = importlib.import_module(package_path)
    package_directory = package.__path__
    
    def import_modules_recursively(directory: str, prefix: str) -> None:
        for _, module_name, is_pkg in pkgutil.iter_modules([directory]):
            if skip_underscore and module_name.startswith("_"):
                continue
            
            full_module_name = f"{prefix}.{module_name}"
            log.debug(f"{'Reloading' if reload else 'Importing'} module {full_module_name}")
            
            import_module(full_module_name, reload=reload)
            
            if is_pkg:
                sub_package_directory = os.path.join(directory, module_name)
                import_modules_recursively(sub_package_directory, full_module_name)
    
    for directory in package_directory:
        import_modules_recursively(directory, package_path)
```

**执行过程**:

1. **导入根包**: 使用 `importlib.import_module()` 导入根包
2. **获取包目录**: 从 `package.__path__` 获取文件系统路径
3. **递归遍历**: 
   - 使用 `pkgutil.iter_modules()` 遍历所有子模块
   - 对每个模块调用 `import_module()`
   - 如果是包（子目录），递归导入
4. **跳过私有模块**: 如果 `skip_underscore=True`，跳过 `_xxx.py` 文件

**导入的目录**:

1. **`cosmos_predict2._src.predict2.configs.video2world.experiment`**:
   - 视频生成的实验配置
   - 通常包含不同规模模型的实验设置

2. **`cosmos_predict2.experiments`**:
   - 项目级别的实验配置
   - 包含 `cosmos_nemo_assets.py` 等实验定义

**实验配置的注册**:

在实验模块中（如 `cosmos_nemo_assets.py`），配置通过以下方式注册：

```python
# cosmos_predict2/experiments/base/cosmos_nemo_assets.py

predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    defaults=[...],
    job=dict(...),
    dataloader_train=...,
    checkpoint=dict(...),
    optimizer=dict(...),
    # ...
)

cs = ConfigStore.instance()
for _item in [predict2_video2world_training_2b_cosmos_nemo_assets]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
```

**注册机制**:
- `group="experiment"`: 配置组为 `experiment`
- `package="_global_"`: 配置会合并到全局（Config对象的根级别）
- `name=experiment_name`: 选项名为变量名（小写）
- `node=_item`: 配置字典

---

### 3.6 返回配置对象

```python
return c
```

返回完全初始化和配置的 `Config` 对象，供训练脚本使用。

---

## 4. 配置类体系

### 4.1 配置类继承关系

```
Config (video2world/config.py)
  └── config.Config (imaginaire/config.py)
```

### 4.2 基础 Config 类

**位置**: `cosmos_predict2/_src/imaginaire/config.py:395`

```python
@make_freezable
@attrs.define(slots=False)
class Config:
    # 核心配置字段
    model: LazyDict
    optimizer: LazyDict
    scheduler: LazyDict
    dataloader_train: LazyDict
    dataloader_val: LazyDict
    
    # 子配置对象
    job: JobConfig = attrs.field(factory=JobConfig)
    trainer: TrainerConfig = attrs.field(factory=TrainerConfig)
    model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)
    
    # 其他
    upload_reproducible_setup: bool = False
    
    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)
    
    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)
    
    def validate(self) -> None:
        # 验证配置的有效性
        assert self.job.project != ""
        assert self.job.group != ""
        assert self.job.name != ""
```

### 4.3 attrs 装饰器

```python
@make_freezable
@attrs.define(slots=False)
```

#### `@attrs.define(slots=False)`

- **attrs** 是一个 Python 库，用于自动生成类的样板代码
- 自动生成 `__init__()`, `__repr__()`, `__eq__()` 等方法
- `slots=False`: 允许动态添加属性（`make_freezable` 需要）

#### `@make_freezable`

自定义装饰器，添加冻结功能：

**位置**: `cosmos_predict2/_src/imaginaire/config.py:57`

```python
def make_freezable(cls: T) -> T:
    original_setattr = cls.__setattr__
    
    def setattr_override(self, key, value) -> None:
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)
    
    cls.__setattr__ = setattr_override
    
    def freeze(self: object) -> None:
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True
    
    cls.freeze = freeze
    return cls
```

**作用**:
- 添加 `freeze()` 方法
- 冻结后不能修改任何属性
- 递归冻结所有嵌套的 attrs 对象
- 防止训练过程中意外修改配置

**使用**:
```python
config.validate()
config.freeze()  # 冻结配置
# config.job.name = "new_name"  # 会抛出 AttributeError
```

### 4.4 子配置类

#### 4.4.1 JobConfig

**位置**: `cosmos_predict2/_src/imaginaire/config.py:180`

```python
@make_freezable
@attrs.define(slots=False)
class JobConfig:
    project: str = ""
    group: str = ""
    name: str = ""
    wandb_mode: str = "offline"
    
    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"
    
    @property
    def path_local(self) -> str:
        local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        return f"{local_root}/{self.path}"
```

**字段说明**:
- `project`: W&B 项目名
- `group`: 实验组名
- `name`: 运行名称（唯一标识）
- `wandb_mode`: W&B 模式（`online`, `offline`, `disabled`）
- `path`: 相对路径（组合 project/group/name）
- `path_local`: 本地文件系统的绝对路径

#### 4.4.2 TrainerConfig

**位置**: `cosmos_predict2/_src/imaginaire/config.py:347`

```python
@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    type: Type[ImaginaireTrainer] = ImaginaireTrainer
    callbacks: LazyDict = LazyDict(...)
    distributed_parallelism: str = "ddp"
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    seed: int = 0
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    max_iter: int = 999999999
    max_val_iter: int | None = None
    logging_iter: int = 100
    run_validation: bool = True
    validation_iter: int = 999999999
    timeout_period: int = 999999999
    memory_format: torch.memory_format = torch.preserve_format
    grad_accum_iter: int = 1
    straggler_detection: StragglerDetectionConfig = ...
    profiling: Profiling = ...
```

**字段说明**:
- `type`: 训练器类
- `distributed_parallelism`: 并行策略（`ddp` 或 `fsdp`）
- `max_iter`: 最大迭代次数
- `logging_iter`: 日志记录间隔
- `validation_iter`: 验证间隔
- `grad_accum_iter`: 梯度累积步数

#### 4.4.3 CheckpointConfig

**位置**: `cosmos_predict2/_src/imaginaire/config.py:260`

```python
@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    type: Optional[Dict] = None
    dcp_async_mode_enabled: bool = False
    save_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    save_iter: int = 999999999
    load_from_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    load_path: str = ""
    load_training_state: bool = False
    only_load_scheduler_state: bool = False
    strict_resume: bool = True
    jit: JITConfig = attrs.field(factory=JITConfig)
    verbose: bool = True
    keys_not_to_resume: list[str] = []
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False
    dcp_allow_mismatched_size: bool = False
```

**字段说明**:
- `save_iter`: 保存检查点的间隔
- `load_path`: 加载检查点的路径
- `load_training_state`: 是否加载优化器状态
- `strict_resume`: 是否严格匹配权重名称
- `save_to_object_store`: 保存到对象存储的配置
- `load_from_object_store`: 从对象存储加载的配置

### 4.5 LazyDict

**LazyDict** 是一种延迟实例化的配置字典，不会立即创建对象，而是保存创建对象所需的信息。

**位置**: `cosmos_predict2/_src/imaginaire/lazy_config`

**使用示例**:

```python
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L

# 定义一个 LazyDict
optimizer_config = L(get_base_optimizer)(
    lr=1e-4,
    weight_decay=0.1,
    optim_type="adamw",
)

# 此时 optimizer_config 不是一个优化器对象，而是一个配置
# 实际创建对象时：
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
optimizer = instantiate(optimizer_config)
```

**优点**:
- 延迟创建：配置阶段不创建对象，节省内存
- 序列化：LazyDict 可以保存和加载
- 灵活性：可以在运行时修改参数再实例化

---

## 5. 注册函数详解

每个 `register_*()` 函数负责将特定模块的配置选项注册到 Hydra ConfigStore。

### 5.1 register_training_and_val_data()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/dataloader.py:128`

**源代码**:
```python
def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
```

**注册的配置**:

| 配置组 | 选项名 | 配置对象 | 说明 |
|--------|--------|----------|------|
| `data_train` | `mock` | `MOCK_DATA_INTERLEAVE_CONFIG` | 混合图像和视频的模拟数据 |
| `data_train` | `mock_image` | `MOCK_DATA_IMAGE_ONLY_CONFIG` | 仅图像的模拟数据 |
| `data_train` | `mock_video` | `MOCK_DATA_VIDEO_ONLY_CONFIG` | 仅视频的模拟数据 |
| `data_val` | `mock` | `MOCK_DATA_INTERLEAVE_CONFIG` | 验证集模拟数据 |

**使用方式**:
```bash
# 命令行选择数据配置
python train.py ... -- data_train=mock_video
```

---

### 5.2 register_optimizer()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/optimizer.py:45`

**源代码**:
```python
def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
    cs.store(group="optimizer", package="optimizer", name="adamw", node=AdamWConfig)
```

**注册的配置**:

| 配置组 | 选项名 | 配置对象 | 说明 |
|--------|--------|----------|------|
| `optimizer` | `fusedadamw` | `FusedAdamWConfig` | 融合的 AdamW 优化器（速度更快） |
| `optimizer` | `adamw` | `AdamWConfig` | 标准 AdamW 优化器 |

**FusedAdamWConfig 定义**:
```python
FusedAdamWConfig = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="fusedadam",
    eps=1e-8,
    master_weights=True,
    capturable=True,
)
```

**参数说明**:
- `lr`: 学习率（默认 1e-4）
- `weight_decay`: 权重衰减（L2正则化）
- `betas`: Adam 的 β1 和 β2 参数
- `optim_type`: 优化器类型
- `master_weights`: 混合精度训练时使用 FP32 master weights
- `capturable`: 支持 CUDA graph 捕获

---

### 5.3 register_scheduler()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/scheduler.py`

**作用**: 注册学习率调度器配置

**典型配置**:
```python
cs.store(
    group="scheduler",
    package="scheduler",
    name="lambdalinear",
    node=LambdaLinearConfig
)
```

**LambdaLinear 调度器**:
- 预热阶段：学习率从0线性增加到最大值
- 衰减阶段：学习率线性衰减到最小值
- 参数：`warm_up_steps`, `f_max`, `f_min`, `cycle_lengths`

---

### 5.4 register_model()

**位置**: `cosmos_predict2/_src/predict2/configs/video2world/defaults/model.py:77`

**源代码**:
```python
def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1", node=FSDP_WAN2PT1_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_rectified_flow", node=FSDP_RECTIFIED_FLOW_CONFIG)
```

**注册的配置**:

| 配置组 | 选项名 | 配置对象 | 说明 |
|--------|--------|----------|------|
| `model` | `ddp` | `DDP_CONFIG` | DDP 并行模式 + Video2WorldModel |
| `model` | `fsdp` | `FSDP_CONFIG` | FSDP 并行模式 + Video2WorldModel |
| `model` | `fsdp_wan2pt1` | `FSDP_WAN2PT1_CONFIG` | FSDP + WAN2.1 模型 |
| `model` | `fsdp_rectified_flow` | `FSDP_RECTIFIED_FLOW_CONFIG` | FSDP + Rectified Flow 模型 |

**DDP_CONFIG 定义**:
```python
DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Video2WorldModel)(
        config=Video2WorldConfig(),
        _recursive_=False,
    ),
)
```

**注意**:
- `package="_global_"` 表示配置会合并到 Config 的根级别
- 包含模型类和训练并行策略

---

### 5.5 register_callbacks()

**位置**: `cosmos_predict2/_src/predict2/configs/video2world/defaults/callbacks.py:48`

**源代码**:
```python
def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="debug", node=DEBUG_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="long", node=LONG_RUNNING_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="cluster_speed", node=SPEED_CALLBACKS)
```

**注册的配置**:

| 配置组 | 选项名 | 说明 |
|--------|--------|------|
| `callbacks` | `basic` | 基础回调（EMA、进度条、设备监控等） |
| `callbacks` | `wandb` | W&B 回调 |
| `callbacks` | `debug` | 调试回调 |
| `callbacks` | `viz_online_sampling` | 在线采样可视化 |
| `callbacks` | `long` | 长时间运行的回调 |
| `callbacks` | `cluster_speed` | 集群速度监控 |

---

### 5.6 register_net()

**位置**: `cosmos_predict2/_src/predict2/configs/video2world/defaults/net.py:103`

**源代码**:
```python
def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B", node=WAN2PT1_1PT3B)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B", node=WAN2PT1_14B)
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_7B", node=COSMOS_V1_7B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_14B", node=COSMOS_V1_14B_NET_MININET)
```

**注册的配置**:

| 配置组 | 选项名 | 网络架构 | 参数量 |
|--------|--------|----------|--------|
| `net` | `cosmos_v1_2B` | MinimalV1LVGDiT | 2B |
| `net` | `cosmos_v1_7B` | MinimalV1LVGDiT | 7B |
| `net` | `cosmos_v1_14B` | MinimalV1LVGDiT | 14B |
| `net` | `wan2pt1_1pt3B` | WanModel | 1.3B |
| `net` | `wan2pt1_14B` | WanModel | 14B |
| `net` | `mini_net` | MinimalV1LVGDiT | 小型测试网络 |

**COSMOS_V1_2B_NET_MININET 定义**:
```python
COSMOS_V1_2B_NET_MININET = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
)
```

---

### 5.7 register_conditioner()

**位置**: `cosmos_predict2/_src/predict2/configs/video2world/defaults/conditioner.py:330`

**源代码**:
```python
def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="wan2pt1_video_prediction_conditioner_empty_string_drop",
        node=VideoConditionerFpsPaddingEmptyStringDrppConfig,
    )
```

**作用**: 注册条件生成器配置

**VideoPredictionConditioner 定义**:
```python
VideoPredictionConditioner = L(Video2WorldConditioner)(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)
```

**条件信息**:
- `fps`: 帧率信息
- `padding_mask`: 填充掩码
- `text`: 文本嵌入（来自T5）
- `use_video_condition`: 是否使用视频条件（用于 Classifier-Free Guidance）

---

### 5.8 register_ema()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/ema.py:27`

**源代码**:
```python
def register_ema():
    cs = ConfigStore.instance()
    cs.store(group="ema", package="model.config.ema", name="power", node=PowerEMAConfig)
```

**PowerEMAConfig 定义**:
```python
PowerEMAConfig = EMAConfig(
    enabled=True,
    rate=0.10,
    iteration_shift=0,
)
```

**作用**: 
- 指数移动平均（EMA）配置
- 维护模型权重的移动平均
- 验证和推理时使用 EMA 权重，通常效果更好

**EMA 更新公式**（Power EMA）:
```
decay = (1 + iteration / s)^-1
θ_ema = decay * θ_ema + (1 - decay) * θ
```

---

### 5.9 register_tokenizer()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/tokenizer.py:26`

**源代码**:
```python
def register_tokenizer():
    cs = ConfigStore.instance()
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer", node=Wan2pt1VAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt1_tokenizer_gcp", node=Wan2pt1VAEConfig_GCP)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt2_tokenizer", node=Wan2pt2VAEConfig)
```

**作用**: 注册 VAE tokenizer 配置

**Tokenizer**:
- 将视频编码到潜在空间（压缩表示）
- 使用变分自编码器（VAE）
- 典型压缩比：8x8x8（空间 8x，时间 8x）

---

### 5.10 register_checkpoint()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/checkpoint.py:65`

**源代码**:
```python
def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="pbss", node=CHECKPOINT_PBSS)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)
```

**注册的配置**:

| 配置组 | 选项名 | 对象存储类型 |
|--------|--------|--------------|
| `checkpoint` | `pbss` | PBSS（内部存储） |
| `checkpoint` | `s3` | AWS S3 |
| `checkpoint` | `gcp` | Google Cloud Storage |

**CHECKPOINT_S3 定义**:
```python
s3_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/s3_checkpoint.secret",
    bucket="bucket",
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    save_iter=1000,
    load_from_object_store=s3_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)
```

---

### 5.11 register_ckpt_type()

**位置**: `cosmos_predict2/_src/predict2/configs/common/defaults/ckpt_type.py`

**作用**: 注册检查点类型（DCP、TorchSave等）

---

## 6. Hydra配置系统

### 6.1 什么是 Hydra？

**Hydra** 是 Facebook Research 开发的配置管理框架，提供：
- 层次化配置
- 配置组合和覆盖
- 命令行接口
- 多运行支持

### 6.2 ConfigStore

**ConfigStore** 是 Hydra 的配置注册中心。

**使用方式**:
```python
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(
    group="optimizer",        # 配置组名
    package="optimizer",      # 目标字段路径
    name="fusedadamw",       # 选项名称
    node=FusedAdamWConfig,   # 配置对象
)
```

**参数说明**:

1. **group**: 配置组名
   - 定义一类相关配置选项
   - 例如：`optimizer`, `model`, `scheduler`

2. **package**: 目标字段路径
   - 配置在 Config 对象中的位置
   - 使用点分路径：`"model.config.net"`
   - 特殊值 `"_global_"` 表示根级别

3. **name**: 选项名称
   - 这个配置的标识符
   - 命令行使用：`group=name`

4. **node**: 配置对象
   - 实际的配置数据
   - 可以是 dict、dataclass、attrs类等

### 6.3 配置组的工作原理

#### 默认配置

在 `Config` 类中定义：

```python
defaults: List[Any] = attrs.field(
    factory=lambda: [
        "_self_",
        {"data_train": "mock"},
        {"optimizer": "fusedadamw"},
        # ...
    ]
)
```

#### 配置选择

命令行覆盖：
```bash
python train.py ... -- optimizer=adamw
```

Hydra 会：
1. 查找 `optimizer` 配置组
2. 选择名为 `adamw` 的配置
3. 将其应用到 `package` 指定的路径

#### 配置合并

多个配置源的优先级（从低到高）：
1. 默认值（代码中定义）
2. 配置组默认值（`defaults` 列表）
3. 实验配置（`experiment=xxx`）
4. 命令行覆盖（`opts`）

### 6.4 `_global_` Package

当 `package="_global_"` 时，配置会合并到 Config 根级别：

```python
cs.store(
    group="model",
    package="_global_",
    name="ddp",
    node={
        "trainer": {"distributed_parallelism": "ddp"},
        "model": L(Video2WorldModel)(...),
    }
)
```

应用后：
```python
config.trainer.distributed_parallelism = "ddp"
config.model = Video2WorldModel(...)
```

---

## 7. 配置覆盖机制

### 7.1 override() 函数

**位置**: `cosmos_predict2/_src/imaginaire/utils/config_helper.py:69`

**函数签名**:
```python
def override(config: Config, overrides: Optional[list[str]] = None) -> Config:
```

**作用**: 使用命令行参数覆盖配置

### 7.2 覆盖流程

**在 train.py 中调用**:

```python
# scripts/train.py:94
overrides = list(args.opts)
if args.smoke:
    overrides.append("job.wandb_mode=disabled")
    overrides.append("trainer.max_iter=1")
    # ...
config = override(config, overrides)
```

**override() 实现**:

```python
def override(config: Config, overrides: Optional[list[str]] = None) -> Config:
    # 1. 转换 Config 为 DictConfig
    config_dict = attrs.asdict(config)
    config_omegaconf = DictConfig(content=config_dict, flags={"allow_objects": True})
    
    # 2. 检查分隔符
    if overrides and overrides[0] != "--":
        raise ValueError('Hydra config overrides must be separated with a "--" token.')
    overrides = overrides[1:]
    
    # 3. 使用 Hydra 处理覆盖
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_omegaconf)
    
    if not GlobalHydra().is_initialized():
        with initialize(version_base=None):
            config_omegaconf = compose(config_name="config", overrides=overrides)
            OmegaConf.resolve(config_omegaconf)
    else:
        config_omegaconf = compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(config_omegaconf)
    
    # 4. 转换回 Config 对象
    config = config_from_dict(config, config_omegaconf)
    return config
```

### 7.3 覆盖语法

#### 基本覆盖

```bash
# 设置单个值
-- optimizer.lr=1e-3

# 覆盖嵌套值
-- model.config.net.num_blocks=32

# 选择配置组
-- optimizer=adamw

# 多个覆盖
-- optimizer=adamw optimizer.lr=1e-3 trainer.max_iter=10000
```

#### 删除配置

```bash
# 删除字段
-- ~trainer.callbacks.wandb
```

#### 列表和字典

```bash
# 列表
-- scheduler.f_max=[0.5,0.3]

# 字典
-- model.config='{key1: value1, key2: value2}'
```

### 7.4 实验配置覆盖

实验配置通过 `experiment=xxx` 指定：

```bash
-- experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

**工作原理**:

1. 实验配置是一个字典，定义在 `cosmos_predict2/experiments/base/cosmos_nemo_assets.py`
2. 字典中的 `defaults` 字段定义继承关系
3. 其他字段会覆盖基础配置

**示例**:

```python
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_nemo_assets",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    # ...
)
```

**覆盖效果**:
- `config.job.project` → `"cosmos_predict_v2p5"`
- `config.job.name` → `"2b_cosmos_nemo_assets"`
- `config.optimizer.lr` → `5.66e-5`
- `config.dataloader_train` → 自定义数据加载器

---

## 8. 完整调用链

### 8.1 从脚本到配置对象

```
run_posttrain_nemo.sh
  └── torchrun scripts/train.py --config=... -- experiment=...
        └── __main__
              ├── argparse.ArgumentParser().parse_args()
              │     └── args.config, args.opts
              │
              ├── get_config_module(args.config)
              │     └── "cosmos_predict2._src.predict2.configs.video2world.config"
              │
              ├── importlib.import_module(config_module)
              │     └── 导入 cosmos_predict2/_src/predict2/configs/video2world/config.py
              │
              ├── .make_config()
              │     ├── Config(model=None, optimizer=None, ...)
              │     │     ├── Config.__init__()
              │     │     │     ├── self.model = None
              │     │     │     ├── self.optimizer = None
              │     │     │     ├── self.scheduler = None
              │     │     │     ├── self.dataloader_train = None
              │     │     │     ├── self.dataloader_val = None
              │     │     │     ├── self.job = JobConfig()
              │     │     │     ├── self.trainer = TrainerConfig()
              │     │     │     ├── self.checkpoint = CheckpointConfig()
              │     │     │     └── self.model_parallel = ModelParallelConfig()
              │     │     └── return Config instance
              │     │
              │     ├── 设置默认值
              │     │     ├── c.job.project = "cosmos_diffusion_v2"
              │     │     ├── c.job.group = "debug"
              │     │     ├── c.job.name = "delete_${now:...}"
              │     │     ├── c.trainer.type = Trainer
              │     │     ├── c.trainer.max_iter = 400_000
              │     │     ├── c.trainer.logging_iter = 10
              │     │     ├── c.trainer.validation_iter = 100
              │     │     └── c.trainer.run_validation = False
              │     │
              │     ├── 注册配置组
              │     │     ├── register_training_and_val_data()
              │     │     │     ├── ConfigStore.instance()
              │     │     │     ├── cs.store(group="data_train", name="mock", ...)
              │     │     │     ├── cs.store(group="data_train", name="mock_image", ...)
              │     │     │     ├── cs.store(group="data_train", name="mock_video", ...)
              │     │     │     └── cs.store(group="data_val", name="mock", ...)
              │     │     │
              │     │     ├── register_optimizer()
              │     │     │     ├── ConfigStore.instance()
              │     │     │     ├── cs.store(group="optimizer", name="fusedadamw", node=FusedAdamWConfig)
              │     │     │     └── cs.store(group="optimizer", name="adamw", node=AdamWConfig)
              │     │     │
              │     │     ├── register_scheduler()
              │     │     │     └── cs.store(group="scheduler", name="lambdalinear", ...)
              │     │     │
              │     │     ├── register_model()
              │     │     │     ├── cs.store(group="model", name="ddp", node=DDP_CONFIG)
              │     │     │     ├── cs.store(group="model", name="fsdp", node=FSDP_CONFIG)
              │     │     │     ├── cs.store(group="model", name="fsdp_wan2pt1", ...)
              │     │     │     └── cs.store(group="model", name="fsdp_rectified_flow", ...)
              │     │     │
              │     │     ├── register_callbacks()
              │     │     │     ├── cs.store(group="callbacks", name="basic", ...)
              │     │     │     ├── cs.store(group="callbacks", name="wandb", ...)
              │     │     │     ├── cs.store(group="callbacks", name="debug", ...)
              │     │     │     ├── cs.store(group="callbacks", name="viz_online_sampling", ...)
              │     │     │     ├── cs.store(group="callbacks", name="long", ...)
              │     │     │     └── cs.store(group="callbacks", name="cluster_speed", ...)
              │     │     │
              │     │     ├── register_net()
              │     │     │     ├── cs.store(group="net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
              │     │     │     ├── cs.store(group="net", name="cosmos_v1_7B", node=COSMOS_V1_7B_NET_MININET)
              │     │     │     ├── cs.store(group="net", name="cosmos_v1_14B", node=COSMOS_V1_14B_NET_MININET)
              │     │     │     ├── cs.store(group="net", name="wan2pt1_1pt3B", ...)
              │     │     │     ├── cs.store(group="net", name="wan2pt1_14B", ...)
              │     │     │     └── cs.store(group="net", name="mini_net", ...)
              │     │     │
              │     │     ├── register_conditioner()
              │     │     │     ├── cs.store(group="conditioner", name="video_prediction_conditioner", ...)
              │     │     │     ├── cs.store(group="conditioner", name="video_prediction_conditioner_v2", ...)
              │     │     │     └── cs.store(group="conditioner", name="wan2pt1_video_prediction_conditioner_empty_string_drop", ...)
              │     │     │
              │     │     ├── register_ema()
              │     │     │     └── cs.store(group="ema", name="power", node=PowerEMAConfig)
              │     │     │
              │     │     ├── register_tokenizer()
              │     │     │     ├── cs.store(group="tokenizer", name="wan2pt1_tokenizer", ...)
              │     │     │     ├── cs.store(group="tokenizer", name="wan2pt1_tokenizer_gcp", ...)
              │     │     │     └── cs.store(group="tokenizer", name="wan2pt2_tokenizer", ...)
              │     │     │
              │     │     ├── register_checkpoint()
              │     │     │     ├── cs.store(group="checkpoint", name="pbss", ...)
              │     │     │     ├── cs.store(group="checkpoint", name="s3", ...)
              │     │     │     └── cs.store(group="checkpoint", name="gcp", ...)
              │     │     │
              │     │     └── register_ckpt_type()
              │     │           └── cs.store(group="ckpt_type", ...)
              │     │
              │     ├── 导入实验配置
              │     │     ├── import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
              │     │     │     ├── importlib.import_module(package_path)
              │     │     │     ├── 遍历包目录
              │     │     │     └── 递归导入所有模块
              │     │     │
              │     │     └── import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
              │     │           ├── 导入 cosmos_predict2/experiments/base/cosmos_nemo_assets.py
              │     │           │     └── 模块加载时执行注册
              │     │           │           ├── predict2_video2world_training_2b_cosmos_nemo_assets = dict(...)
              │     │           │           ├── cs = ConfigStore.instance()
              │     │           │           └── cs.store(group="experiment", name="predict2_video2world_training_2b_cosmos_nemo_assets", ...)
              │     │           └── 导入其他实验模块...
              │     │
              │     └── return c
              │
              └── override(config, overrides=args.opts)
                    ├── attrs.asdict(config) → dict
                    ├── DictConfig(content=config_dict)
                    ├── ConfigStore.instance().store(name="config", node=config_omegaconf)
                    ├── Hydra compose(config_name="config", overrides=overrides)
                    │     ├── 应用 defaults 列表中的配置组
                    │     ├── 应用 experiment=xxx 覆盖
                    │     └── 应用命令行覆盖
                    ├── OmegaConf.resolve() → 解析变量插值
                    ├── config_from_dict() → 重建 Config 对象
                    └── return Config
```

### 8.2 配置应用流程

```
初始 Config (空值)
  └── apply defaults
        ├── data_train=mock → 应用 MOCK_DATA_INTERLEAVE_CONFIG
        ├── optimizer=fusedadamw → 应用 FusedAdamWConfig
        ├── scheduler=lambdalinear → 应用调度器配置
        ├── model=ddp → 应用 DDP_CONFIG
        ├── callbacks=basic → 应用基础回调
        ├── conditioner=video_prediction_conditioner → 应用条件器
        ├── ema=power → 应用 PowerEMAConfig
        ├── tokenizer=cosmos_tokenizer_... → 应用 tokenizer
        └── checkpoint=s3 → 应用 CHECKPOINT_S3
  └── apply experiment
        └── experiment=predict2_video2world_training_2b_cosmos_nemo_assets
              ├── defaults 继承
              │     └── inherit from /experiment/{DEFAULT_CHECKPOINT.experiment}
              ├── job 覆盖
              │     ├── project → "cosmos_predict_v2p5"
              │     ├── group → "video2world"
              │     └── name → "2b_cosmos_nemo_assets"
              ├── dataloader_train 覆盖
              │     └── dataloader_train_cosmos_nemo_assets
              ├── checkpoint 覆盖
              │     ├── save_iter → 200
              │     └── load_path → ".../81edfebe...ema_bf16.pt"
              ├── optimizer 覆盖
              │     ├── lr → 5.66e-5
              │     └── weight_decay → 0.001
              ├── scheduler 覆盖
              │     ├── f_max → [0.5]
              │     ├── f_min → [0.2]
              │     ├── warm_up_steps → [2000]
              │     └── cycle_lengths → [100000]
              ├── trainer 覆盖
              │     ├── logging_iter → 100
              │     ├── max_iter → 1000
              │     └── callbacks 覆盖
              └── model_parallel 覆盖
                    └── context_parallel_size → 1
  └── apply command-line overrides
        └── (任何额外的命令行参数)
  └── 最终 Config (完全配置)
```

---

## 9. 使用示例

### 9.1 基本使用

```python
# 创建配置
from cosmos_predict2._src.predict2.configs.video2world.config import make_config
config = make_config()

# 访问配置
print(config.job.project)  # "cosmos_diffusion_v2"
print(config.trainer.max_iter)  # 400000

# 注意：此时 model, optimizer 等还是 None
# 需要通过 Hydra 应用配置组才会有具体值
```

### 9.2 命令行使用

```bash
# 基本训练
python scripts/train.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets

# 覆盖学习率
python scripts/train.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets \
  optimizer.lr=1e-4

# 选择不同的网络
python scripts/train.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  net=cosmos_v1_7B \
  experiment=my_experiment

# Smoke test（快速调试）
python scripts/train.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --smoke \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

### 9.3 编程方式覆盖

```python
from cosmos_predict2._src.predict2.configs.video2world.config import make_config
from cosmos_predict2._src.imaginaire.utils.config_helper import override

# 创建配置
config = make_config()

# 应用覆盖
overrides = [
    "--",
    "experiment=predict2_video2world_training_2b_cosmos_nemo_assets",
    "optimizer.lr=1e-4",
    "trainer.max_iter=10000",
]
config = override(config, overrides)

# 现在配置已完全设置
print(config.optimizer.lr)  # 1e-4
print(config.trainer.max_iter)  # 10000
```

### 9.4 自定义实验配置

创建新的实验配置文件：

```python
# cosmos_predict2/experiments/my_experiments/my_config.py

from hydra.core.config_store import ConfigStore
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset, get_generic_dataloader, get_sampler

# 定义数据集
my_dataset = L(VideoDataset)(
    dataset_dir="datasets/my_dataset",
    num_frames=121,
    video_size=(720, 1280),
)

my_dataloader = L(get_generic_dataloader)(
    dataset=my_dataset,
    sampler=L(get_sampler)(dataset=my_dataset),
    batch_size=1,
    num_workers=8,
)

# 定义实验配置
my_experiment_config = dict(
    defaults=[
        "/experiment/predict2_video2world_training_2b_cosmos_nemo_assets",
        "_self_",
    ],
    job=dict(
        project="my_project",
        group="my_group",
        name="my_experiment",
    ),
    dataloader_train=my_dataloader,
    optimizer=dict(
        lr=1e-4,
    ),
    trainer=dict(
        max_iter=50000,
    ),
)

# 注册配置
cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="my_experiment_config",
    node=my_experiment_config,
)
```

使用新配置：

```bash
python scripts/train.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=my_experiment_config
```

---

## 10. 总结

### 10.1 `make_config()` 的核心作用

1. **创建配置骨架**: 实例化 `Config` 对象，包含所有必需字段
2. **设置默认值**: 为作业和训练器设置合理的默认值
3. **注册配置选项**: 将所有可用的配置选项注册到 Hydra
4. **导入实验配置**: 加载项目和实验级别的配置

### 10.2 配置系统的优势

✅ **模块化**: 每个组件（优化器、模型、数据等）独立配置  
✅ **可组合**: 通过配置组灵活组合不同配置  
✅ **可覆盖**: 多层覆盖机制，从默认值到命令行  
✅ **类型安全**: 使用 attrs 定义的配置类提供类型检查  
✅ **可扩展**: 通过注册机制轻松添加新配置  
✅ **可重用**: 配置可以继承和共享  

### 10.3 关键概念回顾

| 概念 | 说明 |
|------|------|
| **Config 类** | 配置系统的根对象，包含所有训练配置 |
| **ConfigStore** | Hydra 的配置注册中心 |
| **配置组** | 一类相关配置选项（如 optimizer, model） |
| **defaults** | 配置组的默认选择 |
| **package** | 配置在 Config 对象中的路径 |
| **LazyDict** | 延迟实例化的配置字典 |
| **override()** | 使用 Hydra 应用配置覆盖 |
| **实验配置** | 项目级别的配置，通过 `experiment=xxx` 选择 |

### 10.4 配置流程总结

```
make_config() 创建配置骨架
    ↓
register_*() 注册所有配置选项
    ↓
import_all_modules_from_package() 导入实验配置
    ↓
override() 应用 defaults 和命令行覆盖
    ↓
最终的完全配置的 Config 对象
    ↓
用于训练
```

### 10.5 最佳实践

1. **不要直接修改 make_config()**
   - 通过实验配置或命令行覆盖

2. **创建新实验配置而不是修改现有配置**
   - 保持可重现性

3. **使用有意义的实验名称**
   - 便于跟踪和比较

4. **验证配置后立即冻结**
   - 防止训练中意外修改

5. **保存配置到训练目录**
   - 确保可重现性

---

**文档生成时间**: 2025-10-11  
**项目**: Cosmos Predict2.5  
**版本**: Video2World Config 系统分析

