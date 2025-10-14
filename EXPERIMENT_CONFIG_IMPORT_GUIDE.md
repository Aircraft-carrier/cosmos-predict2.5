# 实验配置导入指南

## 问题背景

在 `make_config()` 函数中，有两行代码会导入所有实验配置：

```python
import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
```

**问题**：如果确定不使用这些预先设置的实验配置，是否可以不导入这些模块？

**答案**：可以，但需要根据使用场景选择合适的方案。

---

## 实验配置导入的作用

### 导入流程

```
import_all_modules_from_package()
  └── 遍历目录下所有 .py 文件
        └── 对每个文件执行 import
              └── 触发模块顶层代码执行
                    └── cs.store() 注册配置到 Hydra ConfigStore
                          └── 配置变为可选项
```

### 示例：`model_14b_reason_1p1.py`

```python
# 定义配置
I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16 = LazyDict(
    dict(
        job=dict(name="Stage-c_pt_4-reason_embeddings-v1p1-..."),
        model=dict(...),
        trainer=dict(...),
        # ...
    )
)

# 注册配置（模块导入时自动执行）
cs = ConfigStore.instance()
cs.store(
    group="experiment", 
    package="_global_", 
    name="Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_gcp",
    node=I2V_REASON_EMBEDDINGS_V1P1_STAGE_C_PT_4_INDEX_43_SIZE_14B_RES_720_FPS16
)
```

**核心要点**：
- ✅ 导入后，可以通过 `experiment=xxx` 选择这个配置
- ❌ 不导入，就不能使用 `experiment=xxx`
- ✅ 不导入不影响其他功能（只要不引用这些实验配置）

---

## 使用场景分析

### 场景 1：使用预定义实验配置

```bash
python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

**需要导入**：✅ 必须保留 `import_all_modules_from_package()`

---

### 场景 2：自定义配置（命令行覆盖）

```bash
python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  optimizer.lr=5e-5 \
  trainer.max_iter=1000 \
  model.config.resolution=720 \
  dataloader_train.batch_size=2
```

**可以不导入**：❌ 不需要 `import_all_modules_from_package()`

---

### 场景 3：自定义配置（编程方式）

```python
from cosmos_predict2._src.predict2.configs.video2world.config import make_config
from cosmos_predict2._src.imaginaire.utils.config_helper import override

config = make_config()

# 手动设置所有参数
overrides = [
    "--",
    "optimizer=fusedadamw",
    "optimizer.lr=5e-5",
    "model=fsdp",
    "net=cosmos_v1_2B",
    # ... 更多参数
]
config = override(config, overrides)
```

**可以不导入**：❌ 不需要 `import_all_modules_from_package()`

---

### 场景 4：只使用特定的实验配置

```bash
# 只使用 cosmos_predict2.experiments 下的配置
# 不使用 cosmos_predict2._src.predict2.configs.video2world.experiment 下的配置

python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

**部分导入**：✅ 只保留一个 `import_all_modules_from_package()`

---

## 修改方案

### 方案 A：完全不导入（适合完全自定义配置）

**修改 `make_config()`**：

```python
def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # 设置默认值
    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    # ...

    # 注册配置组
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

    # ❌ 完全不导入实验配置
    # import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    # import_all_modules_from_package("cosmos_predict2.experiments", reload=True)

    return c
```

**优点**：
- ✅ 启动速度更快
- ✅ 内存占用更少
- ✅ 代码更清晰（只保留必需部分）

**缺点**：
- ❌ 不能使用 `experiment=xxx`
- ❌ 需要手动指定所有配置参数

---

### 方案 B：选择性导入（推荐）

```python
def make_config() -> Config:
    c = Config(...)
    
    # 设置默认值
    # ...
    
    # 注册配置组
    # ...
    
    # ✅ 只导入你需要的实验配置
    # 如果不需要 video2world.experiment 下的大型配置文件，就注释掉
    # import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    
    # 保留 experiments（包含常用的 cosmos_nemo_assets 等）
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    
    return c
```

**优点**：
- ✅ 平衡了灵活性和性能
- ✅ 保留常用的实验配置
- ✅ 去掉不常用的大型配置文件

---

### 方案 C：条件导入（最灵活）

```python
import os

def make_config() -> Config:
    c = Config(...)
    
    # 设置默认值和注册配置组
    # ...
    
    # 通过环境变量控制
    load_exp_configs = os.environ.get("LOAD_EXPERIMENT_CONFIGS", "true").lower() == "true"
    
    if load_exp_configs:
        import_all_modules_from_package(
            "cosmos_predict2._src.predict2.configs.video2world.experiment", 
            reload=True
        )
        import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    
    return c
```

**使用方式**：

```bash
# 不加载实验配置（快速启动）
LOAD_EXPERIMENT_CONFIGS=false python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  optimizer.lr=5e-5 \
  trainer.max_iter=1000

# 加载实验配置（使用预定义配置）
python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_cosmos_nemo_assets
```

**优点**：
- ✅ 最大灵活性
- ✅ 不需要修改代码
- ✅ 适合不同的使用场景

---

### 方案 D：延迟导入（高级）

```python
def make_config() -> Config:
    c = Config(...)
    
    # 设置默认值和注册配置组
    # ...
    
    # ✅ 不在 make_config() 中导入
    # 在 override() 中按需导入
    
    return c
```

**修改 `override()` 或 `train.py`**：

```python
# scripts/train.py

config_module = get_config_module(args.config)
config = importlib.import_module(config_module).make_config()

# 如果命令行参数包含 experiment=xxx，才导入实验配置
if any('experiment=' in opt for opt in args.opts):
    from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
    import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)

config = override(config, list(args.opts))
```

**优点**：
- ✅ 完全按需加载
- ✅ 最优性能

**缺点**：
- ❌ 需要修改 `train.py`
- ❌ 逻辑更复杂

---

## 性能对比

### 测量导入时间

```python
import time

def make_config() -> Config:
    c = Config(...)
    # ...
    
    # 测量第一个导入
    start = time.time()
    import_all_modules_from_package(
        "cosmos_predict2._src.predict2.configs.video2world.experiment", 
        reload=True
    )
    elapsed1 = time.time() - start
    print(f"[Timing] Import video2world.experiment: {elapsed1:.2f}s")
    
    # 测量第二个导入
    start = time.time()
    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    elapsed2 = time.time() - start
    print(f"[Timing] Import experiments: {elapsed2:.2f}s")
    
    print(f"[Timing] Total import time: {elapsed1 + elapsed2:.2f}s")
    
    return c
```

### 预期结果

根据文件大小估算：
- `cosmos_predict2._src.predict2.configs.video2world.experiment`: **3-10秒**
  - 包含多个大型配置文件（如 `state3_14B_index_3.py` 有 4327 行）
- `cosmos_predict2.experiments`: **0.5-2秒**
  - 文件较少且较小

**总计**：约 **3.5-12秒** 的额外启动时间

---

## 推荐方案总结

| 使用场景 | 推荐方案 | 说明 |
|----------|----------|------|
| **生产环境** | 保持默认 | 导入所有配置，确保兼容性 |
| **开发/调试** | 方案 B 或 C | 选择性或条件导入，加快启动 |
| **自定义训练** | 方案 A | 完全不导入，手动配置所有参数 |
| **性能敏感** | 方案 D | 延迟导入，按需加载 |

### 我的具体建议

**对于您的情况（确定不使用预设配置）**：

1. **短期方案**（快速验证）：
   ```python
   # 在 make_config() 中直接注释掉
   # import_all_modules_from_package("cosmos_predict2._src.predict2.configs.video2world.experiment", reload=True)
   # import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
   ```

2. **长期方案**（推荐）：
   - 使用方案 C（条件导入）
   - 通过环境变量控制
   - 保持代码的灵活性

---

## 实际操作示例

### 示例 1：使用自定义配置训练

```bash
# 步骤 1: 修改 make_config()，注释掉导入
# （如上述方案 A 所示）

# 步骤 2: 通过命令行指定所有参数
python scripts/train.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  job.project=my_project \
  job.group=my_group \
  job.name=my_experiment \
  optimizer=fusedadamw \
  optimizer.lr=5e-5 \
  optimizer.weight_decay=0.001 \
  scheduler=lambdalinear \
  scheduler.warm_up_steps=[2000] \
  model=fsdp \
  net=cosmos_v1_2B \
  trainer.max_iter=1000 \
  trainer.logging_iter=100 \
  checkpoint.save_iter=200 \
  checkpoint.load_path=/path/to/checkpoint.pt
```

### 示例 2：创建简化的自定义实验配置

如果你有自己的数据集和训练设置，创建一个简单的实验配置文件：

```python
# cosmos_predict2/experiments/my_experiments/my_simple_config.py

from hydra.core.config_store import ConfigStore

my_simple_training = dict(
    job=dict(
        project="my_project",
        group="my_group",
        name="my_simple_training",
    ),
    trainer=dict(
        max_iter=1000,
        logging_iter=100,
    ),
    optimizer=dict(
        lr=5e-5,
        weight_decay=0.001,
    ),
    checkpoint=dict(
        save_iter=200,
        load_path="/path/to/checkpoint.pt",
    ),
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="my_simple_training", node=my_simple_training)
```

然后只导入你自己的实验配置：

```python
def make_config() -> Config:
    c = Config(...)
    # ...
    
    # 只导入你自己的实验配置目录
    import_all_modules_from_package("cosmos_predict2.experiments.my_experiments", reload=True)
    
    return c
```

---

## 总结

1. **可以不导入**：如果确定不使用预设实验配置，完全可以不导入这些模块
2. **导入开销**：对于大型配置文件，导入可能需要 3-12 秒
3. **推荐做法**：
   - 生产环境：保持默认（确保兼容性）
   - 开发环境：选择性导入或条件导入（提升效率）
   - 自定义训练：完全不导入（简化流程）
4. **灵活性**：使用环境变量控制是最佳平衡

**关键原则**：这些导入只是注册配置选项，不影响核心功能。只要你不引用这些配置名称，就可以安全地跳过导入。

