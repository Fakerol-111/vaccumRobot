# 算法扩展指南

## 目录结构

```
agent/
├── base.py                 # Algorithm 抽象基类 + ActResult / LossInfo
├── registry.py             # 算法注册表 register() / get() / list_available()
├── common/                 # 跨算法共享组件
│   ├── checkpoint.py       # Checkpoint 数据类、RNG 快照、config_snapshot 构建
│   └── functional.py       # 网络构建单元 make_fc / make_conv
├── nn/                     # 网络架构
│   ├── actor_critic.py     # Actor-Critic 双流网络（CNN + MLP，共享编码器）
│   ├── separate_ac.py      # SeparateActorCritic（独立编码器，TRPO 必须）
│   └── factory.py          # create_model(model_type, ...) 工厂函数
├── preprocessor.py         # 环境特征工程（领域相关，算法无关）
├── a2c/                    # A2C 算法
│   ├── algorithm.py        # @register("a2c")，n-step return
│   └── a2c_metrics.py      # 监控指标
├── ppo/                    # PPO 算法
│   ├── algorithm.py        # @register("ppo")，on_step/maybe_update
│   ├── ppo_metrics.py      # 监控指标
│   ├── buffer.py           # RolloutBuffer（变长 episode 缓冲）
│   ├── batch.py            # RolloutBatch 数据类 + GAE 计算
│   └── update.py           # PPO 更新核心（clip 目标、值裁剪、熵）
├── ppo_kl/                 # PPO-KL 算法
│   ├── algorithm.py        # @register("ppo_kl")，KL 自适应惩罚，无 clip
│   └── ppo_kl_metrics.py   # 监控指标（含 KL Beta）
├── reinforce/              # REINFORCE 算法
│   ├── algorithm.py        # @register("reinforce")，纯蒙特卡洛
│   └── reinforce_metrics.py# 监控指标
├── trpo/                   # TRPO 算法
│   ├── algorithm.py        # @register("trpo")，共轭梯度 + 线搜索
│   └── trpo_metrics.py     # 监控指标（Surrogate, KL, Step）
└── grpo/                   # GRPO 算法
    ├── algorithm.py        # @register("grpo")，分支 rollout + 组内归一化
    └── grpo_metrics.py     # 监控指标（Mean Score, Std Score, KL）
```

## 如何添加新算法

### 最小步骤（5 步）

假设你要添加 **DQNAlgorithm**：

**1. 创建算法目录**

```
agent/dqn/
├── __init__.py         → 导出 DQNAlgorithm
├── algorithm.py        → DQNAlgorithm(Algorithm)，加 @register("dqn")
├── replay_buffer.py    → DQN 专用回放缓冲区
└── network.py          → Q 网络（可选，也可复用 agent/nn/ 中的组件）
```

**2. 实现 Algorithm 接口**

只需实现 `agent/base.py` 中定义的抽象方法：

| 方法 | 用途 |
|---|---|
| `act(map_img, vector, legal_mask, mode)` | 统一入口，`mode="explore"` 或 `"exploit"` |
| `explore(map_img, vector, legal_mask)` | 随机采样（训练） |
| `exploit(map_img, vector, legal_mask)` | 贪婪选择（评估） |
| `collect(map_img, vector, ..., done)` | 存入内部 buffer |
| `ready_to_update()` | buffer 是否够一次 update |
| `update(bootstrap_value)` | 从 buffer 学习，返回 LossInfo |
| `compute_value(map_img, vector, legal_mask)` | 计算状态价值（用于 bootstrapping） |
| `save(path)` | 保存模型权重 |
| `load(path)` | 加载模型权重 |
| `save_checkpoint(path, global_step, ...)` | 保存完整断点（含优化器、RNG） |
| `load_checkpoint(path)` | 加载完整断点，返回 Checkpoint |

**可选覆写：**
| 方法 | 默认行为 |
|---|---|
| `on_step(map_img, vector, ..., done)` | 调用 `collect()`，返回 `None` |
| `maybe_update(bootstrap_state)` | 检查 `ready_to_update()`，调用 `update()` |
| `set_env_config(env_config)` | 无操作（环境切换时通知算法） |
| `metrics_reporter` | 返回 `None`（各算法实现自己的监控指标） |

**3. 注册算法**

在 DQNAlgorithm 类上使用装饰器自动注册：

```python
from agent.registry import register

@register("dqn")
class DQNAlgorithm(Algorithm):
    ...
```

也可以通过 `agent/registry.py` 中的 `get("dqn")` 按名称查找。

**4. 配置中指定算法**

在 `configs/train_config.toml` 中：

```toml
[algorithm]
name = "dqn"

# DQN 独立超参
[dqn]
learning_rate = 0.0001
batch_size = 64
...
```

**5. 更新 config 加载器**

在 `configs/runtime_config.py` 中：

- 添加 `_parse_dqn(raw)` 解析 `[dqn]` 节
- `load_train_config_bundle()` 的返回值增加 `dqn` 属性

`core/trainer_runner.py` 中通过 registry 获取算法类：

```python
from agent.registry import get as get_algorithm

algo_cls = get_algorithm(req.algo_name)
algorithm = algo_cls(req.algo_config, device)
```

`req.algo_name` 和 `req.algo_config` 由 `scripts/train.py` 根据 `[algorithm].name` 自动派发：

```python
algo_name = cfg.algo["name"]
algo_config = getattr(cfg, algo_name, None)  # → cfg.ppo / cfg.grpo / cfg.trpo ...
```

### 可复用的共享组件

| 组件 | 位置 | 说明 |
|---|---|---|
| `Checkpoint` | `agent/common/checkpoint.py` | 断点数据结构，任何算法都可直接使用 |
| `capture_rng_state` / `restore_rng_state` | `agent/common/checkpoint.py` | RNG 快照工具 |
| `build_config_snapshot(config, extra)` | `agent/common/checkpoint.py` | 从 config 提取超参快照 |
| `make_fc` / `make_conv` | `agent/common/functional.py` | 正交初始化网络层工厂 |
| `ActorCritic` | `agent/nn/actor_critic.py` | 共享编码器双流网络 |
| `SeparateActorCritic` | `agent/nn/separate_ac.py` | 独立编码器双流网络（TRPO 必须） |
| `create_model(type, ...)` | `agent/nn/factory.py` | 工厂函数，按 `model_type` 创建 |
| `Preprocessor` | `agent/preprocessor.py` | 环境特征工程（完全共享，无需修改） |

### 应保留在算法内部的组件

- **经验回放 / 轨迹 buffer**：PPO 的 `RolloutBatch` vs DQN 的 `ReplayBuffer`，存储结构完全不同
- **更新数学**：PPO 的 clip 目标 vs DQN 的 TD 误差，无法统一
- **网络架构**：虽然 `ActorCritic` 可复用，但 DQN 可能只需要一个 Q 网络（无 value head）
- **超参数集合**：LR、batch_size 等名称相似但语义不同，留在各自 `[ppo]` / `[dqn]` 节中

### 训练流程适配

`core/trainer.py` 已通过 `Algorithm` 接口与具体算法解耦。新算法只要实现了该接口，无需修改 Trainer 即可适配。特殊需求（如 off-policy 的不同 `ready_to_update` 逻辑）通过算法自身的 `collect()` 和 `ready_to_update()` 实现差异。

评估入口 `core/evaluator_runner.py` 同理，通过 registry 获取算法类并调用 `Algorithm` 接口进行推理。
