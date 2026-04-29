# 清扫机器人 PPO 训练

基于 **PyTorch + Gymnasium** 的 PPO 强化学习框架，支持**多地图轮换训练**、**课程学习**、**断点续训**、**实时 Web 监控面板**。

---

## 项目结构

```
vaccumRobot/
├── agent/                    # 智能体核心
│   ├── agent.py              # Agent：封装模型、优化器、前向/训练接口
│   ├── algorithm.py          # PPO 算法实现（clip 损失、GAE）
│   ├── model.py              # ActorCritic 双流网络（CNN 地图编码 + MLP 向量编码）
│   ├── preprocessor.py       # 特征工程 + 奖励函数
│   ├── checkpoint.py         # Checkpoint 数据类（含 RNG 状态、配置快照）
│   ├── definition.py         # RolloutBatch 数据类 + GAE 计算函数
│
├── core/                     # 运行框架
│   ├── trainer.py            # Trainer 类（训练主循环、课程学习、断点续训）
│   ├── evaluator.py          # 评估函数族（单图 / 多图录制 / 汇总输出 + MapEvalResult）
│   ├── trainer_runner.py     # 训练启动编排（设种子、建 Dashboard、组装 Trainer）
│   ├── evaluator_runner.py   # 评估启动编排（定位断点、组装 Agent、汇总结果）
│   ├── paths.py              # 路径工具（查找 run / checkpoint / eval 目录）
│   ├── types.py              # 请求/上下文/结果数据类（TrainRequest, EvalResult 等）
│
├── env/                      # 仿真环境
│   ├── grid_world.py         # GridWorldEnv（Gymnasium Env，128×128 网格）
│   ├── factory.py            # create_env 工厂函数
│   ├── trajectory_recorder.py# 轨迹录制（GIF + 日志）
│
├── services/                 # 服务层
│   ├── dashboard_service.py  # Web Dashboard（Chart.js 实时图表 + 日志流）
│   ├── metrics_service.py    # MetricsLogger（episode/update 记录、EMA、汇总）
│   ├── checkpoint_service.py # 断点查找与自动恢复
│
├── scripts/                  # CLI 入口脚本
│   ├── train.py              # 训练入口：python scripts/train.py [config] [--resume]
│   ├── eval.py               # 评估入口：python scripts/eval.py [--config <path>]
│   ├── export_maps_to_json.py# 地图 .py → .json 导出
│   ├── validate_maps_json.py # 地图 JSON 校验
│   ├── plot_maps.py          # 地图可视化
│
├── configs/                  # 配置文件
│   ├── train_config.toml     # 训练配置（PPO 超参 / 环境 / 课程 / Dashboard）
│   ├── test_config.toml      # 评估配置（地图 / episode / 断点选择）
│   ├── runtime_config.py     # 统一配置加载器（TOML → SimpleNamespace / dict）
│   ├── map_loader.py         # 地图 JSON 加载与校验
│   ├── map_editor.py         # 地图编辑器
│   ├── maps/                 # 地图数据
│       ├── map_1.json ~ map_4.json  # 运行时地图数据
│       └── src/                      # 地图生成源码
│           ├── map_1.py ~ map_4.py
│
├── tests/                    # 单元测试
│   ├── test_checkpoint_service.py
│   ├── test_map_loader.py
│   ├── test_paths.py
│   ├── test_runner_smoke.py
│   ├── test_runtime_config.py
│   └── test_simple_map.py
│
├── main.py                   # 快捷入口（等价于 python scripts/train.py）
├── requirements.txt
└── pyproject.toml
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 训练（使用默认配置）
python main.py
# 或
python scripts/train.py

# 评估（使用默认测试配置）
python scripts/eval.py
```

---

## 训练

### 启动训练

```bash
# 使用默认配置 configs/train_config.toml
python main.py

# 指定配置文件
python scripts/train.py my_config.toml

# 断点续训（自动选择最新的 checkpoint）
python scripts/train.py --resume

# 从指定 run 目录续训（自动找最新 checkpoint）
python scripts/train.py --resume artifacts/multi_map/checkpoints/20260429_120000

# 从指定 checkpoint 文件续训
python scripts/train.py --resume artifacts/multi_map/checkpoints/20260429_120000/checkpoint_5000.pt
```

### 训练配置 (`configs/train_config.toml`)

```toml
[ppo]
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
ppo_epochs = 4
batch_size = 256
mini_batch_size = 64
total_timesteps = 10_000_000_000
save_interval = 2_000
log_interval = 500
num_actions = 8
local_view_size = 21
max_npcs = 5

[general]
seed = 42

[env]
default_map_list = [1, 2, 3, 4]
default_npc_count = 1
default_station_count = 4
map_strategy = "round_robin"

[curriculum]
enabled = false

[[curriculum.stage]]
name = "easy_basic"
maps = [1]
npc_count = 0
station_count = 1
total_steps = 300_000

[[curriculum.stage]]
name = "medium_npc_maze"
maps = [1, 2]
npc_count = 1
station_count = 2
total_steps = 1_000_000

[[curriculum.stage]]
name = "hard_all_maps"
maps = [1, 2, 3, 4]
npc_count = 1
station_count = 4
total_steps = 10_000_000_000

[training]
artifacts_dir = "artifacts"
eval_episodes = 10

[dashboard]
enabled = true
host = "0.0.0.0"
port = 8088

[metrics]
max_updates = 500
max_episodes = 500
```

### 多地图轮换

训练自动轮换 `[env]` 中指定的多张地图：

- **`round_robin`** — 按顺序轮换，每结束一个 episode 切换地图
- **`random`** — 每次随机选择一张地图

### 课程学习

按难度递进分阶段训练：

| 阶段 | 名称 | 地图 | NPC | 充电桩 | 累计步数阈值 |
|------|------|------|-----|--------|-------------|
| 1 | easy_basic | map_1 | 0 | 1 | 300,000 |
| 2 | medium_npc_maze | map_1, map_2 | 1 | 2 | 1,000,000 |
| 3 | hard_all_maps | 全部 | 1 | 4 | 10,000,000,000 |

`total_steps` 为**累计**步数阈值，到达后自动进入下一阶段。设置 `enabled = false` 关闭课程学习。

### 训练产物

```
artifacts/
  └─ multi_map/
       └─ checkpoints/
            └─ 20260429_135229/          ← 一次训练 run
                 ├─ run_info.json          ← 训练元信息（种子 + 超参 + Git 信息）
                 ├─ train_config.toml      ← 配置文件副本
                 ├─ train.log              ← 训练日志
                 ├─ checkpoint_2000.pt     ← 按 save_interval 保存
                 ├─ checkpoint_4000.pt
                 ├─ ...
                 └─ eval_2000/             ← 评估产物（按步数命名）
                      ├─ map_1/
                      │    ├─ ep01_score_15_step_234.gif
                      │    ├─ ep01_score_15_step_234.log
                      │    ├─ ...
                      │    └─ summary.txt
                      ├─ map_2/
                      ├─ ...
                      └─ summary_all.txt   ← 全局汇总
```

### 断点续训

Checkpoint 保存了完整训练状态：

- 模型参数（model_state_dict）
- 优化器状态（optimizer_state_dict）
- 训练进度（global_step, episode_counter）
- 地图轮换位置（current_map_idx, current_map_id）
- 课程阶段（current_stage_name）
- 配置快照（config_snapshot）
- 随机数生成器状态（Python / NumPy / Torch RNG）

支持直接传递 checkpoint 文件或 run 目录路径，自动解析。

---

## 实时训练监控 (Dashboard)

训练时自动启动 Web 监控面板，访问 **`http://localhost:8088`**：

| 图表 | 说明 |
|------|------|
| Policy Loss | Actor 策略损失曲线 |
| Value Loss | Critic 价值损失曲线 |
| Entropy | 策略熵值曲线 |
| Total Loss | Policy + Value 合并损失 |
| Episode Reward & EMA Cleaned | 每 episode 累计奖励 + 清洁分 EMA |
| Episode 指标 | Cleaned / Steps 双轴图 |
| Update Mean Reward | 每次 PPO update 的平均奖励 |
| 实时日志流 | 彩色编码的事件日志（episode / update / 阶段切换） |

配置：

```toml
[dashboard]
enabled = true
host = "0.0.0.0"
port = 8088
```

Dashboard 运行在后台线程中，不影响训练速度。

---

## 评估

### 启动评估

```bash
# 使用默认测试配置
python scripts/eval.py

# 指定测试配置文件
python scripts/eval.py --config configs/test_config.toml
```

### 测试配置 (`configs/test_config.toml`)

```toml
[test]
maps = [1, 2, 3, 4]    # 要测评的地图 ID 列表
episodes = 10           # 每张地图测评的 episode 数量
npc_count = 1           # NPC 数量（空 = 沿用 train_config 默认值）
station_count = 4       # 充电桩数量
run_id = ""             # 空字符串 = 自动选最新 run
step = 0                # 0 = 自动选最新 checkpoint
gif_fps = 10            # GIF 导出帧率
output_dir = ""         # 空字符串 = 自动生成到 run 目录下
```

### 评估产物

```
eval_2000/
├── map_1/
│   ├── ep01_score_15_step_234.gif   # 轨迹动画
│   ├── ep01_score_15_step_234.log   # 轨迹日志
│   ├── ...
│   └── summary.txt                  # 单图统计
├── map_2/
├── ...
└── summary_all.txt                  # 全局汇总（多图对比表）
```

示例汇总输出：

```
=== Per-Map Results ===
map            ep  avg_reward  avg_score  avg_steps  avg_charges
----------------------------------------------------------------
map_1          10       42.50       18.0      189.0         0.20
map_2          10       15.30        8.0      450.0         1.50
map_3          10       28.10       15.0      310.0         0.80
map_4          10       52.00       22.0      220.0         0.00
----------------------------------------------------------------
OVERALL        40       34.47       15.8      292.3         0.62
```

---

## 添加 / 修改地图

### 目录规则

```
configs/maps/map_N.json       ← 运行时数据（训练/测试只读此 JSON）
configs/maps/src/map_N.py     ← 生成源码（含 build 函数，不参与运行）
```

**训练只读 JSON**，修改 `.py` 后必须重新导出才能生效。

### 工作流

```
1. 编辑 configs/maps/src/map_N.py
        ↓
2. 运行 python scripts/export_maps_to_json.py     ← 从 .py → .json
        ↓
3. 运行 python scripts/validate_maps_json.py     ← 校验 .json 完整性
        ↓
4. 在 train_config.toml 中加入新 ID，开始训练
```

### 内置地图

| ID | 地图名 | 源文件 | 特点 |
| -- | ------ | ------ | ---- |
| 1 | simple | `configs/maps/src/map_1.py` | 128×128 方形，边界障碍，内部全脏 |
| 2 | maze | `configs/maps/src/map_2.py` | 128×128 迷宫走廊，狭窄通道 |
| 3 | rooms | `configs/maps/src/map_3.py` | 128×128 四房间+门洞连接 |
| 4 | open_scattered | `configs/maps/src/map_4.py` | 128×128 无边界墙，散布随机障碍块 |

---

## 网络架构

双流 Actor-Critic 网络：

```
map_img (B, 9, 21, 21)
  └─ CNN 编码器（4 层 Conv + GroupNorm + SiLU）
       └─ AdaptiveAvgPool2d(3,3) → Flatten → FC(256)
                                            ─┐
vector_data (B, 10)                           │
  └─ MLP 编码器（2 层 FC + LayerNorm + SiLU） → ┼─ Fusion(256+64→256)
                            ──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           Policy Trunk (128→64)         Value Trunk (128→64)
                    │                              │
                    ▼                              ▼
           ActorHead (64→8)              CriticHead (64→1)
                    │                              │
                    ▼                              ▼
              action_logits                    value
```

- 地图视角大小：21×21（agent 周围 10 格半径）
- 地图通道（9 层）：静态障碍物 / NPC 热度图 / NPC 距离场 / 脏污掩码 / 脏污距离场 / 自标记 / 充电桩位置 / 充电桩距离场 / 已清洁掩码
- 向量特征（10 维）：电池电量 / 清洁进度 / 充电桩归一化距离 / 充电桩方向(x,z) / 脏污归一化距离 / 脏污方向(x,z) / 充电桩可达性 / 充电桩可见性

---

## 奖励函数

多目标复合奖励：

| 分量 | 说明 |
|------|------|
| Cleaning reward | +0.05 / 格（每次清扫脏污） |
| Step penalty | -0.02 / 步 |
| NPC penalty | -0.25（3×3 内）/ -0.05（5×5 内） |
| Charge potential | 引导低电量时靠近充电桩 |
| Full charge bonus | 低电量→充满时 +2.0 |
| Intrinsic reward | η / √(visit_count) 探索奖励 |
| Density reward | 鼓励停留在脏污密集区域 |
| Repeat visit penalty | 重复踩点惩罚 |
| Action consistency | 连续相同动作 +0.01 |

---

## 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_checkpoint_service.py
python -m pytest tests/test_map_loader.py
python -m pytest tests/test_runtime_config.py
```
