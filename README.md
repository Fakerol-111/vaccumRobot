# 清扫机器人 PPO 训练

基于 PyTorch + Gymnasium 的 PPO 强化学习框架，支持**多地图轮换训练**和**课程学习**。

## 快速开始

```bash
pip install -r requirements.txt
python main.py          # 或 python train_sync.py
```

## 实时训练监控 (Dashboard)

训练时自动启动一个 Web 监控面板，可在浏览器中实时查看训练状态。

打开浏览器访问 **`http://localhost:8088`**，页面包含：

- **Loss 曲线**：Policy Loss / Value Loss / Entropy 随 PPO update 次数变化
- **Episode 指标**：每个 episode 的总 Reward、EMA 清洁分数、Cleaned 数、Steps 数
- **Update 指标**：每次 PPO update 的平均 Reward
- **实时日志流**：彩色编码的事件日志（episode / update / 阶段切换）

### 配置

在 `config/train_config.toml` 中：

```toml
[dashboard]
enabled = true          # 是否启用
host = "0.0.0.0"        # 监听地址
port = 8088             # 监听端口
```

训练启动后控制台会打印 `[Dashboard] Server started at http://0.0.0.0:8088`，直接在浏览器打开即可。Dashboard 运行在后台线程中，不影响训练速度。

---

## 多地图训练

训练自动从 `config/train_config.toml` 读取配置，默认使用 4 张不同风格的地图轮换训练。

### 训练配置 (`config/train_config.toml`)

```toml
[env]
default_map_list = [1, 2, 3, 4]    # 默认地图 ID 列表
default_npc_count = 1              # 默认 NPC 数量
default_station_count = 4          # 默认充电桩数量
map_strategy = "round_robin"       # 地图轮换策略：round_robin | random

[general]
seed = 42                          # 全局随机种子，保证训练可复现
```

训练启动时统一设置 `random` / `numpy` / `torch` 种子。每个 episode 通过 `base_seed + episode_index` 规则计算子种子传入环境，确保相同配置重新运行可获得一致的训练轨迹。种子和完整超参会同时写入 `run_meta.json`。

### 课程学习

通过 `[curriculum]` 段可以按难度递进分阶段训练，让模型逐步掌握技能：

```toml
[curriculum]
enabled = true

[[curriculum.stage]]
name = "easy_basic"                # Stage 1: 单地图无 NPC，学会清扫+回充
maps = [1]
npc_count = 0
station_count = 1
total_steps = 300_000

[[curriculum.stage]]
name = "medium_npc_maze"           # Stage 2: 加入迷宫+NPC，学会避让
maps = [1, 2]
npc_count = 1
station_count = 2
total_steps = 1_000_000

[[curriculum.stage]]
name = "hard_all_maps"             # Stage 3: 全地图泛化
maps = [1, 2, 3, 4]
npc_count = 1
station_count = 4
total_steps = 10_000_000_000
```

- `total_steps` 是**累计**步数阈值，到达后自动切换下一阶段
- `maps` 为当前阶段可选的地图 ID 列表
- 设为 `enabled = false` 可关闭课程学习，始终使用 `[env]` 中的默认参数

### 内置地图

| ID | 地图名 | 文件 | 特点 |
| -- | ---- | --- | --- |
| 1 | simple | `config/map_1.py` | 128×128 方形，边界障碍，内部全脏 |
| 2 | maze | `config/map_2.py` | 128×128 迷宫走廊，狭窄通道 |
| 3 | rooms | `config/map_3.py` | 128×128 四房间+门洞连接 |
| 4 | open_scattered | `config/map_4.py` | 128×128 无边界墙，散布随机障碍块 |

### 训练产物

所有产物输出到 `artifacts/multi_map/checkpoints/<run_id>/`：

```
artifacts/
  └─ multi_map/
       └─ checkpoints/
            └─ 20260429_135229/         ← 一次训练 run
                 ├─ run_meta.json         ← 训练元信息 (seed + 超参)
                 ├─ checkpoint_2000.pt   ← 每 save_interval 步保存
                 ├─ checkpoint_4000.pt
                 ├─ train.log            ← 训练日志
                 │
                 └─ eval_2000/           ← 测试产物 (按步数命名)
                      ├─ map_1/          ← 每张地图单独目录
                      │    ├─ ep01_score_15_step_234.gif
                      │    ├─ ep01_score_15_step_234.log
                      │    ├─ ...
                      │    └─ summary.txt
                      ├─ map_2/
                      │    ├─ ep01_score_8_step_456.gif
                      │    └─ summary.txt
                      ├─ ...
                      └─ summary_all.txt ← 全局汇总
```

---

## 测试模型

训练过程中断点自动保存。随时可以运行 `test_model.py` 加载断点在多个地图上评估。

### 一键运行

```bash
# 使用默认测试配置 config/test_config.toml
python test_model.py

# 使用自定义测试配置
python test_model.py --config config/my_test.toml
```

### 测试配置 (`config/test_config.toml`)

所有评估参数集中在这个文件中，无需命令行传参：

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

### 测试产物

- 每个地图的 episode 轨迹 GIF 保存在 `eval_<step>/<map_name>/` 目录下
- 每个地图单独生成 `summary.txt` 统计（reward / score / steps / charges）
- 根目录生成 `summary_all.txt` 全局汇总，包含：
  - Per-Map Averages 表格（横向对比四张地图的得分）
  - Overall Averages（全部地图的加权平均值）

示例输出：

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

## 添加新地图

1. 新建 `config/map_N.py`（N 为下一个数字 ID），遵循约定格式：

```python
MAP_ID = 5

def build_my_map(size: int = 128) -> np.ndarray:
    grid = np.full((size, size), 2, dtype=np.int8)
    # ... 构建地图逻辑 ...
    return grid

MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_my_map(128),
    "agent_spawn_pool": [(2, 64), ...],
    "npc_spawn_pool": [(64, 32), ...],
    "station_pool": [{"x": 2, "z": 2, "dx": 3, "dz": 3}, ...],
    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": MAP_ID,
    "local_view_size": 21,
}
```

2. 在 `train_config.toml` 的 `default_map_list` 和/或课程阶段中加入新 ID 即可自动加载。

**无需修改任何注册代码** — `config/map_loader.py` 会通过 `MAP_ID` 自动发现和导入。

---

## 自定义配置

通过 `--config` 参数可以指定任意训练或测试配置文件：

```bash
# 训练 (train_sync.py 支持，main.py 透传)
python train_sync.py path/to/another.toml

# 测试 (只支持 --config 参数)
python test_model.py --config config/my_test.toml
```

所有配置项说明详见 [config/README.md](config/README.md)。
