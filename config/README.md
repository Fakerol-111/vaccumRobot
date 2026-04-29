# 环境配置说明

## 配置文件总览

| 文件 | 用途 |
| --- | --- |
| `train_config.toml` | 训练入口：`[ppo]` 超参 + `[env]` 多地图参数 + `[curriculum]` 课程学习 + `[training]` 产物路径 + `[dashboard]` 监控面板 |
| `test_config.toml` | **测试入口**：多地图测评参数，独立于训练配置 |
| `map_1.py ~ map_N.py` | 地图定义（每文件一个地图）：网格构建函数 + 完整配置字典 |
| `map_loader.py` | 通用加载器：按数字 ID 自动导入 `map_N.py` |
| `map_editor.py` | 图形化地图编辑器，可导出 `map_N.py` 格式 |

---

## 内置地图

| ID | 地图名 | 文件 | 特点 |
| -- | ---- | --- | --- |
| 1 | simple | `map_1.py` | 128×128 方形，边界障碍，内部全脏 |
| 2 | maze | `map_2.py` | 128×128 迷宫走廊，狭窄通道 |
| 3 | rooms | `map_3.py` | 128×128 四房间+门洞连接 |
| 4 | open_scattered | `map_4.py` | 128×128 无边界墙，散布随机障碍块 |

---

## 自定义地图

创建 `config/map_N.py`（N 为任意整数 ID），约定格式：

```python
MAP_ID = 5                          # 必须：地图唯一 ID

def build_my_map(size: int = 128) -> np.ndarray:
    """构建网格数组（0=障碍 1=干净 2=脏）。"""
    grid = np.full((size, size), 2, dtype=np.int8)
    # ... 地图构建逻辑 ...
    return grid

MAP_CONFIG = {                      # 必须：完整配置字典
    "size": (128, 128),
    "custom_map": build_my_map(128),
    "agent_spawn_pool": [(2, 64), (125, 64)],   # Agent 出生位置池
    "npc_spawn_pool": [(64, 32), (32, 96)],     # NPC 出生位置池
    "station_pool": [                            # 充电桩位置池
        {"x": 2, "z": 2, "dx": 3, "dz": 3},
    ],
    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": MAP_ID,               # 必须引用顶部 MAP_ID
    "local_view_size": 21,
}
```

然后在 `train_config.toml` 的 `default_map_list` 或课程阶段中加入该 ID 即可生效，**无需修改任何注册代码**。

### 分配模式说明

`*_mode` 字段控制每局如何从池中选取位置：

| 值 | 含义 |
| --- | --- |
| `-1` | 每局随机选（使用 Gymnasium 的确定性 RNG，seed 可复现） |
| `0 / 1 / 2 / ...` | 固定选取池中对应索引的位置 |
| `[0, 2]`（列表） | 仅 `station_mode` / `npc_spawn_modes` 支持，指定多个固定索引 |

### 地图网格约定

| 值 | 含义 |
| --- | --- |
| `0` | 障碍物（不可通行） |
| `1` | 干净地面 |
| `2` | 脏地面（清扫后变 1） |

### 出生点安全要求

所有 `agent_spawn_pool` 和 `npc_spawn_pool` 中的坐标必须落在地图的可通行格（值 ≠ 0）上，否则环境初始化会报错。

---

## 训练配置 (`train_config.toml`)

### `[general]` — 全局参数

```toml
[general]
seed = 42               # 全局随机种子，保证训练可复现
```

训练启动时统一设置 `random` / `numpy` / `torch` 种子。每个 episode 的 `GridWorldEnv.reset(seed=episode_seed)` 使用 `base_seed + episode_index` 规则计算，确保按相同配置重新运行时能获得完全一致的训练轨迹。

### `[env]` — 多地图训练参数

```toml
[env]
default_map_list = [1, 2, 3, 4]    # 默认地图 ID 列表（课程关闭时使用）
default_npc_count = 1              # 默认 NPC 数量
default_station_count = 4          # 默认充电桩数量
map_strategy = "round_robin"       # map 轮换策略: round_robin | random
```

### `[curriculum]` — 课程学习

```toml
[curriculum]
enabled = true

[[curriculum.stage]]
name = "easy_basic"
maps = [1]
npc_count = 0
station_count = 1
total_steps = 300_000               # 累计步数阈值

[[curriculum.stage]]
name = "medium"
maps = [1, 2]
npc_count = 1
station_count = 2
total_steps = 1_000_000

[[curriculum.stage]]
name = "hard"
maps = [1, 2, 3, 4]
npc_count = 1
station_count = 4
total_steps = 10_000_000_000
```

- `total_steps` 为累计值，到达后自动进入下一阶段
- 每个阶段的 `maps` 列表中按 `map_strategy` 轮换
- 训练日志中会打印 `[Curriculum] >>> Entering stage: xxx` 标记阶段切换
- 设为 `enabled = false` 则始终使用 `[env]` 节参数

### `[training]` — 基础路径

```toml
[training]
artifacts_dir = "artifacts"
```

### `[dashboard]` — 实时训练监控

```toml
[dashboard]
enabled = true          # 是否启用 Web 监控面板
host = "0.0.0.0"        # HTTP 监听地址
port = 8088             # HTTP 监听端口
```

训练启动后，在浏览器访问 `http://localhost:8088` 即可实时查看 Loss 曲线、Episode 指标、Update 指标和事件日志流。设为 `enabled = false` 可关闭 Dashboard。

---

## 测试配置 (`test_config.toml`)

独立的测评配置文件，**所有参数通过此文件设置，无需命令行传参**：

```toml
[test]
maps = [1, 2, 3, 4]    # 要测评的地图 ID 列表
episodes = 10           # 每张地图测评的 episode 数
npc_count = 1           # NPC 数量（空 = 沿用 train_config 默认值）
station_count = 4       # 充电桩数量
run_id = ""             # 空 = 自动选最新 run
step = 0                # 0 = 自动选最新 checkpoint
gif_fps = 10            # GIF 导出帧率
output_dir = ""         # 空 = 自动生成到 run_dir/eval_<step>/
```

启动：

```bash
python test_model.py                                   # 默认配置
python test_model.py --config config/my_test.toml      # 自定配置
```

---

## 地图加载器 (`map_loader.py`)

自动加载模块，通过数字 ID 动态导入对应地图文件：

```python
from config.map_loader import load_map_config, load_map_configs

cfg = load_map_config(1)              # 加载 map_1.py
cfgs = load_map_configs([1, 2, 3])    # 批量加载
```

---

## GridWorldEnv 构造函数参数完整列表

地图配置文件中的所有 key 会被 `**kwargs` 解包传给 `GridWorldEnv()`。各参数的缺省值如下：

| 参数 | 类型 | 缺省 |
| --- | --- | --- |
| `size` | `(int, int)` | `(128, 128)` |
| `custom_map` | `np.ndarray` | `None` |
| `agent_spawn_pool` | `list[tuple[int,int]]` | `None` |
| `agent_spawn_mode` | `int` | `-1` |
| `npc_spawn_pool` | `list[tuple[int,int]]` | `None` |
| `npc_count` | `int` | `1` |
| `npc_spawn_modes` | `list[int]` | `None` |
| `station_pool` | `list[dict]` | `None` |
| `station_count` | `int` | `4` |
| `station_mode` | `int \| list[int]` | `-1` |
| `npc_walk_radius` | `int` | `10` |
| `max_battery` | `int` | `100` |
| `max_steps` | `int` | `1000` |
| `hero_id` | `int` | `37` |
| `map_id` | `int` | `0` |
| `local_view_size` | `int` | `21` |

---

## 地图编辑器 (map_editor.py)

图形化的地图绘制工具，用鼠标绘制网格地图并导出配置。

### 启动

```bash
python config/map_editor.py            # 默认 128×128
python config/map_editor.py 64         # 指定尺寸 64×64
```

首屏上方标题栏会显示当前画笔类型、放置模式、实体数量等信息。

### 鼠标操作

| 操作 | 功能 |
| --- | --- |
| 左键拖拽 | 涂色（当前选中类型） |
| 右键点击 | 放置选中实体（Agent / NPC / 充电桩） |
| 滚轮 | 缩放地图 |
| 中键拖拽 | 平移视野 |
| 鼠标悬停 | 黄色高亮当前格，标题栏显示坐标及类型 |

### 键盘操作

| 按键 | 功能 |
| --- | --- |
| `0` | 画笔切换为 **障碍** (黑色) |
| `1` | 画笔切换为 **干净** (白色) |
| `2` | 画笔切换为 **脏地面** (金色) |
| `A` | 放置模式切换为 **Agent 出生点**（右键放置，绿色圆） |
| `N` | 放置模式切换为 **NPC 出生点**（右键放置，红色圆） |
| `S` | 放置模式切换为 **充电桩**（右键放置，蓝色矩形，默认 3×3） |
| `D` | 删除光标最近实体（Agent / NPC / 充电桩） |
| `C` | 清除所有出生点和充电桩 |
| `G` | 切换网格显隐 |
| `R` | 重置视野到全局 |
| `+` / `-` | 放大 / 缩小 |
| `L` | **坐标线染色**：输入起止坐标，同行或同列批量填充 |
| `Ctrl+Z` | **撤销**上一次染色操作（最多 50 步） |
| `O` | **载入已有地图**，继续编辑 |
| `E` | **导出**为 `config/<name>_map_config.py` |
| `Q` / Esc | 退出 |

### 导出格式

按 `E` 输入地图名后，生成 `<name>_map_config.py`。导出后建议重命名为 `map_N.py` 并添加 `MAP_ID = N` 头，以适配自动加载器。

### 网格颜色对照

| 颜色 | 值 | 含义 |
| --- | --- | --- |
| 🖤 黑 | `0` | 障碍物（不可通行） |
| 🤍 白 | `1` | 干净地面 |
| 💛 金 | `2` | 脏地面（清扫后变 1） |
