# 环境配置说明

## 目录结构

```
configs/
├── maps/                          # ★ 地图数据（运行时）
│   ├── schema.json                #   JSON Schema 定义
│   ├── map_1.json                 #   地图 1 - 纯正方形
│   ├── map_2.json                 #   地图 2 - 迷宫
│   ├── map_3.json                 #   地图 3 - 四房间
│   ├── map_4.json                 #   地图 4 - 开放散落
│   └── src/                       # ★ 地图生成源码（不参与运行）
│       ├── map_1.py
│       ├── map_2.py
│       ├── map_3.py
│       └── map_4.py
├── train_config.toml              # 训练入口配置
├── test_config.toml               # 测试入口配置
├── map_loader.py                  # 运行时加载器（只读 JSON）
├── map_editor.py                  # 图形化地图编辑器
└── README.md
```

| 路径 | 角色 | 说明 |
| --- | --- | --- |
| `configs/maps/map_N.json` | **运行时数据** | 训练/测试只读此 JSON，不碰 .py |
| `configs/maps/src/map_N.py` | **生成源码** | 含 build 函数和 MAP_CONFIG，导出后才生效 |
| `configs/map_loader.py` | **加载器** | `load_map_config(id)` → 读 JSON → 返回 dict |

> **⚠️ 核心规则**：训练只读 JSON，`configs/maps/src/` 中的 `.py` 不参与运行时加载。

---

## 修改地图的工作流

```
修改 configs/maps/src/map_N.py
        ↓
运行 scripts/export_maps_to_json.py    ← 从 .py 导出到 .json
        ↓
运行 scripts/validate_maps_json.py    ← 校验 .json 完整性
        ↓
开始训练                               ← 通过 map_loader 读 .json
```

### 添加新地图

1. 新建 `configs/maps/src/map_N.py`，遵循地图约定格式（见下方）
2. 运行 `python scripts/export_maps_to_json.py`
3. 运行 `python scripts/validate_maps_json.py`
4. 在 `train_config.toml` 的 `default_map_list` 或课程阶段中加入新 ID

**无需修改 map_loader.py 或任何注册代码**。

---

## 内置地图

| ID | 地图名 | 源文件 | 特点 |
| -- | ---- | --- | --- |
| 1 | simple | `maps/src/map_1.py` | 128×128 方形，边界障碍，内部全脏 |
| 2 | maze | `maps/src/map_2.py` | 128×128 迷宫走廊，狭窄通道 |
| 3 | rooms | `maps/src/map_3.py` | 128×128 四房间+门洞连接 |
| 4 | open_scattered | `maps/src/map_4.py` | 128×128 无边界墙，散布随机障碍块 |

---

## 自定义地图（源文件格式）

创建 `configs/maps/src/map_N.py`（N 为任意整数 ID），约定格式：

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

### `[metrics]` — 滚动窗口大小

```toml
[metrics]
max_updates = 500       # 参与统计的 PPO update 最大记录数
max_episodes = 500      # 参与统计的 episode 最大记录数
```

超出上限后旧数据自动丢弃，仅保留最近条目的统计结果。

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
python test_model.py --config configs/my_test.toml      # 自定配置
```

---

## 地图加载器 (`map_loader.py`)

运行时只读 `configs/maps/map_N.json`，不碰 `.py` 源文件：

```python
from configs.map_loader import load_map_config, load_map_configs

cfg = load_map_config(1)              # 读取 configs/maps/map_1.json
cfgs = load_map_configs([1, 2, 3])    # 批量读取
```

加载时自动校验：
- `schema_version` 是否为 1
- 文件名中的 map_id 是否与文件内容一致
- `custom_map` 行列数是否与 `size` 匹配
- 每个字符是否仅为 `0` / `1` / `2`

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
python configs/map_editor.py            # 默认 128×128
python configs/map_editor.py 64         # 指定尺寸 64×64
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
| `E` | **导出**为 `configs/<name>_map_config.py` |
| `Q` / Esc | 退出 |

### 导出手动迁移

编辑器导出的格式为 `<name>_map_config.py`，需手动处理：

1. 复制到 `configs/maps/src/map_N.py`
2. 添加 `MAP_ID = N` 头部
3. 确保 `MAP_CONFIG` 中的 `map_id` 引用 `MAP_ID`
4. 运行 `python scripts/export_maps_to_json.py`
5. 运行 `python scripts/validate_maps_json.py`

### 网格颜色对照

| 颜色 | 值 | 含义 |
| --- | --- | --- |
| 🖤 黑 | `0` | 障碍物（不可通行） |
| 🤍 白 | `1` | 干净地面 |
| 💛 金 | `2` | 脏地面（清扫后变 1） |
