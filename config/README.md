# 环境配置说明

## 配置文件总览

环境配置由两层组成：

| 层级   | 文件                     | 控制内容                                          |
| ---- | ---------------------- | --------------------------------------------- |
| 训练入口 | `train_config.toml`    | `[ppo]` 超参 + `[env]` 选择地图 + `[training]` 产物路径 |
| 地图定义 | `simple_map_config.py` | 地图网格 + Agent/NPC/充电桩位置池                       |

***

## 自定义地图

在 `config/` 下新建一个 Python 文件（如 `my_map_config.py`），定义一个字典常量：

```python
MY_MAP_CONFIG = {
    "size": (128,128),                          # 必填：地图尺寸 (宽, 高)
    "custom_map": my_map,                      # 可选：numpy int8 数组，0=障碍 1=干净 2=脏

    "agent_spawn_pool": [                      # 必填：Agent 出生位置池
        (32, 1),                               #        坐标格式 (x, z)
    ],
    "agent_spawn_mode": 0,                     # 可选：-1=每局随机一个位置 / >=0=锁定池索引

    "npc_spawn_pool": [                        # 可选：NPC 出生位置池
        (32, 32),
        (48, 48),
    ],
    "npc_count": 2,                            # 可选：实际 NPC 数量 (<= pool 长度)
    "npc_spawn_modes": [-1, -1],              # 可选：每个 NPC 的分配模式，长度==npc_count

    "station_pool": [                          # 可选：充电桩位置池
        {"x": 1,  "z": 1,  "dx": 3, "dz": 3},
        {"x": 60, "z": 1,  "dx": 3, "dz": 3},
    ],
    "station_count": 2,                        # 可选：实际充电桩数量
    "station_mode": -1,                        # 可选：-1=随机选 / [0,1]=锁定编号

    "max_battery": 200,                        # Agent 最大电量
    "max_steps": 500,                          # 单局最大步数
    "hero_id": 37,                             # Agent 编号
    "npc_ids": None,                           # 可选：NPC 编号列表 (None 自动分配)
    "map_id": 2,                               # 地图编号
    "local_view_size": 21,                     # 局部视野大小
}
```

### 分配模式说明

`*_mode` 字段控制每局如何从池中选取位置：

| 值                 | 含义                                               |
| ----------------- | ------------------------------------------------ |
| `-1`              | 每局随机选（使用 Gymnasium 的确定性 RNG，seed 可复现）            |
| `0 / 1 / 2 / ...` | 固定选取池中对应索引的位置                                    |
| `[0, 2]`（列表）      | 仅 `station_mode` / `npc_spawn_modes` 支持，指定多个固定索引 |

### 地图网格约定

`custom_map` 的每个单元格取值为 `np.int8`：

| 值   | 含义          |
| --- | ----------- |
| `0` | 障碍物（不可通行）   |
| `1` | 干净地面        |
| `2` | 脏地面（清扫后变 1） |

<br />

如果未提供 `custom_map`，将使用缺省值（全是 1 的干净地面）。

***

## 注册到 train\_config.toml

在 `train_config.toml` `[env]` section 中指定地图名，并可选覆盖 `npc_count` / `station_count`：

```toml
[env]
map = "my_map"           # 对应 my_map_config.py 中的 MY_MAP_CONFIG
npc_count = 3            # 可覆盖地图配置中的 npc_count
station_count = 2        # 可覆盖地图配置中的 station_count
```

然后在 `train_sync.py` 的 `load_env_config()` 中注册新地图：

```python
if map_name == "simple":
    from config.simple_map_config import SIMPLE_MAP_CONFIG
    kwargs = dict(SIMPLE_MAP_CONFIG)
elif map_name == "my_map":
    from config.my_map_config import MY_MAP_CONFIG
    kwargs = dict(MY_MAP_CONFIG)
```

***

## GridWorldEnv 构造函数参数完整列表

地图配置文件中的所有 key 会被 `**kwargs` 解包传给 `GridWorldEnv()`。各参数的缺省值如下：

| 参数                 | 类型                     | 缺省           |
| ------------------ | ---------------------- | ------------ |
| `size`             | `(int, int)`           | `(128, 128)` |
| `custom_map`       | `np.ndarray`           | `None`       |
| `agent_spawn_pool` | `list[tuple[int,int]]` | `None`       |
| `agent_spawn_mode` | `int`                  | `-1`         |
| `npc_spawn_pool`   | `list[tuple[int,int]]` | `None`       |
| `npc_count`        | `int`                  | `1`          |
| `npc_spawn_modes`  | `list[int]`            | `None`       |
| `station_pool`     | `list[dict]`           | `None`       |
| `station_count`    | `int`                  | `4`          |
| `station_mode`     | `int \| list[int]`     | `-1`         |
| `npc_walk_radius`  | `int`                  | `10`         |
| `max_battery`      | `int`                  | `100`        |
| `max_steps`        | `int`                  | `1000`       |
| `hero_id`          | `int`                  | `37`         |
| `map_id`           | `int`                  | `0`          |
| `local_view_size`  | `int`                  | `21`         |

***

## 地图编辑器 (map_editor.py)

图形化的地图绘制工具，用鼠标绘制网格地图并导出配置。

### 启动

```bash
cd vacumRobot
python config/map_editor.py            # 默认 128×128
python config/map_editor.py 64         # 指定尺寸 64×64
```

首屏上方标题栏会显示当前画笔类型、放置模式、实体数量等信息。

### 鼠标操作

| 操作         | 功能                     |
| ---------- | ---------------------- |
| 左键拖拽       | 涂色（当前选中类型）             |
| 右键点击       | 放置选中实体（Agent / NPC / 充电桩） |
| 滚轮         | 缩放地图                   |
| 中键拖拽       | 平移视野                   |
| 鼠标悬停       | 黄色高亮当前格，标题栏显示坐标及类型    |

### 键盘操作

| 按键         | 功能                                  |
| ---------- | ----------------------------------- |
| `0`        | 画笔切换为 **障碍** (黑色)                   |
| `1`        | 画笔切换为 **干净** (白色)                   |
| `2`        | 画笔切换为 **脏地面** (金色)                  |
| `A`        | 放置模式切换为 **Agent 出生点**（右键放置，绿色圆）     |
| `N`        | 放置模式切换为 **NPC 出生点**（右键放置，红色圆）       |
| `S`        | 放置模式切换为 **充电桩**（右键放置，蓝色矩形，默认 3×3）  |
| `D`        | 删除光标最近实体（Agent / NPC / 充电桩）         |
| `C`        | 清除所有出生点和充电桩                         |
| `G`        | 切换网格显隐                              |
| `R`        | 重置视野到全局                             |
| `+` / `-` | 放大 / 缩小                             |
| `L`        | **坐标线染色**：输入起止坐标，同行或同列批量填充         |
| `Ctrl+Z`   | **撤销**上一次染色操作（最多 50 步）              |
| `O`        | **载入已有地图**，继续编辑                     |
| `E`        | **导出**为 `config/<name>_map_config.py` |
| `Q` / Esc | 退出                                  |

### 坐标线染色 (L 键)

按 `L` 弹出对话框，输入起止坐标，格式非常灵活（仅需包含 4 个数字）：

```
10,5 30,5      # 第5行，第10~30列
50,10 50,60    # 第50列，第10~60行
4,1,1,1        # 同样合法
(4,1)-(1,1)    # 也合法
```

起止坐标必须**同行或同列**（至少共享 x 或 y），否则拒绝染色。输入非 4 个数字时直接拒绝，不会污染已有地图。

### 载入已有地图 (O 键)

按 `O` 弹出文件选择对话框，选一个 `*_map_config.py` 文件即可恢复：
- 自动解析 `build_*_map()` 恢复网格
- 自动解析 `*_MAP_CONFIG` 恢复所有出生点和充电桩
- 尺寸必须匹配（如用 128 编辑器只能载入 128 的地图）
- 载入后撤销栈清空、操作计数重置

### 自动存档与恢复

- 每完成 **10 步**染色操作（左键拖拽或 L 线染色），自动保存一份副本到 `config/.editor_autosave.npy`
- 编辑器**重新启动**时，若检测到自动存档，弹出对话框询问是否恢复，输入 `y` 即可恢复

### 导出格式

按 `E` 输入地图名后，生成 `config/<name>_map_config.py`，包含：
- `build_<name>_map()` — 网格构建函数
- `<NAME>_MAP_CONFIG` — 可直接被 `train_config.toml` 引用的完整配置字典

### 网格颜色对照

| 颜色   | 值   | 含义          |
| ---- | --- | ----------- |
| 🖤 黑 | `0` | 障碍物（不可通行）   |
| 🤍 白 | `1` | 干净地面        |
| 💛 金 | `2` | 脏地面（清扫后变 1） |

