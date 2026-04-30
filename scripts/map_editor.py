"""地图编辑器 —— 用鼠标绘制 GridWorld 地图，导出配置。

操作说明：
    鼠标左键拖拽  — 涂色（当前选中类型）
    鼠标右键点击  — 放置选中实体（Agent / NPC / 充电桩）
    鼠标滚轮     — 缩放地图
    鼠标中键拖拽 — 平移视野
    键盘 0/1/2  — 切换涂色类型 (障碍/干净/脏)
    键盘 A      — 选中 Agent 出生模式，右键添加
    键盘 N      — 选中 NPC 出生模式，右键添加
    键盘 S      — 选中充电桩模式，右键添加
    键盘 D      — 删除鼠标位置最近的实体
    键盘 C      — 清除所有出生点和充电桩
    键盘 G      — 切换网格显隐
    键盘 R      — 重置视野到全局
    键盘 +/-    — 缩放
    键盘 E      — 导出为 configs/<name>_map_config.py
    键盘 L      — 坐标线染色，输入起止坐标批量填充
    键盘 Ctrl+Z — 撤销上一次染色操作
    键盘 O      — 载入已有地图配置继续编辑
    Q / Esc    — 退出
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path
from tkinter import filedialog, simpledialog
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CMAP = mcolors.ListedColormap(["#475569", "#E5E7EB", "#F59E0B", "#DC2626", "#7C3AED", "#2563EB"])

BLOCKED, CLEAN, DIRTY = 0, 1, 2


class MapEditor:
    def __init__(self, size: tuple[int, int] = (128, 128)):
        self.size = size
        self.grid = np.full((size[0], size[1]), DIRTY, dtype=np.int8)
        self._init_boundaries()

        self._paint_value = 0
        self._place_mode = "agent"

        self._agent_spawns: list[tuple[int, int]] = []
        self._npc_spawns: list[tuple[int, int]] = []
        self._stations: list[tuple[int, int, int, int]] = []

        self._dragging = False
        self._panning = False
        self._show_grid = True
        self._hover_col = -1
        self._hover_row = -1
        self._last_hover_draw = 0.0
        self._last_paint_col = -1
        self._last_paint_row = -1
        self._last_paint_draw = 0.0

        self._undo_stack: list[np.ndarray] = []
        self._MAX_UNDO = 50
        self._op_count = 0
        self._AUTO_CHECKPOINT = 10
        self._autosave_path = PROJECT_ROOT / "configs" / ".editor_autosave.npy"

        self._fig, self._ax = plt.subplots(figsize=(9, 9))
        self._fig.patch.set_facecolor("#F1F5F9")
        self._ax.set_facecolor("#E2E8F0")
        self._fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)
        self._img = self._ax.imshow(
            self.grid, cmap=CMAP, vmin=0, vmax=5, origin="upper", interpolation="nearest",
        )
        self._overlay: list[Any] = []

        self._highlight = plt.Rectangle(
            (0, 0), 1, 1, fill=True, facecolor="#0EA5E9", alpha=0.30, edgecolor="#0EA5E9",
            linewidth=2.5, zorder=20, visible=False,
        )
        self._ax.add_patch(self._highlight)

        self._MIN_VIEW = 8
        self._build_grid_lines()
        self._reset_view()
        self._setup_events()
        self._refresh()
        self._try_load_autosave()

    def _init_boundaries(self) -> None:
        h, w = self.size
        self.grid[0, :] = BLOCKED
        self.grid[-1, :] = BLOCKED
        self.grid[:, 0] = BLOCKED
        self.grid[:, -1] = BLOCKED

    # ---------- grid ----------

    def _build_grid_lines(self) -> None:
        h, w = self.size
        self._ax.set_xticks(np.arange(-0.5, w + 0.5, 1), minor=True)
        self._ax.set_yticks(np.arange(-0.5, h + 0.5, 1), minor=True)
        self._ax.set_xticks(np.arange(0, w, max(1, w // 16)))
        self._ax.set_yticks(np.arange(0, h, max(1, h // 16)))
        self._ax.tick_params(which="both", labelsize=7)
        self._ax.grid(True, which="minor", color="#CBD5E1", linewidth=0.3, alpha=0.85)
        self._ax.grid(True, which="major", color="#CBD5E1", linewidth=0.8, alpha=0.95)

    def _toggle_grid(self) -> None:
        self._show_grid = not self._show_grid
        self._ax.grid(self._show_grid, which="minor", color="#CBD5E1", linewidth=0.3, alpha=0.85)
        self._ax.grid(self._show_grid, which="major", color="#CBD5E1", linewidth=0.8, alpha=0.95)
        self._fig.canvas.draw_idle()

    # ---------- events ----------

    def _setup_events(self) -> None:
        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_press(self, event: MouseEvent) -> None:
        if event.inaxes != self._ax:
            return
        col, row = int(event.xdata), int(event.ydata)

        if event.button == MouseButton.LEFT:
            if not self._in_bounds(col, row):
                return
            self._save_undo()
            self._dragging = True
            self._last_paint_col, self._last_paint_row = -1, -1
            self._paint_cell(col, row)
            self._img.set_data(self.grid)
            self._fig.canvas.draw_idle()
        elif event.button == MouseButton.RIGHT:
            if not self._in_bounds(col, row):
                return
            self._place_entity(col, row)
        elif event.button == MouseButton.MIDDLE:
            self._panning = True
            self._pan_x = event.xdata
            self._pan_y = event.ydata

    def _on_release(self, event: MouseEvent) -> None:
        if event.button == MouseButton.LEFT:
            self._dragging = False
            self._img.set_data(self.grid)
            self._fig.canvas.draw_idle()
        elif event.button == MouseButton.MIDDLE:
            self._panning = False

    def _on_motion(self, event: MouseEvent) -> None:
        if self._dragging:
            if event.inaxes == self._ax:
                col, row = int(event.xdata), int(event.ydata)
                if self._in_bounds(col, row):
                    if col == self._last_paint_col and row == self._last_paint_row:
                        return
                    self._paint_cell(col, row)
                    self._last_paint_col, self._last_paint_row = col, row
                    now = time.perf_counter()
                    if now - self._last_paint_draw < 0.03:
                        return
                    self._last_paint_draw = now
                    self._img.set_data(self.grid)
                    self._fig.canvas.draw_idle()
            return
        elif self._panning:
            if event.inaxes == self._ax:
                dx = event.xdata - self._pan_x
                dy = event.ydata - self._pan_y
                xlim = list(self._ax.get_xlim())
                ylim = list(self._ax.get_ylim())
                self._ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
                self._ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
                self._fig.canvas.draw_idle()
                self._pan_x = event.xdata
                self._pan_y = event.ydata
            return

        # Hover highlight
        if event.inaxes == self._ax:
            col, row = int(event.xdata), int(event.ydata)
            if self._in_bounds(col, row):
                if col == self._hover_col and row == self._hover_row:
                    return
                now = time.perf_counter()
                if now - self._last_hover_draw < 0.03:
                    return
                self._highlight.set_xy((col - 0.5, row - 0.5))
                self._highlight.set_visible(True)
                self._hover_col, self._hover_row = col, row
            else:
                if self._hover_col == -1 and self._hover_row == -1:
                    return
                self._highlight.set_visible(False)
                self._hover_col, self._hover_row = -1, -1
        else:
            if self._hover_col == -1 and self._hover_row == -1:
                return
            self._highlight.set_visible(False)
            self._hover_col, self._hover_row = -1, -1
        self._last_hover_draw = time.perf_counter()
        self._fig.canvas.draw_idle()

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes != self._ax:
            return
        scale = 1.0 / 1.3 if event.button == "up" else 1.3
        self._zoom(scale, event.xdata, event.ydata)

    def _on_key(self, event) -> None:
        k = event.key
        if k in ("0", "1", "2"):
            self._paint_value = int(k)
            self._update_title()
        elif k == "a":
            self._place_mode = "agent"
        elif k == "n":
            self._place_mode = "npc"
        elif k == "s":
            self._place_mode = "station"
        elif k == "d":
            self._delete_under_cursor(event)
        elif k == "c":
            self._agent_spawns.clear()
            self._npc_spawns.clear()
            self._stations.clear()
            self._refresh()
        elif k == "g":
            self._toggle_grid()
        elif k == "r":
            self._reset_view()
        elif k in ("+", "="):
            self._zoom(1.0 / 1.3)
        elif k == "-":
            self._zoom(1.3)
        elif k == "e":
            self._export()
        elif k == "l":
            self._paint_line()
        elif k == "ctrl+z":
            self._undo()
        elif k == "o":
            self._load_map()
        elif k in ("q", "escape"):
            plt.close(self._fig)
            return
        self._update_title()

    # ---------- zoom ----------

    def _zoom(self, scale: float, cx: float | None = None, cy: float | None = None) -> None:
        xlim = list(self._ax.get_xlim())
        ylim = list(self._ax.get_ylim())
        w, h = self.size

        if cx is None:
            cx = (xlim[0] + xlim[1]) / 2
        if cy is None:
            cy = (ylim[0] + ylim[1]) / 2

        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[0] - ylim[1]) * scale

        if scale < 1 and new_w < self._MIN_VIEW:
            return
        if scale > 1 and new_w > max(w + 40, self._MIN_VIEW * 2):
            return

        fx = (cx - xlim[0]) / max(xlim[1] - xlim[0], 1)
        fy = (ylim[0] - cy) / max(ylim[0] - ylim[1], 1)

        xlim[0] = cx - new_w * fx
        xlim[1] = cx + new_w * (1 - fx)
        ylim[0] = cy + new_h * fy
        ylim[1] = cy - new_h * (1 - fy)

        pad = 100
        xlim[0] = max(xlim[0], -pad)
        xlim[1] = min(xlim[1], w + pad)
        ylim[0] = min(ylim[0], h + pad)
        ylim[1] = max(ylim[1], -pad)

        if ylim[0] <= ylim[1]:
            return

        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._fig.canvas.draw_idle()

    def _reset_view(self) -> None:
        w, h = self.size
        self._ax.set_xlim(-2, w + 2)
        self._ax.set_ylim(h + 2, -2)
        self._fig.canvas.draw_idle()

    # ---------- paint ----------

    def _save_undo(self) -> None:
        if len(self._undo_stack) >= self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.grid.copy())
        self._op_count += 1
        if self._op_count % self._AUTO_CHECKPOINT == 0:
            self._save_checkpoint()

    def _undo(self) -> None:
        if not self._undo_stack:
            print("  没有可撤销的操作")
            return
        self.grid = self._undo_stack.pop()
        self._img.set_data(self.grid)
        self._fig.canvas.draw_idle()
        print(f"  已撤销 (剩余 {len(self._undo_stack)} 步)")

    def _save_checkpoint(self) -> None:
        np.save(str(self._autosave_path), self.grid)
        print(f"  [checkpoint #{self._op_count}] 自动存档 -> {self._autosave_path.name}")

    def _load_map(self) -> None:
        path_str = filedialog.askopenfilename(
            title="选择地图配置文件",
            initialdir=str(PROJECT_ROOT / "configs"),
            filetypes=[("Map Config", "*_map_config.py"), ("All", "*")],
        )
        if not path_str:
            print("  cancelled")
            return
        src = Path(path_str)
        if not src.exists():
            print(f"  文件不存在: {src}")
            return

        try:
            spec = importlib.util.spec_from_file_location("_loaded_map", src)
            if spec is None or spec.loader is None:
                print(f"  无法解析模块: {src}")
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  加载模块失败: {e}")
            return

        build_func = None
        for name in dir(module):
            if name.startswith("build_") and name.endswith("_map"):
                build_func = getattr(module, name)
                break
        if build_func is None:
            print("  未找到 build_*_map() 函数")
            return

        loaded = build_func()
        if not isinstance(loaded, np.ndarray):
            print(f"  build_*_map() 返回类型不是 ndarray: {type(loaded)}")
            return
        if loaded.shape != self.size:
            print(f"  地图尺寸不匹配: 文件={loaded.shape} vs 编辑器={self.size}，请用对应的尺寸启动")
            return

        self.grid = loaded.copy()
        self._agent_spawns.clear()
        self._npc_spawns.clear()
        self._stations.clear()
        self._undo_stack.clear()
        self._op_count = 0

        for attr in dir(module):
            if attr.endswith("_MAP_CONFIG"):
                cfg = getattr(module, attr)
                if isinstance(cfg, dict):
                    for x, z in cfg.get("agent_spawn_pool", []):
                        if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                            self._agent_spawns.append((int(x), int(z)))
                    for x, z in cfg.get("npc_spawn_pool", []):
                        if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                            self._npc_spawns.append((int(x), int(z)))
                    for s in cfg.get("station_pool", []):
                        if isinstance(s, dict):
                            self._stations.append((
                                int(s.get("x", 0)), int(s.get("z", 0)),
                                int(s.get("dx", 3)), int(s.get("dz", 3)),
                            ))
                break

        self._img.set_data(self.grid)
        self._reset_view()
        self._refresh()
        print(f"  已载入 {src.name} (op count 已重置)")

    def _try_load_autosave(self) -> bool:
        if not self._autosave_path.exists():
            return False
        choice = simpledialog.askstring(
            "恢复存档",
            f"检测到自动存档 ({self._autosave_path.name})，输入 y 恢复，其他键跳过:",
        )
        if not choice or choice.strip().lower() != "y":
            return False
        try:
            saved = np.load(str(self._autosave_path))
        except Exception as e:
            print(f"  读取自动存档失败: {e}")
            return False
        if not isinstance(saved, np.ndarray) or saved.shape != self.size:
            print(f"  存档尺寸不匹配: {saved.shape if isinstance(saved, np.ndarray) else 'N/A'} vs {self.size}")
            return False
        self.grid = saved.copy()
        self._undo_stack.clear()
        self._op_count = 0
        self._img.set_data(self.grid)
        self._reset_view()
        self._refresh()
        print("  已恢复自动存档")
        return True

    def _paint_cell(self, col: int, row: int) -> None:
        """只写入数据，不立即刷新画布（由 _on_motion / _on_release 统一刷新）"""
        self.grid[row, col] = self._paint_value

    def _paint_line(self) -> None:
        raw = simpledialog.askstring(
            "坐标线染色",
            "起止坐标 (格式: x1,y1 x2,y2  例: 10,5 30,5):",
        )
        if not raw:
            print("  cancelled")
            return
        nums = re.findall(r"-?\d+", raw)
        if len(nums) != 4:
            print(f"  需要恰好4个数字（x1,y1,x2,y2），实际收到 {len(nums)} 个: {nums}")
            return
        x1, y1, x2, y2 = map(int, nums)

        w, h = self.size[1], self.size[0]
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            print(f"  坐标越界，地图范围: (0~{w - 1}, 0~{h - 1})")
            return
        if x1 != x2 and y1 != y2:
            print(f"  起止坐标必须至少共享 x 或 y（同列或同行），目前 ({x1},{y1}) → ({x2},{y2})")
            return

        self._save_undo()
        if x1 == x2:
            rng = range(min(y1, y2), max(y1, y2) + 1)
            self.grid[rng, x1] = self._paint_value
        else:
            rng = range(min(x1, x2), max(x1, x2) + 1)
            self.grid[y1, rng] = self._paint_value
        self._img.set_data(self.grid)
        self._fig.canvas.draw_idle()
        paint_label = {0: "obstacle", 1: "clean", 2: "dirt"}[self._paint_value]
        print(f"  已用 [{paint_label}] 填充 ({x1},{y1}) → ({x2},{y2})")

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.size[1] and 0 <= row < self.size[0]

    # ---------- place ----------

    def _place_entity(self, col: int, row: int) -> None:
        if self._place_mode == "agent":
            self._agent_spawns.append((col, row))
        elif self._place_mode == "npc":
            self._npc_spawns.append((col, row))
        elif self._place_mode == "station":
            self._stations.append((col, row, 3, 3))
        self._refresh()

    def _delete_under_cursor(self, event) -> None:
        if event.inaxes != self._ax:
            return
        col, row = int(event.xdata), int(event.ydata)
        self._agent_spawns = [p for p in self._agent_spawns if max(abs(p[0] - col), abs(p[1] - row)) > 2]
        self._npc_spawns = [p for p in self._npc_spawns if max(abs(p[0] - col), abs(p[1] - row)) > 2]
        self._stations = [s for s in self._stations if max(abs(s[0] - col), abs(s[1] - row)) > 2]
        self._refresh()

    # ---------- render ----------

    def _refresh(self) -> None:
        for obj in self._overlay:
            obj.remove()
        self._overlay.clear()

        if self._agent_spawns:
            xs, zs = zip(*self._agent_spawns)
            objs = self._ax.scatter(xs, zs, c="#2563EB", s=100, marker="o", edgecolors="white",
                                    linewidths=1.5, zorder=10)
            self._overlay.append(objs)
        if self._npc_spawns:
            xs, zs = zip(*self._npc_spawns)
            objs = self._ax.scatter(xs, zs, c="#DC2626", s=80, marker="o", edgecolors="white",
                                    linewidths=1.5, zorder=10)
            self._overlay.append(objs)
        for sx, sz, dx, dz in self._stations:
            rect = plt.Rectangle((sx - 0.5, sz - 0.5), dx, dz, fill=True,
                                 facecolor="#7C3AED", alpha=0.20, zorder=4)
            self._ax.add_patch(rect)
            self._overlay.append(rect)
            rect2 = plt.Rectangle((sx - 0.5, sz - 0.5), dx, dz, fill=False,
                                  edgecolor="#7C3AED", linewidth=2, zorder=5)
            self._ax.add_patch(rect2)
            self._overlay.append(rect2)

        self._update_title()
        self._fig.canvas.draw_idle()

    def _update_title(self) -> None:
        place_label = {"agent": "Agent", "npc": "NPC", "station": "Station"}[self._place_mode]
        paint_label = {0: "obstacle", 1: "clean", 2: "dirt"}[self._paint_value]
        xl = self._ax.get_xlim()
        yl = self._ax.get_ylim()
        view_w = int(xl[1] - xl[0])
        view_h = int(yl[0] - yl[1])
        grid_status = "ON" if self._show_grid else "OFF"
        grid_val = self.grid[self._hover_row, self._hover_col] if self._in_bounds(self._hover_col, self._hover_row) else "-"
        grid_labels = {0: "obstacle", 1: "clean", 2: "dirty"}
        cell_info = f"({self._hover_col}, {self._hover_row})={grid_labels.get(grid_val, '-')}" \
            if self._in_bounds(self._hover_col, self._hover_row) else "(--, --)"
        title = (
            f"Paint: [{paint_label}]  |  Place: [{place_label}] (right-click)  |  "
            f"Agent:{len(self._agent_spawns)}  NPC:{len(self._npc_spawns)}  "
            f"Stations:{len(self._stations)}  |  View: {view_w}x{view_h}  Grid: {grid_status}  |  "
            f"Undo:{len(self._undo_stack)}  |  "
            f"Pos: {cell_info}"
        )
        self._ax.set_title(title, fontsize=11)

    # ---------- export ----------

    def _export(self) -> None:
        name = simpledialog.askstring("Map Export", "Map name (e.g. 'my_map'):")
        if not name:
            print("  cancelled")
            return
        name = name.strip()

        out = PROJECT_ROOT / "configs" / f"{name}_map_config.py"

        grid_lines = self._format_grid()
        agent_pool = [f"        ({x}, {z})," for x, z in self._agent_spawns]
        npc_pool = [f"        ({x}, {z})," for x, z in self._npc_spawns]
        station_pool = []
        seen = set()
        for sx, sz, dx, dz in self._stations:
            cfg = f'        {{"x": {sx}, "z": {sz}, "dx": {dx}, "dz": {dz}}},'
            if cfg not in seen:
                seen.add(cfg)
                station_pool.append(cfg)

        content = f'''from __future__ import annotations

import numpy as np


def build_{name}_map(size: tuple[int, int] = ({self.size[0]}, {self.size[1]})) -> np.ndarray:
    grid = np.array(
{chr(10).join(grid_lines)},
        dtype=np.int8,
    )
    return grid


{name.upper()}_MAP_CONFIG = {{
    "size": ({self.size[0]}, {self.size[1]}),
    "custom_map": build_{name}_map(),

    "agent_spawn_pool": [
{chr(10).join(agent_pool) if agent_pool else "        (32, 1),"}
    ],

    "npc_spawn_pool": [
{chr(10).join(npc_pool) if npc_pool else "        (32, 32),"}
    ],

    "station_pool": [
{chr(10).join(station_pool) if station_pool else '        {{"x": 1, "z": 1, "dx": 3, "dz": 3}},'}
    ],

    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": 2,
    "local_view_size": 21,
}}
'''

        out.write_text(content, encoding="utf-8")
        print(f"  exported to {out}")

    def _format_grid(self) -> list[str]:
        lines = []
        for i, row in enumerate(self.grid):
            vals = ", ".join(str(x) for x in row)
            prefix = "        ["
            suffix = "]," if i < len(self.grid) - 1 else "]"
            lines.append(f"{prefix}{vals}{suffix}")
        return lines

    def run(self) -> None:
        plt.show()


def main():
    size = 128
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    editor = MapEditor(size=(size, size))
    editor.run()


if __name__ == "__main__":
    main()
