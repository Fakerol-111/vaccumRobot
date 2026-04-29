#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np
from collections import deque


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _safe_dir(dx, dz):
    """Return normalized direction components (dir_x, dir_z).

    返回归一化方向分量（dir_x, dir_z）。
    """
    dist = float(np.sqrt(dx * dx + dz * dz))
    if dist <= 1e-6:
        return 0.0, 0.0
    return float(dx / dist), float(dz / dist)


def _inv_dist_field(target_mask, max_dist):
    """Compute inverse-normalized distance field: max(0, 1 - d/max_dist)."""
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)
    target_coords = np.argwhere(mask)
    grid_y, grid_x = np.indices(mask.shape, dtype=np.float32)
    dy = grid_y[..., None] - target_coords[:, 0].astype(np.float32)
    dx = grid_x[..., None] - target_coords[:, 1].astype(np.float32)
    dist = np.sqrt(dx * dx + dy * dy).min(axis=-1)
    field = 1.0 - dist / float(max_dist)
    return np.clip(field, 0.0, 1.0).astype(np.float32, copy=False)


def _global_points_to_local_mask(points_xz, center_x, center_z, half=10, size=21):
    """Project global points to local view mask with vectorized indexing."""
    mask = np.zeros((size, size), dtype=np.float32)
    if points_xz.size == 0:
        return mask

    dx = points_xz[:, 0] - int(center_x)
    dz = points_xz[:, 1] - int(center_z)
    cols = half + dx
    rows = half + dz

    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    if np.any(valid):
        mask[rows[valid], cols[valid]] = 1.0
    return mask


def _sum_3x3_neighbors(mask):
    """Vectorized 3x3 neighborhood sum with zero padding."""
    arr = np.asarray(mask, dtype=np.float32)
    padded = np.pad(arr, 1, mode="constant", constant_values=0.0)
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    ACTION_OFFSETS = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.last_battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.last_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global grid visit counter for intrinsic reward (exploration bonus)
        # 全局网格访问计数器，用于内在奖励（探索奖励）
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.recent_positions = deque(maxlen=15)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._last_map_img = np.zeros((9, 21, 21), dtype=np.float32)
        self._last_npc_danger_map = np.zeros((21, 21), dtype=np.float32)
        self._cached_dirt_feature = (200.0, 0.0, 0.0)
        self.nearest_charger_dist = 1e9
        self.last_nearest_charger_dist = 1e9
        self._legal_act = [1] * 8
        self.stuck_steps = 0
        self.prev_action = -1
        self.curr_action = -1
        self.npc_masked_action_count = 0
        self.npc_mask_candidate_action_count = 0
        # Per-charger cooldown memory: key=(x,z), value=remaining steps.
        # 充电桩独立冷却记忆：key=(x,z)，value=剩余步数。
        self.charger_cooldown = {}
        self.charger_cooldown_steps = 20
        self.active_charger_key = None
        self.charger_potential_enabled = 1.0

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        hero = frame_state.get("heroes") or {}

        if not hero:
            return

        self.step_no = int(observation["step_no"])
        self.last_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))
        self.recent_positions.append(self.cur_pos)

        # Update visit counter for current position (intrinsic reward / exploration bonus).
        # 更新当前位置的访问计数（用于内在奖励）。
        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            self.visit_count[x, z] += 1

        # Track action history for inertia-style reward shaping.
        # 记录动作历史，用于动作惯性奖励。
        self.prev_action = self.curr_action
        self.curr_action = int(last_action) if last_action is not None else -1

        # Track consecutive stagnant steps so the agent can switch to a backup action.
        # 记录连续未移动步数，供执行层切换次优动作。
        if self.step_no > 0 and self.cur_pos == self.last_pos:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        # Battery / 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _calc_nearest_dirt_feature(self, dirt_coords=None):
        """Get nearest dirt distance and direction in local view coordinates.

        返回最近污渍的距离与方向（相对机器人本地视野坐标）。
        """
        view = self._view_map
        if view is None:
            return 200.0, 0.0, 0.0

        if dirt_coords is None:
            dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0, 0.0, 0.0

        center = self.VIEW_HALF
        dz_all = dirt_coords[:, 0].astype(np.float32) - float(center)
        dx_all = dirt_coords[:, 1].astype(np.float32) - float(center)
        dists = np.sqrt(dx_all * dx_all + dz_all * dz_all)

        idx = int(np.argmin(dists))
        dx = float(dx_all[idx])
        dz = float(dz_all[idx])
        dist = float(dists[idx])
        dir_x, dir_z = _safe_dir(dx, dz)
        return dist, dir_x, dir_z

    def _calc_charger_reachable(self, static_obstacle, charger_map):
        """Return 1.0 if any charger is reachable in local 21x21 map, else 0.0.

        在局部 21x21 视野内，用 8 连通 BFS 判断是否存在可达充电桩。
        """
        obstacle = np.asarray(static_obstacle, dtype=bool)
        target = np.asarray(charger_map, dtype=np.float32) > 0.5
        if obstacle.shape != (21, 21) or target.shape != (21, 21):
            return 0.0
        if not np.any(target):
            return 0.0

        center = self.VIEW_HALF
        passable = ~obstacle
        passable[center, center] = True
        if not passable[center, center]:
            return 0.0

        q = deque([(center, center)])
        visited = np.zeros((21, 21), dtype=bool)
        visited[center, center] = True

        while q:
            x, z = q.popleft()
            if target[z, x]:
                return 1.0

            for dx, dz in self.ACTION_OFFSETS:
                nx = x + dx
                nz = z + dz
                if 0 <= nx < 21 and 0 <= nz < 21 and not visited[nz, nx] and passable[nz, nx]:
                    visited[nz, nx] = True
                    q.append((nx, nz))

        return 0.0

    def _decay_charger_cooldown(self):
        """Decay per-charger cooldowns by one step and remove expired keys."""
        if not self.charger_cooldown:
            return
        next_cd = {}
        for key, val in self.charger_cooldown.items():
            v = int(val) - 1
            if v > 0:
                next_cd[key] = v
        self.charger_cooldown = next_cd

    def _calc_reachable_to_local_target(self, static_obstacle, target_row, target_col):
        """Reachability from center to a specific local target cell (8-neighbor BFS)."""
        obstacle = np.asarray(static_obstacle, dtype=bool)
        if obstacle.shape != (21, 21):
            return 0.0
        if not (0 <= target_row < 21 and 0 <= target_col < 21):
            return 0.0

        center = self.VIEW_HALF
        passable = ~obstacle
        passable[center, center] = True
        if not passable[target_row, target_col]:
            return 0.0

        q = deque([(center, center)])
        visited = np.zeros((21, 21), dtype=bool)
        visited[center, center] = True
        while q:
            x, z = q.popleft()
            if z == target_row and x == target_col:
                return 1.0
            for dx, dz in self.ACTION_OFFSETS:
                nx = x + dx
                nz = z + dz
                if 0 <= nx < 21 and 0 <= nz < 21 and not visited[nz, nx] and passable[nz, nx]:
                    visited[nz, nx] = True
                    q.append((nx, nz))
        return 0.0

    def get_legal_action(self, map_img=None):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        base_mask = np.array(self._legal_act, dtype=np.float32)
        wall_mask = self._build_wall_action_mask(map_img)
        merged_mask = base_mask * wall_mask

        # Fallback: if NPC masking removes all actions, keep original env mask.
        # 兜底：如果叠加 NPC 掩码后 8 个动作全部不可用，则保留原始环境掩码。
        if np.sum(merged_mask) <= 0:
            return base_mask.astype(np.int32).tolist()

        return merged_mask.astype(np.int32).tolist()

    def get_npc_danger_mask_rate(self):
        """Return accumulated NPC-danger mask rate in current episode."""
        denom = max(self.npc_mask_candidate_action_count, 1)
        return float(self.npc_masked_action_count) / float(denom)

    def _build_wall_action_mask(self, map_img=None):
        """Build action mask that blocks moves stepping directly into static walls.

        基于 map[0]（静态墙体）构建动作掩码，直接走进墙体的动作置 0。
        """
        mask = np.ones(8, dtype=np.float32)

        # Prefer current-step map[0]; fallback to cached map[0].
        # 优先使用当前步 map[0]，否则退回到缓存 map[0]。
        wall_map = None
        if map_img is not None and isinstance(map_img, np.ndarray) and map_img.shape == (9, 21, 21):
            wall_map = map_img[0]
        elif self._last_map_img is not None and self._last_map_img.shape == (9, 21, 21):
            wall_map = self._last_map_img[0]

        if wall_map is None:
            return mask

        center_x = self.VIEW_HALF
        center_z = self.VIEW_HALF
        for action_idx, (dx, dz) in enumerate(self.ACTION_OFFSETS):
            local_x = center_x + dx
            local_z = center_z + dz
            if 0 <= local_x < 21 and 0 <= local_z < 21:
                if wall_map[local_z, local_x] >= 0.5:
                    mask[action_idx] = 0.0
            else:
                mask[action_idx] = 0.0

        return mask

    def _build_npc_safe_action_mask(self, map_img=None):
        """Build action mask from NPC danger channel at robot-centered offsets.

        基于 map[1]（NPC 禁区）和中心偏移构建动作掩码。
        """
        mask = np.ones(8, dtype=np.float32)

        # Prefer current-step map[1]; fallback to cached map[1].
        # 优先使用当前步 map[1]，否则退回到缓存 map[1]。
        block_map = None
        if map_img is not None and isinstance(map_img, np.ndarray) and map_img.shape == (9, 21, 21):
            block_map = map_img[1]
        elif self._last_map_img is not None and self._last_map_img.shape == (9, 21, 21):
            block_map = self._last_map_img[1]

        if block_map is None:
            return mask

        center_x = self.VIEW_HALF
        center_z = self.VIEW_HALF
        for action_idx, (dx, dz) in enumerate(self.ACTION_OFFSETS):
            local_x = center_x + dx
            local_z = center_z + dz
            if 0 <= local_x < 21 and 0 <= local_z < 21:
                if block_map[local_z, local_x] >= 0.5:
                    mask[action_idx] = 0.0
            else:
                # Out-of-view moves are conservatively blocked here; the env legal mask
                # still handles hard boundaries and obstacles.
                mask[action_idx] = 0.0

        return mask

    def feature_process(self, env_obs, last_action):
        """Generate two-stream model inputs, legal action mask, and scalar reward.

        生成双流模型输入、合法动作掩码和标量奖励。

        feature 字段包含：
                    - map_img: (9, 21, 21)
                    - vector_data: (10,)
        """
        self.pb2struct(env_obs, last_action)

        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        extra_info = env_obs.get("extra_info") or {}
        extra_state = extra_info.get("frame_state") or {}

        # Base map (21x21): 0 obstacle, 1 cleaned, 2 dirt
        map_info = observation.get("map_info")
        view_src = self._view_map if map_info is None else map_info
        view_map = np.array(view_src, dtype=np.float32)
        if view_map.shape != (21, 21):
            view_map = np.zeros((21, 21), dtype=np.float32)

        robot_pos = env_info.get("pos") or {}
        hx = int(robot_pos.get("x", self.cur_pos[0]))
        hz = int(robot_pos.get("z", self.cur_pos[1]))

        # CH1: static obstacles only (pure map walls/blocks).
        static_obstacle = (view_map == 0)

        # CH2/CH3: NPC danger zone and NPC distance field.
        npcs = frame_state.get("npcs") or extra_state.get("npcs") or []
        if npcs:
            npc_points = np.array(
                [
                    [int((npc.get("pos") or {}).get("x", -10**9)), int((npc.get("pos") or {}).get("z", -10**9))]
                    for npc in npcs
                ],
                dtype=np.int32,
            )
        else:
            npc_points = np.empty((0, 2), dtype=np.int32)
        npc_point_map = _global_points_to_local_mask(npc_points, hx, hz, half=self.VIEW_HALF, size=21)
        npc_danger_map = (_sum_3x3_neighbors(npc_point_map) > 0).astype(np.float32)
        npc_dist_field = _inv_dist_field(npc_danger_map > 0, max_dist=10.0)

        # CH4/CH5: dirt and dirt distance field.
        dirt_mask = (view_map == 2)
        dirt_coords = np.argwhere(dirt_mask)
        dirt_dist_field = _inv_dist_field(dirt_mask, max_dist=15.0)

        # CH6: self position one-hot.
        self_map = np.zeros((21, 21), dtype=np.float32)
        self_map[self.VIEW_HALF, self.VIEW_HALF] = 1.0

        # CH7/CH8: charger and charger distance field.
        organs = frame_state.get("organs") or extra_state.get("organs") or []
        if organs:
            organ_points = np.array(
                [
                    [int((organ.get("pos") or {}).get("x", -10**9)), int((organ.get("pos") or {}).get("z", -10**9))]
                    for organ in organs
                ],
                dtype=np.int32,
            )
        else:
            organ_points = np.empty((0, 2), dtype=np.int32)
        charger_map = _global_points_to_local_mask(organ_points, hx, hz, half=self.VIEW_HALF, size=21)
        charger_dist_field = _inv_dist_field(charger_map > 0, max_dist=15.0)

        # CH9: cleaned map.
        cleaned_mask = (view_map == 1)

        map_img = np.stack(
            [
                static_obstacle.astype(np.float32),
                npc_danger_map,
                npc_dist_field,
                dirt_mask.astype(np.float32),
                dirt_dist_field,
                self_map,
                charger_map,
                charger_dist_field,
                cleaned_mask.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        # Charger target selection with per-charger cooldown memory.
        # 带“按桩独立冷却记忆”的充电目标选择。
        self._decay_charger_cooldown()
        self.last_nearest_charger_dist = self.nearest_charger_dist
        charger_reachable = 0.0
        charger_visible = 0.0
        selected_key = None
        selected_rel = np.array([0.0, 0.0], dtype=np.float32)
        selected_dist = 1e9

        if organ_points.size > 0:
            rel = organ_points.astype(np.float32) - np.array([float(hx), float(hz)], dtype=np.float32)
            dists = np.sqrt(rel[:, 0] * rel[:, 0] + rel[:, 1] * rel[:, 1])
            order = np.argsort(dists)

            reachable_cache = {}
            in_view_cache = {}
            for i in order:
                ox = int(organ_points[i, 0])
                oz = int(organ_points[i, 1])
                key = (ox, oz)
                li = int(oz - hz + self.VIEW_HALF)  # row
                lj = int(ox - hx + self.VIEW_HALF)  # col
                in_view = 0 <= li < 21 and 0 <= lj < 21
                in_view_cache[key] = in_view

                # Only evaluate/refresh unreachable cooldown for chargers in local view.
                # 只对视野内充电桩做可达判定与冷却刷新。
                if in_view:
                    r = self._calc_reachable_to_local_target(static_obstacle, li, lj)
                    reachable_cache[key] = r
                    if r > 0.5:
                        self.charger_cooldown.pop(key, None)
                    else:
                        self.charger_cooldown[key] = int(self.charger_cooldown_steps)

            # 1) Prefer nearest charger that is currently reachable in view and not cooling down.
            chosen_idx = None
            for i in order:
                key = (int(organ_points[i, 0]), int(organ_points[i, 1]))
                if key in self.charger_cooldown:
                    continue
                if reachable_cache.get(key, 0.0) > 0.5:
                    chosen_idx = int(i)
                    break

            # 2) Otherwise choose nearest charger not in cooldown (including out-of-view candidates).
            if chosen_idx is None:
                for i in order:
                    key = (int(organ_points[i, 0]), int(organ_points[i, 1]))
                    if key not in self.charger_cooldown:
                        chosen_idx = int(i)
                        break

            # 3) Final fallback: nearest charger (all candidates are cooling down).
            if chosen_idx is None:
                chosen_idx = int(order[0])

            selected_key = (int(organ_points[chosen_idx, 0]), int(organ_points[chosen_idx, 1]))
            selected_rel = rel[chosen_idx]
            selected_dist = float(dists[chosen_idx])
            selected_in_view = bool(in_view_cache.get(selected_key, False))
            charger_visible = 1.0 if selected_in_view else 0.0

            # Reachable feature is tied to selected charger and suppressed during cooldown.
            # selected charger 在冷却期时 reachable 强制为 0。
            if selected_key in self.charger_cooldown:
                charger_reachable = 0.0
            else:
                charger_reachable = float(reachable_cache.get(selected_key, 0.0))
        else:
            self.charger_cooldown = {}

        # When switching target, avoid artificial potential jump by resetting delta baseline.
        # 切换目标时重置势能差分基线，避免奖励跳变。
        if selected_key is not None and self.active_charger_key != selected_key:
            self.last_nearest_charger_dist = selected_dist
        self.active_charger_key = selected_key
        self.nearest_charger_dist = selected_dist
        self.charger_potential_enabled = float(charger_reachable > 0.5)
        charger_dir_x, charger_dir_z = _safe_dir(float(selected_rel[0]), float(selected_rel[1]))

        # Cache nearest dirt feature from local view.
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        dirt_dist, dirt_dir_x, dirt_dir_z = self._calc_nearest_dirt_feature(dirt_coords)
        self._cached_dirt_feature = (dirt_dist, dirt_dir_x, dirt_dir_z)
        self.nearest_dirt_dist = dirt_dist

        # Vector stream:
        # [battery_ratio, cleaning_progress]
        # + nearest_charger(dist, dir_x, dir_z)
        # + nearest_dirt(dist, dir_x, dir_z)
        # + charger_reachable(0/1)
        # + charger_visible(0/1)
        battery_max = float(max(self.battery_max, 1))
        battery_signal = _norm(self.battery, battery_max)
        cleaning_progress = float(self.dirt_cleaned) / float(max(self.total_dirt, 1))

        charger_dist_norm = _norm(self.nearest_charger_dist, 200.0)
        dirt_dist_norm = _norm(self.nearest_dirt_dist, 30.0)

        vector_data = np.array(
            [
                battery_signal,
                cleaning_progress,
                charger_dist_norm,
                charger_dir_x,
                charger_dir_z,
                dirt_dist_norm,
                dirt_dir_x,
                dirt_dir_z,
                charger_reachable,
                charger_visible,
            ],
            dtype=np.float32,
        )

        legal_action = self.get_legal_action(map_img)  # 8D
        feature = {
            "map_img": map_img,
            "vector_data": vector_data,
        }

        # Cache latest map_img for reward shaping.
        # 缓存最新 map_img，用于奖励塑形。
        self._last_map_img = map_img
        self._last_npc_danger_map = npc_danger_map

        reward = self.reward_process()

        return feature, legal_action, reward

    def _compute_intrinsic_reward(self, eta=0.1):
        """Compute intrinsic (exploration) reward based on visit count.
        
        Formula: r_int = eta / sqrt(visit_count[x][y] + epsilon)
        where epsilon=1 prevents division by zero and gives reward for unvisited cells.
        
        计算内在奖励（探索奖励），基于访问计数。
        公式: r_int = eta / sqrt(visit_count[x][y] + epsilon)
        其中 epsilon=1 防止除以零，也给予未访问格子奖励。
        """
        x, z = self.cur_pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return 0.0
        
        count = float(self.visit_count[x, z])
        epsilon = 1.0  # Smoothing constant: eta/sqrt(1)=eta for unvisited cells
        intrinsic_reward = eta / np.sqrt(count + epsilon)
        return float(intrinsic_reward)

    def _compute_local_dirt_density_reward(self, weight=0.1):
        """Reward moving toward dirt-dense local areas.

        根据局部 7×7 视野中的污渍密度提供奖励，引导机器人朝 dirt 更密集的区域移动。
        """
        view = self._view_map
        if view is None:
            return 0.0

        center = self.VIEW_HALF
        half = self.LOCAL_HALF
        crop = view[center - half : center + half + 1, center - half : center + half + 1]
        if crop.size == 0:
            return 0.0

        local_dirt_density = float(np.mean(crop == 2))
        return float(weight * local_dirt_density)

    def _compute_repeat_visit_penalty(self):
        """Penalty for repeated visits within a 15-step sliding window.

        仅统计最近 15 步的位置；当前位置在窗口内出现次数越多，惩罚越大。
        """
        recent_positions = getattr(self, "recent_positions", None)
        if not recent_positions:
            return 0.0

        x, z = self.cur_pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return 0.0

        count = sum(1 for pos in recent_positions if pos == self.cur_pos)
        if count <= 1:
            return 0.0

        
        # Progressive penalty by repetition count in the sliding window.
        # 窗口内重复次数越多，惩罚线性增强，并做上限保护。
        penalty = -0.08 * float(count - 1)
        return float(max(penalty, -0.24))

    def _compute_stuck_penalty(self):
        """Penalty for consecutive stagnant steps.

        连续卡死惩罚：每步 -0.1，最多到 -0.5。
        """
        stuck_penalty = 0.0
        stuck_steps = getattr(self, "stuck_steps", 0)
        if stuck_steps > 0:
            stuck_penalty = -0.15 * min(stuck_steps, 7)
        return float(stuck_penalty)

    def reward_process(self):
        # Incremental reward selector (switch this single line manually).
        # 增量训练奖励选择器（手动改这一行即可切换阶段）。
        return self._reward_stage_1_bootstrap()
        # return self._reward_stage_2()

    def _reward_stage_1_bootstrap(self):
        """Stage 1: minimal reward set for stable warm-up training.

        保留项：清扫奖励、步数惩罚、NPC 靠近惩罚、基础寻桩势能、低电回满一次性奖励。
        新增项：探索奖励（内在奖励）。
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.05 * cleaned_this_step
        step_penalty = -0.02

        # NPC proximity penalty (same danger-map rule as stage2, but no extra terms).
        # NPC 靠近惩罚（与 stage2 使用同一危险图规则）。
        npc_penalty = 0.0
        danger = self._last_npc_danger_map
        if danger is not None:
            if np.any(danger[8:13, 8:13] == 1.0):
                npc_penalty = -0.25
            elif np.any(danger[7:14, 7:14] == 1.0):
                npc_penalty = -0.05

        battery_max = float(max(self.battery_max, 1))
        battery = float(self.battery)
        low_battery = 110.0 / 200.0 * battery_max
        stop_charge_battery = 140.0 / 200.0 * battery_max
        full_charge_battery = 190.0 / 200.0 * battery_max
        mid_battery = 50.0 / 200.0 * battery_max

        dist_change = self.last_nearest_charger_dist - self.nearest_charger_dist
        charge_coef = 0.0
        
        if battery < low_battery:
            charge_coef = (low_battery - battery) / low_battery
        if battery > stop_charge_battery:
            charge_coef = 0.0
        if battery >= low_battery:
            charge_coef = charge_coef * 0.2
        elif battery >= mid_battery:
            charge_coef = charge_coef * 0.5
        charge_potential_reward = dist_change * charge_coef * float(self.charger_potential_enabled)

        instant_full_charge_reward = 0.0

        if self.last_battery <= 0.5 * battery_max and battery > full_charge_battery:
            instant_full_charge_reward = 2.0 - 0.01 * self.last_battery


        # Intrinsic reward for exploration (eta=0.1).
        # 探索奖励（内在奖励）。
        intrinsic_reward = self._compute_intrinsic_reward(eta=0.1)

        # Density reward for moving toward dirt-dense areas.
        # 朝污渍更密集的区域移动的奖励。
        density_reward = self._compute_local_dirt_density_reward(weight=0.1)

        # Repeat-visit penalty based on global visit_count.
        # 基于全局 visit_count 的重复访问惩罚。
        repeat_visit_penalty = self._compute_repeat_visit_penalty()

        # Action consistency reward: small reward for repeating the same action.
        # 动作一致奖励：对重复相同动作给予小奖励。
        action_consistency_reward = 0.01 if self.prev_action == self.curr_action and self.prev_action != -1 else 0.0

        return (
            cleaning_reward
            + step_penalty
            + npc_penalty
            + charge_potential_reward
            + instant_full_charge_reward
            + intrinsic_reward
            + density_reward
            + repeat_visit_penalty
            + action_consistency_reward
        )
    
    def _reward_stage_2(self):
        """Stage 2: minimal reward set for stable warm-up training.

        保留项：清扫奖励、步数惩罚、NPC 靠近惩罚、基础寻桩势能、低电回满一次性奖励。
        新增项：探索奖励（内在奖励）。
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.05 * cleaned_this_step
        step_penalty = -0.02

        # NPC proximity penalty (same danger-map rule as stage2, but no extra terms).
        # NPC 靠近惩罚（与 stage2 使用同一危险图规则）。
        npc_penalty = 0.0
        danger = self._last_npc_danger_map
        if danger is not None:
            if np.any(danger[8:13, 8:13] == 1.0):
                npc_penalty = -0.2
            elif np.any(danger[7:14, 7:14] == 1.0):
                npc_penalty = -0.02

        battery_max = float(max(self.battery_max, 1))
        battery = float(self.battery)
        low_battery = 160.0 / 200.0 * battery_max
        stop_charge_battery = 190.0 / 200.0 * battery_max
        full_charge_battery = 190.0 / 200.0 * battery_max
        mid_battery = 100.0 / 200.0 * battery_max

        dist_change = self.last_nearest_charger_dist - self.nearest_charger_dist
        charge_coef = 0.0
        if battery < low_battery:
            charge_coef = (low_battery - battery) / low_battery * 1.5
        if battery > stop_charge_battery:
            charge_coef = 0.0
        if battery >= low_battery:
            charge_coef = charge_coef * 0.2
        elif battery >= mid_battery:
            charge_coef = charge_coef * 0.5
        charge_potential_reward = dist_change * charge_coef * float(self.charger_potential_enabled)

        instant_full_charge_reward = 0.0

        if self.last_battery <= 0.5 * battery_max and battery > full_charge_battery:
            instant_full_charge_reward = 2.0 - 0.01 * self.last_battery

        # Intrinsic reward for exploration (eta=0.1).
        # 探索奖励（内在奖励）。
        intrinsic_reward = self._compute_intrinsic_reward(eta=0.1)

        # Density reward for moving toward dirt-dense areas.
        # 朝污渍更密集的区域移动的奖励。
        density_reward = self._compute_local_dirt_density_reward(weight=0.1)

        # Repeat-visit penalty based on global visit_count.
        # 基于全局 visit_count 的重复访问惩罚。
        repeat_visit_penalty = self._compute_repeat_visit_penalty()

        # Stuck penalty for consecutive no-move steps.
        # 连续不移动惩罚（仅 stage2 生效）。
        stuck_penalty = self._compute_stuck_penalty()

        # Action consistency reward: small reward for repeating the same action.
        # 动作一致奖励：对重复相同动作给予小奖励。
        action_consistency_reward = 0.01 if self.prev_action == self.curr_action and self.prev_action != -1 else 0.0

        return (
            cleaning_reward
            + step_penalty
            + npc_penalty
            + charge_potential_reward
            + instant_full_charge_reward
            + intrinsic_reward
            + density_reward
            + repeat_visit_penalty
            + stuck_penalty
            + action_consistency_reward
        )


