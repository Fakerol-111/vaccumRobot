from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _safe_dir(dx, dz):
    dist = float(np.sqrt(dx * dx + dz * dz))
    if dist <= 1e-6:
        return 0.0, 0.0
    return float(dx / dist), float(dz / dist)


def _inv_dist_field(target_mask, max_dist):
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
    arr = np.asarray(mask, dtype=np.float32)
    padded = np.pad(arr, 1, mode="constant", constant_values=0.0)
    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )


class Preprocessor:
    GRID_SIZE = 128
    VIEW_HALF = 10
    LOCAL_HALF = 3
    ACTION_OFFSETS = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.battery = 600
        self.last_battery = 600
        self.battery_max = 600
        self.cur_pos = (0, 0)
        self.last_pos = (0, 0)
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.recent_positions = deque(maxlen=15)
        self.nearest_charger_dist = 1e9
        self.last_nearest_charger_dist = 1e9
        self._legal_act = [1] * 8
        self.stuck_steps = 0
        self.prev_action = -1
        self.curr_action = -1
        self._last_map_img = np.zeros((9, 21, 21), dtype=np.float32)
        self._last_npc_danger_map = np.zeros((21, 21), dtype=np.float32)
        self.charger_cooldown = {}
        self.charger_cooldown_steps = 20
        self.active_charger_key = None
        self.charger_potential_enabled = 1.0

    # ---------- parsing ----------

    def pb2struct(self, env_obs: dict[str, Any], last_action: int | None):
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

        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            self.visit_count[x, z] += 1

        self.prev_action = self.curr_action
        self.curr_action = int(last_action) if last_action is not None else -1

        if self.step_no > 0 and self.cur_pos == self.last_pos:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)

    # ---------- feature process ----------

    def feature_process(self, env_obs: dict[str, Any], last_action: int | None):
        self.pb2struct(env_obs, last_action)

        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        extra_info = env_obs.get("extra_info") or {}
        extra_state = extra_info.get("frame_state") or {}

        map_info = observation.get("map_info")
        view_map = np.array(map_info, dtype=np.float32) if map_info is not None else np.zeros((21, 21), dtype=np.float32)
        if view_map.shape != (21, 21):
            view_map = np.zeros((21, 21), dtype=np.float32)

        robot_pos = env_info.get("pos") or {}
        hx = int(robot_pos.get("x", self.cur_pos[0]))
        hz = int(robot_pos.get("z", self.cur_pos[1]))

        static_obstacle = (view_map == 0)

        npcs = frame_state.get("npcs") or extra_state.get("npcs") or []
        if npcs:
            npc_points = np.array(
                [[int((npc.get("pos") or {}).get("x", -10**9)), int((npc.get("pos") or {}).get("z", -10**9))]
                 for npc in npcs], dtype=np.int32)
        else:
            npc_points = np.empty((0, 2), dtype=np.int32)
        npc_point_map = _global_points_to_local_mask(npc_points, hx, hz, half=self.VIEW_HALF, size=21)
        npc_danger_map = (_sum_3x3_neighbors(npc_point_map) > 0).astype(np.float32)
        npc_dist_field = _inv_dist_field(npc_danger_map > 0, max_dist=10.0)

        dirt_mask = (view_map == 2)
        dirt_coords = np.argwhere(dirt_mask)
        dirt_dist_field = _inv_dist_field(dirt_mask, max_dist=15.0)

        self_map = np.zeros((21, 21), dtype=np.float32)
        self_map[self.VIEW_HALF, self.VIEW_HALF] = 1.0

        organs = frame_state.get("organs") or extra_state.get("organs") or []
        if organs:
            organ_points = np.array(
                [[int((organ.get("pos") or {}).get("x", -10**9)), int((organ.get("pos") or {}).get("z", -10**9))]
                 for organ in organs], dtype=np.int32)
        else:
            organ_points = np.empty((0, 2), dtype=np.int32)
        charger_map = _global_points_to_local_mask(organ_points, hx, hz, half=self.VIEW_HALF, size=21)
        charger_dist_field = _inv_dist_field(charger_map > 0, max_dist=15.0)

        cleaned_mask = (view_map == 1)

        map_img = np.stack([
            static_obstacle.astype(np.float32),
            npc_danger_map,
            npc_dist_field,
            dirt_mask.astype(np.float32),
            dirt_dist_field,
            self_map,
            charger_map,
            charger_dist_field,
            cleaned_mask.astype(np.float32),
        ], axis=0).astype(np.float32, copy=False)

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
            for i in order:
                ox = int(organ_points[i, 0])
                oz = int(organ_points[i, 1])
                key = (ox, oz)
                li = int(oz - hz + self.VIEW_HALF)
                lj = int(ox - hx + self.VIEW_HALF)
                in_view = 0 <= li < 21 and 0 <= lj < 21
                if in_view:
                    r = self._calc_reachable_to_local_target(static_obstacle, li, lj)
                    reachable_cache[key] = r
                    if r > 0.5:
                        self.charger_cooldown.pop(key, None)
                    else:
                        self.charger_cooldown[key] = int(self.charger_cooldown_steps)

            chosen_idx = None
            for i in order:
                key = (int(organ_points[i, 0]), int(organ_points[i, 1]))
                if key in self.charger_cooldown:
                    continue
                if reachable_cache.get(key, 0.0) > 0.5:
                    chosen_idx = int(i)
                    break
            if chosen_idx is None:
                for i in order:
                    key = (int(organ_points[i, 0]), int(organ_points[i, 1]))
                    if key not in self.charger_cooldown:
                        chosen_idx = int(i)
                        break
            if chosen_idx is None:
                chosen_idx = int(order[0])

            selected_key = (int(organ_points[chosen_idx, 0]), int(organ_points[chosen_idx, 1]))
            selected_rel = rel[chosen_idx]
            selected_dist = float(dists[chosen_idx])
            li = int(selected_key[1] - hz + self.VIEW_HALF)
            lj = int(selected_key[0] - hx + self.VIEW_HALF)
            charger_visible = 1.0 if (0 <= li < 21 and 0 <= lj < 21) else 0.0
            if selected_key in self.charger_cooldown:
                charger_reachable = 0.0
            else:
                charger_reachable = float(reachable_cache.get(selected_key, 0.0))
        else:
            self.charger_cooldown = {}

        if selected_key is not None and self.active_charger_key != selected_key:
            self.last_nearest_charger_dist = selected_dist
        self.active_charger_key = selected_key
        self.nearest_charger_dist = selected_dist
        self.charger_potential_enabled = float(charger_reachable > 0.5)
        charger_dir_x, charger_dir_z = _safe_dir(float(selected_rel[0]), float(selected_rel[1]))

        dirt_dist, dirt_dir_x, dirt_dir_z = self._calc_nearest_dirt_feature(view_map, dirt_coords)

        battery_max = float(max(self.battery_max, 1))
        battery_signal = _norm(self.battery, battery_max)
        cleaning_progress = float(self.dirt_cleaned) / float(max(self.total_dirt, 1))
        charger_dist_norm = _norm(self.nearest_charger_dist, 200.0)
        dirt_dist_norm = _norm(dirt_dist, 30.0)

        vector_data = np.array([
            battery_signal, cleaning_progress,
            charger_dist_norm, charger_dir_x, charger_dir_z,
            dirt_dist_norm, dirt_dir_x, dirt_dir_z,
            charger_reachable, charger_visible,
        ], dtype=np.float32)

        legal_action = self._build_legal_action(map_img)

        self._last_map_img = map_img
        self._last_npc_danger_map = npc_danger_map

        reward = self.reward_process()

        return map_img, vector_data, legal_action, reward

    # ---------- legal action ----------

    def _build_legal_action(self, map_img):
        base_mask = np.array(self._legal_act, dtype=np.float32)
        wall_mask = self._build_wall_action_mask(map_img)
        merged = base_mask * wall_mask
        if np.sum(merged) <= 0:
            return base_mask.astype(np.int32).tolist()
        return merged.astype(np.int32).tolist()

    def _build_wall_action_mask(self, map_img):
        mask = np.ones(8, dtype=np.float32)
        wall_map = None
        if map_img is not None and map_img.shape == (9, 21, 21):
            wall_map = map_img[0]
        if wall_map is None:
            return mask
        cx, cz = self.VIEW_HALF, self.VIEW_HALF
        for i, (dx, dz) in enumerate(self.ACTION_OFFSETS):
            lx, lz = cx + dx, cz + dz
            if 0 <= lx < 21 and 0 <= lz < 21:
                if wall_map[lz, lx] >= 0.5:
                    mask[i] = 0.0
            else:
                mask[i] = 0.0
        return mask

    # ---------- dirt / charger helpers ----------

    def _calc_nearest_dirt_feature(self, view_map, dirt_coords=None):
        if dirt_coords is None:
            dirt_coords = np.argwhere(view_map == 2)
        if len(dirt_coords) == 0:
            return 200.0, 0.0, 0.0
        center = self.VIEW_HALF
        dz_all = dirt_coords[:, 0].astype(np.float32) - float(center)
        dx_all = dirt_coords[:, 1].astype(np.float32) - float(center)
        dists = np.sqrt(dx_all * dx_all + dz_all * dz_all)
        idx = int(np.argmin(dists))
        dx, dz = float(dx_all[idx]), float(dz_all[idx])
        dist = float(dists[idx])
        dir_x, dir_z = _safe_dir(dx, dz)
        return dist, dir_x, dir_z

    def _calc_reachable_to_local_target(self, static_obstacle, target_row, target_col):
        obstacle = np.asarray(static_obstacle, dtype=bool)
        if obstacle.shape != (21, 21):
            return 0.0
        if not (0 <= target_row < 21 and 0 <= target_col < 21):
            return 0.0
        cx, cz = self.VIEW_HALF, self.VIEW_HALF
        passable = ~obstacle
        passable[cz, cx] = True
        if not passable[target_row, target_col]:
            return 0.0
        q = deque([(cx, cz)])
        visited = np.zeros((21, 21), dtype=bool)
        visited[cz, cx] = True
        while q:
            x, z = q.popleft()
            if z == target_row and x == target_col:
                return 1.0
            for dx, dz in self.ACTION_OFFSETS:
                nx, nz = x + dx, z + dz
                if 0 <= nx < 21 and 0 <= nz < 21 and not visited[nz, nx] and passable[nz, nx]:
                    visited[nz, nx] = True
                    q.append((nx, nz))
        return 0.0

    def _decay_charger_cooldown(self):
        if not self.charger_cooldown:
            return
        next_cd = {}
        for key, val in self.charger_cooldown.items():
            v = int(val) - 1
            if v > 0:
                next_cd[key] = v
        self.charger_cooldown = next_cd

    # ---------- reward ----------

    def reward_process(self):
        return self._reward_stage_1_bootstrap()

    def _reward_stage_1_bootstrap(self):
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.05 * cleaned_this_step
        step_penalty = -0.02

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

        intrinsic_reward = self._compute_intrinsic_reward(eta=0.1)
        density_reward = self._compute_local_dirt_density_reward(weight=0.1)
        repeat_visit_penalty = self._compute_repeat_visit_penalty()
        action_consistency_reward = 0.01 if self.prev_action == self.curr_action and self.prev_action != -1 else 0.0

        return (
            cleaning_reward + step_penalty + npc_penalty
            + charge_potential_reward + instant_full_charge_reward
            + intrinsic_reward + density_reward
            + repeat_visit_penalty + action_consistency_reward
        )

    def _compute_intrinsic_reward(self, eta=0.1):
        x, z = self.cur_pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return 0.0
        count = float(self.visit_count[x, z])
        epsilon = 1.0
        return float(eta / np.sqrt(count + epsilon))

    def _compute_local_dirt_density_reward(self, weight=0.1):
        view = self._view_map if hasattr(self, '_view_map') else None
        if view is None or view.shape != (21, 21):
            return 0.0
        center = self.VIEW_HALF
        half = self.LOCAL_HALF
        crop = view[center - half:center + half + 1, center - half:center + half + 1]
        if crop.size == 0:
            return 0.0
        local_dirt_density = float(np.mean(crop == 2))
        return float(weight * local_dirt_density)

    def _compute_repeat_visit_penalty(self):
        if not self.recent_positions:
            return 0.0
        x, z = self.cur_pos
        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return 0.0
        count = sum(1 for pos in self.recent_positions if pos == self.cur_pos)
        if count <= 1:
            return 0.0
        penalty = -0.08 * float(count - 1)
        return float(max(penalty, -0.24))
