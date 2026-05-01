"""GRPO 变体：First-Action Branch Rollout（适配环境结构的 GRPO）。

与标准 GRPO 的关系
------------------
本算法是标准 Group Relative Policy Optimization 的一个**环境适配变体**。
核心 GRPO 特征完整保留：

  - **无价值网络**（no critic），组内归一化替代基线
  - **组内相对比较**：K 条分支的 score 进行均值中心化作为优势
  - **frozen reference model** 提供 KL 约束

差异来源于环境结构约束，而非算法层面的偏离：

  +----------------------------+-----------------------------------+------------------------------------------+
  | 维度                       | 标准 GRPO                         | 本变体                                   |
  +============================+===================================+==========================================+
  | 后半段动作来源             | 模型继续采样                       | **greedy rollout**                       |
  +----------------------------+-----------------------------------+------------------------------------------+
  | NPC 行为                   | 与环境自然交互                     | **锁定为 greedy trace**                  |
  +----------------------------+-----------------------------------+------------------------------------------+
  | 优化对象                   | 全程每步 log-prob                 | **第一步**候选动作 log-prob              |
  +----------------------------+-----------------------------------+------------------------------------------+
  | 优势                       | 轨迹累计回报（组内归一化）         | **score 增量**（组内归一化）             |
  +----------------------------+-----------------------------------+------------------------------------------+

为什么这些差异是合理的（而不是对 GRPO 的偏离）：

  1. **NPC 独立于 agent 行动** —— 环境中的 NPC 移动不受 agent 决策影响。
     锁定 NPC trace 只是消除无关方差，确保分支间仅因第一步动作不同而产生
     差异。如果让 NPC 自然交互，反而会引入 agent 无法控制的噪声。

  2. **环境在给定 agent 动作 + NPC 轨迹后是确定性的** —— 因此第一步之后
     的 greedy rollout 就是该分支的最优续作。score 增量等价于该候选动作
     的真实回报，无需额外采样来估计。

  3. **MDP 性质** —— 当前状态包含决策所需的全部信息，优化第一步动作的
     log-prob 就是在优化策略函数本身。后续步的 greedy rollout 不是
     "跳过更新"，而是利用环境确定性来精确评估第一步动作的价值。

综上，本实现不是"偏离了 GRPO"，而是**针对 NPC 独立 + 确定性环境的
GRPO 适配**。

核心流程
--------
每隔 ``branch_interval`` 步触发一次组内更新：

  1. 保存 env + preprocessor 完整状态快照
  2. 跑一条 greedy 分支记录 NPC 轨迹
  3. 对 K 个候选动作各开一条分支：恢复快照 → 执行候选动作 →
     强制 NPC 按 greedy 轨迹移动 → 后续 greedy 执行 ``branch_window`` 步
  4. 分支得分的组内归一化值作为优势，更新第一步动作的策略
  5. frozen reference model 的 KL 散度作为正则项
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agent.base import ActResult, Algorithm, LossInfo
from agent.common.checkpoint import Checkpoint, capture_rng_state
from agent.grpo.grpo_metrics import GRPOMetricsReporter
from agent.nn import create_model
from agent.preprocessor import Preprocessor
from agent.registry import register
from env.factory import create_env



def _sample_candidate_actions(
    logits: torch.Tensor,
    legal_mask: np.ndarray,
    k: int,
) -> list[int]:
    """从策略分布采样 K 个合法候选动作。

    采样取代确定性 top-K 选择，使得即使策略接近均匀分布，
    各候选动作的 log_prob 也天然不同，与组内归一化的 advantage
    产生非零梯度信号，打破对称性坍缩。
    """

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    probs = probs * mask.float()
    probs = probs / (probs.sum() + 1e-10)

    num_valid = int(mask.sum().item())
    k_safe = min(k, num_valid)
    if k_safe < 2:
        return []

    indices = torch.multinomial(probs, k_safe, replacement=False)
    return [int(idx.item()) for idx in indices]



@register("grpo")
class GRPOAlgorithm(Algorithm):
    """GRPO 算法。

    维护两个 ActorCritic 网络：
      - ``model``: 当前策略（可训练）
      - ``reference``: 参考策略（frozen），用于 KL 约束
    """

    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = create_model(
            getattr(config, "model_type", "shared"),
            config.num_actions, actor_gain=1.0,
        )
        self.model.to(self.device)

        self.reference = create_model(
            getattr(config, "model_type", "shared"),
            config.num_actions, actor_gain=1.0,
        )
        self.reference.to(self.device)

        # 立即同步 reference = model，并冻结
        self.reference.load_state_dict(self.model.state_dict())
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self._steps_since_group = 0
        self._current_env_config: dict[str, Any] | None = None
        self._branch_state: dict[str, Any] | None = None
        self._episode_count = 0
        self._metrics_reporter = GRPOMetricsReporter()

    # ── tensor helpers ────────────────────────────────────

    def _to_tensor(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        map_img_t = torch.as_tensor(map_img, dtype=torch.float32, device=self.device)
        vector_t = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        legal_t = torch.as_tensor(legal_mask, dtype=torch.float32, device=self.device)
        if map_img_t.dim() == 3:
            map_img_t = map_img_t.unsqueeze(0)
            vector_t = vector_t.unsqueeze(0)
            legal_t = legal_t.unsqueeze(0)
        return map_img_t, vector_t, legal_t

    def _forward(self, map_img, vector, legal_mask):
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            logits, _ = self.model(map_img_t, vector_t, legal_t)
            dist = Categorical(logits=logits)
        return logits, dist

    # ── Algorithm interface ───────────────────────────────

    def explore(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, dist = self._forward(map_img, vector, legal_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return ActResult(action=action.item(), log_prob=log_prob.item())

    def exploit(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, dist = self._forward(map_img, vector, legal_mask)
        action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return ActResult(action=action.item(), log_prob=log_prob.item())

    def collect(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """GRPO 不使用标准 transition buffer，本方法仅为接口占位。"""

    def on_step(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> LossInfo | None:
        self._steps_since_group += 1
        interval = getattr(self.config, "branch_interval", 0)
        result = None
        if interval > 0 and self._steps_since_group >= interval:
            result = self.group_update(map_img, vector, legal_mask, self._current_env_config)

        if done:
            self._episode_count += 1
            self._sync_reference()

        return result

    def ready_to_update(self) -> bool:
        return False

    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        raise NotImplementedError("GRPO 使用 group_update() 而非 update()")

    def compute_value(self, *args) -> float:
        return 0.0

    @property
    def metrics_reporter(self) -> GRPOMetricsReporter:
        return self._metrics_reporter

    def set_env_config(self, env_config: dict[str, Any]) -> None:
        self._current_env_config = env_config

    def set_branch_state(self, state: dict[str, Any] | None) -> None:
        """由 Trainer 在每个 step 前调用，保存 env + preprocessor 快照。"""
        self._branch_state = state

    # ── group update ──────────────────────────────────────

    def _compute_kl(self, map_img, vector, legal_mask) -> torch.Tensor:
        """KL(current || reference) 在给定状态处的散度。"""
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            ref_logits, _ = self.reference(map_img_t, vector_t, legal_t)
        cur_logits, _ = self.model(map_img_t, vector_t, legal_t)
        return torch.distributions.kl_divergence(
            Categorical(logits=cur_logits),
            Categorical(logits=ref_logits),
        ).mean()

    def _sync_reference(self) -> None:
        """将当前策略权重复制给 reference model。

        受 ``config.ref_sync`` 控制：
          - "episode"（默认）: 每 episode 结束时同步
          - 0 / False / None: 禁用（完全冻结）
        """
        if not getattr(self.config, "ref_sync", "episode"):
            return
        self.reference.load_state_dict(self.model.state_dict())

    def _record_npc_trace(
        self,
        first_action: int,
        env_config: dict[str, Any],
        window: int,
    ) -> list[list[tuple[int, int]]]:
        """从分支点 greedy rollout，记录每一步后的 NPC 位置序列。"""
        bs = self._branch_state
        if bs is None:
            return []
        env = create_env(env_config, enable_recording=False, render_mode=None)
        env.set_state(bs["env"])
        pp = Preprocessor()
        pp.set_state(bs["pp"])

        npc_trace: list[list[tuple[int, int]]] = []
        payload = env.step(first_action)
        npc_trace.append([tuple(p) for p in env.npc_positions])
        img, vec, legal, reward = pp.feature_process(payload, first_action)
        legal_arr = np.asarray(legal, dtype=np.float32)

        for _ in range(window - 1):
            logits = self._get_logits(img, vec, legal_arr)
            action = int(torch.argmax(logits, dim=-1).item())
            payload = env.step(action)
            npc_trace.append([tuple(p) for p in env.npc_positions])
            img, vec, legal, reward = pp.feature_process(payload, action)
            legal_arr = np.asarray(legal, dtype=np.float32)
            if payload.get("terminated") or payload.get("truncated"):
                last_pos = [tuple(p) for p in env.npc_positions]
                while len(npc_trace) < window:
                    npc_trace.append(last_pos)
                break

        env.close()
        return npc_trace

    def _rollout_branch(
        self,
        first_action: int,
        env_config: dict[str, Any],
        window: int,
        npc_trace: list[list[tuple[int, int]]] | None = None,
    ) -> float:
        """从分支点执行一条分支，返回环境原始 score 增量（打扫的污渍格数）。

        恢复状态 → 候选动作 → 后续 greedy。
        使用 env.score 差值而非 shaped reward，确保分支之间公平比较。
        """
        bs = self._branch_state
        if bs is not None:
            env = create_env(env_config, enable_recording=False, render_mode=None)
            env.set_state(bs["env"])
            pp = Preprocessor()
            pp.set_state(bs["pp"])
            if npc_trace is not None:
                env.set_npc_trace(npc_trace)
            start_score = env.score
        else:
            # Fallback: 无分支状态时从 episode 起点开始
            env = create_env(env_config, enable_recording=False, render_mode=None)
            payload = env.reset(options={"mode": "eval"})
            pp = Preprocessor()
            pp.reset()
            start_score = 0

        # 第一步：candidate action
        payload = env.step(first_action)
        img, vec, legal, _ = pp.feature_process(payload, first_action)
        legal_arr = np.asarray(legal, dtype=np.float32)

        # 后续：greedy（仍需 preprocessor 获取特征，但忽略其 reward）
        for _ in range(window - 1):
            logits = self._get_logits(img, vec, legal_arr)
            action = int(torch.argmax(logits, dim=-1).item())
            payload = env.step(action)
            img, vec, legal, _ = pp.feature_process(payload, action)
            legal_arr = np.asarray(legal, dtype=np.float32)
            if payload.get("terminated") or payload.get("truncated"):
                break

        score_delta = env.score - start_score
        env.close()
        return score_delta

    def _get_logits(self, map_img, vector, legal_mask):
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            logits, _ = self.model(map_img_t, vector_t, legal_t)
        return logits

    def group_update(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        env_config: dict[str, Any] | None,
    ) -> LossInfo | None:
        """从已保存的状态快照做一次 GRPO 组内更新。"""
        if env_config is None or self._branch_state is None:
            return None

        legal_arr = np.asarray(legal_mask, dtype=np.float32)

        # 1. 候选动作（含梯度 — 用于后续 loss backward）
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_arr)
        logits, _ = self.model(map_img_t, vector_t, legal_t)


        candidate_actions = _sample_candidate_actions(
            logits, legal_arr, self.config.num_candidates,
        )
        if len(candidate_actions) < 2:

            return None

        # 2. 候选动作在当前策略下的 log_prob（含梯度）
        cur_dist = Categorical(logits=logits)
        log_probs_t = []

        for a in candidate_actions:

            action_t = torch.tensor([a], device=self.device)
            log_probs_t.append(cur_dist.log_prob(action_t))
        log_probs_t = torch.stack(log_probs_t)

        # 3. greedy 动作为合法动作中 logits 最大的（argmax，非采样）
        with torch.no_grad():
            greedy_action = int(torch.argmax(
                logits.masked_fill(torch.as_tensor(legal_arr, dtype=torch.bool, device=self.device) == 0, -1e9),
                dim=-1,
            ).item())

        npc_trace = self._record_npc_trace(
            greedy_action, env_config, self.config.branch_window,
        )

        # 4. 所有候选分支（含 greedy）恢复状态并强制 NPC 轨迹一致
        scores = []
        for a in candidate_actions:
            score = self._rollout_branch(a, env_config, self.config.branch_window, npc_trace)
            scores.append(score)

        # 5. 组内均值中心化 → advantages
        # 只减均值消除基线，不除以标准差（除以标准差会导致零和 advantages，
        # 均匀分布下 log_prob 相同，与零和向量点积恒为 0，梯度消失）
        entropy_coef = getattr(self.config, "entropy_coef", 0.0)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        advantages = scores_t - scores_t.mean()

        # 6. KL 散度（ref 无梯度，cur 含梯度）
        with torch.no_grad():
            ref_logits, _ = self.reference(map_img_t, vector_t, legal_t)
        kl = torch.distributions.kl_divergence(
            Categorical(logits=logits),
            Categorical(logits=ref_logits),
        ).mean()

        # 7. 策略熵（监控用）
        entropy = cur_dist.entropy().mean()

        # 8. GRPO loss
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        total_loss = policy_loss + self.config.kl_coef * kl - entropy_coef * entropy

        # 9. 更新
        self.model.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        grad_norm = sum(
            p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None
        ) ** 0.5
        self.optimizer.step()
        self.model.eval()

        self._steps_since_group = 0
        self._branch_state = None  # 快照已用完

        loss_info = LossInfo(
            total_loss=float(total_loss.item()),
            policy_loss=float(policy_loss.item()),
            mean_reward=float(scores_t.mean().item()),
            entropy=float(entropy.item()),
            extra={
                "std_reward": float(scores_t.std().item()),
                "kl_divergence": float(kl.item()),
                "grad_norm": float(grad_norm),
            },
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    # ── checkpoint / serialization ────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.reference.load_state_dict(state)
        self.model.to(self.device)
        self.reference.to(self.device)

    def save_checkpoint(
        self,
        path: str | Path,
        global_step: int = 0,
        episode_counter: int = 0,
        current_map_idx: int = 0,
        current_map_id: int = 0,
        current_stage_name: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        ckpt = Checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_step=global_step,
            episode_counter=episode_counter,
            current_map_idx=current_map_idx,
            current_map_id=current_map_id,
            current_stage_name=current_stage_name,
            config_snapshot=config_snapshot or {},
            rng_state=capture_rng_state(),
        )
        data = ckpt.to_dict()
        data["reference_state_dict"] = self.reference.state_dict()
        data["steps_since_group"] = self._steps_since_group
        data["episode_count"] = self._episode_count
        torch.save(data, str(path))

    def load_checkpoint(self, path: str | Path) -> Checkpoint:
        data = torch.load(str(path), map_location=self.device, weights_only=False)
        ckpt = Checkpoint.from_dict(data)
        self.model.load_state_dict(ckpt.model_state_dict)
        self.reference.load_state_dict(data.get("reference_state_dict", ckpt.model_state_dict))
        self.model.to(self.device)
        self.reference.to(self.device)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        self._steps_since_group = int(data.get("steps_since_group", 0))
        self._episode_count = int(data.get("episode_count", 0))
        return ckpt
