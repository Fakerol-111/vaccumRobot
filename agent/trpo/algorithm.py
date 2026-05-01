"""Trust Region Policy Optimization (TRPO).

TRPO constrains the policy update by bounding the KL divergence:

    max   E[ (π_θ(a|s) / π_θ_old(a|s)) * A(s,a) ]
    s.t.  E[ KL[π_θ_old(·|s) || π_θ(·|s)] ] ≤ δ

The constrained optimisation is solved via:
  1. Conjugate gradient to compute the natural gradient direction: F⁻¹g
  2. Line search along that direction to enforce the KL constraint

Actor and critic use *separate* networks (``SeparateActorCritic``) so that
value-function updates never perturb the policy's trust region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent.base import ActResult, Algorithm, LossInfo
from agent.common.checkpoint import Checkpoint, capture_rng_state
from agent.nn import SeparateActorCritic
from agent.ppo.batch import compute_gae
from agent.ppo_kl.algorithm import kl_div_categorical
from agent.registry import register
from agent.trpo.trpo_metrics import TRPOMetricsReporter


@register("trpo")
class TRPOAlgorithm(Algorithm):
    """Trust Region Policy Optimisation with separate actor/critic networks.

    Always uses ``SeparateActorCritic`` so that value-function updates never
    perturb the policy's trust region.
    """

    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = SeparateActorCritic(num_actions=config.num_actions)
        self.model.to(self.device)

        self._policy_params = list(self.model.actor.parameters())
        self._value_params = list(self.model.critic.parameters())

        # Value-only optimiser only touches critic — no impact on trust region
        self.value_optimizer = optim.Adam(self._value_params, lr=config.learning_rate)
        self._metrics_reporter = TRPOMetricsReporter()

        # ── internal buffer ──────────────────────────────────
        self._map_imgs: list[np.ndarray] = []
        self._vectors: list[np.ndarray] = []
        self._legal_masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._old_log_probs: list[float] = []
        self._old_logits: list[np.ndarray] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

        self._last_logits: np.ndarray | None = None

    # ── Metrics ─────────────────────────────────────────────

    @property
    def metrics_reporter(self) -> TRPOMetricsReporter:
        return self._metrics_reporter

    # ── Tensor helpers ──────────────────────────────────────

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

    def _run_model(self, map_img, vector, legal_mask):
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            logits, value = self.model(map_img_t, vector_t, legal_t)
            dist = Categorical(logits=logits)
        return logits, value, dist

    # ── Action selection ────────────────────────────────────

    def explore(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self._last_logits = logits.squeeze(0).detach().cpu().numpy()
        return ActResult(action=action.item(), log_prob=log_prob.item(), value=value.item())

    def exploit(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return ActResult(action=action.item(), log_prob=log_prob.item(), value=value.item())

    def compute_value(self, map_img: np.ndarray, vector: np.ndarray, legal_mask: np.ndarray) -> float:
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            _, value = self.model(map_img_t, vector_t, legal_t)
        return value.item()

    # ── Experience collection ───────────────────────────────

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
        self._map_imgs.append(map_img)
        self._vectors.append(vector)
        self._legal_masks.append(np.array(legal_mask, dtype=np.float32))
        self._actions.append(action)
        self._old_log_probs.append(log_prob)
        self._old_logits.append(self._last_logits.copy())
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

    def _buffer_clear(self) -> None:
        self._map_imgs.clear()
        self._vectors.clear()
        self._legal_masks.clear()
        self._actions.clear()
        self._old_log_probs.clear()
        self._old_logits.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()

    def ready_to_update(self) -> bool:
        return len(self._rewards) >= self.config.batch_size

    # ── Natural-gradient helpers ────────────────────────────

    @staticmethod
    def _flat_params(params: list[nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in params])

    @staticmethod
    def _set_params(params: list[nn.Parameter], flat_params: torch.Tensor) -> None:
        idx = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat_params[idx: idx + n].view(p.shape))
            idx += n

    @staticmethod
    def _flat_grad(
        loss: torch.Tensor,
        params: list[nn.Parameter],
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _fisher_vector_product(
        self,
        vec: torch.Tensor,
        old_logits: torch.Tensor,
        map_imgs: torch.Tensor,
        vectors: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.model(map_imgs, vectors, legal_masks)
        kl = kl_div_categorical(old_logits, logits).mean()

        grads = torch.autograd.grad(kl, self._policy_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])

        gv = (flat_grad * vec).sum()
        hvp = torch.autograd.grad(gv, self._policy_params, retain_graph=False)
        flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp])

        return flat_hvp + self.config.cg_damping * vec

    @staticmethod
    def _conjugate_gradient(
        matvec: callable,
        b: torch.Tensor,
        nsteps: int,
        residual_tol: float = 1e-10,
    ) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = r.dot(r)

        for _ in range(nsteps):
            Avp = matvec(p)
            alpha = rdotr / p.dot(Avp)
            x.add_(alpha * p)
            r.sub_(alpha * Avp)
            new_rdotr = r.dot(r)
            if new_rdotr.sqrt() < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    # ── Update ──────────────────────────────────────────────

    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        rewards = np.array(self._rewards, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.int8)

        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda, bootstrap_value,
        )
        mean_reward = float(rewards.mean())
        n = len(rewards)

        map_imgs_t = torch.as_tensor(
            np.stack(self._map_imgs), dtype=torch.float32, device=self.device,
        )
        vectors_t = torch.as_tensor(
            np.stack(self._vectors), dtype=torch.float32, device=self.device,
        )
        legal_masks_t = torch.as_tensor(
            np.stack(self._legal_masks), dtype=torch.float32, device=self.device,
        )
        actions_t = torch.as_tensor(
            np.array(self._actions, dtype=np.int64), device=self.device,
        )
        old_log_probs_t = torch.as_tensor(
            np.array(self._old_log_probs, dtype=np.float32), device=self.device,
        )
        old_logits_t = torch.as_tensor(
            np.stack(self._old_logits), dtype=torch.float32, device=self.device,
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self._buffer_clear()

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── 3. Policy update via natural gradient ──────────────────
        self.model.train()

        old_policy_flat = self._flat_params(self._policy_params)

        logits, _ = self.model(map_imgs_t, vectors_t, legal_masks_t)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_t)
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surrogate_loss = (ratio * advantages_t).mean()

        policy_g = self._flat_grad(surrogate_loss, self._policy_params)

        def fisher_product(vec: torch.Tensor) -> torch.Tensor:
            return self._fisher_vector_product(
                vec, old_logits_t, map_imgs_t, vectors_t, legal_masks_t,
            )

        step_dir = self._conjugate_gradient(
            fisher_product, policy_g, nsteps=self.config.cg_iterations,
        )

        fv = fisher_product(step_dir)
        shs = (step_dir * fv).sum(0, keepdim=True).clamp(min=1e-8)
        full_step = step_dir * torch.sqrt(2.0 * self.config.max_kl / shs)

        # ── 4. Line search ────────────────────────────────────────
        old_surrogate = surrogate_loss.item()
        accepted_step: float | None = None

        for frac in (0.5 ** i for i in range(self.config.line_search_steps)):
            new_params_flat = old_policy_flat + frac * full_step
            self._set_params(self._policy_params, new_params_flat)

            with torch.no_grad():
                logits_new, _ = self.model(map_imgs_t, vectors_t, legal_masks_t)
                kl = kl_div_categorical(old_logits_t, logits_new).mean().item()

                if kl > self.config.max_kl:
                    continue

                dist_new = Categorical(logits=logits_new)
                ratio_new = torch.exp(dist_new.log_prob(actions_t) - old_log_probs_t)
                new_surrogate = (ratio_new * advantages_t).mean().item()

                if new_surrogate >= old_surrogate - 1e-6:
                    accepted_step = frac
                    break
        else:
            self._set_params(self._policy_params, old_policy_flat)

        # ── 4a. Evaluate final policy after line search ──────────
        with torch.no_grad():
            final_logits_ls, _ = self.model(map_imgs_t, vectors_t, legal_masks_t)
            final_kl = kl_div_categorical(old_logits_t, final_logits_ls).mean().item()
            final_dist_ls = Categorical(logits=final_logits_ls)
            final_ratio = torch.exp(final_dist_ls.log_prob(actions_t) - old_log_probs_t)
            final_surrogate = (final_ratio * advantages_t).mean().item()

        # ── 5. Value-function update (multi-epoch, mini-batch) ────
        total_value_loss = 0.0
        n_value_updates = 0

        for _ in range(self.config.value_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.value_mini_batch_size):
                idx = indices[start:start + self.config.value_mini_batch_size]

                m_i = map_imgs_t[idx]
                v_i = vectors_t[idx]
                l_i = legal_masks_t[idx]
                ret_i = returns_t[idx]

                _, v_pred = self.model(m_i, v_i, l_i)
                value_loss = F.mse_loss(v_pred.squeeze(-1), ret_i)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self._value_params, self.config.max_grad_norm)
                self.value_optimizer.step()

                total_value_loss += value_loss.item()
                n_value_updates += 1

        avg_value_loss = total_value_loss / max(n_value_updates, 1)

        with torch.no_grad():
            final_logits, _ = self.model(map_imgs_t, vectors_t, legal_masks_t)
            final_dist = Categorical(logits=final_logits)
            entropy = final_dist.entropy().mean().item()

        self.model.eval()

        loss_info = LossInfo(
            total_loss=0.0,
            policy_loss=final_surrogate,
            value_loss=avg_value_loss,
            entropy=entropy,
            mean_reward=mean_reward,
            extra={
                "surrogate_loss": final_surrogate,  # for metrics reporter compatibility
                "surrogate_before": old_surrogate,
                "surrogate_after": final_surrogate,
                "kl": final_kl,
                "max_kl": self.config.max_kl,
                "line_search_step": accepted_step,
                "line_search_accepted": accepted_step is not None,
            },
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    # ── Serialisation ───────────────────────────────────────

    def save(self, path: str | Path) -> None:
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.value_optimizer.state_dict(),
        }
        torch.save(state, str(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                self.value_optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device)

    def save_checkpoint(
        self,
        path: str | Path,
        global_step: int,
        episode_counter: int = 0,
        current_map_idx: int = 0,
        current_map_id: int = 0,
        current_stage_name: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        ckpt = Checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.value_optimizer.state_dict(),
            global_step=global_step,
            episode_counter=episode_counter,
            current_map_idx=current_map_idx,
            current_map_id=current_map_id,
            current_stage_name=current_stage_name,
            config_snapshot=config_snapshot or {},
            rng_state=capture_rng_state(),
        )
        torch.save(ckpt.to_dict(), str(path))

    def load_checkpoint(self, path: str | Path) -> Checkpoint:
        data = torch.load(str(path), map_location=self.device, weights_only=False)
        ckpt = Checkpoint.from_dict(data)
        self.model.load_state_dict(ckpt.model_state_dict)
        self.model.to(self.device)
        self.value_optimizer.load_state_dict(ckpt.optimizer_state_dict)
        return ckpt
