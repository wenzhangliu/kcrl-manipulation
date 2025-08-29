# MTSAC-SB3: Disentangled Alphas for Multi-Task SAC on top of SB3
from typing import Optional, Union
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update


class MTSAC(SAC):
    """
    Multi-Task SAC (disentangled temperature coefficients): requires the observation to include a task one-hot.
    Convention: if the observation is a vector, the one-hot is appended at the "end" of the observation
                and its dimension equals num_tasks;
                if the observation is a Dict, then the one-hot (or index) is stored under the key `task_key`.

    Additional parameters:
      - num_tasks: number of tasks
      - task_id_pos: 'tail' | 'head', the location of the one-hot when observation is a vector
      - task_key: the key name when the observation is a Dict
    """
    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        *,
        num_tasks: int,
        task_id_pos: str = "tail",
        task_key: str = "task",
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",     # Recommended "auto"
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        **kwargs,
    ):
        assert ent_coef == "auto" or isinstance(ent_coef, float), \
            "It is recommended to use ent_coef='auto' to learn entropy coefficient; a constant value works but is not disentangled"
        self.num_tasks = int(num_tasks)
        self.task_id_pos = task_id_pos
        self.task_key = task_key

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            **kwargs,
        )

        # Extra multi-task entropy coefficient optimizer
        # self.log_ent_coef_vec: Optional[th.Tensor] = None
        # self.mt_ent_coef_optimizer: Optional[th.optim.Optimizer] = None

    # ---- Replace scalar α with a vector α after the architecture is built ----
    def _setup_model(self) -> None:
        super()._setup_model()  # Sets target_entropy / actor/critic / base ent_coef etc.

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Initialize consistent with SB3 (you can specify init value via 'auto_0.1')
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0

            # Disentangled: log α parameters of length num_tasks
            self.log_ent_coef_vec = th.log(
                th.ones(self.num_tasks, device=self.device) * init_value
            ).requires_grad_(True)

            # Disable parent class scalar α; switch to our own optimizer
            self.log_ent_coef = None
            if self.ent_coef_optimizer is not None:
                # Clear parent optimizer's params to avoid duplicate steps
                for g in self.ent_coef_optimizer.param_groups:
                    g["params"] = []
            self.ent_coef_optimizer = None
            self.mt_ent_coef_optimizer = th.optim.Adam([self.log_ent_coef_vec], lr=self.lr_schedule(1))
        else:
            # Constant α: still works, but not disentangled
            self.log_ent_coef_vec = None
            self.mt_ent_coef_optimizer = None
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    # ---- Extract task one-hot from observations (or convert index -> one-hot) ----
    def _extract_one_hot(self, obs):
        if isinstance(obs, dict):
            x = obs[self.task_key]
            if x.ndim == 1:  # Index -> one-hot
                return F.one_hot(x.long(), num_classes=self.num_tasks).float()
            return x.float()
        # Vector observation
        if self.task_id_pos == "tail":
            return obs[:, -self.num_tasks:].float()  # 需环境观测包含任务ID（修改环境）
        elif self.task_id_pos == "head":
            return obs[:, :self.num_tasks].float()
        raise ValueError("Unknown task_id_pos, expected 'tail' or 'head'.")

    # ---- Training loop: replace scalar α with per-sample selected α ----
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.mt_ent_coef_optimizer is not None:
            optimizers.append(self.mt_ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for g in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            # Actions and log_prob under current policy
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # --- α (temperature) ---
            ent_coef_loss = None
            if self.log_ent_coef_vec is not None and self.mt_ent_coef_optimizer is not None:
                one_hot = self._extract_one_hot(replay_data.observations).to(self.device)  # (B, T)
                log_alpha_sel = one_hot @ self.log_ent_coef_vec.unsqueeze(1)               # (B, 1)
                ent_coef = th.exp(log_alpha_sel.detach())                                  # (B, 1)

                target = (log_prob + float(self.target_entropy)).detach()                  # (B, 1)
                # Key: only the selected log_alpha receives gradients; other tasks' gradients are zero
                ent_coef_loss = -(log_alpha_sel * target).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor  # Broadcast scalar

            ent_coefs.append(ent_coef.mean().item() if isinstance(ent_coef, th.Tensor) else float(ent_coef))

            if ent_coef_loss is not None:
                self.mt_ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.mt_ent_coef_optimizer.step()

            # --- Target Q calculation, note next_obs must also use the corresponding α ---
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                if self.log_ent_coef_vec is not None:
                    one_hot_next = self._extract_one_hot(replay_data.next_observations).to(self.device)
                    ent_coef_next = th.exp((one_hot_next @ self.log_ent_coef_vec.unsqueeze(1)).detach())
                else:
                    ent_coef_next = ent_coef

                next_q_values = next_q_values - ent_coef_next * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # --- Critic update ---
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor update (using per-sample α) ---
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(float(actor_loss.item()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Soft-update target networks
            if g % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Synchronize BN running stats (following SB3)
                polyak_update(
                    getattr(self, "batch_norm_stats", []),
                    getattr(self, "batch_norm_stats_target", []),
                    1.0,
                )

        self._n_updates += gradient_steps
        # Log some scalars
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
