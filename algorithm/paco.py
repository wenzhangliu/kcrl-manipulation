import os
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import BasePolicy

# Assumed available from your project
from .layer import CompositionalFC


# -----------------------------
# Task Encoder
# -----------------------------
class TaskEncodingNetwork(nn.Module):
    """Task encoder producing composition weights over K parameter sets.

    Expects either integer task ids or a one-hot (last dim) embedded in obs.
    """

    def __init__(self, num_tasks: int, num_param_sets: int, emb_dim: int = 32):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_param_sets = num_param_sets
        self.embedding = nn.Embedding(num_tasks, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ReLU(),
            nn.Linear(64, num_param_sets), nn.Softmax(dim=-1)
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        emb = self.embedding(task_ids.long())  # [B, emb]
        return self.mlp(emb)  # [B, K], simplex over K


# -----------------------------
# Actor (Gaussian, tanh-squashed), task-conditioned via CompositionalFC
# -----------------------------
class CompositionalGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_param_sets: int,
                 log_std_bounds: Tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.layers = nn.ModuleList([
            CompositionalFC(obs_dim, 256, num_param_sets),
            CompositionalFC(256, 256, num_param_sets),
        ])
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.log_std_bounds = log_std_bounds
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std.bias, -0.5)

    def forward(self, obs: torch.Tensor, comp_w: torch.Tensor):
        x = obs
        for layer in self.layers:
            x, _ = layer((x, comp_w))
            x = F.relu(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        low, high = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = low + 0.5 * (log_std + 1.0) * (high - low)
        std = torch.exp(log_std)
        return mu, std, log_std

    @staticmethod
    def _tanh_squash(mu: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        dist = torch.distributions.Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        # log_prob with tanh correction
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [B,1]
        return action, log_prob, pre_tanh


# -----------------------------
# Critic Q(s,a) with compositional layers
# -----------------------------
class CompositionalQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_param_sets: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CompositionalFC(obs_dim + act_dim, 256, num_param_sets),
            CompositionalFC(256, 256, num_param_sets),
        ])
        self.out = nn.Linear(256, 1)
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, comp_w: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        for layer in self.layers:
            x, _ = layer((x, comp_w))
            x = F.relu(x)
        return self.out(x)  # [B,1]


# -----------------------------
# PaCo-style SAC (continuous actions) in SB3-style shell
# -----------------------------
class PACO(BasePolicy):
    """PACO (parameter compositional multi-task SAC) for Box action spaces.

    Differences vs vanilla SAC:
      • Task-conditioned actor/critics via composition weights from a task encoder.
      • Per-task temperature (alpha) selected by one-hot task_id inside obs (last num_tasks dims)
        or by explicit task_ids passed to .forward/.predict.
      • Optional online resetting for extreme tasks.
    """

    def __init__(self,
                 env,
                 num_tasks: int = 1,
                 num_param_sets: int = 8,
                 learning_rate: float = 3e-4,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 target_entropy: Optional[float] = None,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
                 tensorboard_log: Optional[str] = None,
                 mask_extreme_task: bool = False,
                 reset_task_while_extreme: bool = False,
                 task_loss_threshold: float = 3e3,
                 reset_task_method: str = "interpolate"):
        self.env = env
        super().__init__(
            env.observation_space,
            env.action_space,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            optimizer_class=torch.optim.Adam,
            squash_output=True,
        )
        assert len(env.action_space.shape) == 1, "PaCoSAC supports Box actions"

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.shape[0]

        # ---- task encoder & actor/critics ----
        self.task_encoder = TaskEncodingNetwork(num_tasks, num_param_sets)
        self.actor = CompositionalGaussianActor(obs_dim, act_dim, num_param_sets, log_std_bounds)
        self.q1 = CompositionalQ(obs_dim, act_dim, num_param_sets)
        self.q2 = CompositionalQ(obs_dim, act_dim, num_param_sets)
        self.target_q1 = CompositionalQ(obs_dim, act_dim, num_param_sets)
        self.target_q2 = CompositionalQ(obs_dim, act_dim, num_param_sets)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

        # ---- optimizers ----
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.task_encoder.parameters()), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )
        # per-task log alpha
        self.log_alpha = nn.Parameter(torch.zeros(num_tasks))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        # ---- hyperparams ----
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_param_sets = num_param_sets
        self.learning_rate = learning_rate
        self.mask_extreme_task = mask_extreme_task
        self.reset_task_while_extreme = reset_task_while_extreme
        self.task_loss_threshold = task_loss_threshold
        assert reset_task_method in ("interpolate", "copy")
        self.reset_task_method = reset_task_method

        # ---- action bounds as buffers ----
        self.register_buffer("act_low", torch.as_tensor(self.action_space.low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(self.action_space.high, dtype=torch.float32))

        # ---- entropy target ----
        self.target_entropy = (
            -float(act_dim) if target_entropy is None else float(target_entropy)
        )

        # ---- rb, logs ----
        self.replay_buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space, device=self.device)
        self._n_updates = 0
        self.num_timesteps = 0
        log_path = os.path.join(tensorboard_log, 'logs') if tensorboard_log else None
        self.logger = configure(log_path, ["stdout", "tensorboard"])

    # ------------- utils -------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def soft_update(self, online: nn.Module, target: nn.Module):
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _extract_task_one_hot(self, obs: torch.Tensor) -> torch.Tensor:
        """Assume last num_tasks dims represent one-hot task id.
        If not present, fall back to task 0 and warn once.
        """
        if obs.shape[-1] >= self.observation_space.shape[0] and self.num_tasks > 1:
            # try last dims as one-hot
            maybe = obs[..., -self.num_tasks:]
            # basic sanity: values close to {0,1}
            if (maybe.max() <= 1.05) and (maybe.min() >= -1e-3):
                return (maybe > 0.5).float()
        # fallback: all samples -> task0
        oh = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device)
        oh[:, 0] = 1.0
        return oh

    def _current_log_alpha(self, task_one_hot: torch.Tensor) -> torch.Tensor:
        # [B,T]x[Ntasks] @ [Ntasks] -> [B,1]
        la = (task_one_hot @ self.log_alpha.unsqueeze(-1))  # [B,1]
        return la

    def _scale_to_env(self, a: torch.Tensor) -> torch.Tensor:
        # assume input a in [-1,1]
        return self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)

    # ------------- policy forward -------------
    def policy(self, obs: torch.Tensor, task_ids: Optional[torch.Tensor] = None):
        B = obs.shape[0]
        if task_ids is None:
            task_one_hot = self._extract_task_one_hot(obs)
            task_ids = task_one_hot.argmax(dim=-1)
        comp_w = self.task_encoder(task_ids)  # [B,K]
        mu, std, _ = self.actor(obs, comp_w)
        a_norm, logp, pre_tanh = self.actor._tanh_squash(mu, std)
        a = self._scale_to_env(a_norm)
        return a, logp, mu, std, comp_w

    @torch.no_grad()
    def _policy_no_grad(self, obs: torch.Tensor, task_ids: Optional[torch.Tensor] = None):
        return self.policy(obs, task_ids)

    def _q_all(self, obs: torch.Tensor, act: torch.Tensor, comp_w: torch.Tensor):
        q1 = self.q1(obs, act, comp_w)
        q2 = self.q2(obs, act, comp_w)
        return q1, q2

    # ------------- BasePolicy API -------------
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            B = observation.shape[0]
            task_one_hot = self._extract_task_one_hot(observation)
            task_ids = task_one_hot.argmax(dim=-1)
            comp_w = self.task_encoder(task_ids)
            mu, std, _ = self.actor(observation, comp_w)
            if deterministic:
                a_norm = torch.tanh(mu)
                a = self._scale_to_env(a_norm)
            else:
                a, _, _, = self.actor._tanh_squash(mu, std)
                a = self._scale_to_env(a)
            return a

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=False):
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            act = self._predict(obs_t, deterministic)
        if observation.shape == self.observation_space.shape:
            return act.squeeze(0).cpu().numpy(), state
        return act.cpu().numpy(), state

    # ------------- training -------------
    def update(self, batch):
        obs = batch.observations.float()
        actions = batch.actions.float()
        rewards = batch.rewards.float()
        dones = batch.dones.float()
        next_obs = batch.next_observations.float()

        # task ids from obs
        task_one_hot = self._extract_task_one_hot(obs)
        task_ids = task_one_hot.argmax(dim=-1)
        comp_w = self.task_encoder(task_ids)  # [B,K]

        # --- Next action for target ---
        with torch.no_grad():
            next_task_one_hot = self._extract_task_one_hot(next_obs)
            next_task_ids = next_task_one_hot.argmax(dim=-1)
            next_comp_w = self.task_encoder(next_task_ids)
            mu2, std2, _ = self.actor(next_obs, next_comp_w)
            next_a_norm, next_logp, _ = self.actor._tanh_squash(mu2, std2)
            next_a = self._scale_to_env(next_a_norm)
            tq1 = self.target_q1(next_obs, next_a, next_comp_w)
            tq2 = self.target_q2(next_obs, next_a, next_comp_w)
            target_q = torch.min(tq1, tq2)
            curr_log_alpha = self._current_log_alpha(next_task_one_hot)  # [B,1]
            target = rewards + (1.0 - dones) * self.gamma * (target_q - curr_log_alpha.exp() * next_logp)

        # --- Critic loss ---
        q1 = self.q1(obs, actions, comp_w)
        q2 = self.q2(obs, actions, comp_w)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.critic_optimizer.step()

        # --- Actor loss ---
        act, logp, mu, std, comp_w = self.policy(obs, task_ids)
        q1_pi, q2_pi = self._q_all(obs, act, comp_w)
        q_pi = torch.min(q1_pi, q2_pi)
        curr_log_alpha = self._current_log_alpha(task_one_hot)
        actor_loss = (curr_log_alpha.exp() * logp - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.task_encoder.parameters()), 1.0)
        self.actor_optimizer.step()

        # --- Temperature (per-task) ---
        curr_log_alpha_alpha = self._current_log_alpha(task_one_hot)  # 不 detach，这里需要对 α 求导
        alpha_loss = (-(curr_log_alpha_alpha * (logp.detach() + self.target_entropy))).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update ---
        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)

        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 4):
        obs, _ = self.env.reset()
        episode_rewards = 0.0
        episode_length = 0
        num_episodes = 0

        if callback is not None:
            callback.init_callback(self)

        al_hist, cl_hist, a_hist = [], [], []

        for step in range(total_timesteps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            with torch.no_grad():
                action_t = self._predict(obs_t, deterministic=False)
                action = action_t.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            episode_length += 1

            infos = [info] if isinstance(info, dict) else info
            self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

            if self.replay_buffer.size() > self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                actor_loss, critic_loss, alpha_loss = self.update(batch)
                al_hist.append(actor_loss); cl_hist.append(critic_loss); a_hist.append(alpha_loss)
                self._n_updates += 1

            self.num_timesteps += 1
            if callback is not None and callback.on_step() is False:
                break

            if done:
                self.logger.record("rollout/ep_rew_mean", episode_rewards)
                self.logger.record("rollout/ep_rew_len", episode_length)
                obs, _ = self.env.reset()
                episode_rewards = 0.0
                episode_length = 0
                num_episodes += 1

                if num_episodes % log_interval == 0:
                    self.logger.record("train/n_updates", self._n_updates)
                    if al_hist:
                        self.logger.record("train/actor_loss", float(np.mean(al_hist)))
                        self.logger.record("train/critic_loss", float(np.mean(cl_hist)))
                        self.logger.record("train/alpha_loss", float(np.mean(a_hist)))
                    self.logger.dump(step)
                    al_hist, cl_hist, a_hist = [], [], []
            else:
                obs = next_obs

        return self

    # ------------- saving shared params (like PaCo) -------------
    def save_shared_params(self, path: str) -> None:
        shared = {
            'actor_comp_base': [m.base_weights.data.clone() for m in self.actor.layers],
            'critic1_comp_base': [m.base_weights.data.clone() for m in self.q1.layers],
            'critic2_comp_base': [m.base_weights.data.clone() for m in self.q2.layers],
            'task_encoder': self.task_encoder.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu()
        }
        torch.save(shared, path)

    def load_shared_params(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        for lst, key in [(self.actor.layers, 'actor_comp_base'), (self.q1.layers, 'critic1_comp_base'), (self.q2.layers, 'critic2_comp_base')]:
            for i, layer in enumerate(lst):
                if isinstance(layer, CompositionalFC):
                    layer.base_weights.data.copy_(ckpt[key][i])
        self.task_encoder.load_state_dict(ckpt['task_encoder'])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'].to(self.device))

    # ------------- optional: online resetting for extreme tasks -------------
    def reset_extreme_tasks(self, task_ids: torch.Tensor):
        """Reset task-specific parameters for given task ids (1D list tensor).
        Here we simply reinitialize their embeddings; interpolation can be implemented as needed.
        """
        with torch.no_grad():
            if self.reset_task_method == "copy":
                # copy from a random *non-extreme* task (here: task 0 fallback)
                src = int((set(range(self.num_tasks)) - set(task_ids.tolist()) or {0}).pop())
                for tid in task_ids.tolist():
                    self.task_encoder.embedding.weight[tid].copy_(self.task_encoder.embedding.weight[src])
            else:  # interpolate
                mean_vec = self.task_encoder.embedding.weight.mean(dim=0, keepdim=True)
                for tid in task_ids.tolist():
                    noise = torch.randn_like(mean_vec) * 0.05
                    self.task_encoder.embedding.weight[tid].copy_(mean_vec[0] + noise[0])

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_tasks=self.num_tasks,
            num_param_sets=self.num_param_sets,
            learning_rate=self.learning_rate,
            tau=self.tau,
            gamma=self.gamma,
        )
