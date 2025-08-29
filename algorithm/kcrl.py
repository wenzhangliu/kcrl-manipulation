import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from stable_baselines3.common import policies
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .layer import CompositionalFC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
import os
import time


class TaskEncodingNetwork(nn.Module):

    def __init__(self, num_tasks: int, num_param_sets: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding = nn.Embedding(num_tasks, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_param_sets),
            nn.Softmax(dim=-1)
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        if task_ids.dim() == 0:
            task_ids = task_ids.unsqueeze(0)
        emb = self.embedding(task_ids.long())
        return self.mlp(emb)


class KCRL(BasePolicy):

    def __init__(self,
                 env,
                 num_tasks: int = 3,
                 num_param_sets: int = 5,
                 learning_rate: float = 3e-4,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 use_sde: bool = False,
                 tensorboard_log: Optional[str] = None,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256):

        self.env = env
        super().__init__(
            env.observation_space,
            env.action_space,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            optimizer_class=torch.optim.Adam,
            squash_output=True,
        )

        self.tensorboard_log = tensorboard_log

        self.task_encoder = TaskEncodingNetwork(num_tasks, num_param_sets)
        self.actor = self._build_actor(env.observation_space.shape[0], env.action_space.shape[0], num_param_sets)
        self.critic1, self.critic2, self.target_critic1, self.target_critic2 = self._build_critics(
            env.observation_space.shape[0], env.action_space.shape[0], num_param_sets)

        self.actor_optimizer = self.optimizer_class(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = self.optimizer_class(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate
        )

        self.tau = tau
        self.gamma = gamma
        self.num_param_sets = num_param_sets
        self.use_sde = use_sde
        self.action_space_high = torch.tensor(self.action_space.high, dtype=torch.float32)
        self.action_space_low = torch.tensor(self.action_space.low, dtype=torch.float32)
        self.batch_size = batch_size
        self.action_space_high = self.action_space_high.to(self.device)
        self.action_space_low = self.action_space_low.to(self.device)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device
        )
        self.num_timesteps = 0
        self._n_updates = 0
        log_path = os.path.join(self.tensorboard_log, 'logs') if self.tensorboard_log else None
        self.logger = configure(log_path, ["stdout", "tensorboard"])

    def _build_actor(self, obs_dim: int, act_dim: int, num_param_sets: int) -> nn.Module:
        return nn.ModuleList([
            CompositionalFC(obs_dim, 256, num_param_sets),
            CompositionalFC(256, 256, num_param_sets),
            nn.Linear(256, act_dim)
        ])

    def _build_critics(self, obs_dim: int, act_dim: int, num_param_sets: int) -> Tuple[nn.Module, ...]:
        def build_q_net():
            return nn.ModuleList([
                CompositionalFC(obs_dim + act_dim, 256, num_param_sets),
                CompositionalFC(256, 256, num_param_sets),
                nn.Linear(256, 1)
            ])
        critic1 = build_q_net()
        critic2 = build_q_net()
        target_critic1 = build_q_net()
        target_critic2 = build_q_net()
        target_critic1.load_state_dict(critic1.state_dict())
        target_critic2.load_state_dict(critic2.state_dict())
        for param in target_critic1.parameters():
            param.requires_grad = False
        for param in target_critic2.parameters():
            param.requires_grad = False

        return critic1, critic2, target_critic1, target_critic2

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            batch_size = obs.shape[0]
            task_ids = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            actions, _ = self.forward(obs, task_ids, deterministic=deterministic)
            return actions

    def forward(self,
                obs: torch.Tensor,
                task_ids: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, TensorDict]:
        comp_weights = self.task_encoder(task_ids)
        x = obs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        for layer in self.actor[:-1]:
            if isinstance(layer, CompositionalFC):
                x, _ = layer((x, comp_weights))
                x = F.relu(x)
            else:
                x = layer(x)
        actions = self.actor[-1](x)

        actions = torch.tanh(actions)
        actions = self.action_space_low + (self.action_space_high - self.action_space_low) * (actions + 1) / 2

        if single_sample:
            actions = actions.squeeze(0)
        if actions.size(-1) != self.action_space.shape[0]:
            raise ValueError(f"Expected action dimension to be {self.action_space.shape[0]}, but got {actions.size(-1)}.")
        return actions.detach(), {
            "comp_weights": comp_weights.detach()
        }

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
        for net in [self.critic1, self.critic2, self.target_critic1, self.target_critic2]:
            net.train(mode)

    def update(self, replay_buffer, batch_size: int):
        obs, actions, next_obs, dones, rewards, _ = replay_buffer.sample(batch_size)
        task_ids = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        obs = obs.clone().detach().to(self.device).to(torch.float32)
        actions = actions.clone().detach().to(self.device).to(torch.float32)
        rewards = rewards.clone().detach().to(self.device).to(torch.float32)
        next_obs = next_obs.clone().detach().to(self.device).to(torch.float32)
        dones = dones.clone().detach().to(self.device).to(torch.float32)

        if rewards.dim() > 1:
            rewards = torch.sum(rewards, dim=1)
        with torch.no_grad():
            comp_weights = self.task_encoder(task_ids)
        with torch.no_grad():
            next_actions, _ = self.forward(next_obs, task_ids)
            target_q1 = self._q_forward(next_obs, next_actions, self.target_critic1, comp_weights)
            target_q2 = self._q_forward(next_obs, next_actions, self.target_critic2, comp_weights)
            target_q = torch.min(target_q1, target_q2)

            target_q = rewards + (1 - dones.squeeze()) * self.gamma * target_q

        current_q1 = self._q_forward(obs, actions, self.critic1, comp_weights)
        current_q2 = self._q_forward(obs, actions, self.critic2, comp_weights)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
        actions_pred, _ = self.forward(obs, task_ids)
        q1 = self._q_forward(obs, actions_pred, self.critic1, comp_weights)
        actor_loss = -q1.mean()
        actor_loss.requires_grad = True
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return actor_loss.item(), critic_loss.item()

    def _q_forward(self,
                   obs: torch.Tensor,
                   actions: torch.Tensor,
                   q_net: nn.ModuleList,
                   comp_weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        for layer in q_net:
            if isinstance(layer, CompositionalFC):
                x, _ = layer((x, comp_weights))
                x = F.relu(x)
            else:
                x = layer(x)
        return x.squeeze(-1)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def predict(self,
                observation: np.ndarray,
                state: Optional[Tuple] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple]]:

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        task_ids = torch.zeros(obs_tensor.size(0), dtype=torch.long).to(self.device)

        with torch.no_grad():
            actions, _ = self.forward(obs_tensor, task_ids, deterministic=deterministic)
        if observation.shape == self.observation_space.shape:
            return actions.squeeze(0).cpu().numpy(), state
        return actions.cpu().numpy(), state

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save_shared_params(self, path: str) -> None:
        shared_params = {
            'base_weights': [layer.base_weights.data.clone() for layer in self.actor if
                             isinstance(layer, CompositionalFC)],
            'task_encoder': self.task_encoder.state_dict()
        }
        torch.save(shared_params, path)

    def load_shared_params(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        for i, layer in enumerate(self.actor):
            if isinstance(layer, CompositionalFC) and i < len(checkpoint['base_weights']):
                layer.base_weights.data.copy_(checkpoint['base_weights'][i])
        self.task_encoder.load_state_dict(checkpoint['task_encoder'])

    def reset_noise(self, batch_size: int = 1) -> None:
        if self.use_sde:
            for layer in self.actor:
                if isinstance(layer, CompositionalFC):
                    layer.reset_noise(batch_size)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_tasks=self.task_encoder.num_tasks,
            num_param_sets=self.num_param_sets,
            learning_rate=self.learning_rate,
            tau=self.tau,
            gamma=self.gamma,
            use_sde=self.use_sde
        )

    def learn(self, total_timesteps, callback=None, log_interval=4):
        obs, _ = self.env.reset()
        if callback is not None:
            callback.init_callback(self)

        actor_losses = []
        critic_losses = []
        ent_coefs = []
        ent_coef_losses = []

        for step in range(total_timesteps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                task_ids = torch.zeros(1, dtype=torch.long).to(self.device)

                action_tensor, _ = self.forward(obs_tensor, task_ids)
                action = action_tensor.cpu().numpy()
            if action.shape != self.action_space.shape:
                raise ValueError(f"Expected action shape to be {self.action_space.shape}, but got {action.shape}.")


            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            infos = [info] if isinstance(info, dict) else info

            self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

            if self.replay_buffer.size() > self.batch_size:
                actor_loss, critic_loss = self.update(self.replay_buffer, self.batch_size)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                self._n_updates += 1
            self.num_timesteps += 1

            if callback is not None:
                if callback.on_step() is False:
                    break

            if step % log_interval == 0:
                actor_loss_mean = np.mean(actor_losses) if actor_losses else 0
                critic_loss_mean = np.mean(critic_losses) if critic_losses else 0
                ent_coef_mean = np.mean(ent_coefs) if ent_coefs else 0
                ent_coef_loss_mean = np.mean(ent_coef_losses) if ent_coef_losses else 0

                self.logger.record("train/n_updates", self._n_updates)
                self.logger.record("train/ent_coef", ent_coef_mean)
                self.logger.record("train/actor_loss", actor_loss_mean)
                self.logger.record("train/critic_loss", critic_loss_mean)
                if len(ent_coef_losses) > 0:
                    self.logger.record("train/ent_coef_loss", ent_coef_loss_mean)

                self.logger.dump(step)

                actor_losses = []
                critic_losses = []
                ent_coefs = []
                ent_coef_losses = []

            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        return self
