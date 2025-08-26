import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TaskEncodingNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_tasks: int,
                 num_param_sets: int,
                 hidden_dims: tuple = (64, 64),
                 activation_fn: nn.Module = nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_fn())
            prev_dim = dim
        self.base_net = nn.Sequential(*layers)
        self.weight_layer = nn.Linear(prev_dim, num_param_sets * num_tasks)
        self.num_param_sets = num_param_sets
        self.num_tasks = num_tasks

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        features = self.base_net(task_ids)
        weights = self.weight_layer(features)
        weights = weights.view(-1, self.num_param_sets, self.num_tasks)  # [B, K, M]
        return F.softmax(weights, dim=-1)


class CompositionalActor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 num_param_sets: int,
                 hidden_dims: tuple = (256, 256)):
        super().__init__()
        from .layer import CompositionalFC

        self.layers = nn.ModuleList()
        input_dim = obs_dim
        for dim in hidden_dims:
            self.layers.append(
                CompositionalFC(input_dim, dim, num_param_sets)
            )
            input_dim = dim
        self.output_layer = CompositionalFC(
            input_dim, action_dim * 2, num_param_sets
        )

    def forward(self,
                obs: torch.Tensor,
                comp_weights: torch.Tensor) -> torch.distributions.Distribution:
        for layer in self.layers:
            obs, _ = layer((obs, comp_weights))
        mu_logstd = self.output_layer((obs, comp_weights))[0]
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        return torch.distributions.Normal(mu, log_std.exp())


class CompositionalCritic(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 num_param_sets: int,
                 hidden_dims: tuple = (256, 256)):
        super().__init__()
        from .layer import CompositionalFC

        self.layers = nn.ModuleList()
        input_dim = obs_dim + action_dim
        for dim in hidden_dims:
            self.layers.append(
                CompositionalFC(input_dim, dim, num_param_sets)
            )
            input_dim = dim

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor,
                comp_weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        for layer in self.layers:
            x, _ = layer((x, comp_weights))
        return self.output_layer(x)