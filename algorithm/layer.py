import torch
import torch.nn as nn
import numpy as np

class CompositionalFC(nn.Module):
    def __init__(self, in_features, out_features, num_param_sets, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_param_sets = num_param_sets
        self.base_weights = nn.Parameter(torch.randn(num_param_sets, out_features, in_features))
        if bias:
            self.base_bias = nn.Parameter(torch.zeros(num_param_sets, out_features))
        else:
            self.register_parameter('base_bias', None)
        nn.init.orthogonal_(self.base_weights, gain=np.sqrt(2))
        if bias:
            nn.init.constant_(self.base_bias, 0.1)

    def forward(self, inputs):
        x, comp_weights = inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if comp_weights.dim() == 1:
            comp_weights = comp_weights.unsqueeze(0)
        combined_weights = torch.einsum(
            'bk,koh->boh',
            comp_weights,
            self.base_weights
        )
        output = torch.einsum('bi,boh->bo', x, combined_weights)
        if self.base_bias is not None:
            combined_bias = torch.einsum(
                'bk,ko->bo',
                comp_weights,
                self.base_bias
            )
            output += combined_bias

        assert output.size(-1) == self.out_features, f"Expected output dimension to be {self.out_features}, but got {output.size(-1)}."

        return output, combined_weights
