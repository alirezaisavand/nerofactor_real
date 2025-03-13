import torch
import torch.nn as nn
import torch.nn.functional as F

from network.seq_brdf import SequentialNetwork

class MLPNetwork(SequentialNetwork):
    def __init__(self, widths, act=None, skip_at=None):
        super(MLPNetwork, self).__init__()
        depth = len(widths)
        self.skip_at = skip_at or []

        if act is None:
            act = [None] * (depth-1)

        assert len(act) == (depth-1), "If not `None`, `act` must have the same length as `widths`"

        activation_map = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'softmax': nn.Softmax,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
            None: nn.Identity  # Handle case where no activation is specified
        }

        for i in range(depth - 1):
            input_dim = widths[i]
            if i-1 in self.skip_at:
                input_dim += widths[0]  # Account for concatenation with input

            output_dim = widths[i + 1]
            a = act[i]
            activation = activation_map.get(a.lower() if a else None, None)
            if activation is None:
                raise ValueError(f"Unsupported activation function: {a}")

            layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),  # Dense layer
                activation()  # Add activation function
            )
            self.layers.append(layer)


    def forward(self, x):
        if self.skip_at is None:
            return super().forward(x)

        # Handle skip connections
        x_ = x.clone()  # Make a copy of the input tensor
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if self.skip_at and i in self.skip_at:
                y = torch.cat((y, x), dim=-1)  # Concatenate input with output
            x_ = y
        return y
