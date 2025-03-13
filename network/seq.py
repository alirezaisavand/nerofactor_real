import torch.nn as nn

from network.base_brdf import Network

class SequentialNetwork(Network):
    """Assuming simple sequential flow."""
    def build(self, input_shape):
        # Ensure all layers are properly added to a sequential model
        self.seq = nn.Sequential(*self.layers)

        # Initialize the layers (in PyTorch, layers are lazy-initialized)
        dummy_input = torch.randn(*input_shape)
        self.forward(dummy_input)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
