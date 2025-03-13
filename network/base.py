import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError("The 'forward' method must be implemented by subclasses.")
