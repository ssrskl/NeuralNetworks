import torch
from torch import nn


class MaoyanNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x + 1
        return output


maoyan = MaoyanNeuralNetwork()
x = torch.tensor(1)
output = maoyan(x)
print(output)
