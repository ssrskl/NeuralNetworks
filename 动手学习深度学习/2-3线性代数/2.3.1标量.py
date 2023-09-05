import torch

X = torch.arange(6).reshape((2, 3, 1))
print(X)
Y = torch.arange(2).reshape((1, 1, 2))
print(Y)
print(X + Y)
