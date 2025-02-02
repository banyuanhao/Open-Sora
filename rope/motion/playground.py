import torch

W = torch.randn(16,1,1)
print(W)

Q = W.expand(16, 16, 16)
print(Q)