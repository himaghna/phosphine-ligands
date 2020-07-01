import torch
a = [1, 2, 1]
a = torch.tensor(a, dtype=torch.float)
print(a)
print(a.std(dim=0))