import torch

bootstrap_batchsize = 5
dt_sections = torch.tensor([4, 2, 1, 8, 3])

# Run multiple times to check distribution
for _ in range(8):
    t = (torch.rand(bootstrap_batchsize) * dt_sections.float()).floor().long()
    print(t.tolist())