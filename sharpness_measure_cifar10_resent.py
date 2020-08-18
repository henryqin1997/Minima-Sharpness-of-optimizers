import torch

ckpt = './checkpoint/novograd0.05_ckptworst.pt'

checkpoint = torch.jit.load(ckpt)

print(checkpoint)