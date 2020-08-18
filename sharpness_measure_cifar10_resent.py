import torch

ckpt = './checkpoint/novograd0.05_ckptworst.pth'

checkpoint = torch.jit.load(ckpt)

print(checkpoint)