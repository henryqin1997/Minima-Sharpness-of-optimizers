import torch

ckpt = './checkpoint/novograd0.05_ckptworst.pth'

checkpoint = torch.load(ckpt)

print(checkpoint)