import torch

ckpt = './checkpoint/novograd0.05_ckptworst.pth'

print(torch.__version__)

checkpoint = torch.load(ckpt)

print(checkpoint)