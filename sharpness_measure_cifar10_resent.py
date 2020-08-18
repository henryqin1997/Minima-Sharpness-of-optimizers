import torch

#ckpt = './checkpoint/novograd0.05_ckptbest.pth'
ckpt = './checkpoint/ckpt1.pth'
print(torch.__version__)

checkpoint = torch.load(ckpt)

print(checkpoint)