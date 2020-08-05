import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

#rev = np.linalg.pinv
fc = np.random.normal(loc=0,size=[10,1])

x = torch.tensor(np.random.random([100,10]),dtype=torch.float)
y = torch.tensor([[sum([x[j][i] if i&1 else -x[j][i] for i in range(10)])] for j in range(100)],dtype=torch.float)

def loss(ypred,ytrue):
    return sum(abs(ytrue-ypred))/len(ypred)

def loss_backward(yp,ytrue):
    return (ytrue-yp)/len(yp)

def fc_forward(x,fc):
    return np.dot(x,fc)
def fc_backward(dA,x):
    return np.dot(x.T,dA)

l_list = []
sharpness_list = []


for i in range(10000):
    ypred = fc_forward(x,fc)
    l = loss(ypred,y)
    l_list.append(l)
#    print(l)
    da=loss_backward(ypred,y)
    dw=fc_backward(da,x)
    fc+=0.1*dw
    if i%100==99:
        numerator = 1 + l
        da = loss_backward(ypred, y)
        dw = fc_backward(da, x)
        sign = dw / abs(dw)
        nfc = fc - 0.0005 * np.multiply(sign, 1 + fc)
        newy = fc_forward(x, nfc)
        newl = loss(newy, y)
        sharpness = 100 * (newl - l) / numerator
        sharpness_list.append(sharpness)

print(sharpness_list)
plt.plot(sharpness_list)
plt.show()
plt.plot(l_list)
plt.show()