import json
import math
from matplotlib import pyplot as plt
from helper import train_loader

prefix = 'mnist_lenet_batchsize32_'
postfix = '.json'
namelist = ['sgd','sgdwm','adam','rmsprop','radam','lars','lamb','novograd']
low = math.log2(1e-5)
high = math.log2(10)
x = [2**(low+(high-low)*i/len(train_loader)/3) for i in range(len(train_loader)*3)]
x = x[:int(len(x)*(math.log2(1e-3)-low)/(high-low))]
for name in namelist[0:1]:
    loss = json.load(open(prefix+name+postfix))
    loss = loss[:int(len(loss) * (math.log2(1e-3) - low) / (high - low))]
    plt.plot(x,loss)
    plt.xlabel('lr')
    plt.xscale('log')
    plt.ylabel('loss')
    plt.title(name)
    plt.show()