import json
import math
import matplotlib
from matplotlib import pyplot as plt
from helper import train_loader

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

# prefix = 'mnist_lenet_batchsize32_'
# postfix = '.json'
# namelist = ['sgd','sgdwm','adam','rmsprop','radam','lars','lamb','novograd']
# low = math.log2(1e-5)
# high = math.log2(10)
# x = [2**(low+(high-low)*i/len(train_loader)/3) for i in range(len(train_loader)*3)]
# x = x[:int(len(x)*(math.log2(0.01)-low)/(high-low))]
# for name in namelist[2:4]:
#     loss = json.load(open(prefix+name+postfix))
#     loss = loss[:int(len(loss) * (math.log2(0.01) - low) / (high - low))]
#     plt.plot(x,loss)
#     plt.xlabel('lr')
#     plt.xscale('log')
#     plt.ylabel('loss')
#     plt.title(name)
#     plt.show()

prefix = 'mnist_lenet_batchsize_8192'
postfix = '.json'
namelist = ['sgd','sgdwm','adam','rmsprop','radam','lars','lamb','novograd']
low = math.log2(1e-5)
high = math.log2(10)
x = [2**(low+(high-low)*i/len(train_loader)/5) for i in range(len(train_loader)*5)]
x = x[:int(len(x)*(math.log(1)-low)/(high-low))]
for name in namelist[6:7]:
    loss = json.load(open(prefix+name+postfix))
    loss = loss[:int(len(loss) * (math.log(1) - low) / (high - low))]
    plt.plot(x,loss)
    plt.xlabel('lr')
    plt.xscale('log')
    plt.ylabel('loss')
    plt.title(name)
    plt.show()