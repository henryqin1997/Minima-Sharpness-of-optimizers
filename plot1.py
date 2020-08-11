from matplotlib import pyplot as plt
import json
import math
# train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
# print(valid_acc.index(max(valid_acc[:150])))
# print((valid_acc.index(max(valid_acc[150:250]))))
# print((valid_acc.index(max(valid_acc[250:]))))
low = math.log2(1e-5)
high = math.log2(10)
name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']
for opt in name_list[8:]:
    trainloss = json.load(open(opt+'_lr_range_find_minibatch.json'))
    i = trainloss.index(min(trainloss))
    print(i,2**(low+(high-low)*i/391/5))
    x = [2**(low+(high-low)*i/391/5) for i in range(int(391*5*(0-low)/(high-low)))]
    plt.plot(x,trainloss[:int(391*5*(0-low)/(high-low))])
    plt.xscale('log')
    plt.title(opt+'_lr_range_test')
    plt.show()
