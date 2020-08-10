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
for opt in name_list:
    trainloss,validloss = json.load(open(opt+'lr_range_find.json'))
    x = [2**(low+(high-low)*i/200) for i in range(100)]
    plt.plot(x,trainloss[:100])
    plt.xscale('log')
    plt.title(opt+'_lr_range_test')
    plt.show()
