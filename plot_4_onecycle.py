from matplotlib import pyplot as plt
import glob
import json
import numpy as np
names = ['lamb0.0008-0.02*','lars0.092-2.3*','novograd0.002-0.05*','radam0.0006-0.015*']

for name in names[2:3]:
    fn = glob.glob(name)
    valid_list = []
    for f in fn[3:4]:
        train_acc,valid_acc = json.load(open(f,'r'))
        # valid_list.append(valid_acc)
        plt.plot(train_acc)
        plt.title(name)
        plt.show()
        plt.plot(valid_acc)
        plt.title(name)
        plt.show()
    # valid_list = np.array(valid_list)
    # print(name)
    # print('max of max')
    # print(np.max(valid_list))
    # print('max of avg')
    # print(np.max(np.average(valid_list,axis=0)))
    # print('avg of max')
    # print(np.average(np.max(valid_list,axis=1)))
    # print('max of stop')
    # print(np.max([v[-1] for v in valid_list]))
    # print('avg of stop')
    # print(np.average([v[-1] for v in valid_list]))
    # plt.plot(np.average(valid_list,axis=0))
    # plt.title(name+'average of 5 run')
    # plt.show()