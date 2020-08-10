from matplotlib import pyplot as plt
import json
# train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
# print(valid_acc.index(max(valid_acc[:150])))
# print((valid_acc.index(max(valid_acc[150:250]))))
# print((valid_acc.index(max(valid_acc[250:]))))

y = [(n+1)/(n+2) for n in range(100)]
x = [1.1**i for i in range(-50,50)]
plt.plot(x,y)
plt.xscale('log')
plt.show()