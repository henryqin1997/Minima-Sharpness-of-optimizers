from matplotlib import pyplot as plt
import json
train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
plt.plot(valid_acc)
plt.show()
