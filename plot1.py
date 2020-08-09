from matplotlib import pyplot as plt
import json
train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
print(valid_acc.index(max(valid_acc[:150])))
print((valid_acc.index(max(valid_acc[150:250]))))
print((valid_acc.index(max(valid_acc[250:]))))