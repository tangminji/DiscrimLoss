import os
import re
import pandas as pd
import matplotlib.pyplot as plt

res = []
for root, dirs , files in os.walk('./CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results_ablation/fract0.40'):
    for csv in filter(lambda x: re.match(".*\.csv",x),files):
        label = root.rsplit('/',2)[-2]
        kind, extend = label[0],float(label[1:])
        with open(os.path.join(root,csv),"r") as f:
            val_acc = [float(x.strip().split(',')[-1]) for x in f.readlines()[1:]]
            res.append((kind, extend, max(val_acc)))
        
res.sort()
df = pd.DataFrame(res)
print(df)
plt.plot(df[2])
plt.show()