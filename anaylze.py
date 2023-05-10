# -*- coding: utf-8 -*-
# Author: jlgao HIT-SCIR
import json
from collections import defaultdict

datas=json.load(open("MINIST/ea_emak_tanh_newq_lr1e-1_epo20_30times_results/fract0.20/seed1/logs/hy_best_params.json",'r',encoding="utf-8"))

vil=[]
for epoch in datas['trials']:
    best_acc=epoch['best_acc']
    params=epoch['params']
    vil.append((params['tanh_q'],best_acc))
vil.sort(key=lambda x:x[0])
for t in vil:
    print(t)
