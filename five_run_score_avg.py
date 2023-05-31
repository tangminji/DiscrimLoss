# -*- coding: utf-8 -*-
# Author: jlgao HIT-SCIR
import os
import re
import numpy

def cal_avg(path):
    histo = []
    for seed_n in os.listdir(path):
        res_path = os.path.join(path, "%s/weights" % (seed_n))
        res_path = os.path.join(res_path, list(filter(lambda x: re.match(".*\.csv", x), os.listdir(res_path)))[0])
        datas = [line.strip().split(",") for line in open(res_path, 'r', encoding="utf-8").readlines()]
        score = 100000000000
        # print(len(datas))
        for t in datas[:-1]:
            if t[-1] == datas[-1][-1]:
                score = float(t[-1])
        assert score != 100000000000
        histo.append((seed_n, score))
    histo.sort(key=lambda x: x[-1])
    histo = histo[:5]
    histo_score = [t[-1] for t in histo]
    avg_score = sum(histo_score) / len(histo)
    print("%s\t%.2f±%.2f" % (path, avg_score, numpy.std(histo_score)))
    for seed_n, sc in histo:
        print(seed_n, sc)

# def cal_avg(path,reverse):
#     histo = []
#     for seed_n in os.listdir(path):
#         res_path = os.path.join(path, "%s/weights" % (seed_n))
#         #print(res_path, end="\t")
#         res_path = os.path.join(res_path, list(filter(lambda x: re.match(".*\.csv", x), os.listdir(res_path)))[0])
#         #print(res_path, end='\n')
#         last_line = list(filter(lambda x: "," in x, open(res_path, 'r', encoding="utf-8").readlines()))[-1]
#         score = float(last_line.strip().split(",")[-1])
#         histo.append((seed_n,score))

#     histo.sort(key=lambda x: x[-1],reverse=reverse)
#     histo = histo[:3]

#     histo_value = [t[-1] for t in histo]
#     avg_score = sum(histo_value) / len(histo_value)
#     print("%s\t%.2f±%.2f" % (path, avg_score, numpy.std(histo_value)))

#     for seed_n, sc in histo:
#         print(seed_n, sc)




if __name__ == '__main__':
    cal_avg("CIFAR10/sin/fract0.40")
    cal_avg("CIFAR10/sin/fract0.60")
    cal_avg("CIFAR10/exp/fract0.40")
    cal_avg("CIFAR10/exp/fract0.60")

    cal_avg("CIFAR100/sin/fract0.40")
    cal_avg("CIFAR100/sin/fract0.60")
    cal_avg("CIFAR100/exp/fract0.40")
    cal_avg("CIFAR100/exp/fract0.60")
    # cal_avg("DIGITSUM/mae_noscheduler_lr0.1_epo100_lstm_results/fract0.40")
    # cal_avg("DIGITSUM/box_ea_GAK_tanh_noscheduler_lr1e-1_epo100_lstm_results/fract0.40")
    # cal_avg("WIKIHOW/GOAL/roberta_ce_ses1000",reverse=True)
    # cal_avg("WIKIHOW/ORDER/roberta_ce_ses1000",reverse=True)
    
    # for i in [0, 20, 40, 60, 80]:
    #     cal_avg("MNIST/nocl_lr1e-1_epo20_results/fract0.%s" % (str(i)),reverse=True)
    # for i in [0, 20, 40, 60, 80]:
    #     cal_avg("MNIST/superloss_lr1e-1_epo20_results/fract0.%s" % (str(i)),reverse=True)
    # for i in [0, 20, 40, 80]:
    #     cal_avg("UTKFACE/ea_emak_tanh_L1_results/fract0.%d"%i,reverse=False)
    # for i in [0, 20, 40, 60, 80]:
    #     cal_avg("UTKFACE/ea_emak_tanh_L2_results/fract0.%d"%i,reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.20",reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.20",reverse=False)

    # cal_avg("MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.0",reverse=True)
    # cal_avg("MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.20", reverse=True)
    # cal_avg("MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.40", reverse=True)
    # cal_avg("MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.60", reverse=True)
    # cal_avg("MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.80", reverse=True)

    # cal_avg("CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.0", reverse=True)
    # cal_avg("CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.20", reverse=True)
    # cal_avg("CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.40", reverse=True)
    # cal_avg("CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.60", reverse=True)

    # cal_avg("CIFAR100/box_ea_emak_tanh_newq_lr1e-1_results/fract0.0", reverse=True)
    # cal_avg("CIFAR100/box_ea_emak_tanh_newq_lr1e-1_results/fract0.20", reverse=True)
    # cal_avg("CIFAR100/box_ea_emak_tanh_newq_lr1e-1_results/fract0.40", reverse=True)
    # cal_avg("CIFAR100/box_ea_emak_tanh_newq_lr1e-1_results/fract0.60", reverse=True)

    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.0",reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.20", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.40", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.60", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L2_results_5run/fract0.80", reverse=False)

    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.0", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.20", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.40", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.60", reverse=False)
    # cal_avg("UTKFACE/ea_emak_tanh_L1_results_5run/fract0.80", reverse=False)

    # cal_avg("MNIST/box_ea_emak_tanh_wo_es_newq_lr1e-1_epo20/fract0.40", reverse=True)
    # cal_avg("MNIST/box_ea_emak_tanh_wo_ea_newq_lr1e-1_epo20/fract0.40", reverse=True)


