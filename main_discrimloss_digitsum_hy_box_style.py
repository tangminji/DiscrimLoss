# our method main.py
# to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import hyperopt
from hyperopt import hp
from tensorboard_logger import log_value
from transformers import get_linear_schedule_with_warmup, AdamW

from common import utils
from common.cmd_args_digitsum import args
from dataset.digitsum_dataset import get_DIGITSUM_train_val_test_loader, get_DIGITSUM_model_and_loss_criterion
from timeit import default_timer as timer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # "0,1,2"

best_acc = 0
best_mae = 65536
best_test_mae = 65536
ITERATION = 0
# 模型，数据单独配置, model and data
MD_CLASSES = {
    'DIGITSUM': (get_DIGITSUM_train_val_test_loader, get_DIGITSUM_model_and_loss_criterion)
}


def main(params):
    global ITERATION
    ITERATION += 1
    params['ITERATION'] = ITERATION
    json.dump(params, open(args.digitsum_with_params_path_box, 'w+', encoding="utf-8"), ensure_ascii=False)
    sig = os.system("sh %s" % args.digitsum_subp_script)
    assert sig == 0
    res = json.load(open(args.digitsum_out_box, 'r', encoding="utf-8"))
    return res


if __name__ == '__main__':
    # Global variable
    MAX_EVALS = 50
    # MAX_EVALS = 50
    space = {
        'lr_inst_param': hp.loguniform('lr_inst_param', np.log(1e-3), np.log(100)),
        'wd_inst_param': hp.loguniform('wd_inst_param', np.log(1e-8), np.log(1)),
        # 'tanh_a': hp.uniform('tanh_a', 0.1, 0.3),
        'tanh_a': hp.uniform('tanh_a', 0.1, 2),
        'tanh_p': hp.uniform('tanh_p', 0.1, 5.0),
        'tanh_q': hp.quniform('tanh_q', 40, 80, 1),
        'clamp_min': hp.uniform('clamp_min', 0.05, 0.7)
    }

    trials = hyperopt.Trials()
    best = hyperopt.fmin(fn=main, space=space, algo=hyperopt.tpe.suggest,
                         max_evals=MAX_EVALS, trials=trials, rstate=np.random.RandomState(args.seed))

    print(best)
    print(trials.results)

    json.dump({"best": best, "trials": trials.results},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False)
    os.remove(args.digitsum_with_params_path_box)
    os.remove(args.digitsum_out_box)
