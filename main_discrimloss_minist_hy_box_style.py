# our method main.py
# to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
import os

import numpy as np
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe

from common.cmd_args import args

best_acc = 0
ITERATION = 0

def main(params):
    global ITERATION
    ITERATION += 1
    params['ITERATION'] = ITERATION
    json.dump(params, open(args.minist_with_params_path_box, 'w+', encoding="utf-8"), ensure_ascii=False)
    sig = os.system("sh %s" % args.minist_subp_script)
    assert sig == 0
    res = json.load(open(args.minist_out_box, 'r', encoding="utf-8"))
    return res

if __name__ == '__main__':
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()



    assert args.dataset == "MNIST"
    MAX_EVALS = 30
    space = {
        'lr_inst_param': hp.loguniform('lr_inst_param', np.log(0.05), np.log(10)),
        'wd_inst_param': hp.loguniform('wd_inst_param', np.log(1e-100), np.log(0.1)),
        'sup_eps': hp.quniform('sup_eps', 2, 5, 1),
        'tanh_a': hp.uniform('tanh_a', 0.1, 0.5),
        'tanh_p': hp.uniform('tanh_p', 0.5, 2.0),
        'tanh_q': hp.quniform('tanh_q', 2, 18, 1),
        # self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.
        'clamp_min': hp.uniform('clamp_min', 0.05, 0.7)
    }

    best = fmin(fn=main, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(args.seed))

    print(best)
    print(bayes_trials.results)

    json.dump({"best": best, "trials": bayes_trials.results},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False)
    os.remove(args.minist_with_params_path_box)
    os.remove(args.minist_out_box)
