import argparse

import numpy as np
from hyperopt import hp
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
# TODO
parser.add_argument('--digitsum_loss_type', type=str,default="ea_emak_tanh_newq",choices=["no_cl","ea_gak_tanh_newq","ea_emak_tanh_newq","ea_tanh_newq"])
parser.add_argument('--digitsum_subp_script', default=None,type=str)
parser.add_argument('--digitsum_out_box', default=None,type=str)
parser.add_argument('--digitsum_with_params_path_box', default=None,type=str)
parser.add_argument('--digitsum_with_params_path', default=None,type=str)
parser.add_argument('--no_save_model', default=False, const=True, action='store_const',
                    help="use scheduler or not.")
parser.add_argument('--baby_step_p', default=10,type=int,
                    help="p of baby step, only work for main_discrimloss_digitsum_babystep.py")
parser.add_argument('--reg_loss_type', default='L2', type=str, help="Regression loss type selected in the list: [L1, L2]")
parser.add_argument('--hidden_size', default=256,type=int,
                    help="hidden size of lstm and digit embedding")
parser.add_argument('--rand_fraction', default=0.4,type=float,
                    help="Fraction of data we will corrupt")

parser.add_argument('--is_discrimloss', default=False, const=True, action='store_const',
                    help="discrimloss or CrossEntropyLoss")
parser.add_argument('--no_scheduler', default=False, const=True, action='store_const',
                    help="use scheduler or not.")

parser.add_argument('--dataset', default='DIGITSUM', type=str,
                    help="Model type selected in the list: [MNIST, CIFAR10, CIFAR100, UTKFACE]")
parser.add_argument('--task_type', default='regression', type=str,
                    help="Task type selected in the list: [classification, regression, ]")
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run selected in the list: [120, 20, 100]')  # 120 for classification, 20 for mnist, 100 for utkface
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')
parser.add_argument('--lr_inst_param', default=0.001, type=float, help='Learning rate for instance parameters')
parser.add_argument('--wd_inst_param', default=1e-6, type=float, help='Weight decay for instance parameters')  # 0.0
parser.add_argument('--per_gpu_train_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for training. (default: 128)')  # 128/32
parser.add_argument('--per_gpu_eval_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for evaluation. (default: 100)')  # 32
parser.add_argument('--per_gpu_test_batch_size', default=24, type=int,
                    metavar='N', help='Batch size per GPU/CPU for test. (default: 100)')  # 32
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')

# wikihow
parser.add_argument('--model_path',
                    default="roberta-base", type=str,
                    help='Path to roberta_pretrain_model, used in wikihow task.') # roberta_base_2.4.1
parser.add_argument('--log_dir', default='WIKIHOW/GOAL/%s_results/logs_roberta_ea_tanh', type=str)
parser.add_argument('--save_dir', default='WIKIHOW/GOAL/%s_results/weights_roberta_ea_tanh', type=str)

parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')  # False
parser.add_argument("-local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("-no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=True, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')


args = parser.parse_args()


CURRENT_PATH = os.getcwd()

DIGITSUM_PATH = './dataset/digitsum'