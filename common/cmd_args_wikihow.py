import argparse

import numpy as np
from hyperopt import hp
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
# TODO
parser.add_argument('--wikihow_subp_script', default=None,type=str)
parser.add_argument('--wikihow_out_box', default=None,type=str)
parser.add_argument('--wikihow_with_params_path_box', default=None,type=str)
parser.add_argument('--wikihow_with_params_path', default=None,type=str)
parser.add_argument('--train_data_path', default='dataset/wikihow/goal', type=str)
parser.add_argument('--adam_epsilon', default=1e-08, type=float)
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--warmup_steps', default=0, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--a', default=0.2, type=float)
parser.add_argument('--p', default=1.2, type=float)
parser.add_argument('--newq', default=3, type=int)
parser.add_argument('--sup_eps', default=3, type=int)
parser.add_argument('--es_type', default='linear', type=str, choices=['linear', 'sin', 'exp', 'piecewise'],help='early suppression type')

parser.add_argument('--wikihow_loss_type', type=str,choices=["no_cl","ea_gak_tanh_newq","ea_emak_tanh_newq","ea_tanh_newq"])

parser.add_argument('--dataset', default='WIKIHOW', type=str,
                    help="Model type selected in the list: [MNIST, CIFAR10, CIFAR100, UTKFACE]")

parser.add_argument('--nr_classes', default=4, type=int,
                    help="If have (for classification task), Number classes in the dataset")
parser.add_argument('--task_type', default='classification', type=str,
                    help="Task type selected in the list: [classification, regression, ]")
parser.add_argument('--num_train_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run selected in the list: [120, 20, 100]')  # 120 for classification, 20 for mnist, 100 for utkface
parser.add_argument('--sub_epochs_step', default=20000, type=int, metavar='N')
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

parser.add_argument('--cache_dir', default='',type=str, help='the path where the model was downloaded')
parser.add_argument('--model_type', default='bert',type=str)
parser.add_argument('--log_dir', default='WIKIHOW/GOAL/%s_results/logs_roberta_ea_tanh', type=str)
parser.add_argument('--save_dir', default='WIKIHOW/GOAL/%s_results/weights_roberta_ea_tanh', type=str)

parser.add_argument('--max_seq_length', default=80, type=int, help='max_seq_length, used in wikihow task.')
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

WIKIHOW_GOAL_PATH = os.path.join(CURRENT_PATH, 'dataset/wikihow/goal')
WIKIHOW_ORDER_PATH = os.path.join(CURRENT_PATH, 'dataset/wikihow/order')
WIKIHOW_STEP_PATH = os.path.join(CURRENT_PATH, 'dataset/wikihow/step')

MODEL = {
    'bert': 'bert-base-uncased',
    'xlnet': 'xlnet-base-cased',
    'roberta': 'roberta-base',
    'gpt2': 'gpt2'
}
args.model_name = MODEL[args.model_type]
args.model_path = os.path.join(args.cache_dir,args.model_name)