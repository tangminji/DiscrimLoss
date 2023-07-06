import argparse
import numpy as np
from hyperopt import hp
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--clothing1m_subp_script', default=None, type=str)
parser.add_argument('--clothing1m_out_box', default=None, type=str)
parser.add_argument('--clothing1m_with_params_path_box', default=None, type=str)
parser.add_argument('--clothing1m_with_params_path', default=None, type=str)


parser.add_argument('--log_dir', type=str)
parser.add_argument('--save_dir', type=str)

parser.add_argument('--lam', default=None, type=float, help='only for main_superloss_clothing1m.py')

parser.add_argument('--dataset', default='Clothing1M', type=str,
                    help="Model type selected in the list: [MNIST, CIFAR10, CIFAR100, UTKFACE]")
parser.add_argument('--task_type', default='classification', type=str,
                    help="Task type selected in the list: [classification, regression, ]")
parser.add_argument('--nr_classes', default=14, type=int,
                    help="If have (for classification task), Number classes in the dataset")
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run selected in the list: [120, 20, 100]')  # 120 for classification, 20 for mnist, 100 for utkface
# TODO:1000 steps for es(early suppression)
# parser.add_argument('--sub_epochs_step', default=1000, type=int, metavar='N')

# TODO:lr_inst_param=1e-4 for MSE(l2_loss), lr_inst_param=1e-3 for MSE(l1_loss), others image datasets=0.1
parser.add_argument('--lr_inst_param', default=1e-3, type=float, help='Learning rate for instance parameters')  # 0.001
parser.add_argument('--wd_inst_param', default=1e-6, type=float, help='Weight decay for instance parameters')  # 0.0
parser.add_argument('--sup_eps', default=1, type=float, help='early suppression epochs')

parser.add_argument('--clothing1m_loss_type', default='ea_emak_tanh_newq', type=str,
                    choices=["no_cl", "ea_gak_tanh_newq", "ea_emak_tanh_newq", "ea_tanh_newq"])

# TODO:不常调整的超参(3个)
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')  # 0.001
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
# L2 penalty
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')

# parser.add_argument('--correction', type=int, default=1, help='correction start epoch')#1
parser.add_argument('--num_per_class', type=int, default=18976,
                    help='num samples per class, if -1, use all samples.')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')  # 32
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')  # False
parser.add_argument('--per_gpu_train_batch_size', default=32, type=int,
                    metavar='N', help='Batch size per GPU/CPU for training. (default: 32)')
parser.add_argument('--per_gpu_eval_batch_size', default=128, type=int,
                    metavar='N', help='Batch size per GPU/CPU for evaluation. (default: 4*32)')
parser.add_argument('--per_gpu_test_batch_size', default=128, type=int,
                    metavar='N', help='Batch size per GPU/CPU for test. (default: 4*32)')

parser.add_argument("-local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("-no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
# parser.add_argument('--lr_tanh', default=0.001, type=float, help='Learning rate for tanh')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--lr_class_param', default=0.1, type=float, help='Learning rate for class parameters')
parser.add_argument('--wd_class_param', default=0.0, type=float, help='Weight decay for class parameters')
parser.add_argument('--init_class_param', default=1.0, type=float, help='Initial value for class parameters')
parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')

args = parser.parse_args()

CURRENT_PATH = os.getcwd()
Clothing1M_PATH = os.path.join(CURRENT_PATH,'dataset/Clothing_1M/')

# used in hyperopt
# https://blog.csdn.net/qq_34139222/article/details/60322995
