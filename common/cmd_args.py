import argparse

import numpy as np
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
# Ablation
parser.add_argument('--cifar_ablation_a', default=None, type=float)
parser.add_argument('--cifar_ablation_p', default=None, type=float)
parser.add_argument('--cifar_ablation_newq', default=None, type=int)


parser.add_argument('--minist_ablation_a', default=None, type=float)
parser.add_argument('--minist_ablation_p', default=None, type=float)
parser.add_argument('--minist_ablation_newq', default=None, type=int)

parser.add_argument('--minist_subp_script', default=None, type=str)
parser.add_argument('--minist_out_box', default=None, type=str)
parser.add_argument('--minist_with_params_path_box', default=None, type=str)
parser.add_argument('--minist_with_params_path', default=None, type=str)

parser.add_argument('--cifar_subp_script', default=None, type=str)
parser.add_argument('--cifar_out_box', default=None, type=str)
parser.add_argument('--cifar_with_params_path_box', default=None, type=str)
parser.add_argument('--cifar_with_params_path', default=None, type=str)
parser.add_argument('--utkface_with_params_path', default=None, type=str)

parser.add_argument('--utkface_loss_type', type=str, default="ea_emak_tanh", choices=["ea_gak_tanh", "ea_emak_tanh"])
parser.add_argument('--cifar_loss_type', type=str, default="ea_emak_tanh_newq",
                    choices=["ea_gak_tanh_newq", "ea_emak_tanh_newq", "ea_tanh_newq", "no_cl",
                             "ea_emak_tanh_wo_es_newq", "ea_emak_tanh_wo_ea_newq","ea_emak_tanh_fixk_newq"])
parser.add_argument('--minist_loss_type', type=str, default="no_cl",
                    choices=["ea_gak_tanh_newq", "ea_emak_tanh_newq", "ea_tanh_newq", "no_cl",
                             "ea_emak_tanh_wo_es_newq", "ea_emak_tanh_wo_ea_newq","ea_emak_tanh_fixk_newq"])
# parser.add_argument('--minist_no_save', default=False, const=True, action='store_const')
parser.add_argument('--log_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--dataset', default='MINIST', type=str,
                    help="Model type selected in the list: [MINIST, CIFAR10, CIFAR100, UTKFACE]")

# dataset related
# classification

parser.add_argument('--nr_classes', default=10, type=int,
                    help="If have (for classification task), Number classes in the dataset")
# regression
parser.add_argument('--reg_loss_type', default='L1', type=str,
                    help="Regression loss type selected in the list: [L1, L2]")
parser.add_argument('--t', default=10.0, type=float,
                    help="Regression task, t for Truncated absolute error for UTKFACE")  # selected in the list: [1.0,10.0] for MNIST and utkface in paper, but mnist is not used for regression here

parser.add_argument('--task_type', default='classification', type=str,
                    help="Task type selected in the list: [classification, regression, ]")
# noise
parser.add_argument('--rand_fraction', default=0.6, type=float, help='Fraction of data we will corrupt')  # 0.4
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run selected in the list: [120, 20, 100]')  # 120 for classification, 20 for mnist, 100 for utkface
# lr
# TODO: lr=0.1 for UTKFACE smooth_l1_loss, l2=1e-3 for UTKFACE l2_loss, others image datasets=0.001
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')  # 0.001
# TODO:lr_inst_param=1e-4 for MSE(l2_loss), others image datasets=1e-3
parser.add_argument('--lr_inst_param', default=1e-3, type=float, help='Learning rate for instance parameters')  # 0.001
parser.add_argument('--wd_inst_param', default=1e-6, type=float, help='Weight decay for instance parameters')  # 0.0
parser.add_argument('--sup_eps', default=3, type=float, help='early suppression epochs')
parser.add_argument('--es_type', default='linear', type=str, choices=['linear', 'sin', 'exp', 'piecewise'],help='early suppression type')
parser.add_argument('--a', default=0.2, type=float, help='a*tanh(p*epoch+q)+a+1')
parser.add_argument('--p', default=1.5, type=float, help='a*tanh(p*epoch+q)+a+1')
parser.add_argument('--q', default=-50, type=float, help='a*tanh(p*epoch+q)+a+1')

# TODO
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
# L2 penalty
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
# TODO:just note, not used,0.5 for MNIST, ln(10) for CIFAR10, ln(100) for CIFAR100
parser.add_argument('--k', default=10., type=float, help='threshold to distinguish easy/hard samples'
                                                         'selected in the list:[0.5,ln(10),ln(100)]')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')  # 32
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')  # False
parser.add_argument('--per_gpu_train_batch_size', default=128, type=int,
                    metavar='N', help='Batch size per GPU/CPU for training. (default: 128)')  # 128/32
parser.add_argument('--per_gpu_eval_batch_size', default=128, type=int,
                    metavar='N', help='Batch size per GPU/CPU for evaluation. (default: 100)')  # 32
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

learning_rate_schedule = np.array([80, 100, 160])  # for CIFAR10/CIFAR100
# learning_rate_schedule_utkface = np.array([60, 80, 100])  # for UTKFACE,but has been deprecated!

CURRENT_PATH = os.getcwd()
CIFAR100_PATH = '{}/dataset'.format(CURRENT_PATH)
CIFAR10_PATH = '{}/dataset'.format(CURRENT_PATH)
MNIST_PATH = '{}/dataset'.format(CURRENT_PATH)
UTKFACE_IMGPATH = '{}/dataset/UTKface_Aligned_cropped/faces_aligned_cropped/'.format(
    CURRENT_PATH)
UTKFACE_TRAINPATH = '{}/dataset/UTKface_Aligned_cropped/faces_aligned_cropped/train'.format(
    CURRENT_PATH)
UTKFACE_TESTPATH = '{}/dataset/UTKface_Aligned_cropped/faces_aligned_cropped/test'.format(
    CURRENT_PATH)

# used in hyperopt
# https://blog.csdn.net/qq_34139222/article/details/60322995
