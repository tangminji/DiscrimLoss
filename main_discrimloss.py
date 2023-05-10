#our method main.py
#to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2,7"#

import time
import argparse

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed

from tensorboard_logger import log_value

from common import utils
from common.cmd_args import args, CURRENT_PATH, learning_rate_schedule
from dataset.mnist_dataset import get_MNIST_train_and_val_loader, get_MNIST_model_and_loss_criterion
from dataset.cifar100_dataset import get_CIFAR100_train_and_val_loader, get_CIFAR100_model_and_loss_criterion
from dataset.cifar10_dataset import get_CIFAR10_train_and_val_loader, get_CIFAR10_model_and_loss_criterion
from dataset.utkface_dataset import get_UTKFACE_train_and_val_loader, get_UTKFACE_model_and_loss_criterion
import csv
THRESHOLD = 0.417#0.667#0.417#0.667#0.95#0.90#0.85#switch time
best_acc = 0
best_mae = 65536

#模型，数据单独配置, model and data
MD_CLASSES = {
    'MNIST':(get_MNIST_train_and_val_loader, get_MNIST_model_and_loss_criterion),
    'CIFAR10':(get_CIFAR10_train_and_val_loader, get_CIFAR10_model_and_loss_criterion),
    'CIFAR100':(get_CIFAR100_train_and_val_loader, get_CIFAR100_model_and_loss_criterion),
    'UTKFACE':(get_UTKFACE_train_and_val_loader, get_UTKFACE_model_and_loss_criterion)
}

def validate(args, val_loader, model, criterion, epoch):
    """Evaluates model on validation set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    global best_acc
    global best_mae
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')#for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")#for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(val_loader):
            # if args.device == 'cuda':
            #     inputs = inputs.cuda()
            #     target = target.cuda()
            inputs, target = inputs.to(args.device), target.to(args.device)

            # compute output
            logits = model(inputs)
            if args.task_type == 'regression':
                logits_ = logits.squeeze(dim=-1)
                loss = criterion(logits_, target)
            elif args.task_type == 'classification':
                loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))

            if args.task_type == 'regression':
                mae = utils.compute_MAE(args, logits, target)
                MAE.update(mae.item(), inputs.size(0))
            elif args.task_type == 'classification':
                acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
                top1.update(acc1[0].item(), inputs.size(0))
        if args.task_type == 'regression':
            print('Test-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))
        elif args.task_type == 'classification':
            print('Test-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))


    log_value('val/loss', losses.avg, step=epoch)
    # Logging results on tensorboard
    if args.task_type == 'regression':
        log_value('val/mean absolute error', MAE.avg, step=epoch)
        # Save checkpoint.
        mae = MAE.avg
        if mae < best_mae:#取最小
            best_mae = mae
            utils.checkpoint(mae, epoch, model, args.save_dir)

        return losses.avg, MAE.avg
    elif args.task_type == 'classification':
        log_value('val/accuracy', top1.avg, step=epoch)
        # Save checkpoint.
        acc = top1.avg
        if acc > best_acc:
            best_acc = acc
            utils.checkpoint(acc, epoch, model, args.save_dir)

        return losses.avg, top1.avg


def train_for_one_epoch(args,
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        optimizer_data_parameters,
                        data_parameters,
                        config,
                        switch=False):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        train_loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer (torch.optim.SGD): optimizer for model parameters.
        epoch (int): current epoch.
        global_iter (int): current iteration count.
        optimizer_data_parameters (tuple SparseSGD): SparseSGD optimizer for class and instance data parameters.
        data_parameters (tuple of torch.Tensor): class and instance level data parameters.
        config (dict): config file for the experiment.

    Returns:
        global iter (int): updated iteration count after 1 epoch.
    """
    # loss-for-each-sample
    loss_parameters = torch.tensor(np.zeros(len(train_loader.dataset)),
                                   dtype=torch.float32,
                                   requires_grad=False,
                                   device=args.device)
    # Initialize counters
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')#for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")#for regression

    # Unpack data parameters
    optimizer_inst_param = optimizer_data_parameters
    inst_parameters, exp_avg = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    for i, (inputs, target, index_dataset) in enumerate(train_loader):
        global_iter = global_iter + 1
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        optimizer_inst_param.zero_grad()
        # Compute logits
        logits = model(inputs)

        # Compute data parameters for instances in the minibatch
        inst_parameter_minibatch = inst_parameters[index_dataset]
        data_parameter_minibatch = torch.exp(inst_parameter_minibatch)


        #logits = logits / data_parameter_minibatch
        loss = criterion(args, logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)#ea/ea_wo_es/ea_wo_bc/ea_gak/ea_emak/ea_0/ea_tanh
        #loss, tanh_a, tanh_p, tanh_q = criterion(logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)#ea_tanh/ea_exp
        #loss = criterion(logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch, switch, 'switch')#ea_2
        #loss = criterion(logits, target, data_parameter_minibatch)#base
        #loss = criterion(logits, target, data_parameter_minibatch, epoch)#es

        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        if args.wd_inst_param > 0.0:
            loss = loss + 0.5 * args.wd_inst_param * (inst_parameter_minibatch ** 2).sum()#2,4

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.skip_clamp_data_param is False:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])


        # Measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        if args.task_type == 'regression':
            mae = utils.compute_MAE(args, logits, target)
            MAE.update(mae.item(), inputs.size(0))
        elif args.task_type == 'classification':
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            if args.task_type == 'regression':
                utils.log_discrim_reg_intermediate_iteration_stats(epoch,
                                                       global_iter,
                                                       losses,
                                                       inst_parameters=inst_parameters,
                                                       mae=MAE)
            elif args.task_type == 'classification':
                utils.log_discrim_cl_intermediate_iteration_stats(epoch,
                                                       global_iter,
                                                       losses,
                                                       inst_parameters=inst_parameters,
                                                       top1=top1)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    log_value('train/loss', losses.avg, step=epoch)

    if args.task_type == 'regression':
        print('Train-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))
        log_value('train/mean absolute error', MAE.avg, step=epoch)
    elif args.task_type == 'classification':
        print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
        log_value('train/accuracy', top1.avg, step=epoch)

    #add tanh's parameters on tensorboard
    # if criterion._get_name() in ['DiscrimEA_TANHLoss']:
    #     log_value('two-phases-tanh/a', tanh_a, step=epoch)
    #     log_value('two-phases-tanh/p', tanh_p, step=epoch)
    #     log_value('two-phases-tanh/q', tanh_q, step=epoch)

    utils.save_loss(args.save_dir, loss_parameters, epoch)
    utils.save_sigma(args.save_dir, inst_parameters, epoch)
    if args.task_type == 'regression':
        train_metric = MAE.avg
    elif args.task_type == 'classification':
        train_metric = top1.avg
    return global_iter, losses.avg, train_metric#top1.avg


def main_worker(args, config):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global best_acc
    global best_mae
    global_iter = 0
    #learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    model, loss_criterion, loss_criterion_val, logname = mdl_loss(args)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#args.lr

    # Get train and validation dataset loader
    #train_loader, val_loader = get_CIFAR100_train_and_val_loader(args)
    train_loader, val_loader = loaders(args)

    # Initialize class and instance based temperature
    (inst_parameters, optimizer_inst_param, exp_avg) = utils.get_inst_conf_n_optimizer(
                                                        args=args,
                                                        nr_instances=len(train_loader.dataset),
                                                        device='cuda')

    # var_sigmas = []
    # var_sigmas.append(torch.var(inst_parameters).item())
    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if args.task_type == 'regression':
                logwriter.writerow(
                    ['epoch', 'train loss', 'train mae', 'val loss', 'val mae'])
            elif args.task_type == 'classification':
                logwriter.writerow(
                    ['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])
    start_epoch_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        # Adjust learning rate for model parameters
        if epoch in learning_rate_schedule:
            utils.adjust_learning_rate(model_initial_lr=args.lr,
                                 optimizer=optimizer,
                                 gamma=0.1,
                                 step=np.sum(epoch >= learning_rate_schedule))

        if loss_criterion._get_name() in ['DiscrimEA_2Loss']:
            if epoch >= int(args.epochs*THRESHOLD-1):
                switch = True
            else:
                switch = False
        else:
            switch = False
        # Train for one epoch
        # TODO: train_metric: acc for classification, MAE for regression
        global_iter, train_loss, train_metric = train_for_one_epoch(
                            args=args,
                            train_loader=train_loader,
                            model=model,
                            criterion=loss_criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_iter=global_iter,
                            optimizer_data_parameters=(optimizer_inst_param),
                            data_parameters=(inst_parameters, exp_avg),
                            config=config,
                            switch=switch)

        #var_sigmas.append(torch.var(inst_parameters).item())
        # Evaluate on validation set
        #TODO: val_metric: acc for classification, MAE for regression
        val_loss, val_metric = validate(args, val_loader, model, loss_criterion_val, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_metric, val_loss, val_metric])

        # Save artifacts
        #utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

        # Log temperature stats over epochs
        utils.log_stats(data=torch.exp(inst_parameters),
                        name='epoch_stats_inst_parameter',
                        step=epoch)

        if args.rand_fraction > 0.0:
            # We have corrupted labels in the train data; plot instance parameter stats for clean and corrupt data
            nr_corrupt_instances = int(np.floor(len(train_loader.dataset) * args.rand_fraction))
            # Corrupt data is in the top-fraction of dataset
            utils.log_stats(data=torch.exp(inst_parameters[:nr_corrupt_instances]),
                            name='epoch_stats_corrupt_inst_parameter',
                            step=epoch)
            utils.log_stats(data=torch.exp(inst_parameters[nr_corrupt_instances:]),
                            name='epoch_stats_clean_inst_parameter',
                            step=epoch)

    #record best_acc/best_mae
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        if args.task_type == 'regression':
            logwriter.writerow([-1, -1, -1, -1, best_mae])
        elif args.task_type == 'classification':
            logwriter.writerow([-1, -1, -1, -1, best_acc])

    print('total time: {}'.format(time.time() - start_epoch_time))

    # with open(os.path.join(CURRENT_PATH, 'var_sigmas.txt'), 'a+') as f:
    #     f.write('inst only, all 1.0\n')
    #     f.write(str(var_sigmas))
    #     f.write('\n')


def main():
    #args = parser.parse_args()
    #args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.log_dir = CURRENT_PATH + '/UTKFACE/discrimloss_results/logs_n40_bt128_cn1_ea_emak_tanh_wd6_lr0.001_L1'
    args.save_dir = CURRENT_PATH + '/UTKFACE/discrimloss_results/weights_n40_bt128_cn1_ea_emak_tanh_wd6_lr0.001_L1'
    utils.generate_log_dir(args)
    utils.generate_save_dir(args)

    config = {}
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = np.log(0.05)#1/20,0.5,0.6,0.4,0.3,0.2,0.1
    config['clamp_inst_sigma']['max'] = np.log(20)
    utils.save_config(args.save_dir, config)

    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    main_worker(args, config)

if __name__ == '__main__':
    main()
'''
python main_cifar_parallel.py --rand_fraction 0.4 --init_inst_param 1.0 --lr_inst_param 0.2 --wd_inst_param 0.0 --learn_inst_parameters 
'''
'''
python main_cifar_parallel.py --init_class_param 1.0 --lr_class_param 0.1 --wd_class_param 1e-4 --learn_class_parameters
'''
'''
nohup python main_cifar_parallel.py --init_class_param 1.0 --lr_class_param 0.1 --wd_class_param 1e-4 --init_inst_param 0.001 --lr_inst_param 0.8 --wd_inst_param 1e-8 --learn_class_parameters --learn_inst_parameters >> ./datapara_results/log_class_inst_no_noise.log 2>&1 &

nohup python main_cifar_parallel.py --init_class_param 1.0 --lr_class_param 0.1 --wd_class_param 1e-4 --init_inst_param 0.001 --lr_inst_param 0.8 --wd_inst_param 1e-8 --learn_class_parameters --learn_inst_parameters >> ./CIFAR100/datapara_results/log_n40_bt128.log 2>&1 &

nohup python main_discrimloss.py --wd_inst_param 1e-6 >> ./CIFAR100/discrimloss_results/log_n0_bt128_cn1_ea_tanh_wd6.log 2>&1 &

'''