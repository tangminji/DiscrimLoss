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
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed

from tensorboard_logger import log_value

from common import utils
from dataset.cifar100_dataset import CIFAR100WithIdx
from models.wide_resnet import WideResNet28_10
from common.cmd_args import args, CURRENT_PATH, learning_rate_schedule, space
from dataset.cifar100_dataset import get_CIFAR100_train_and_val_loader

from common.DiscrimLoss import (DiscrimLoss, DiscrimESLoss, DiscrimEALoss,
                                DiscrimEA_WO_ESLoss, DiscrimEA_WO_BCLoss,
                                DiscrimEA_GAKLoss, DiscrimEA_EMAKLoss,
                                DiscrimEA_2Loss, DiscrimEA_0Loss,
                                DiscrimEAEXPLoss, DiscrimEA_TANHLoss)
import csv
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from timeit import default_timer as timer

THRESHOLD = 0.417#0.667#0.417#0.667#0.95#0.90#0.85#switch time
best_acc = 0


def get_model_and_loss_criterion(args, params):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building WideResNet28_10')
    args.arch = 'WideResNet28_10'
    model = WideResNet28_10(num_classes=args.nr_classes)
    logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    model.to(args.device)
    #criterion = DiscrimEALoss(k1=args.nr_classes).to(args.device)#
    #criterion = DiscrimEAEXPLoss(k1=args.nr_classes).to(args.device)
    criterion = DiscrimEA_TANHLoss(k1=args.nr_classes, a=params['tanh_a'], p=params['tanh_p'], q=params['tanh_q']).to(args.device)
    #criterion = DiscrimEA_0Loss(k1=args.nr_classes).to(args.device)
    #criterion = DiscrimEA_2Loss(k1=args.nr_classes).to(args.device)
    #criterion = DiscrimEA_GAKLoss().to(args.device)
    #criterion = DiscrimEA_EMAKLoss().to(args.device)
    #criterion = DiscrimEA_WO_ESLoss(k1=args.nr_classes).to(args.device)
    #criterion = DiscrimEA_WO_BCLoss(k1=args.nr_classes).to(args.device)
    #criterion = DiscrimLoss(k1 = args.nr_classes).to(args.device)
    #criterion = DiscrimESLoss(k1=args.nr_classes).to(args.device)

    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    return model, criterion, criterion_val, logname


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
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
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
            loss = criterion(logits, target)
            loss = loss.mean()
            # measure accuracy and record loss
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))

        print('Test-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))

    # Logging results on tensorboard
    log_value('val/accuracy', top1.avg, step=epoch)
    log_value('val/loss', losses.avg, step=epoch)
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
                        params,
                        ITERATION,
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
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

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
        loss = criterion(logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)#ea/ea_wo_es/ea_wo_bc/ea_gak/ea_emak/ea_0/ea_tanh
        #loss, tanh_a, tanh_p, tanh_q = criterion(logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)#ea_tanh/ea_exp
        #loss = criterion(logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch, switch, 'switch')#ea_2
        #loss = criterion(logits, target, data_parameter_minibatch)#base
        #loss = criterion(logits, target, data_parameter_minibatch, epoch)#es

        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        # if args.wd_inst_param > 0.0:
        if params['wd_inst_param'] > 0.0:
            loss = loss + 0.5 * params['wd_inst_param'] * (inst_parameter_minibatch ** 2).sum()#2,4

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
        acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            utils.log_discrim_intermediate_iteration_stats(epoch,
                                                   global_iter,
                                                   losses,
                                                   inst_parameters=inst_parameters,
                                                   top1=top1)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)
    log_value('train/loss', losses.avg, step=epoch)
    #add tanh's parameters on tensorboard
    # if criterion._get_name() in ['DiscrimEA_TANHLoss']:
    #     log_value('two-phases-tanh/a', tanh_a, step=epoch)
    #     log_value('two-phases-tanh/p', tanh_p, step=epoch)
    #     log_value('two-phases-tanh/q', tanh_q, step=epoch)

    utils.save_loss_hyp(args.save_dir, loss_parameters, epoch, ITERATION)
    utils.save_sigma_hyp(args.save_dir, inst_parameters, epoch, ITERATION)
    return global_iter, losses.avg, top1.avg


def main_worker(args, config, params, ITERATION):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global best_acc
    global_iter = 0
    #learning_rate_schedule = np.array([80, 100, 160])

    # Create model
    model, loss_criterion, loss_criterion_val, logname = get_model_and_loss_criterion(args, params)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#args.lr

    # Get train and validation dataset loader
    train_loader, val_loader = get_CIFAR100_train_and_val_loader(args)

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
        global_iter, train_loss, train_acc = train_for_one_epoch(
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
                            params=params,
                            ITERATION=ITERATION,
                            switch=switch)

        #var_sigmas.append(torch.var(inst_parameters).item())
        # Evaluate on validation set
        val_loss, val_acc = validate(args, val_loader, model, loss_criterion_val, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

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

    #record best_acc
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, best_acc])

    print('total time: {}'.format(time.time() - start_epoch_time))
    return best_acc
    # with open(os.path.join(CURRENT_PATH, 'var_sigmas.txt'), 'a+') as f:
    #     f.write('inst only, all 1.0\n')
    #     f.write(str(var_sigmas))
    #     f.write('\n')


def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    global best_acc
    best_acc=0

    start = timer()

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

    args.log_dir = CURRENT_PATH + '/CIFAR100/discrimloss_results/logs_n40_bt128_cn1_ea_tanh_hyperopt'
    args.save_dir = CURRENT_PATH + '/CIFAR100/discrimloss_results/weights_n40_bt128_cn1_ea_tanh_hyperopt'
    args.nr_classes = 100 # Number classes in CIFAR100DiscrimLoss
    utils.generate_log_dir_hyp(args, ITERATION)
    utils.generate_save_dir_hyp(args, ITERATION)

    config = {}
    config['clamp_inst_sigma'] = {}
    #config['clamp_inst_sigma']['min'] = np.log(0.05)#1/20,0.5,0.6,0.4,0.3,0.2,0.1
    config['clamp_inst_sigma']['min'] = params['clamp_min']
    config['clamp_inst_sigma']['max'] = np.log(20)
    utils.save_config_hyp(args.save_dir, {"args": str(args.__dict__), "config": config,"params":params}, ITERATION)

    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    best_acc = main_worker(args, config, params, ITERATION)
    print('best acc: {}, params: {}, iteration: {}, status: {}'.
        format(best_acc, params, ITERATION, STATUS_OK))

    # Dictionary with information for evaluation, otherwise will report errors
    run_time = utils.format_time(timer() - start)
    loss = 1 - best_acc
    return {'loss': loss, 'best_acc': best_acc, \
            'params': params, 'iteration': ITERATION, \
            'train_time': run_time, 'status': STATUS_OK}

if __name__ == '__main__':
    #main()
    # optimization algorithm
    tpe_algorithm = tpe.suggest
    # Keep track of results
    bayes_trials = Trials()

    # Global variable
    global ITERATION
    ITERATION = 0
    MAX_EVALS = 2
    best = fmin(fn=main, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50))

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