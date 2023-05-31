# our method main.py
# to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
import os

import time
import argparse

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
from tqdm import tqdm, trange
from tensorboard_logger import log_value

from common import utils
from common.cmd_args import args, CURRENT_PATH
from dataset.utkface_dataset import get_UTKFACE_train_and_val_loader, get_UTKFACE_model_and_loss_criterion
import csv
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from timeit import default_timer as timer

best_mae = 65536
ITERATION=0
# model and data
MD_CLASSES = {
    'UTKFACE': (get_UTKFACE_train_and_val_loader, get_UTKFACE_model_and_loss_criterion)
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
    global best_mae
    losses = utils.AverageMeter('Loss', ':.4e')
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression
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
            logits_ = logits.squeeze(dim=-1)
            loss = criterion(logits_, target)

            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            mae = utils.compute_MAE(args, logits, target)
            MAE.update(mae.item(), inputs.size(0))
        print('Test-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))

    log_value('val/loss', losses.avg, step=epoch)
    # Logging results on tensorboard
    log_value('val/mean absolute error', MAE.avg, step=epoch)
    # Save checkpoint.
    mae = MAE.avg
    if mae < best_mae:
        best_mae = mae
        utils.checkpoint(mae, epoch, model, args.save_dir)

    return losses.avg, MAE.avg


def train_for_one_epoch(args,
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        scheduler,
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
        scheduler (torch.optim.SGD): optimizer for model parameters.
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
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression

    # Unpack data parameters
    optimizer_inst_param = optimizer_data_parameters
    inst_parameters, exp_avg = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    # epoch_iterator = tqdm(train_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for i, (inputs, target, index_dataset) in enumerate(train_loader):
        global_iter = global_iter + 1
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Flush the gradient buffer for model and data-parameters. https://cloud.tencent.com/developer/article/1710864
        optimizer.zero_grad()
        optimizer_inst_param.zero_grad()
        # Compute logits
        logits = model(inputs)

        # Compute data parameters for instances in the minibatch
        inst_parameter_minibatch = inst_parameters[index_dataset]
        data_parameter_minibatch = torch.exp(inst_parameter_minibatch)

        # logits = logits / data_parameter_minibatch
        loss = criterion(args, logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)
        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        if params['wd_inst_param'] > 0.0:
            loss = loss + 0.5 * params['wd_inst_param'] * (inst_parameter_minibatch ** 2).sum()  # 2,4

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.skip_clamp_data_param is False:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])

        # Measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        mae = utils.compute_MAE(args, logits, target)
        MAE.update(mae.item(), inputs.size(0))
        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            utils.log_discrim_reg_intermediate_iteration_stats(epoch,
                                                               global_iter,
                                                               losses,
                                                               inst_parameters=inst_parameters,
                                                               mae=MAE)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    log_value('train/loss', losses.avg, step=epoch)
    print('Train-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))
    log_value('train/mean absolute error', MAE.avg, step=epoch)

    # add tanh's parameters on tensorboard
    # if criterion._get_name() in ['DiscrimEA_TANHLoss']:
    #     log_value('two-phases-tanh/a', tanh_a, step=epoch)
    #     log_value('two-phases-tanh/p', tanh_p, step=epoch)
    #     log_value('two-phases-tanh/q', tanh_q, step=epoch)

    utils.save_loss_hyp(args.save_dir, loss_parameters, epoch, ITERATION)
    utils.save_sigma_hyp(args.save_dir, inst_parameters, epoch, ITERATION)
    train_metric = MAE.avg
    return global_iter, losses.avg, train_metric  # top1.avg


def main_worker(args, config, params, ITERATION):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global best_mae
    best_mae = 65536
    global_iter = 0
    # learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    model, loss_criterion, loss_criterion_val, logname = mdl_loss(args, params=params, ITERATION=ITERATION)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,  # 0.1 for l1_loss/0.001 for l2_loss,#args.lr,
                                momentum=args.momentum)  # 0.9,
    # weight_decay=args.weight_decay)#args.lr

    # Get train and validation dataset loader
    # train_loader, val_loader = get_CIFAR100_train_and_val_loader(args)
    train_loader, val_loader = loaders(args)

    # Initialize class and instance based temperature
    (inst_parameters, optimizer_inst_param, exp_avg) = utils.get_inst_conf_n_optimizer(
        args=args,
        nr_instances=len(train_loader.dataset),
        device='cuda',
        params=params)

    # var_sigmas = []
    # var_sigmas.append(torch.var(inst_parameters).item())
    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train mae', 'val loss', 'val mae'])
    start_epoch_time = time.time()
    train_iterator = trange(int(args.start_epoch), int(args.epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:

        # Adjust learning rate for model parameters
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, np.exp(-0.01))

        # Train for one epoch
        # TODO: train_metric: acc for classification, MAE for regression
        global_iter, train_loss, train_metric = train_for_one_epoch(
            args=args,
            train_loader=train_loader,
            model=model,
            criterion=loss_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_iter=global_iter,
            optimizer_data_parameters=(optimizer_inst_param),
            data_parameters=(inst_parameters, exp_avg),
            config=config,
            params=params,
            ITERATION=ITERATION)

        # var_sigmas.append(torch.var(inst_parameters).item())
        # Evaluate on validation set
        val_loss, val_metric = validate(args, val_loader, model, loss_criterion_val, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_metric, val_loss, val_metric])

        # Save artifacts
        # utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

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

    # record best_mae
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, best_mae])

    print('total time: {}'.format(time.time() - start_epoch_time))
    return best_mae
    # with open(os.path.join(CURRENT_PATH, 'var_sigmas.txt'), 'a+') as f:
    #     f.write('inst only, all 1.0\n')
    #     f.write(str(var_sigmas))
    #     f.write('\n')


def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    start = timer()

    # args = parser.parse_args()
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    utils.generate_log_dir_hyp(args, ITERATION)
    utils.generate_save_dir_hyp(args, ITERATION)

    config = {}
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = params['clamp_min']  # 1/20,0.5,0.6,0.4,0.3,0.2,0.1
    config['clamp_inst_sigma']['max'] = np.log(20)

    utils.save_config_hyp(args.save_dir, {"args": str(args.__dict__), "config": config, "params": params}, ITERATION)
    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    best_mae = main_worker(args, config, params, ITERATION)
    print('best mae: {}, params: {}, iteration: {}, status: {}'.
          format(best_mae, params, ITERATION, STATUS_OK))

    # Dictionary with information for evaluation, otherwise will report errors
    run_time = utils.format_time(timer() - start)
    loss = best_mae
    return {'loss': loss, 'best_mae': best_mae,
            'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


if __name__ == '__main__':
    print("load params from : ", args.utkface_with_params_path)
    params = json.load(open(args.utkface_with_params_path, 'r', encoding="utf-8"))['best']
    main(params=params)
