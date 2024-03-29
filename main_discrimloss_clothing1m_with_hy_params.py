# our method main.py
# to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
from timeit import default_timer as timer
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0,1,2,7"#
from tqdm import tqdm
import time
import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
from hyperopt import STATUS_OK
from tensorboard_logger import log_value

from common import utils
from common.cmd_args_clothing1m import args
from dataset.clothing1m_dataset import get_Clothing1M_train_and_val_loader, \
    get_Clothing1M_model_and_loss_criterion, learning_rate
import csv

val_best = 0
test_at_best = 0
ITERATION = 0
# 模型，数据单独配置, model and data
MD_CLASSES = {
    'Clothing1M': (get_Clothing1M_train_and_val_loader, get_Clothing1M_model_and_loss_criterion),
}


def validate(args, val_loader, model, criterion, epoch, mode='val'):
    """Evaluates model on validation/test set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(val_loader):
            inputs, target = inputs.to(args.device), target.to(args.device)

            # compute output
            logits = model(inputs)
            loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs.size(0))
        print('{}-Epoch-{}: Acc:{}, Loss:{}'.format('Test' if mode == 'test' else 'Val', epoch, top1.avg, losses.avg))

    # Logging results on tensorboard
    log_value('{}/loss'.format(mode), losses.avg, step=epoch)
    log_value('{}/accuracy'.format(mode), top1.avg, step=epoch)
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
                        ITERATION):
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
    top1 = utils.AverageMeter('Acc@1', ':6.2f')  # for classification

    # Unpack data parameters
    optimizer_inst_param = optimizer_data_parameters
    inst_parameters, exp_avg = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    for i, (inputs, target, index_dataset) in enumerate(tqdm(train_loader, unit='batch')):
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

        # logits = logits / data_parameter_minibatch
        loss = criterion(args, logits, target, data_parameter_minibatch, exp_avg, index_dataset, epoch)

        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        # if args.wd_inst_param > 0.0:
        if params['wd_inst_param'] > 0.0:
            loss = loss + 0.5 * params['wd_inst_param'] * (inst_parameter_minibatch ** 2).sum()  # 2,4

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
        acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
        top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            utils.log_discrim_cl_intermediate_iteration_stats(epoch,
                                                              global_iter,
                                                              losses,
                                                              inst_parameters=inst_parameters,
                                                              top1=top1)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    log_value('train/loss', losses.avg, step=epoch)
    print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)

    utils.save_loss_hyp(args.save_dir, loss_parameters, epoch, ITERATION)
    utils.save_sigma_hyp(args.save_dir, inst_parameters, epoch, ITERATION)
    train_metric = top1.avg
    return global_iter, losses.avg, train_metric  # top1.avg


def main_worker(args, config, params, ITERATION):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global val_best, test_at_best
    global_iter = 0
    # learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]

    # Create model
    model, loss_criterion, loss_criterion_val, loss_criterion_test, logname = mdl_loss(args, params, ITERATION)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  # args.lr

    # Get train and validation dataset loader
    train_loader, val_loader, test_loader = loaders(args)

    # Initialize class and instance based temperature
    (inst_parameters, optimizer_inst_param, exp_avg) = utils.get_inst_conf_n_optimizer(
        args=args,
        nr_instances=len(train_loader.dataset),
        device=args.device, params=params)

    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'val loss', 'val acc', 'test loss', 'test acc'])
    start_epoch_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        # Adjust learning rate for model parameters
        lr = learning_rate(args.lr, epoch + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
            params=params,
            ITERATION=ITERATION,
        )

        # Evaluate on validation set
        # TODO: val_metric: acc for classification, MAE for regression
        val_loss, val_metric = validate(args, val_loader, model, loss_criterion_val, epoch, mode='val')
        test_loss, test_metric = validate(args, test_loader, model, loss_criterion_test, epoch, mode='test')

        # Save checkpoint.
        if val_metric > val_best:
            val_best = val_metric
            test_at_best = test_metric
            utils.checkpoint(val_metric, epoch, model, args.save_dir)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric])

        # Save artifacts
        # utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

        # Log temperature stats over epochs
        utils.log_stats(data=torch.exp(inst_parameters),
                        name='epoch_stats_inst_parameter',
                        step=epoch)

    # record best_acc
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, val_best, -1, test_at_best])

    print('total time: {}'.format(time.time() - start_epoch_time))
    return val_best, test_at_best


def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals

    global ITERATION
    ITERATION += 1

    global best_acc
    best_acc = 0

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
    # config['clamp_inst_sigma']['min'] = np.log(0.05)#1/20,0.5,0.6,0.4,0.3,0.2,0.1
    # todo
    config['clamp_inst_sigma']['min'] = params['clamp_min']
    config['clamp_inst_sigma']['max'] = np.log(20)

    utils.save_config_hyp(args.save_dir, {"args": str(args.__dict__), "config": config, "params": params}, ITERATION)
    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    best_acc, test_at_best = main_worker(args, config, params, ITERATION)
    print('best acc: {}, test at best: {}, params: {}, iteration: {}, status: {}'.
          format(best_acc, test_at_best, params, ITERATION, STATUS_OK))

    # Dictionary with information for evaluation, otherwise will report errors
    # 必须有loss
    run_time = utils.format_time(timer() - start)
    loss = - best_acc
    return {'loss': loss, 'best_acc': best_acc, 'test_at_best': test_at_best,
            'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


if __name__ == '__main__':
    print("load params from : ", args.clothing1m_with_params_path)
    params = json.load(open(args.clothing1m_with_params_path, 'r', encoding="utf-8"))['best']
    assert params is not None
    # clothing1m_ablation_a
    # params['tanh_a'] = params['tanh_a'] if args.clothing1m_ablation_a is None else args.clothing1m_ablation_a
    # params['tanh_p'] = params['tanh_p'] if args.clothing1m_ablation_p is None else args.clothing1m_ablation_p
    # params['tanh_q'] = params['tanh_q'] if args.clothing1m_ablation_newq is None else args.clothing1m_ablation_newq
    main(params=params)
