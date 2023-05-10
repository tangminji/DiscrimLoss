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


def validate(args, val_loader, model, criterion, epoch):
    """Evaluates model on validation set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """

    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # input_ids, input_mask, segment_ids, label_ids, index_dataset
        for i, batch in enumerate(val_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, lengths, _, target = batch
            logits = model(inputs, lengths)

            logits_ = logits.squeeze(dim=-1)
            loss = criterion(logits_, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), target.size(0))
            mae = utils.compute_MAE(args, logits, target)
            MAE.update(mae.item(), inputs.size(0))
        print('Val-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))

    log_value('val/loss', losses.avg, step=epoch)
    # Logging results on tensorboard
    log_value('val/mean absolute error', MAE.avg, step=epoch)
    # Save checkpoint.
    mae = MAE.avg



    return losses.avg, MAE.avg


def test(args, test_loader, model, criterion, epoch):
    """Evaluates model on test set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        test_loader (torch.utils.data.dataloader): dataloader for test set.
        model (torch.nn.Module):
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # input_ids, input_mask, segment_ids, label_ids, index_dataset
        for i, batch in enumerate(test_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, lengths, _, target = batch
            logits = model(inputs, lengths)

            logits_ = logits.squeeze(dim=-1)
            loss = criterion(logits_, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), target.size(0))
            mae = utils.compute_MAE(args, logits, target)
            MAE.update(mae.item(), inputs.size(0))
        print('Test-Epoch-{}: MAE:{}, Loss:{}'.format(epoch, MAE.avg, losses.avg))

    log_value('test/loss', losses.avg, step=epoch)
    # Logging results on tensorboard
    log_value('test/mean absolute error', MAE.avg, step=epoch)

    return losses.avg, MAE.avg


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
                        scheduler=None,
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
    top1 = utils.AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression

    # Unpack data parameters
    optimizer_inst_param = optimizer_data_parameters
    inst_parameters, exp_avg = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    # all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_index_dataset
    print("start train one epoch ... ")
    for i, batch in enumerate(train_loader):

        global_iter = global_iter + 1

        # todo 更新，根据model和criterion
        batch = tuple(t.to(args.device) for t in batch)
        inputs, lengths, index_dataset, target = batch

        optimizer.zero_grad()
        optimizer_inst_param.zero_grad()
        logits = model(inputs, lengths)

        # Compute data parameters for instances in the minibatch
        inst_parameter_minibatch = inst_parameters[index_dataset]
        data_parameter_minibatch = torch.exp(inst_parameter_minibatch)

        if args.is_discrimloss:
            loss = criterion(args, logits, target, data_parameter_minibatch, exp_avg, index_dataset,
                             epoch)
        else:
            loss = criterion(logits.squeeze(dim=-1), target)

        # 只是用于记录log信息
        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        if params["wd_inst_param"] > 0.0:
            loss = loss + 0.5 * params["wd_inst_param"] * (inst_parameter_minibatch ** 2).sum()  # 2,4

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer_inst_param.step()
        if not args.no_scheduler:
            scheduler.step()
        # Clamp class and instance level parameters within certain bounds
        if args.skip_clamp_data_param is False:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])

        # Measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        mae = utils.compute_MAE(args, logits, target)
        MAE.update(mae.item(), target.size(0))

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
    global best_acc
    global best_mae
    global best_test_mae
    best_acc = 0
    best_mae = 65536
    best_test_mae = 65536

    global_iter = 0
    # learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    model, loss_criterion, loss_criterion_val, logname = mdl_loss(args, params, ITERATION)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Get train and validation dataset loader
    train_loader, val_loader, test_loader = loaders(args)

    # Initialize class and instance based temperature
    (inst_parameters, optimizer_inst_param, exp_avg) = utils.get_inst_conf_n_optimizer(
        args=args,
        nr_instances=len(train_loader.dataset),
        device=args.device)

    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            assert args.task_type == 'regression'
            logwriter.writerow(
                ['epoch', 'train loss', 'train mae', 'val loss', 'val mae', 'test loss', 'test mae'])
    start_epoch_time = time.time()
    if not args.no_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_loader),
            num_training_steps=len(train_loader) * args.epochs
        )
    else:
        scheduler = None
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
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
            scheduler=scheduler,
            params=params,
            ITERATION=ITERATION,
            switch=False)

        # Evaluate on validation set
        val_loss, val_metric = validate(args, val_loader, model, loss_criterion_val, epoch)
        # Evaluate on test set
        test_loss, test_metric = test(args, test_loader, model, loss_criterion_val, epoch)
        if val_metric < best_mae:  # 取最小
            best_mae = val_metric
            best_test_mae = test_metric
            if not args.no_save_model:
                utils.checkpoint(val_metric, epoch, model, args.save_dir)
        # log model metrics
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric])

        # Log temperature stats over epochs
        utils.log_stats(data=torch.exp(inst_parameters),
                        name='epoch_stats_inst_parameter',
                        step=epoch)

    # record best_acc/best_mae
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, best_mae, -1, best_test_mae])

    print('total time: {}'.format(time.time() - start_epoch_time))
    return best_test_mae


def main(params):
    global ITERATION
    ITERATION += 1
    start = timer()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    utils.generate_log_dir_hyp(args, ITERATION)
    utils.generate_save_dir_hyp(args, ITERATION)

    config = {}
    config['clamp_inst_sigma'] = {}
    # config['clamp_inst_sigma']['min'] = np.log(0.05)#1/20,0.5,0.6,0.4,0.3,0.2,0.1
    config['clamp_inst_sigma']['min'] = params['clamp_min']
    config['clamp_inst_sigma']['max'] = np.log(20)

    utils.save_config_hyp(args.save_dir, {"args": str(args.__dict__), "config": config, "params": params}, ITERATION)
    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    best_mae = main_worker(args, config, params, ITERATION)
    print('best mae: {}, params: {}, iteration: {}, status: {}'.
          format(best_mae, params, ITERATION, hyperopt.STATUS_OK))

    run_time = utils.format_time(timer() - start)
    return {'loss': best_mae, 'best_mae': best_mae,
            'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': hyperopt.STATUS_OK}


if __name__ == '__main__':
    # Global variable
    # MAX_EVALS = 3
    MAX_EVALS = 50
    space = {
        'wd_inst_param': hp.loguniform('wd_inst_param', np.log(1e-8), np.log(0.1)),
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
