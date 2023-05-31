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
import re
import shutil
import time

import hyperopt
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboard_logger import log_value
from transformers import get_linear_schedule_with_warmup, AdamW

from common import utils
from common.cmd_args import args
from dataset.wikihow_dataset import get_WIKIHOW_train_val_test_loader, get_WIKIHOW_model_and_loss_criterion
from timeit import default_timer as timer

best_acc = 0
best_mae = 65536
best_test_acc=0
ITERATION=0

MD_CLASSES = {
    'WIKIHOW': (get_WIKIHOW_train_val_test_loader, get_WIKIHOW_model_and_loss_criterion)
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
    top1 = utils.AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = utils.AverageMeter("MAE", ":6.2f")  # for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # input_ids, input_mask, segment_ids, label_ids, index_dataset
        for i, batch in enumerate(val_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            target = batch[3]
            logits, = model(**inputs)
            loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs["input_ids"].size(0))
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs["input_ids"].size(0))
        print('Val-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))

    log_value('val/loss', losses.avg, step=epoch)
    # Logging results on tensorboard

    log_value('val/accuracy', top1.avg, step=epoch)
    # Save checkpoint.


    acc = top1.avg

    return losses.avg, top1.avg


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
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            target = batch[3]
            logits, = model(**inputs)
            loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs["input_ids"].size(0))
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs["input_ids"].size(0))
        print('Test-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))

    log_value('test/loss', losses.avg, step=epoch)
    # Logging results on tensorboard

    log_value('test/accuracy', top1.avg, step=epoch)
    return losses.avg, top1.avg


def train_for_one_epoch(args,
                        train_loader,
                        train_loader_iter,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        optimizer_data_parameters,
                        data_parameters,
                        config,
                        params=None,
                        ITERATION=None,
                        scheduler=None,
                        switch=False):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        train_loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer : optimizer for model parameters.
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
    is_train_loader_iter_empty = False
    print("start train one epoch ... ")
    for i in range(args.sub_epochs_step):
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            is_train_loader_iter_empty = True
            break
        global_iter = global_iter + 1

        # print("cudaing batch ... ")
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        target, index_dataset = batch[3:5]
        # print("index_dataset_in_train_loop",index_dataset)
        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        optimizer_inst_param.zero_grad()
        # Compute logits
        # print("cal model ... ")
        logits, = model(**inputs)

        # Compute data parameters for instances in the minibatch
        inst_parameter_minibatch = inst_parameters[index_dataset]
        data_parameter_minibatch = torch.exp(inst_parameter_minibatch)

        if args.wikihow_loss_type != "no_cl":
            loss = criterion(args, logits, target, data_parameter_minibatch, exp_avg, index_dataset,
                             epoch)
        else:
            loss = criterion(logits, target)

        loss_parameters[index_dataset] = loss
        loss = loss.mean()

        # Apply weight decay on data parameters
        if params["wd_inst_param"] > 0.0:
            loss = loss + 0.5 * params["wd_inst_param"] * (inst_parameter_minibatch ** 2).sum()  # 2,4

        # Compute gradient and do SGD step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer_inst_param.step()

        scheduler.step()
        # Clamp class and instance level parameters within certain bounds
        if args.skip_clamp_data_param is False:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])

        # Measure accuracy and record loss
        losses.update(loss.item(), inputs["input_ids"].size(0))

        acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
        top1.update(acc1[0].item(), inputs["input_ids"].size(0))

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
    return global_iter, losses.avg, train_metric, is_train_loader_iter_empty  # top1.avg


def main_worker(args, config):
    """Trains model on ImageNet using data parameters
    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global best_acc
    global best_mae
    global best_test_acc
    best_acc = 0
    best_mae = 65536
    best_test_acc = 0


    global_iter = 0
    # learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    model, loss_criterion, loss_criterion_val, logname = mdl_loss(args,params, ITERATION)

    # Define optimizer

    param_mo = model.module if hasattr(model, "module") else model
    optimizer = AdamW(param_mo.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)

    # Get train and validation dataset loader
    train_loader, val_loader, test_loader = loaders(args)

    # Initialize class and instance based temperature
    (inst_parameters, optimizer_inst_param, exp_avg) = utils.get_inst_conf_n_optimizer(
        args=args,
        nr_instances=len(train_loader.dataset),
        params=params,
        device='cuda')

    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            assert args.task_type == 'classification'
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'val loss', 'val acc', 'test loss', 'test acc'])
    start_epoch_time = time.time()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.num_train_epochs
    )
    epoch = -1
    for num_train_e in range(args.start_epoch, args.num_train_epochs):
        train_loader_iter = iter(train_loader)
        is_train_loader_iter_empty = False
        while not is_train_loader_iter_empty:
            epoch += 1
            # Train for one epoch
            global_iter, train_loss, train_metric, is_train_loader_iter_empty = train_for_one_epoch(
                args=args,
                train_loader=train_loader,
                train_loader_iter=train_loader_iter,
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
            if val_metric > best_acc:
                best_acc = val_metric
                best_test_acc=test_metric
                print('Saving..')
                model_to_save = model.module if hasattr(model, "module") else model

                model_output_file = args.save_dir + '/epoch_{}/model'.format(epoch)
                os.makedirs(model_output_file, exist_ok=True)
                model_to_save.save_pretrained(model_output_file)
                os.system("cp %s %s" % (os.path.join(args.model_path, "merges.txt"), model_output_file))
                os.system("cp %s %s" % (os.path.join(args.model_path, "vocab.json"), model_output_file))
                for last_e in range(epoch):
                    last_e_path = args.save_dir + '/epoch_{}/model'.format(last_e)
                    if os.path.exists(last_e_path):
                        shutil.rmtree(last_e_path, ignore_errors=True)

            state = {
                'acc': val_metric,
                'epoch': epoch,
                'rng_state': torch.get_rng_state()
            }
            os.makedirs(args.save_dir + '/epoch_{}'.format(epoch), exist_ok=True)
            torch.save(obj=state, f=args.save_dir + '/epoch_{}/state'.format(epoch))


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
        logwriter.writerow([-1, -1, -1, -1, best_acc, -1, best_test_acc])

    print('total time: {}'.format(time.time() - start_epoch_time))
    return best_test_acc
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
    # TODO
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = params['clamp_min']
    config['clamp_inst_sigma']['max'] = np.log(20)

    utils.save_config_hyp(args.save_dir, {"args": str(args.__dict__), "config": config, "params": params}, ITERATION)

    # Set seed for reproducibility
    utils.set_seed(args)
    # Simply call main_worker function
    best_acc = main_worker(args, config, params, ITERATION)
    print('best acc: {}, params: {}, iteration: {}, status: {}'.
          format(best_acc, params, ITERATION, hyperopt.STATUS_OK))

    run_time = utils.format_time(timer() - start)
    return {'loss': -best_acc, 'best_acc': best_acc,
            'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': hyperopt.STATUS_OK}


if __name__ == '__main__':
    print("load params from : ", args.wikihow_with_params_path)
    params = json.load(open(args.wikihow_with_params_path, 'r', encoding="utf-8"))['best']
    assert params is not None
    main(params=params)
