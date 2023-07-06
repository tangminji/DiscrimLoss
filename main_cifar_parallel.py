#
# Code For Data Parameter
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,1,0"#"0,1,2"

import time
import argparse

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboard_logger import log_value

from common import utils
from dataset.cifar100_dataset import CIFAR100WithIdx
from models.wide_resnet import WideResNet28_10
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader)
from common.cmd_args import args, CURRENT_PATH, CIFAR_100_PATH, learning_rate_schedule
import csv

best_acc = 0
def get_train_and_val_loader(args):
    """"Constructs data loaders for train and val on CIFAR100

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for CIFAR100 train data.
        val_loader (torch.utils.data.DataLoader): data loader for CIFAR100 val data.
    """
    print('==> Preparing data for CIFAR100..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR100WithIdx(root=CIFAR_100_PATH,
                               train=True,
                               download=True,#True
                               transform=transform_train,
                               rand_fraction=args.rand_fraction)
    valset = CIFAR100WithIdx(root=CIFAR_100_PATH,
                             train=False,
                             download=True,#True
                             transform=transform_val)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                           batch_size=args.train_batch_size,
                           shuffle=True,)
                           #num_workers=args.workers)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # val_sampler = SequentialSampler(valset) if args.local_rank == -1 else DistributedSampler(valset)
    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,)#num_workers=args.workers)

    return train_loader, val_loader


def get_model_and_loss_criterion(args):
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
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # if args.device == 'cuda':
    #     criterion = nn.CrossEntropyLoss().cuda()
    # else:
    #     criterion = nn.CrossEntropyLoss()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    return model, criterion, logname


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
                        config):
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
    optimizer_class_param, optimizer_inst_param = optimizer_data_parameters
    class_parameters, inst_parameters = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    for i, (inputs, target, index_dataset) in enumerate(train_loader):
        global_iter = global_iter + 1
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        if args.learn_class_parameters:
            optimizer_class_param.zero_grad()
        if args.learn_inst_parameters:
            optimizer_inst_param.zero_grad()

        # Compute logits
        logits = model(inputs)

        if args.learn_class_parameters or args.learn_inst_parameters:
            # Compute data parameters for instances in the minibatch
            class_parameter_minibatch = class_parameters[target]
            inst_parameter_minibatch = inst_parameters[index_dataset]
            data_parameter_minibatch = utils.get_data_param_for_minibatch(
                                            args,
                                            class_param_minibatch=class_parameter_minibatch,
                                            inst_param_minibatch=inst_parameter_minibatch)

            # Compute logits scaled by data parameters
            logits = logits / data_parameter_minibatch

        loss = criterion(logits, target)
        loss_parameters[index_dataset] = loss
        loss = loss.mean()
        # Apply weight decay on data parameters
        if args.learn_class_parameters or args.learn_inst_parameters:
            loss = utils.apply_weight_decay_data_parameters(args, loss,
                                                            class_parameter_minibatch=class_parameter_minibatch,
                                                            inst_parameter_minibatch=inst_parameter_minibatch)
        # if args.n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if args.learn_class_parameters:
            optimizer_class_param.step()
        if args.learn_inst_parameters:
            optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.learn_class_parameters or args.learn_inst_parameters:
            utils.clamp_data_parameters(args, class_parameters, config, inst_parameters)

        # Measure accuracy and record loss
        acc1 = utils.compute_topk_accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            utils.log_intermediate_iteration_stats(args, epoch,
                                                   global_iter,
                                                   losses,
                                                   class_parameters=class_parameters,
                                                   inst_parameters=inst_parameters,
                                                   top1=top1)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)
    log_value('train/loss', losses.avg, step=epoch)

    utils.save_loss(args.save_dir, loss_parameters, epoch)
    return global_iter, losses.avg, top1.avg


def main_worker(args, config):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config (dict): config file for the experiment.
    """
    global best_acc
    global_iter = 0
    #learning_rate_schedule = np.array([80, 100, 160])

    # Create model
    model, loss_criterion, logname = get_model_and_loss_criterion(args)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Get train and validation dataset loader
    train_loader, val_loader = get_train_and_val_loader(args)

    # Initialize class and instance based temperature
    (class_parameters, inst_parameters,
     optimizer_class_param, optimizer_inst_param) = utils.get_class_inst_data_params_n_optimizer(
                                                        args=args,
                                                        nr_classes=args.nr_classes,
                                                        nr_instances=len(train_loader.dataset),
                                                        device='cuda'
                                                        )
    # var_sigmas = []
    # var_sigmas.append(torch.var(inst_parameters).item())
    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])

    for epoch in range(args.start_epoch, args.epochs):

        # Adjust learning rate for model parameters
        if epoch in learning_rate_schedule:
            utils.adjust_learning_rate(model_initial_lr=args.lr,
                                 optimizer=optimizer,
                                 gamma=0.1,
                                 step=np.sum(epoch >= learning_rate_schedule))

        # Train for one epoch
        global_iter, train_loss, train_acc = train_for_one_epoch(
                            args=args,
                            train_loader=train_loader,
                            model=model,
                            criterion=loss_criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_iter=global_iter,
                            optimizer_data_parameters=(optimizer_class_param, optimizer_inst_param),
                            data_parameters=(class_parameters, inst_parameters),
                            config=config)

        #var_sigmas.append(torch.var(inst_parameters).item())
        # Evaluate on validation set
        val_loss, val_acc = validate(args, val_loader, model, loss_criterion, epoch)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Save artifacts
        utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

        # Log temperature stats over epochs
        if args.learn_class_parameters:
            utils.log_stats(data=torch.exp(class_parameters),
                            name='epochs_stats_class_parameter',
                            step=epoch)
        if args.learn_inst_parameters:
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

    args.log_dir = CURRENT_PATH + '/CIFAR100/datapara_results/logs_n40_bt128'
    args.save_dir = CURRENT_PATH + '/CIFAR100/datapara_results/weights_n40_bt128'
    args.nr_classes = 100 # Number classes in CIFAR100
    utils.generate_log_dir(args)
    utils.generate_save_dir(args)

    config = {}
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = np.log(1/20)
    config['clamp_inst_sigma']['max'] = np.log(20)
    config['clamp_cls_sigma'] = {}
    config['clamp_cls_sigma']['min'] = np.log(1/20)
    config['clamp_cls_sigma']['max'] = np.log(20)
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
'''