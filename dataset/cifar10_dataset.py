#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from common.DiscrimLoss import (DiscrimEA_TANHLoss_newQ, DiscrimEA_EMAK_TANHLoss_newQ, DiscrimEA_GAK_TANHLoss_newQ,
                                DiscrimEA_EMAK_TANHWO_EALoss_newQ, DiscrimEA_EMAK_TANH_WO_ESLoss_newQ,DiscrimEA_EMAK_TANHLoss_FIXK_newQ)
from common.cmd_args import CIFAR10_PATH
from models.wide_resnet import WideResNet28_10


class CIFAR10WithIdx(CIFAR10):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0,
                 seed=10):
        super(CIFAR10WithIdx, self).__init__(root=root,
                                             train=train,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)

        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction

        if self.rand_fraction > 0.0:
            self.seed = seed
            self.data = self.corrupt_fraction_of_data()

    def corrupt_fraction_of_data(self):
        """Corrupts fraction of train data by permuting image-label pairs."""

        # Check if we are not corrupting test data
        assert self.train is True, 'We should not corrupt test data.'

        nr_points = len(self.data)
        nr_corrupt_instances = int(np.floor(nr_points * self.rand_fraction))
        print('Randomizing {} fraction of data == {} / {}'.format(self.rand_fraction,
                                                                  nr_corrupt_instances,
                                                                  nr_points))
        # We will corrupt the top fraction data points
        corrupt_data = self.data[:nr_corrupt_instances, :, :, :]
        clean_data = self.data[nr_corrupt_instances:, :, :, :]

        # Corrupting data
        np.random.seed(self.seed)
        rand_idx = np.random.permutation(np.arange(len(corrupt_data)))
        corrupt_data = corrupt_data[rand_idx, :, :, :]

        # Adding corrupt and clean data back together
        return np.vstack((corrupt_data, clean_data))

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index


def get_CIFAR10_train_and_val_loader(args):
    """"Constructs data loaders for train and val on CIFAR10
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for CIFAR100 train data.
        val_loader (torch.utils.data.DataLoader): data loader for CIFAR100 val data.
    """
    print('==> Preparing data for CIFAR10..')

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

    trainset = CIFAR10WithIdx(root=CIFAR10_PATH,
                              train=True,
                              download=True,  # True
                              transform=transform_train,
                              rand_fraction=args.rand_fraction,
                              seed=args.seed)
    valset = CIFAR10WithIdx(root=CIFAR10_PATH,
                            train=False,
                            download=True,  # True
                            transform=transform_val)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # args.per_gpu_train_batch_size
    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              shuffle=True, )
    # num_workers=args.workers)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size
    # val_sampler = SequentialSampler(valset) if args.local_rank == -1 else DistributedSampler(valset)
    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False, )  # num_workers=args.workers)

    return train_loader, val_loader


def get_CIFAR10_model_and_loss_criterion(args, params=None, ITERATION=None):
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
    if ITERATION is None:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    else:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '_hy_iter_%d.csv' % ITERATION
    model.to(args.device)

    if params is None:
        if args.cifar_loss_type == "ea_gak_tanh_newq":
            criterion = DiscrimEA_GAK_TANHLoss_newQ().to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_newQ().to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_wo_ea_newq":
            criterion = DiscrimEA_EMAK_TANHWO_EALoss_newQ().to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_wo_es_newq":
            criterion = DiscrimEA_EMAK_TANH_WO_ESLoss_newQ().to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_fixk_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_FIXK_newQ(classes=args.nr_classes).to(args.device)
        elif args.cifar_loss_type == "ea_tanh_newq":
            criterion = DiscrimEA_TANHLoss_newQ(k1=math.log(args.nr_classes)).to(args.device)
        elif args.cifar_loss_type == "no_cl":
            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        else:
            assert False
    else:
        if args.cifar_loss_type == "ea_gak_tanh_newq":
            criterion = DiscrimEA_GAK_TANHLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                    q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                     q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)

        elif args.cifar_loss_type == "ea_emak_tanh_wo_ea_newq":
            criterion = DiscrimEA_EMAK_TANHWO_EALoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                          q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_wo_es_newq":
            criterion = DiscrimEA_EMAK_TANH_WO_ESLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                           q=params['tanh_q'], sup_eps=params['sup_eps']).to(
                args.device)
        elif args.cifar_loss_type == "ea_emak_tanh_fixk_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_FIXK_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                           q=params['tanh_q'], sup_eps=params['sup_eps'],classes=args.nr_classes).to(args.device)
        elif args.cifar_loss_type == "ea_tanh_newq":
            criterion = DiscrimEA_TANHLoss_newQ(k1=math.log(args.nr_classes), a=params['tanh_a'], p=params['tanh_p'],
                                                q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.cifar_loss_type == "no_cl":
            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        else:
            assert False

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
