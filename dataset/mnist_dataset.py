#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from common.DiscrimLoss import (DiscrimEA_TANHLoss_newQ, DiscrimEA_EMAK_TANHLoss_newQ, DiscrimEA_GAK_TANHLoss_newQ,
                                DiscrimEA_EMAK_TANHWO_EALoss_newQ, DiscrimEA_EMAK_TANH_WO_ESLoss_newQ)
from common.cmd_args import MNIST_PATH
from models.lenet import LeNet


class MNISTWithIdx(MNIST):
    """
    Extends MNIST dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0,
                 seed=10):
        super(MNISTWithIdx, self).__init__(root=root,
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
        corrupt_data = self.data[:nr_corrupt_instances, :, :]
        clean_data = self.data[nr_corrupt_instances:, :, :]

        # Corrupting data
        np.random.seed(self.seed)
        rand_idx = np.random.permutation(np.arange(len(corrupt_data)))
        corrupt_data = corrupt_data[rand_idx, :, :]

        # Adding corrupt and clean data back together
        return torch.vstack((corrupt_data, clean_data))

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index


def get_MNIST_train_and_val_loader(args):
    """"Constructs data loaders for train and val on CIFAR10
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for CIFAR100 train data.
        val_loader (torch.utils.data.DataLoader): data loader for CIFAR100 val data.
    """
    print('==> Preparing data for MNIST..')
    # https://blog.csdn.net/kane7csdn/article/details/109038429
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = MNISTWithIdx(root=MNIST_PATH,
                            train=True,
                            download=False,  # True
                            transform=transform,
                            rand_fraction=args.rand_fraction,
                            seed=args.seed)

    valset = MNISTWithIdx(root=MNIST_PATH,
                          train=False,
                          download=False,  # True
                          transform=transform)

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


def get_MNIST_model_and_loss_criterion(args, params=None, ITERATION=None):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building LeNet')
    args.arch = 'LeNet'  # 20epochs with SGD
    model = LeNet()
    if ITERATION is None:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    else:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '_hy_iter_%d.csv' % ITERATION

    model.to(args.device)
    # https://zhuanlan.zhihu.com/p/108489655

    if params is None:
        if args.minist_loss_type == "ea_gak_tanh_newq":
            criterion = DiscrimEA_GAK_TANHLoss_newQ().to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_newQ().to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_wo_ea_newq":
            criterion = DiscrimEA_EMAK_TANHWO_EALoss_newQ().to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_wo_es_newq":
            criterion = DiscrimEA_EMAK_TANH_WO_ESLoss_newQ().to(args.device)
        elif args.minist_loss_type == "ea_tanh_newq":
            criterion = DiscrimEA_TANHLoss_newQ(k1=0.5).to(args.device)
        elif args.minist_loss_type == "no_cl":
            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    else:
        if args.minist_loss_type == "ea_gak_tanh_newq":
            criterion = DiscrimEA_GAK_TANHLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                    q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_newq":
            criterion = DiscrimEA_EMAK_TANHLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                     q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_wo_ea_newq":
            criterion = DiscrimEA_EMAK_TANHWO_EALoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                     q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.minist_loss_type == "ea_emak_tanh_wo_es_newq":
            criterion = DiscrimEA_EMAK_TANH_WO_ESLoss_newQ(a=params['tanh_a'], p=params['tanh_p'],
                                                     q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.minist_loss_type == "ea_tanh_newq":
            criterion = DiscrimEA_TANHLoss_newQ(k1=0.5, a=params['tanh_a'], p=params['tanh_p'],
                                                q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.minist_loss_type == "no_cl":
            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

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
