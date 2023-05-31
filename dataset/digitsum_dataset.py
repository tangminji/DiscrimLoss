#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import utils
from common.DiscrimLoss import (DiscrimEA_TANHLoss, DiscrimEA_EMAK_TANHLoss, DiscrimEA_GAK_TANHLoss)
from common.cmd_args_digitsum import DIGITSUM_PATH
from models.digitsum_lstm import DigitsumLstm

def get_DIGITSUM_train_val_test_loader(args):
    """"
    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader
        val_loader (torch.utils.data.DataLoader): data loader
        test_loader (torch.utils.data.DataLoader): data loader
    """
    print('==> Preparing data for DIGITSUM..')
    utils.set_seed(args)
    trainset = load_dataset(os.path.join(DIGITSUM_PATH, "train.txt"), rand_fraction=args.rand_fraction, cache_dir=args.save_dir)
    # trainset = load_dataset(os.path.join(DIGITSUM_PATH, "train.txt"), rand_fraction=args.rand_fraction)
    # if args.rand_fraction == 0.0:
    #     trainset = load_dataset(os.path.join(DIGITSUM_PATH, "train.txt"))
    # else:
    #     trainset = load_dataset(os.path.join(DIGITSUM_PATH, "noisy", f"{args.rand_fraction:.2f}.txt"))
    valset = load_dataset(os.path.join(DIGITSUM_PATH, "val.txt"))
    testset = load_dataset(os.path.join(DIGITSUM_PATH, "test.txt"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # args.per_gpu_train_batch_size
    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              shuffle=True, )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size

    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False, )  # num_workers=args.workers)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size
    test_loader = DataLoader(testset,
                             batch_size=args.test_batch_size,
                             shuffle=False, )  # num_workers=args.workers)
    return train_loader, val_loader, test_loader


def get_DIGITSUM_train_val_test_loader_with_bucket(args):
    """"
    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader
        val_loader (torch.utils.data.DataLoader): data loader
        test_loader (torch.utils.data.DataLoader): data loader
    """
    print('==> Preparing data for DIGITSUM..')
    utils.set_seed(args.seed)
    trainset = load_dataset(os.path.join(DIGITSUM_PATH, "train.txt"), rand_fraction=args.rand_fraction)
    
    valset = load_dataset(os.path.join(DIGITSUM_PATH, "val.txt"))
    testset = load_dataset(os.path.join(DIGITSUM_PATH, "test.txt"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # args.per_gpu_train_batch_size

    length_diff_index = []
    last_len = -1
    train_lengths = trainset.tensors[1].tolist()
    # for it_len_id, it_len in enumerate(train_lengths):
    #     assert last_len <= it_len
    #     if it_len != last_len:
    #         length_diff_index.append(it_len_id)
    #     last_len = it_len

    length_diff_index = list(range(0, len(train_lengths), len(train_lengths) // 5))

    if length_diff_index[-1] != len(train_lengths):
        length_diff_index.append(len(train_lengths))

    train_loader_list = []
    for iid, i in enumerate(length_diff_index[1:]):
        train_loader_list.append(DataLoader(trainset,
                                            batch_size=args.train_batch_size,
                                            sampler=torch.utils.data.SubsetRandomSampler(list(range(0, i)))))

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size

    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False, )  # num_workers=args.workers)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)  # args.per_gpu_eval_batch_size
    test_loader = DataLoader(testset,
                             batch_size=args.test_batch_size,
                             shuffle=False, )  # num_workers=args.workers)
    return train_loader_list, val_loader, test_loader


def get_DIGITSUM_model_and_loss_criterion(args, params=None, ITERATION=None):
    """

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module):
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building DIGITSUM')
    args.arch = 'digitsum_lstm'  # 20epochs with SGD
    model = DigitsumLstm(hidden_size=args.hidden_size)
    if ITERATION is None:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    else:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '_hy_iter_%d.csv' % ITERATION

    model.to(args.device)

    if args.is_discrimloss:
        if params is None:
            # parser.add_argument('--digitsum_loss_type', type=str,default="ea_emak_tanh_newq",choices=["no_cl","ea_gak_tanh_newq","ea_emak_tanh_newq","ea_tanh_newq"])
            if args.digitsum_loss_type == "ea_gak_tanh_newq":
                criterion = DiscrimEA_GAK_TANHLoss().to(args.device)
            elif args.digitsum_loss_type == "ea_emak_tanh_newq":
                criterion = DiscrimEA_EMAK_TANHLoss().to(args.device)
            elif args.digitsum_loss_type == "ea_tanh_newq":
                criterion = DiscrimEA_TANHLoss(k1=0.5).to(args.device)
            else:
                assert False

        else:

            if args.digitsum_loss_type == "ea_gak_tanh_newq":
                criterion = DiscrimEA_GAK_TANHLoss(a=params['tanh_a'], p=params['tanh_p'],
                                                   q=-params['tanh_q'] * params['tanh_p']).to(args.device)
            elif args.digitsum_loss_type == "ea_emak_tanh_newq":
                criterion = DiscrimEA_EMAK_TANHLoss(a=params['tanh_a'], p=params['tanh_p'],
                                                    q=-params['tanh_q'] * params['tanh_p']).to(args.device)
            elif args.digitsum_loss_type == "ea_tanh_newq":
                criterion = DiscrimEA_TANHLoss(k1=0.5, a=params['tanh_a'], p=params['tanh_p'],
                                               q=-params['tanh_q'] * params['tanh_p']).to(args.device)
            else:
                assert False
    else:
        assert args.digitsum_loss_type == "no_cl"
        criterion = nn.MSELoss(reduction='none').to(args.device)
    criterion_val = nn.MSELoss(reduction='none').to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    assert args.local_rank == -1
    return model, criterion, criterion_val, logname

# def load_dataset(data_path):
#     print("load data from:\t", data_path)
#     assert os.path.exists(data_path)
#     datas = [[int(tt) for tt in t.rstrip().split(" ")] for t in open(data_path, 'r', encoding="utf-8").readlines()]

#     # datas = sorted(datas, key=lambda x: len(x))
#     X = [t[:-1] for t in datas]
#     Y = [t[-1] for t in datas]

#     lengths = [len(t) for t in X]
#     max_len = max(lengths)

#     for x in X:
#         while len(x) < max_len:
#             x.append(10)

#     X = torch.tensor(X, dtype=torch.long)
#     Y = torch.tensor(Y, dtype=torch.float)
#     lengths = torch.tensor(lengths, dtype=torch.int64)
#     index_set = torch.tensor(range(X.size(0)), dtype=torch.long)
#     return TensorDataset(X, lengths, index_set, Y)

def load_dataset(data_path, rand_fraction=0., cache_dir=None):
    print("load data from:\t", data_path)
    assert 0 <= rand_fraction <= 1

    datas = [[int(tt) for tt in t.rstrip().split(" ")] for t in open(data_path, 'r', encoding="utf-8").readlines()]

    datas = sorted(datas, key=lambda x: len(x))
    X = [t[:-1] for t in datas]
    Y = [t[-1] for t in datas]

    oldX = [t[:-1] for t in datas]

    lengths = [len(t) for t in X]
    max_len = max(lengths)

    for x in X:
        while len(x) < max_len:
            x.append(10)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    nr_corrupt_instances = int(math.floor(X.size(0) * rand_fraction))

    shuffle_index = random.sample(list(range(X.size(0))), nr_corrupt_instances)
    origin_index = sorted(shuffle_index)

    Y[origin_index] = Y[shuffle_index]

    print('randomizing {} fraction of data == {} / {}'.format(rand_fraction,
                                                              nr_corrupt_instances,
                                                              X.size(0)))
    index_set = torch.tensor(range(X.size(0)), dtype=torch.long)

    if rand_fraction>0 and cache_dir is not None:
        cache_file = os.path.join(cache_dir,f"{rand_fraction:.2f}.txt")
        with open(cache_file,"w",encoding="utf-8") as f:
            for x,y in zip(oldX,Y):
                f.write(" ".join([str(xx) for xx in x])+f" {int(y)}\n")
        print(f"Save noisy label to {cache_file}\n")

    return TensorDataset(X, lengths, index_set, Y)


def create_digitsum_data():
    # args
    train_num = 1000
    val_num = 200
    test_num = 200
    min_len, max_len = 2, 20
    val_len = 20
    test_len = 20

    # warning if there has existed data.
    for n in ("train", "val", "test"):
        f_n = os.path.join(DIGITSUM_PATH, "%s.txt" % n)
        if os.path.exists(f_n):
            yn = input("Data already exists: %s, do you want to overwrite it?" % f_n)
            if yn not in ("yes", "y", "YES", "Y"):
                return

    # run
    with open(os.path.join(DIGITSUM_PATH, "train.txt"), 'w+', encoding="utf-8") as w:
        for i in range(min_len, max_len + 1):
            print(i)
            visited_set = set()
            while len(visited_set) < min(train_num, pow(10, i)):
                print(i, len(visited_set), end="\r")
                cur_nums = []
                for j in range(i):
                    cur_nums.append(random.randint(0, 9))
                cur_seq = " ".join([str(t) for t in cur_nums])
                if cur_seq in visited_set:
                    continue
                visited_set.add(cur_seq)
                w.write(cur_seq + " " + str(sum(cur_nums)) + "\n")

    with open(os.path.join(DIGITSUM_PATH, "val.txt"), 'w+', encoding="utf-8") as w:
        visited_set = set()
        while len(visited_set) < min(val_num, pow(10, i)):
            cur_nums = []
            for j in range(val_len):
                cur_nums.append(random.randint(0, 9))
            cur_seq = " ".join([str(t) for t in cur_nums])
            if cur_seq in visited_set:
                continue
            visited_set.add(cur_seq)
            w.write(cur_seq + " " + str(sum(cur_nums)) + "\n")

    with open(os.path.join(DIGITSUM_PATH, "test.txt"), 'w+', encoding="utf-8") as w:
        visited_set = set()
        while len(visited_set) < min(test_num, pow(10, i)):
            cur_nums = []
            for j in range(test_len):
                cur_nums.append(random.randint(0, 9))
            cur_seq = " ".join([str(t) for t in cur_nums])
            if cur_seq in visited_set:
                continue
            visited_set.add(cur_seq)
            w.write(cur_seq + " " + str(sum(cur_nums)) + "\n")


if __name__ == '__main__':
    create_digitsum_data()
