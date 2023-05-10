#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import math
import os
import random

import torch
from torch.utils.data import TensorDataset

DIGITSUM_PATH = '.'


def load_dataset(data_path, rand_fraction=0.):
    print("load data from:\t", data_path)
    assert 0 <= rand_fraction <= 1

    datas = [[int(tt) for tt in t.rstrip().split(" ")] for t in open(data_path, 'r', encoding="utf-8").readlines()]

    datas = sorted(datas, key=lambda x: len(x))
    X = [t[:-1] for t in datas]
    Y = [t[-1] for t in datas]

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
    # 这里由于交换的是Y标签，因此index仍然为0,1,2...
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

def init_noisy_data(rand_fraction=0.):
    data_path='train.txt'
    datas = [[tt for tt in t.rstrip().split(" ")] for t in open(data_path, 'r', encoding="utf-8").readlines()]
    
    X = [t[:-1] for t in datas]
    Y = [t[-1] for t in datas]
    old_y = [y for y in Y]
    length = len(X)
    nr_corrupt_instances = int(math.floor(length * rand_fraction))

    shuffle_index = random.sample(list(range(length)), nr_corrupt_instances)
    origin_index = sorted(shuffle_index)
    labels = [Y[i] for i in shuffle_index]
    for i in range(len(origin_index)):
        Y[origin_index[i]] = labels[i]

    print('randomizing {} fraction of data == {} / {}'.format(rand_fraction,
                                                              nr_corrupt_instances,
                                                              length))
    match = 0
    for i in range(length):
        if Y[i]==old_y[i]:
            match+=1
    print(f'Acc: {match/length:.3f}')
    folder = 'noisy'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f'{rand_fraction:.2f}.txt'),'w') as f:
        for x,y in zip(X,Y):
            f.write(" ".join(x)+f" {y}\n")

if __name__ == '__main__':
    for fr in [0.2,0.4,0.6,0.8]:
        init_noisy_data(fr)
    # create_digitsum_data()
