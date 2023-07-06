import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torch.utils.data import DataLoader
from common.cmd_args_clothing1m import Clothing1M_PATH
import models.resnet as resnet
from common.DiscrimLoss import (DiscrimEA_TANHLoss,
                                 DiscrimEA_EMAK_TANHLoss,
                                 DiscrimEA_GAK_TANHLoss)
from common.SuperLoss import SuperLoss_gak

class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)

        if mode == 'train':
            flist = os.path.join(root, "annotations/noisy_train.txt")#100w,265664(18976*14)
        if mode == 'val':
            flist = os.path.join(root, "annotations/clean_val.txt")#14313
        if mode == 'test':
            flist = os.path.join(root, "annotations/clean_test.txt")#10526

        self.impaths, self.targets = self.flist_reader(flist)

        if num_per_class > 0:
            impaths, targets = [], []
            num_each_class = np.zeros(14)
            indexs = np.arange(len(self.impaths))
            random.shuffle(indexs)

            for i in indexs:
                if num_each_class[self.targets[i]] < num_per_class:
                    impaths.append(self.impaths[i])
                    targets.append(self.targets[i])
                    num_each_class[self.targets[i]] += 1

            self.impaths, self.targets = impaths, targets
            print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class],
                                                                       int(sum(num_each_class))))

    #         # for quickly ebug
    #         self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]

    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0][7:])#remove "image/"
                targets.append(int(row[1]))
        return impaths, targets


class Clothing1MWithIdx(Clothing1M):
    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 num_per_class=-1):
        super(Clothing1MWithIdx, self).__init__(root=root,
                                                mode=mode,
                                                transform=transform,
                                                target_transform=target_transform,
                                                num_per_class=num_per_class)
    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index

def get_Clothing1M_train_and_val_loader(args):
    '''
    batch_size: train(32),val/test(128)
    '''
    print('==> Preparing data for Clothing1M..')

    train_transform = transforms.Compose([transforms.Resize((256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                          ])
    test_transform = transforms.Compose([transforms.Resize((256)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                         ])

    train_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                      mode='train',
                                      transform=train_transform,
                                      num_per_class=args.num_per_class)

    val_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                    mode='val',
                                    transform=test_transform)
    test_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                     mode='test',
                                     transform=test_transform)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    args.val_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_Clothing1M_model_and_loss_criterion(args):
    print('Building ResNet50')
    args.arch = 'ResNet50'
    model = create_model()
    logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    model.to(args.device)

    # todo 选择 discrimloss
    a = args.a
    p = args.p
    q = -args.newq * args.p
    sup_eps = args.sup_eps

    if args.clothing1m_loss_type == "ea_gak_tanh_newq":
        criterion = DiscrimEA_GAK_TANHLoss(a=a, p=p,
                                           q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "ea_emak_tanh_newq":
        criterion = DiscrimEA_EMAK_TANHLoss(a=a, p=p,
                                            q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "ea_tanh_newq":
        criterion = DiscrimEA_TANHLoss(k1=0.5, a=a, p=p,
                                       q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "no_cl":
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    else:
        assert False

    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)
    criterion_test = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    return model, criterion, criterion_val, criterion_test, logname

def get_Clothing1M_model_and_loss_criterion_ontop(args):
    '''
    robust loss+discrimloss
    '''
    print('Building ResNet50')
    args.arch = 'ResNet50'
    model = create_model()
    logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    model.to(args.device)

    # todo 选择 discrimloss
    a = args.a
    p = args.p
    q = -args.newq * args.p
    sup_eps = args.sup_eps
    #TODO: 切换为ontopclassification: Superloss+DiscrimLoss
    args.task_type = 'ontopclassification'
    if args.clothing1m_loss_type == "ea_gak_tanh_newq":
        criterion = DiscrimEA_GAK_TANHLoss(a=a, p=p,
                                           q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "ea_emak_tanh_newq":
        criterion = DiscrimEA_EMAK_TANHLoss(a=a, p=p,
                                            q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "ea_tanh_newq":
        criterion = DiscrimEA_TANHLoss(k1=0.5, a=a, p=p,
                                       q=q, sup_eps=sup_eps).to(args.device)
    elif args.clothing1m_loss_type == "no_cl":
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    else:
        assert False

    criterion_val = SuperLoss_gak(lam=args.lam, batch_size=args.val_batch_size).to(args.device)
    criterion_test = SuperLoss_gak(lam=args.lam, batch_size=args.test_batch_size).to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    return model, criterion, criterion_val, criterion_test, logname

def learning_rate(lr_init, epoch):
    optim_factor = 0
    if (epoch > 5):
        optim_factor = 1
    return lr_init * math.pow(0.1, optim_factor)

# model
def create_model(ema=False):
    net = resnet.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, 14)#14 classes for clothing1m
    #net = net.cuda()

    if ema:
        for param in net.parameters():
            param.detach_()
    return net