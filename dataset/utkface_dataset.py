#https://susanqq.github.io/UTKFace/
#[age]_[gender]_[race]_[date&time].jpg
# [age] is an integer from 0 to 116, indicating the age
# [gender] is either 0 (male) or 1 (female)
# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
# [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
# ls -l | grep "^[d-]" | wc -l  ...23708

import glob
import os
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from torch.utils import data
from torchvision.datasets.folder import pil_loader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import shutil
from models.resnet import resnet18
import torch.optim as optim
import sys
#get current root directory
# o_path = '/home/zhh/code/ml-data-parameters-master/'
# sys.path.insert(0, o_path)#load local transformer not the package one

from common.cmd_args import args, UTKFACE_TRAINPATH, UTKFACE_TESTPATH, UTKFACE_IMGPATH
from PIL import Image
from common.DiscrimLoss import (DiscrimLoss, DiscrimESLoss, DiscrimEALoss,
                                DiscrimEA_WO_ESLoss, DiscrimEA_WO_BCLoss,
                                DiscrimEA_GAKLoss, DiscrimEA_EMAKLoss,
                                DiscrimEA_2Loss, DiscrimEA_0Loss,
                                DiscrimEAEXPLoss, DiscrimEA_TANHLoss,
                                DiscrimEA_EMAK_TANHLoss, DiscrimEA_GAK_TANHLoss)

class UTKFACEWithIdx(data.Dataset):

    gender_map = dict(male=0, female=1)
    race_map = dict(white=0, black=1, asian=2, indian=3, others=4)

    def __init__(self,
                 train=True,
                 transform = None,
                 rand_fraction=0.0,
                 seed=10):
        self.train = train
        self.data = []
        self.targets = []
        if self.train:
            root = UTKFACE_TRAINPATH
        else:
            root = UTKFACE_TESTPATH
        paths = glob.glob(os.path.join(root, '*/*'))
        for path in paths:
            try:
                self.data.append(path)
                self.targets.append(self._load_label(path))
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue
        self.transform = transform
        #TODO: original function
        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction

        if self.rand_fraction > 0.0:
            self.seed = seed
            self.data = self.corrupt_fraction_of_data()
        #TODO: 已实现的功能，不会再实现
        #self.root = root#传入的root为UTKFACE_IMGPATH
        #self.loader = transforms.Compose([transforms.ToTensor()])
        #self.seed = seed
        #self._prepare_samples(root)
        #self.__preprocess_mean_std_caculate()
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
        corrupt_data = self.data[:nr_corrupt_instances]
        clean_data = self.data[nr_corrupt_instances:]

        # Corrupting data
        np.random.seed(self.seed)
        rand_idx = np.random.permutation(np.arange(len(corrupt_data)))

        # self.data = self.data[rand_idx, :, :, :]
        # #self.targets = self.targets[rand_idx]
        corrupt_data_ = []
        for idx in rand_idx:
             corrupt_data_.append(corrupt_data[idx])
        # Adding corrupt and clean data back together
        corrupt_data_.extend(clean_data)
        return corrupt_data_

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)#3,128,128

        return image, label, index

    def __len__(self):
        return len(self.data)

    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        #age, gender, race = map(int, str_list[:3])
        age = float(str_list[0])#整形报错
        label = age#dict(age=age, gender=gender, race=race)
        return label

    def _load_datetime(self, s):
        return datetime.strptime(s, '%Y%m%d%H%M%S%f')

    def _prepare_samples(self, root):
        '''
        file split into 9/1 and save into file folder train/test, original folder UTKFace is empty after that
        to get original files, you can unzip the file: faces_aligned_cropped.zip
        '''
        samples = []

        paths = glob.glob(os.path.join(root, '*/*'))

        for path in paths:
            try:
                samples.append(path)
                #label = self._load_label(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue

            #samples.append((path, label))
        #shuffle
        np.random.seed(self.seed)
        rand_idx = np.random.permutation(np.arange(len(samples)))
        total_nums = len(samples)
        train_nums = int(np.floor(total_nums * 0.9))
        for i in rand_idx[:train_nums]:
            shutil.move(samples[i], os.path.join(UTKFACE_TRAINPATH, os.path.basename(samples[i])))
        for i in rand_idx[train_nums:]:
            shutil.move(samples[i], os.path.join(UTKFACE_TESTPATH, os.path.basename(samples[i])))
        return

    def _image_to_tensor(self, image_pth):
        image = Image.open(image_pth).convert('RGB')
        img_tensor = self.loader(image)#.to(self.device)#.unsqueeze(0)
        return img_tensor#1,3,200,200

    def _preprocess_mean_std_caculate(self):
        '''
        preprocess for UTKFace from jpg into tensor file(.pt): not used, take too much memory
        caculate the dataset's mean and std
        used only once
        '''
        self.data = []
        self.targets = []
        paths = glob.glob(os.path.join(self.root, '*/*'))
        count = 0
        for path in paths:
            count +=1
            if count % 1000 == 0:
                print("Loading {} images".format(count))
            try:
                #label = self._load_label(path)
                img_lst = self._image_to_tensor(path).numpy().tolist()
                self.data.append(img_lst)
                #self.targets.append(label)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue

        #tensor list to numpy
        self.data = np.array(self.data)#N,3,200,200
        mean_c1, mean_c2, mean_c3 = 0, 0, 0
        std_c1, std_c2, std_c3 = 0, 0, 0
        for x in self.data:
            mean_c1 += np.mean(x[0, :, :])
            mean_c2 += np.mean(x[1, :, :])
            mean_c3 += np.mean(x[2, :, :])
        mean_c1 /= len(self.data)
        mean_c2 /= len(self.data)
        mean_c3 /= len(self.data)
        self.data[:, 0, :, :] -= mean_c1
        self.data[:, 1, :, :] -= mean_c2
        self.data[:, 2, :, :] -= mean_c3
        for x in self.data:
            std_c1 += np.mean(np.square(x[0, :, :]).flatten())
            std_c2 += np.mean(np.square(x[1, :, :]).flatten())
            std_c3 += np.mean(np.square(x[2, :, :]).flatten())
        std_c1 = np.sqrt(std_c1 / len(self.data))
        std_c2 = np.sqrt(std_c2 / len(self.data))
        std_c3 = np.sqrt(std_c3 / len(self.data))
        print('mean:({},{},{}),std:({},{},{})'.format(
            mean_c1, mean_c2, mean_c3, std_c1, std_c2, std_c3
        ))
        #mean:(0.5960904839657701,0.4563706087169738,0.390623407629947),std:(0.25924394640268644,0.23129983400841414,0.22683888527326607)
        #save samples into file
        #shuffle the dataset and split into train/test with ratio 90%:10%
        # assert len(self.data)==len(self.targets)
        # total_nums = len(self.data)
        # train_nums = int(np.floor(total_nums * 0.9))
        # print('Training {} fraction of data == {} / {}'.format(0.9,
        #                                                       train_nums,
        #                                                       total_nums))
        # #shuffle data
        # np.random.seed(self.seed)
        # rand_idx = np.random.permutation(np.arange(total_nums))
        # self.data = self.data[rand_idx, :, :, :]
        # #self.targets = self.targets[rand_idx]
        # self.targets_ = []
        # for idx in rand_idx:
        #      self.targets_.append(self.targets[idx])
        # # We will corrupt the top fraction data points
        # train_data = self.data[:train_nums, :, :, :]
        # train_targets = self.targets_[:train_nums]
        # test_data = self.data[train_nums:, :, :, :]
        # test_targets = self.targets_[train_nums:]
        # #file path
        # train_pth = os.path.join(UTKFACE_PATH, 'train')
        # torch.save((train_data, train_targets), train_pth)
        # test_pth = os.path.join(UTKFACE_PATH, 'test')
        # torch.save((test_data, test_targets), test_pth)
        # result = torch.load(train_pth)
        # print(result)
        return


def get_UTKFACE_train_and_val_loader(args):
    """"Constructs data loaders for train and val on UTKFACE
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for CIFAR100 train data.
        val_loader (torch.utils.data.DataLoader): data loader for CIFAR100 val data.
    """
    print('==> Preparing data for UTKFACE..')
    #https://blog.csdn.net/kane7csdn/article/details/109038429
    #https://github.com/Jooong/Face-Aging-CAAE-Pytorch/dataloader.py
    #mean:(0.5960904839657701,0.4563706087169738,0.390623407629947),std:(0.25924394640268644,0.23129983400841414,0.22683888527326607)
    # transform = transforms.Compose([
    #     transforms.Resize(128),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.596, 0.456, 0.391), (0.259, 0.231, 0.227)),
    # ])
    #TODO:依照作者邮件设置，同imagenet
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = UTKFACEWithIdx(train=True,
                              transform = transform,
                              rand_fraction=args.rand_fraction,
                              seed=args.seed)
    valset = UTKFACEWithIdx(train=False,
                            transform=transform)

    args.train_batch_size =  args.per_gpu_train_batch_size* max(1, args.n_gpu)#args.per_gpu_train_batch_size
    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                           batch_size=args.train_batch_size,
                           shuffle=True,)
                           #num_workers=args.workers)
    args.eval_batch_size =  args.per_gpu_eval_batch_size* max(1, args.n_gpu)#args.per_gpu_eval_batch_size
    # val_sampler = SequentialSampler(valset) if args.local_rank == -1 else DistributedSampler(valset)
    val_loader = DataLoader(valset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,)#num_workers=args.workers)

    return train_loader, val_loader

def get_UTKFACE_model_and_loss_criterion(args, params=None, ITERATION=None):
    """Initializes DNN model and loss function.
    #https://blog.csdn.net/LXX516/article/details/80124768
    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building pretrained ResNet18')
    args.arch = 'pretrained ResNet18'#100epochs with SGD
    #https://pytorch.org/vision/stable/models.html
    #resnet18 pretrained by ImageNet
    model = resnet18(num_classes=1)
    model_dict = model.state_dict()
    pretrained_model = models.resnet18(pretrained=True)#模型下载，只一次即可
    pretrained_dict = pretrained_model.state_dict()
    #1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if ITERATION is None:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '.csv'
    else:
        logname = os.path.join(args.save_dir, model.__class__.__name__) + '_hy_iter_%d.csv' % ITERATION


    model.to(args.device)
    #原文中并未将tau设置为固定值

    if params is None:
        if args.utkface_loss_type == "ea_gak_tanh":
            criterion = DiscrimEA_GAK_TANHLoss().to(args.device)
        elif args.utkface_loss_type == "ea_emak_tanh":
            criterion = DiscrimEA_EMAK_TANHLoss().to(args.device)
    else:
        if args.utkface_loss_type == "ea_gak_tanh":
            criterion = DiscrimEA_GAK_TANHLoss(a=params['tanh_a'], p=params['tanh_p'],
                                               q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)
        elif args.utkface_loss_type == "ea_emak_tanh":
            criterion = DiscrimEA_EMAK_TANHLoss(a=params['tanh_a'], p=params['tanh_p'],
                                                q=params['tanh_q'], sup_eps=params['sup_eps']).to(args.device)

    if args.reg_loss_type == 'L1':
        criterion_val = nn.SmoothL1Loss(reduction='none').to(args.device)
    elif args.reg_loss_type == 'L2':
        criterion_val = nn.MSELoss(reduction='none').to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    return model, criterion, criterion_val, logname

#TODO:作者提供，暂时未使用
class RegressionError(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
        if type == 'MeanAbsolute':
            self.shortname = 'mae'
            self.err_func = lambda pred, label: (pred - label.float()).abs().sum()
            self.agg_func = lambda err,n: err/n
        elif type == 'RootMeanSquare':
            self.shortname = 'rmse'
            self.err_func = lambda pred, label: ((pred - label.float())**2).sum()
            self.agg_func = lambda err,n: (err/n) ** 0.5
        elif type.startswith('Truncated_'):
            self.thr = float(type.split('_')[1])
            self.shortname = f'trunc{self.thr:g}'
            self.err_func = lambda pred, label: (pred - label.float()).abs().clamp(max=self.thr).sum()
            self.agg_func = lambda err,n: err/n
        else:
            raise ValueError(f'bad type: {type}')

    def reset(self):
        self.error = 0.0
        self.total = 0.0

    def forward(self, outputs, labels, **kw):
        error = self.err_func(outputs.squeeze(), labels)
        self.error += error
        self.total += len(labels)
        return dict(error = self.agg_func(error, len(labels)).item() )

    def gather_all(self):
        result = dict(error = self.agg_func(self.error,self.total).item() )
        return result


# loss = RegressionLoss(nn.SmoothL1Loss())
# eval = RegressionError('MeanAbsolute')
# optimizer = optim.SGD(params, lr=0.1, momentum=0.9)



