from __future__ import print_function
import torch.optim as optim

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2"

import torchvision.transforms as transforms
import time
import csv

from models.wide_resnet import WideResNet28_10
from common.utils import *

#from data.cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from dataset.cifar100_dataset import CIFAR100WithIdx
from common.SuperLoss import SuperLoss
from common.cmd_args import args, CIFAR_100_PATH, CURRENT_PATH, learning_rate_schedule


best_acc = 0
def main():
    global best_acc
    global_iter = 0
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

    # load dataset

    if args.dataset == 'cifar10':
        num_classes = 10
        lam = 1
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        val_dataset = CIFAR10(root='./data/',
                              download=True,
                              train=False,
                              transform=transform_val,
                              noise_type=args.noise_type,
                              noise_rate=args.noise_rate
                              )

    if args.dataset == 'cifar100':
        num_classes = 100
        lam = 0.25#1e-3#
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

        train_dataset = CIFAR100WithIdx(root=CIFAR_100_PATH,
                                        train=True,
                                        download=True,
                                        transform=transform_train,
                                        rand_fraction=args.rand_fraction,
                                        seed=args.seed)

        val_dataset = CIFAR100WithIdx(root=CIFAR_100_PATH,
                                      train=False,
                                      download=True,
                                      transform=transform_val)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    valloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    trainloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Model
    print('==> Building model.. (Default : WideResNet28_10)')
    net = WideResNet28_10(num_classes=num_classes)
    logname = os.path.join(args.save_dir, net.__class__.__name__)  + '.csv'

    net.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        net = torch.nn.DataParallel(net)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()
    criterion = SuperLoss(C=num_classes, lam=lam, batch_size=args.train_batch_size).to(args.device)
    criterion_val = SuperLoss(C=num_classes, lam=lam, batch_size=args.eval_batch_size).to(args.device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.schedule, gamma=args.gamma)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])

    for epoch in range(args.start_epoch, args.epochs):

        # Adjust learning rate for model parameters
        if epoch in learning_rate_schedule:
            adjust_learning_rate(model_initial_lr=args.lr,
                                 optimizer=optimizer,
                                 gamma=0.1,
                                 step=np.sum(epoch >= learning_rate_schedule))

        global_iter, train_loss, train_acc = train(
                                            args=args,
                                            epoch=epoch,
                                            trainloader=trainloader,
                                            net=net,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            global_iter=global_iter)
        val_loss, val_acc = validate(args=args,
                                     epoch=epoch,
                                     valloader=valloader,
                                     net=net,
                                     criterion=criterion_val)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
    #record best_acc
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, best_acc])



# Training
def train(args, epoch, trainloader, net, criterion, optimizer, global_iter):
    print('\nEpoch: %d' % epoch)
    # Initialize counters
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    net.train()
    start_epoch_time = time.time()

    # loss-for-each-sample
    loss_parameters = torch.tensor(np.zeros(len(trainloader.dataset)),
                                   dtype=torch.float32,
                                   requires_grad=False,
                                   device=args.device)
    sigma_parameters = torch.tensor(np.zeros(len(trainloader.dataset)),
                                   dtype=torch.float32,
                                   requires_grad=False,
                                   device=args.device)

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        global_iter = global_iter + 1
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        outputs = net(inputs)
        loss, sigma = criterion(outputs, targets)
        #record loss/sigma for each sample
        loss_parameters[indexes] = loss
        sigma_parameters[indexes] = sigma
        #loss average
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure accuracy and record loss
        acc1 = compute_topk_accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        if batch_idx % args.print_freq == 0:
            log_intermediate_iteration_stats(args, epoch, global_iter, losses, top1=top1)

    # Print and log stats for the epoch
    print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)
    log_value('train/loss', losses.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_sigma(args.save_dir, sigma_parameters, epoch)
    return global_iter, losses.avg, top1.avg


def validate(args, epoch, valloader, net, criterion):
    global best_acc
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(valloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss, sigma = criterion(outputs, targets)
            loss = loss.mean()
            # measure accuracy and record loss
            acc1 = compute_topk_accuracy(outputs, targets, topk=(1,))
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
        checkpoint(acc, epoch, net, args.save_dir)

    return losses.avg, top1.avg


if __name__ == '__main__':
    args.log_dir = CURRENT_PATH + '/CIFAR100/superloss_results/logs_n0_bt128'
    args.save_dir = CURRENT_PATH + '/CIFAR100/superloss_results/weights_n0_bt128'
    generate_log_dir(args)
    generate_save_dir(args)
    # Set seed for reproducibility
    set_seed(args)
    main()
'''
nohup python main_superloss.py >> ./superloss_results/log_no_noise.log 2>&1 &
nohup python main_superloss.py >> ./CIFAR100/superloss_results/log_n40_bt128_wd3.log 2>&1 &
'''