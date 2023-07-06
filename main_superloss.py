from __future__ import print_function
import torch.optim as optim

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"#"0,1,2"

import math
import time
import csv
from common.utils import *

from common.SuperLoss import SuperLoss
from common.cmd_args import args, CURRENT_PATH, learning_rate_schedule
from dataset.mnist_dataset import get_MNIST_train_and_val_loader, get_MNIST_model_and_loss_criterion
from dataset.cifar100_dataset import get_CIFAR100_train_and_val_loader, get_CIFAR100_model_and_loss_criterion
from dataset.cifar10_dataset import get_CIFAR10_train_and_val_loader, get_CIFAR10_model_and_loss_criterion


best_acc = 0

MD_CLASSES = {
    'MNIST':(get_MNIST_train_and_val_loader, get_MNIST_model_and_loss_criterion),
    'CIFAR10':(get_CIFAR10_train_and_val_loader, get_CIFAR10_model_and_loss_criterion),
    'CIFAR100':(get_CIFAR100_train_and_val_loader, get_CIFAR100_model_and_loss_criterion),
    'UTKFACE':()
}

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
    loaders, mdl_loss = MD_CLASSES[args.dataset]

    if args.dataset == 'CIFAR10':
        num_classes = 10
        C=math.log(num_classes)
        lam = 1
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        C = math.log(num_classes)
        lam = 0.25
    elif args.dataset == 'MNIST':
        num_classes = 10
        C=0.5
        lam = 1#is not mentioned in paper superloss

    # Get train and validation dataset loader
    train_loader, val_loader = loaders(args)
    # Create model
    net, loss_criterion, loss_criterion_val, logname = mdl_loss(args)
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
    criterion = SuperLoss(C=C, lam=lam, batch_size=args.train_batch_size).to(args.device)
    criterion_val = SuperLoss(C=C, lam=lam, batch_size=args.eval_batch_size).to(args.device)
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
                                            trainloader=train_loader,
                                            net=net,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            global_iter=global_iter)
        val_loss, val_acc = validate(args=args,
                                     epoch=epoch,
                                     valloader=val_loader,
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

    generate_log_dir(args)
    generate_save_dir(args)
    # Set seed for reproducibility
    set_seed(args)
    main()
'''
nohup python main_superloss.py >> ./superloss_results/log_no_noise.log 2>&1 &
nohup python main_superloss.py >> ./CIFAR100/superloss_results/log_n40_bt128_wd3.log 2>&1 &
'''