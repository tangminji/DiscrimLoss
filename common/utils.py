#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
""" Utility functions for training DNNs with data parameters"""
import os
import json
import shutil
import random
import datetime
import torch
import numpy as np
from tensorboard_logger import configure, log_value, log_histogram
import tensorboard_logger
from optimizer.sparse_sgd import SparseSGD
import torch.nn.functional as F


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def adjust_learning_rate(model_initial_lr, optimizer, gamma, step):
    """Sets the learning rate to the initial learning rate decayed by 10 every few epochs.

    Args:
        model_initial_lr (int) : initial learning rate for model parameters
        optimizer (class derived under torch.optim): torch optimizer.
        gamma (float): fraction by which we are going to decay the learning rate of model parameters
        step (int) : number of steps in staircase learning rate decay schedule
    """
    lr = model_initial_lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_topk_accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): N*C tensor, contains logits for N samples over C classes.
        target (torch.Tensor):  labels for each row in prediction.
        topk (tuple of int): different values of k for which top-k accuracy should be computed.

    Returns:
        result (tuple of float): accuracy at different top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

def compute_MAE(args, prediction, target):
    '''
    compute MAE(mean absolute error) in regression task

    truncated absolute error: min(t, |y-y^|), for mnist:t=1.0, for UTKFACE:t=10.0
    '''
    prediction_ = prediction.squeeze(dim=-1)
    loss = F.l1_loss(prediction_, target, reduction='none')
    # t = torch.tensor([args.t] * target.shape[0]).requires_grad_().type_as(loss)
    # loss = torch.vstack((t, loss))
    # loss = torch.min(loss, dim=0)[0]
    MAE = loss.mean()
    return MAE


def save_artifacts(args, epoch, model, class_parameters, inst_parameters):
    """Saves model and data parameters.

    Args:
        args (argparse.Namespace):
        epoch (int): current epoch
        model (torch.nn.Module): DNN model.
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters.
    """
    artifacts = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'class_parameters': class_parameters.cpu().detach().numpy(),
            'inst_parameters': inst_parameters.cpu().detach().numpy()
             }

    file_path = args.save_dir + '/epoch_{}.pth.tar'.format(epoch)
    torch.save(obj=artifacts, f=file_path)

def checkpoint(acc, epoch, net, save_dir):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    file_path = save_dir + '/epoch_{}.pth.tar'.format(epoch)
    torch.save(obj=state, f=file_path)
    for last_e in range(epoch):
        last_e_path = save_dir + '/epoch_{}.pth.tar'.format(last_e)
        if os.path.exists(last_e_path):
            os.remove(last_e_path) #删除文件应该用os.remove
            # shutil.rmtree(last_e_path, ignore_errors=True)

def save_config(save_dir, cfg):
    """Save config file to disk at save_dir.

    Args:
        save_dir (str): path to directory.
        cfg (dict): config file.
    """
    save_path = save_dir + '/config.json'
    if os.path.isfile(save_path):
        raise Exception("Expected an empty folder but found an existing config file.., aborting")
    with open(save_path,  'w') as outfile:
        json.dump(cfg, outfile)

def save_config_hyp(save_dir, cfg, ITERATION):
    """Save config file to disk at save_dir.

    Args:
        save_dir (str): path to directory.
        cfg (dict): config file.
    """
    '''for hyperopt iteration for MAX_EVALS times'''
    save_dir = os.path.join(save_dir, str(ITERATION))
    save_path = save_dir + '/config.json'
    if os.path.isfile(save_path):
        raise Exception("Expected an empty folder but found an existing config file.., aborting")
    with open(save_path,  'w') as outfile:
        json.dump(cfg, outfile)

def save_loss(save_dir, loss, epoch):
    '''
    save loss for each sample in one epoch
    '''
    save_path = save_dir + '/loss_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch):loss.tolist()}
        outfile.write('{}{}'.format(loss_ep,'\n'))

def save_sigma(save_dir, sigma, epoch):
    '''
    save sigma for each sample in one epoch
    '''
    save_path = save_dir + '/sigma_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        sigma_ep = {'epoch:{}'.format(epoch):sigma.tolist()}
        outfile.write('{}{}'.format(sigma_ep,'\n'))

def save_loss_hyp(save_dir, loss, epoch, ITERATION):
    '''
    save loss for each sample in one epoch
    '''
    '''for hyperopt iteration for MAX_EVALS times'''
    save_dir = os.path.join(save_dir, str(ITERATION))
    save_path = save_dir + '/loss_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch):loss.tolist()}
        outfile.write('{}{}'.format(loss_ep,'\n'))

def save_sigma_hyp(save_dir, sigma, epoch, ITERATION):
    '''
    save sigma for each sample in one epoch
    '''
    '''for hyperopt iteration for MAX_EVALS times'''
    save_dir = os.path.join(save_dir, str(ITERATION))
    save_path = save_dir + '/sigma_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        sigma_ep = {'epoch:{}'.format(epoch):sigma.tolist()}
        outfile.write('{}{}'.format(sigma_ep,'\n'))

def generate_save_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nModel artifacts (checkpoints and config) are going to be saved in: {}'.format(args.save_dir))
    if os.path.exists(args.save_dir):
        if args.restart:
            print('Deleting old model artifacts found in: {}'.format(args.save_dir))
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)
        else:
            error='Old artifacts found; pass --restart flag to erase'.format(args.save_dir)
            raise Exception(error)
    else:
        os.makedirs(args.save_dir)


def generate_log_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nLog is going to be saved in: {}'.format(args.log_dir))

    if os.path.exists(args.log_dir):
        if args.restart:
            print('Deleting old log found in: {}'.format(args.log_dir))
            shutil.rmtree(args.log_dir)
            configure(args.log_dir, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(args.log_dir)
            raise Exception(error)
    else:
        configure(args.log_dir, flush_secs=10)


def generate_save_dir_hyp(args, ITERATION):
    """Generate directory to save artifacts and tensorboard log files."""
    '''for hyperopt iteration for MAX_EVALS times'''
    save_pth = os.path.join(args.save_dir, str(ITERATION))
    print('\nModel artifacts (checkpoints and config) are going to be saved in: {}'.format(save_pth))
    if os.path.exists(save_pth):
        if args.restart:
            print('Deleting old model artifacts found in: {}'.format(save_pth))
            shutil.rmtree(save_pth)
            os.makedirs(save_pth)
        else:
            error='Old artifacts found; pass --restart flag to erase'.format(save_pth)
            raise Exception(error)
    else:
        os.makedirs(save_pth)


def generate_log_dir_hyp(args, ITERATION):
    """Generate directory to save artifacts and tensorboard log files."""
    '''for hyperopt iteration for MAX_EVALS times'''
    log_pth = os.path.join(args.log_dir, str(ITERATION))
    print('\nLog is going to be saved in: {}'.format(log_pth))

    if os.path.exists(log_pth):
        if args.restart:
            print('Deleting old log found in: {}'.format(log_pth))
            shutil.rmtree(log_pth)
            configure(log_pth, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(log_pth)
            raise Exception(error)
    else:
        #https://blog.csdn.net/webmater2320/article/details/105831920
        #/users6/ttwu/.local/lib/python3.7/site-packages/tensorboard_logger/
        #/home/zhh/anaconda3/lib/python3.8/site-packages/tensorboard_logger/

        tensorboard_logger.clean_default_logger()
        configure(log_pth, flush_secs=10)

def set_seed(args):
    """Set seed to ensure deterministic runs.

    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_inst_data_params_n_optimizer(args,
                                           nr_classes,
                                           nr_instances,
                                           device):
    """Returns class and instance level data parameters and their corresponding optimizers.

    Args:
        args (argparse.Namespace):
        nr_classes (int):  number of classes in dataset.
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.

    Returns:
        class_parameters (torch.Tensor): class level data parameters.
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_class_param (SparseSGD): Sparse SGD optimizer for class parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """
    # class-parameter
    class_parameters = torch.tensor(np.ones(nr_classes) * np.log(args.init_class_param),
                                    dtype=torch.float32,
                                    requires_grad=args.learn_class_parameters,
                                    device=device)
    optimizer_class_param = SparseSGD([class_parameters],
                                      lr=args.lr_class_param,
                                      momentum=0.9,
                                      skip_update_zero_grad=True)
    if args.learn_class_parameters:
        print('Initialized class_parameters with: {}'.format(args.init_class_param))
        print('optimizer_class_param:')
        print(optimizer_class_param)

    # instance-parameter
    inst_parameters = torch.tensor(np.ones(nr_instances) * np.log(args.init_inst_param),
                                   dtype=torch.float32,
                                   requires_grad=args.learn_inst_parameters,
                                   device=device)
    # inst_parameters = torch.tensor(np.log(np.random.uniform(size=nr_instances)),
    #                                dtype=torch.float32,
    #                                requires_grad=args.learn_inst_parameters,
    #                                device=device)

    optimizer_inst_param = SparseSGD([inst_parameters],
                                     lr=args.lr_inst_param,
                                     momentum=0.9,
                                     skip_update_zero_grad=True)
    if args.learn_inst_parameters:
        print('Initialized inst_parameters with: {}'.format(args.init_inst_param))
        print('optimizer_inst_param:')
        print(optimizer_inst_param)

    return class_parameters, inst_parameters, optimizer_class_param, optimizer_inst_param

def get_inst_conf_n_optimizer(args,
                              nr_instances,
                              device,
                              params=None):
    """Returns instance level data parameters and their corresponding optimizers.

    Args:
        args (argparse.Namespace):
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.

    Returns:
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """
    # exp_avg
    exp_avg = torch.tensor(np.zeros(nr_instances),
                                   dtype=torch.float32,
                                   requires_grad=False,
                                   device=device)
    # instance-parameter
    inst_parameters = torch.tensor(np.ones(nr_instances) * np.log(args.init_inst_param),
                                   dtype=torch.float32,
                                   requires_grad=True,
                                   device=device)
    # inst_parameters = torch.tensor(np.log(np.random.uniform(size=nr_instances)),
    #                                dtype=torch.float32,
    #                                requires_grad=True,
    #                                device=device)

    optimizer_inst_param = SparseSGD([inst_parameters],
                                     lr=params['lr_inst_param'] if params != None else args.lr_inst_param,
                                     momentum=0.9,
                                     skip_update_zero_grad=True)
    if args.learn_inst_parameters:
        print('Initialized inst_parameters with: {}'.format(args.init_inst_param))
        print('optimizer_inst_param:')
        print(optimizer_inst_param)

    return inst_parameters, optimizer_inst_param, exp_avg

def get_tanh_optimizer():
    pass

def get_inst_conf_n_optimizer_tune(args,
                              nr_instances,
                              device,
                              hypa_config):
    """Returns instance level data parameters and their corresponding optimizers.

    Args:
        args (argparse.Namespace):
        nr_instances (int): number of instances in dataset.
        device (str): device on which data parameters should be placed.

    Returns:
        inst_parameters (torch.Tensor): instance level data parameters
        optimizer_inst_param (SparseSGD): Sparse SGD optimizer for instance parameters
    """
    # exp_avg
    exp_avg = torch.tensor(np.zeros(nr_instances),
                                   dtype=torch.float32,
                                   requires_grad=False,
                                   device=device)
    # instance-parameter
    inst_parameters = torch.tensor(np.ones(nr_instances) * np.log(args.init_inst_param),
                                   dtype=torch.float32,
                                   requires_grad=True,
                                   device=device)
    # inst_parameters = torch.tensor(np.log(np.random.uniform(size=nr_instances)),
    #                                dtype=torch.float32,
    #                                requires_grad=True,
    #                                device=device)

    optimizer_inst_param = SparseSGD([inst_parameters],
                                     lr=hypa_config['sig_lr'],
                                     momentum=0.9,
                                     skip_update_zero_grad=True)
    if args.learn_inst_parameters:
        print('Initialized inst_parameters with: {}'.format(args.init_inst_param))
        print('optimizer_inst_param:')
        print(optimizer_inst_param)

    return inst_parameters, optimizer_inst_param, exp_avg

def get_data_param_for_minibatch(args,
                                 class_param_minibatch,
                                 inst_param_minibatch):
    """Returns the effective data parameter for instances in a minibatch as per the specified curriculum.

    Args:
        args (argparse.Namespace):
        class_param_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_param_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        effective_data_param_minibatch (torch.Tensor): data parameter for samples in the minibatch.
    """
    sigma_class_minibatch = torch.exp(class_param_minibatch).view(-1, 1)
    sigma_inst_minibatch = torch.exp(inst_param_minibatch).view(-1, 1)

    if args.learn_class_parameters and args.learn_inst_parameters:
        # Joint curriculum
        effective_data_param_minibatch = sigma_class_minibatch + sigma_inst_minibatch
    elif args.learn_class_parameters:
        # Class level curriculum
        effective_data_param_minibatch = sigma_class_minibatch
    elif args.learn_inst_parameters:
        # Instance level curriculum
        effective_data_param_minibatch = sigma_inst_minibatch
    else:
        # This corresponds to the baseline case without data parameters
        effective_data_param_minibatch = 1.0

    return effective_data_param_minibatch


def apply_weight_decay_data_parameters(args, loss, class_parameter_minibatch, inst_parameter_minibatch):
    """Applies weight decay on class and instance level data parameters.

    We apply weight decay on only those data parameters which participate in a mini-batch.
    To apply weight-decay on a subset of data parameters, we explicitly include l2 penalty on these data
    parameters in the computational graph. Note, l2 penalty is applied in log domain. This encourages
    data parameters to stay close to value 1, and prevents data parameters from obtaining very high or
    low values.

    Args:
        args (argparse.Namespace):
        loss (torch.Tensor): loss of DNN model during forward.
        class_parameter_minibatch (torch.Tensor): class level parameters for samples in minibatch.
        inst_parameter_minibatch (torch.Tensor): instance level parameters for samples in minibatch.

    Returns:
        loss (torch.Tensor): loss augmented with l2 penalty on data parameters.
    """

    # Loss due to weight decay on instance-parameters
    if args.learn_inst_parameters and args.wd_inst_param > 0.0:
        loss = loss + 0.5 * args.wd_inst_param * (inst_parameter_minibatch ** 2).sum()

    # Loss due to weight decay on class-parameters
    if args.learn_class_parameters and args.wd_class_param > 0.0:
        # (We apply weight-decay to only those classes which are present in the mini-batch)
        loss = loss + 0.5 * args.wd_class_param * (class_parameter_minibatch ** 2).sum()

    return loss


def clamp_data_parameters(args, class_parameters, config, inst_parameters):
    """Clamps class and instance level parameters within specified range.

    Args:
        args (argparse.Namespace):
        class_parameters (torch.Tensor): class level parameters.
        inst_parameters (torch.Tensor): instance level parameters.
        config (dict): config file for the experiment.
    """
    if args.skip_clamp_data_param is False:
        if args.learn_inst_parameters:
            # Project the sigma's to be within certain range
            inst_parameters.data = inst_parameters.data.clamp_(
                min=config['clamp_inst_sigma']['min'],
                max=config['clamp_inst_sigma']['max'])
        if args.learn_class_parameters:
            # Project the sigma's to be within certain range
            class_parameters.data = class_parameters.data.clamp_(
                min=config['clamp_cls_sigma']['min'],
                max=config['clamp_cls_sigma']['max'])


def log_stats(data, name, step):
    """Logs statistics on tensorboard for data tensor.

    Args:
        data (torch.Tensor): torch tensor.
        name (str): name under which stats for the tensor should be logged.
        step (int): step used for logging
    """
    log_value('{}/highest'.format(name), torch.max(data).item(), step=step)
    log_value('{}/lowest'.format(name), torch.min(data).item(),  step=step)
    log_value('{}/mean'.format(name), torch.mean(data).item(),   step=step)
    log_value('{}/std'.format(name), torch.std(data).item(),     step=step)
    try:
        log_histogram('{}'.format(name), data.data.cpu().numpy(),    step=step)
    except:
        print('xxx')


def log_intermediate_iteration_stats(args, epoch, global_iter,
                                    losses, class_parameters=None,
                                     inst_parameters=None, top1=None, top5=None):
    """Log stats for data parameters and loss on tensorboard."""
    if top5 is not None:
        log_value('train_iteration_stats/accuracy_top5', top5.avg, step=global_iter)
    if top1 is not None:
        log_value('train_iteration_stats/accuracy_top1', top1.avg, step=global_iter)
    log_value('train_iteration_stats/loss', losses.avg, step=global_iter)
    log_value('train_iteration_stats/epoch', epoch, step=global_iter)

    # Log temperature stats
    if args.learn_class_parameters:
        log_stats(data=torch.exp(class_parameters),
                  name='iter_stats_class_parameter',
                  step=global_iter)
    if args.learn_inst_parameters:
        log_stats(data=torch.exp(inst_parameters),
                  name='iter_stats_inst_parameter',
                  step=global_iter)

def log_discrim_cl_intermediate_iteration_stats(epoch, global_iter,
                                    losses, inst_parameters=None, top1=None, top5=None):
    """
    dicrimloss for classification
    Log stats for data parameters and loss on tensorboard.
    """
    if top5 is not None:
        log_value('train_iteration_stats/accuracy_top5', top5.avg, step=global_iter)
    if top1 is not None:
        log_value('train_iteration_stats/accuracy_top1', top1.avg, step=global_iter)
    log_value('train_iteration_stats/loss', losses.avg, step=global_iter)
    log_value('train_iteration_stats/epoch', epoch, step=global_iter)

    # Log temperature stats
    log_stats(data=torch.exp(inst_parameters),
              name='iter_stats_inst_parameter',
              step=global_iter)

def log_discrim_reg_intermediate_iteration_stats(epoch, global_iter,
                                    losses, inst_parameters=None, mae=None):
    """
    dicrimloss for regression
    Log stats for data parameters and loss on tensorboard.
    """
    if mae is not None:
        log_value('train_iteration_stats/mean absolute error', mae.avg, step=global_iter)

    log_value('train_iteration_stats/loss', losses.avg, step=global_iter)
    log_value('train_iteration_stats/epoch', epoch, step=global_iter)

    # Log temperature stats
    log_stats(data=torch.exp(inst_parameters),
              name='iter_stats_inst_parameter',
              step=global_iter)

