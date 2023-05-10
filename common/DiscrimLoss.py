import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import AverageMeter

beta = 0.9  # for 1st phase to seperate easy/hard sample, one sample loss(ea)
rho = 0.9  # for k
gamma = 1.7  # for 2nd phase to seperate hard/noisy sample
TOTAL_STEP = 5  # for 2nd phase with iteration mode


# SUPPRESSION_EPS = 3
def ES_linear(ce, SUPPRESSION_EPS):
    '''
    ce:current epoch
    '''
    if ce < SUPPRESSION_EPS:
        return (ce + 1) / 10.0
    else:
        return 1.0

def ES_piecewise(ce, SUPPRESSION_EPS):
    '''
    ce:current epoch
    '''
    if ce < SUPPRESSION_EPS:
        return 0.2
    else:
        return 1.0


class DiscrimLoss(nn.Module):

    def __init__(self, k1=10):
        super(DiscrimLoss, self).__init__()
        self.k1 = k1

    def forward(self, logits, targets, data_parameter_minibatch):
        loss = F.cross_entropy(logits, targets, reduction='none')
        loss.sub_(self.k1)
        # Compute losses scaled by data parameters
        newloss = loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return newloss


class DiscrimESLoss(nn.Module):
    '''
    discrim loss with early suppression
    '''

    def __init__(self, k1=10):
        super(DiscrimESLoss, self).__init__()
        self.k1 = k1  # base:e

    def forward(self, logits, targets, data_parameter_minibatch, epoch):
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')
        newloss = torch.sub(loss, self.k1)  # .sub_()
        newloss.mul_(es)
        # Compute losses scaled by data parameters
        newloss = newloss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return newloss


class DiscrimEALoss(nn.Module):
    def __init__(self, k1=10):
        super(DiscrimEALoss, self).__init__()
        self.k1 = k1

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.k1)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEAEXPLoss(nn.Module):
    '''
    TODO: UNTEST
    L=exp(ES(s)(AVG(l^i)-gamma*k)*sigma),keep the domain of L in [0, +infinite], other DiscrimxxLoss are all in [-infinite, +inifinite]
    gamma is implemented by tanh, see statistic_graph.py for details
    '''

    def __init__(self, k1=10):
        super(DiscrimEAEXPLoss, self).__init__()
        self.k1 = k1  # base:e
        self.a = 0.2  # nn.Parameter(torch.FloatTensor(0.2))#1+0.2+0.2
        self.p = 1.5  # nn.Parameter(torch.FloatTensor(1.5))
        self.q = -50  # nn.Parameter(torch.FloatTensor(50))

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        self.gamma = torch.mul(self.a, nn.Tanh(self.p * epoch + self.q)) + self.a + 1.
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.gamma * self.k1)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = torch.exp(new_loss * data_parameter_minibatch)
        # loss = loss.sum()/self.batch_size
        return new_loss


# class DiscrimEA_TANHLoss(nn.Module):
#     '''
#     L=(ES(s)(AVG(l^i)-gamma*k)/sigma)+0.5*wd(ln(sigma))^2
#     gamma is implemented by tanh, see statistic_graph.py for details
#     '''
#
#     def __init__(self, k1=10):
#
#         super(DiscrimEA_TANHLoss, self).__init__()
#         self.k1 = torch.tensor(math.log(k1))  # base:e
#         self.a = torch.tensor([0.2])#nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
#         self.p = torch.tensor([1.5])#nn.parameter.Parameter(torch.FloatTensor([1.5]))
#         self.q = torch.tensor([-50])#nn.parameter.Parameter(torch.FloatTensor([-50]))
#         self.tanh = nn.Tanh()
#     def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
#         self.gamma = torch.mul(self.a, self.tanh(self.p * epoch + self.q)) + self.a + 1.
#         es = ES_linear(epoch)
#         loss = F.cross_entropy(logits, targets, reduction='none')
#
#         new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
#         exp_avg[index_dataset] = new_loss.detach()
#         # loss.data = exp_avg[index_dataset]
#         # bias correction
#         bias_cor = 1.0 - beta ** (epoch + 1)
#         new_loss.div_(bias_cor)
#         new_loss.sub_(self.gamma * self.k1)
#         new_loss.mul_(es)
#         # Compute losses scaled by data parameters
#         new_loss = new_loss / data_parameter_minibatch
#         return new_loss

class DiscrimEA_TANHLoss(nn.Module):
    '''
    L=(ES(s)(AVG(l^i)-gamma*k)/sigma)+0.5*wd(ln(sigma))^2
    gamma is implemented by tanh, see statistic_graph.py for details
    '''

    def __init__(self, k1=10, a=0.2, p=1.5, q=-50, sup_eps=3):

        super(DiscrimEA_TANHLoss, self).__init__()
        self.k1 = torch.tensor(k1)  # base:e
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        self.gamma = torch.mul(self.a, self.tanh(self.p * epoch + self.q)) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        # operation on loss
        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1.type_as(new_loss))
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        return new_loss


class DiscrimEA_TANHLoss_newQ(nn.Module):
    '''
    L=(ES(s)(AVG(l^i)-gamma*k)/sigma)+0.5*wd(ln(sigma))^2
    gamma is implemented by tanh, see statistic_graph.py for details
    '''

    def __init__(self, k1=10, a=0.2, p=1.5, q=-50, sup_eps=3):

        super(DiscrimEA_TANHLoss_newQ, self).__init__()
        self.k1 = torch.tensor(k1)  # base:e
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        # operation on loss
        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1.type_as(new_loss))
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        return new_loss


# model = DiscrimEA_TANHLoss()
# for parameter in model.parameters():
#     print('tanh:',parameter)
#
# model = DiscrimEALoss()
# for parameter in model.parameters():
#     print('ea:',parameter)

class DiscrimEA_0Loss(nn.Module):
    '''
    bias correction from l-k to l, keep minimum of loss to zero instead of a negative number
    to guarantee the regular term can work to bound the sigma variance
    '''

    def __init__(self, k1=10):
        super(DiscrimEA_0Loss, self).__init__()
        self.k1 = k1

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.k1)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        bc0 = self.k1 / data_parameter_minibatch.detach()
        new_loss += bc0
        new_loss.mul_(es)
        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEA_2Loss(nn.Module):
    '''
    discrim loss with 2 phases: 1st discrim_ea, 2nd threshold moving(k->gamma*k)
    '''

    def __init__(self, k1=10):
        '''
        step: record current step in iteration mode
        '''
        super(DiscrimEA_2Loss, self).__init__()
        self.k1 = k1
        self.step = 0
        self.has_visited = []

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch, switch, type='switch'):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        switch: bool, whether to switch into the 2nd phase
        type: string, switch type: switch/iteration
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        if switch:
            if type == 'switch':
                new_loss.sub_(self.k1 * gamma)
            elif self.step <= TOTAL_STEP:
                if epoch not in self.has_visited:
                    self.step += 1
                    self.has_visited.append(epoch)
                new_loss.sub_(self.k1 + self.step * (gamma - 1) / TOTAL_STEP)

            else:
                new_loss.sub_(self.k1 * gamma)
        else:
            new_loss.sub_(self.k1)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEA_GAKLoss(nn.Module):
    '''
    K：global average loss/unit: one batch
    '''

    def __init__(self):
        super(DiscrimEA_GAKLoss, self).__init__()
        self.k1 = AverageMeter('GAK', ':.4e')

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')
        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        self.k1.update(new_loss.detach().mean().item(), n=targets.size(0))
        new_loss.sub_(self.k1.avg)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size

        return new_loss


class DiscrimEA_GAK_TANHLoss(nn.Module):
    '''
    K：global average loss/unit: one batch
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_GAK_TANHLoss, self).__init__()
        self.k1 = AverageMeter('GAK', ':.4e')
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        self.gamma = torch.mul(self.a, self.tanh(self.p * epoch + self.q)) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        self.k1.update(new_loss.detach().mean().item(), n=targets.size(0))
        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1.avg)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size

        return new_loss


class DiscrimEA_GAK_TANHLoss_newQ(nn.Module):
    '''
    K：global average loss/unit: one batch
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_GAK_TANHLoss_newQ, self).__init__()
        self.k1 = AverageMeter('GAK', ':.4e')
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        self.k1.update(new_loss.detach().mean().item(), n=targets.size(0))
        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1.avg)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size

        return new_loss


class DiscrimEA_EMAKLoss(nn.Module):
    '''
    K：exponential moving average/unit: one batch
    '''

    def __init__(self):
        super(DiscrimEA_EMAKLoss, self).__init__()
        self.first = True

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')
        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        if self.first:
            self.k1 = new_loss.mean().item()
            self.first = False
        else:
            self.k1 = rho * self.k1 + (1 - rho) * new_loss.mean().item()
        # bias correction
        # bias_cor_k = 1.0 - rho ** (epoch + 1)
        # self.k1 /= bias_cor_k

        new_loss.sub_(self.k1)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch

        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEA_EMAK_TANHLoss(nn.Module):
    '''
    K：exponential moving average/unit: one batch
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_EMAK_TANHLoss, self).__init__()
        self.first = True
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        self.gamma = torch.mul(self.a, self.tanh(self.p * epoch + self.q)) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            targets = targets.type_as(logits_)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        if self.first:
            self.k1 = new_loss.mean().item()
            self.first = False
        else:
            self.k1 = rho * self.k1 + (1 - rho) * new_loss.mean().item()
        # bias correction
        # bias_cor_k = 1.0 - rho ** (epoch + 1)
        # self.k1 /= bias_cor_k

        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1)
        new_loss.mul_(torch.tensor(es, dtype=new_loss.dtype))
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch

        # loss = loss.sum()/self.batch_size
        return new_loss

# DiscrimLoss
class DiscrimEA_EMAK_TANHLoss_newQ(nn.Module):
    '''
    K：exponential moving average/unit: one batch
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_EMAK_TANHLoss_newQ, self).__init__()
        self.first = True
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''

        self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            targets = targets.type_as(logits_)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        if self.first:
            self.k1 = new_loss.mean().item()
            self.first = False
        else:
            self.k1 = rho * self.k1 + (1 - rho) * new_loss.mean().item()
        # bias correction
        # bias_cor_k = 1.0 - rho ** (epoch + 1)
        # self.k1 /= bias_cor_k

        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1)
        new_loss.mul_(torch.tensor(es, dtype=new_loss.dtype))
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch

        # loss = loss.sum()/self.batch_size
        return new_loss

class DiscrimEA_EMAK_TANH_WO_ESLoss_newQ(nn.Module):
    '''
    determine the performance improvement by es module
    used for ablation study
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_EMAK_TANH_WO_ESLoss_newQ, self).__init__()
        self.first = True
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.

        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            targets = targets.type_as(logits_)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # loss.data = exp_avg[index_dataset]
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)

        if self.first:
            self.k1 = new_loss.mean().item()
            self.first = False
        else:
            self.k1 = rho * self.k1 + (1 - rho) * new_loss.mean().item()
        # bias correction
        # bias_cor_k = 1.0 - rho ** (epoch + 1)
        # self.k1 /= bias_cor_k

        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch

        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEA_EMAK_TANHWO_EALoss_newQ(nn.Module):
    '''
    determine the performance improvement by loss history module
    used for ablation study
    '''

    def __init__(self, a=0.2, p=1.5, q=-50, sup_eps=3):
        super(DiscrimEA_EMAK_TANHWO_EALoss_newQ, self).__init__()
        self.first = True
        self.a = torch.tensor([a])  # nn.parameter.Parameter(torch.FloatTensor([0.2]))  # 1+0.2+0.2
        self.p = torch.tensor([p])  # nn.parameter.Parameter(torch.FloatTensor([1.5]))
        self.q = torch.tensor([q])  # nn.parameter.Parameter(torch.FloatTensor([-50]))
        self.sup_eps = sup_eps
        self.tanh = nn.Tanh()

    def forward(self, args, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        self.gamma = torch.mul(self.a, self.tanh(self.p * (epoch - self.q))) + self.a + 1.
        es = ES_linear(epoch, self.sup_eps)
        # loss definition
        if args.task_type == 'classification':
            loss = F.cross_entropy(logits, targets, reduction='none')
        elif args.task_type == 'regression':
            logits_ = logits.squeeze(dim=-1)
            targets = targets.type_as(logits_)
            if args.reg_loss_type == 'L1':
                loss = F.smooth_l1_loss(logits_, targets, reduction='none')
            elif args.reg_loss_type == 'L2':
                loss = F.mse_loss(logits_, targets, reduction='none')

        new_loss = loss

        if self.first:
            self.k1 = new_loss.mean().item()
            self.first = False
        else:
            self.k1 = rho * self.k1 + (1 - rho) * new_loss.mean().item()
        # bias correction
        # bias_cor_k = 1.0 - rho ** (epoch + 1)
        # self.k1 /= bias_cor_k

        new_loss.sub_(self.gamma.type_as(new_loss) * self.k1)
        new_loss.mul_(torch.tensor(es, dtype=new_loss.dtype))
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch

        # loss = loss.sum()/self.batch_size
        return new_loss

class DiscrimEA_WO_BCLoss(nn.Module):
    '''
    exponential moving average without bias correction:L1=l1 instead L1=(1-beta)l1
    '''

    def __init__(self, k1=10):
        super(DiscrimEA_WO_BCLoss, self).__init__()
        self.k1 = k1

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        es = ES_linear(epoch)
        loss = F.cross_entropy(logits, targets, reduction='none')

        if epoch == 0:
            new_loss = loss
        else:
            new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()

        new_loss.sub_(self.k1)
        new_loss.mul_(es)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return new_loss


class DiscrimEA_WO_ESLoss(nn.Module):
    '''
    determine the performance improvement by es module
    '''

    def __init__(self, k1=10):
        super(DiscrimEA_WO_ESLoss, self).__init__()
        self.k1 = k1

    def forward(self, logits, targets, data_parameter_minibatch, exp_avg, index_dataset, epoch):
        '''
        exp_avg: exponential moving average for each sample
        index_dataset: batch of samples' indexes
        '''
        loss = F.cross_entropy(logits, targets, reduction='none')
        new_loss = torch.add(torch.mul(exp_avg[index_dataset], beta), loss, alpha=1.0 - beta)
        exp_avg[index_dataset] = new_loss.detach()
        # bias correction
        bias_cor = 1.0 - beta ** (epoch + 1)
        new_loss.div_(bias_cor)
        new_loss.sub_(self.k1)
        # Compute losses scaled by data parameters
        new_loss = new_loss / data_parameter_minibatch
        # loss = loss.sum()/self.batch_size
        return new_loss
