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