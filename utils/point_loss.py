import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            # at variable may be trainable
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class SupConLoss(nn.Module):
    # https://zhuanlan.zhihu.com/p/442415516
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # already normalize the feature
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row  = torch.sum(positives_mask , axis=1)
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class FocalSupConLoss(nn.Module):
    # https://zhuanlan.zhihu.com/p/442415516
    def __init__(self, temperature=0.5, scale_by_temperature=True, gamma=2, alpha=None, size_average=True):
        super(FocalSupConLoss, self).__init__()
        self.focal_fn = FocalLoss(gamma, alpha, size_average)
        self.con_fn = SupConLoss(temperature, scale_by_temperature)

    def forward(self, input, features, labels=None, mask=None):
        loss_focal = self.focal_fn(input, labels)
        loss_con = self.con_fn(features, labels=labels)
        loss = loss_focal + loss_con
        return loss

class MMD_Loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_Loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def linear_kernel(self, x, y):
        return torch.matmul(x, y.t())

    def rbf_kernel(self, x, y, sigma=1):
        norm = torch.sum(x**2, dim=1, keepdim=True) + \
               torch.sum(y**2, dim=1, keepdim=True).t() - \
               2.0 * torch.matmul(x, y.t())
        return torch.exp(-norm / (2.0 * sigma**2))

    def poly_kernel(self, x, y, sigma=1, degree=3):
        return (torch.matmul(x, y.t()) + sigma)**degree

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # L2 dist may decrease to zero
        L2_distance = ((total0-total1)**2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp+1e-6)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, kernel='guassian'):
        batch_size = int(source.size()[0])
        if kernel == 'guassian':
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        elif kernel == 'poly':
            kernels = self.poly_kernel(source, target, sigma=1)
        elif kernel == 'rbf':
            kernels = self.rbf_kernel(source, target, sigma=1)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class CMM_Loss(nn.Module):
    def __init__(self):
        super(CMM_Loss, self).__init__()

    def forward(self, source, target):
        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)

        # 计算中心矩匹配损失
        loss = torch.sum((source - source_mean)**2) + torch.sum((target - target_mean)**2)

        return loss

# https://github.com/syorami/DDC-transfer-learning/blob/9090f7fdd7c149c6158c5145b516abee97bb0944/mmd.py#L3
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.0)
    Ep = log_2 - F.softplus(-p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.0)
    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc):

    sup_point = l_enc.shape[1]
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0] * sup_point
    l_enc = l_enc.view(num_nodes, -1)

    device = g_enc.device

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)

    graph_id = torch.arange(num_graphs).repeat_interleave(sup_point).to(device)

    nodeidx = torch.arange(len(graph_id))
    pos_mask[nodeidx, graph_id] = 1.0
    neg_mask[nodeidx, graph_id] = 0.0

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

class UnsupInfoLoss(nn.Module):
    def __int__(self):
        super().__int__()
    def __call__(self, local_h, global_h, **kwargs):
        loss = local_global_loss_(local_h, global_h)
        return loss
