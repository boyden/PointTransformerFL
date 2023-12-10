import os.path

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from dgl.dataloading import GraphDataLoader
import torch.optim as optim

import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from .sam import SAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False, epsilon=0.01):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epsilon = epsilon
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, worker_models=[], ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            if abs(score - self.best_score) > self.epsilon:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, worker_models=worker_models)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name, worker_models=[]):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation metric decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        for worker_model in worker_models:
            basename = os.path.splitext(ckpt_name)[0]
            torch.save(worker_model.state_dict(), f"{basename}_{worker_model.institute}.pt")
        self.val_loss_min = val_loss


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    return [img, label, event_time, c]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=4, model_type='point_transformer'):
    if model_type.lower() in ['sage', 'slidegraph', 'gat', 'patch_gcn']:
        DataLoader_Type = GraphDataLoader
    else:
        DataLoader_Type = DataLoader
    kwargs = {'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader_Type(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                             collate_fn=None, **kwargs)
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, task_type='classification',
                     batch_size=8, num_workers=4, model_type='point_transformer'):
    """
		return either the validation loader or training loader 
	"""
    if model_type.lower() in ['sage', 'slidegraph', 'gat', 'patch_gcn']:
        DataLoader_Type = GraphDataLoader
    else:
        DataLoader_Type = DataLoader
    if task_type == 'classification':
        collate = collate_MIL
    elif task_type == 'survival':
        collate = collate_MIL_survival
    else:
        collate = None
    kwargs = {'num_workers': num_workers, 'batch_size': batch_size} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader_Type(split_dataset, sampler=WeightedRandomSampler(weights, len(weights)),
                                         drop_last=True,
                                         collate_fn=collate, **kwargs)
            else:
                loader = DataLoader_Type(split_dataset, sampler=RandomSampler(split_dataset),
                                         drop_last=True,
                                         collate_fn=collate, **kwargs)
        else:
            loader = DataLoader_Type(split_dataset, sampler=SequentialSampler(split_dataset),
                                     collate_fn=collate, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
        loader = DataLoader_Type(split_dataset, sampler=SubsetSequentialSampler(ids), collate_fn=collate,
                                 **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    elif args.opt == 'sam':
        optimizer = SAM(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=args.lr,
                        weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=149. / 300.,
                                    num_warmup_steps=0,
                                    min_lr=1e-7,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return max(_lr, min_lr)

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: {:,}'.format(num_params))
    print('Total number of trainable parameters: {:,}'.format(num_params_train))


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def generate_split_mutli_site(cls_ids, val_ratio, test_ratio, samples, n_splits=5,
                              seed=7, label_frac=1.0, custom_test_ids=None, **kwargs):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)
        for site in cls_ids.keys():
            for c in range(len(cls_ids[site])):
                val_num = round(len(cls_ids[site][c]) * val_ratio)
                possible_indices = np.intersect1d(cls_ids[site][c], indices)  # all indices of this class
                val_ids = np.random.choice(possible_indices, val_num, replace=False)  # validation ids

                remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
                all_val_ids.extend(val_ids)

                if custom_test_ids is None:  # sample test split
                    test_num = round(len(cls_ids[site][c]) * test_ratio)
                    test_ids = np.random.choice(remaining_ids, test_num, replace=False)
                    remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                    all_test_ids.extend(test_ids)

                if label_frac == 1:
                    sampled_train_ids.extend(remaining_ids)

                else:
                    sample_num = math.ceil(len(remaining_ids) * label_frac)
                    slice_ids = np.arange(sample_num)
                    sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [0] * len(dataset.slide_cls_ids)
    for c in range(len(dataset.slide_cls_ids)):
        if len(dataset.slide_cls_ids[c]) > 0:
            weight_per_class[c] = N / len(dataset.slide_cls_ids[c])
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def nll_loss(hazards, Y, c, S=None, alpha=0.15, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
        torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, Y, c, S=None, alpha=0.15, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


@torch.no_grad()
def bernouli_mask(x):
    return torch.bernoulli(x.detach()).float()


class CrossEntropySurvLoss(object):
    def __call__(self, hazards, Y, c, S=None, alpha=0.15):
        return ce_loss(hazards, Y, c, S, alpha)


class NLLSurvLoss(object):
    def __call__(self, hazards, Y, c, S=None, alpha=0.15):
        return nll_loss(hazards, Y, c, S, alpha)
