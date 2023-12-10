import glob
import re
import time
import os
import h5py
import copy
import sys

import math
import pandas as pd
import wandb
import numpy as np
import torch
import pickle
import pdb
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# FL library
from datasets.dataset_generic import save_splits
from utils.utils import EarlyStopping, print_network, get_optim, get_cosine_schedule_with_warmup, get_split_loader, \
    get_simple_loader, bernouli_mask
from utils.fl_utils import solve_centered_w, solve_capped_w, sync_models, federated_averging

# Pointnet library
from models import provider
from models import get_model
from .point_loss import FocalLoss, SupConLoss, FocalSupConLoss, UnsupInfoLoss, MMD_Loss


# import syft as sy
def math_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def train_point(datasets, cur, args, ind_dataset=None):
    """
        train for a single fold
    """
    # number of institutions
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    print('\nInit loss function...', end=' ')
    if args.loss == "focal":
        # alpha is a weight factor for classes, [alpha, 1-alpha] for [0, 1]
        loss_fn = FocalLoss(gamma=2, alpha=None, size_average=True)
    elif args.loss == "focal_con":
        loss_fn = FocalSupConLoss(temperature=0.5, scale_by_temperature=True, gamma=2, alpha=None, size_average=True)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print('Done!')

    print('\nInit Model...', end=' ')

    model_dict = {'model_type': args.model_type,
                  'dropout': args.drop_out,
                  'num_class': args.n_classes,
                  'batch_size': args.batch_size,
                  'feature_dim': 3 + args.feat_dim,
                  'base_dim': args.feat_dim,
                  'norm': args.norm,
                  'aux': args.aux, 'ihc': args.ihc, 'mutual_info': args.mutual_info, 'fast_sim': args.fast_sim
                  }

    model = get_model(**model_dict)
    model.relocate()
    print_network(model)
    print('Done!')

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.max_epochs, num_warmup_steps=10, last_epoch=-1)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                    weighted=False, task_type=args.task, batch_size=args.batch_size,
                                    model_type=args.model_type)
    val_loader = get_split_loader(val_split, task_type=args.task, testing=args.testing, batch_size=args.batch_size,
                                  model_type=args.model_type)
    test_loader = get_simple_loader(test_split, batch_size=args.batch_size, num_workers=args.num_worker,
                                    model_type=args.model_type)
    if ind_dataset is not None:
        ind_dataloader = {}
        for k, v in ind_dataset.items():
            ind_dataloader[k] = get_simple_loader(v, batch_size=args.batch_size, num_workers=args.num_worker,
                                                  model_type=args.model_type)
    val_site_labels = val_split.site_labels
    test_site_labels = test_split.site_labels
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=100, verbose=True)
    else:
        early_stopping = None
    print('Done!')
    print('\n')

    if not args.eval:
        for epoch in range(args.max_epochs):
            train_loop_point(epoch, model, train_loader, optimizer,
                             args.n_classes, loss_fn, worker_schedules=scheduler, model_type=args.model_type)
            stop, val_info = validate_point(cur, epoch, model, val_loader, args.n_classes,
                                            early_stopping, loss_fn, args.results_dir, site_labels=val_site_labels)

            wandb_info = {}
            for k, v in val_info.items():
                wandb_info[f'val/{k}'] = v
            if wandb.run:
                wandb.log(wandb_info, commit=True)
            print('\n')

            if stop:
                break

    if args.early_stopping:
        model.load_state_dict(
            torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)), map_location='cpu'))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_auc = summary_point(model, val_loader, args.n_classes, site_labels=val_site_labels)
    print('Val auc: {:.4f}'.format(val_auc))

    results_dict, test_auc = summary_point(model, test_loader, args.n_classes, site_labels=test_site_labels, return_res=True)
    print('Test auc: {:.4f}'.format(test_auc))
    np.save(f'{args.results_dir}/s_{cur}_site.npy', test_site_labels)
    for k in ['her2', 'prob', 'ihc']:
        res_val = results_dict.pop(k)
        np.save(f'{args.results_dir}/s_{cur}_{k}.npy', res_val)

    if ind_dataset is not None:
        for ind_name, ind_loader in ind_dataloader.items():
            ind_dict, ind_auc = summary_point(model, ind_loader, args.n_classes, site_labels=None)
            for k, v in ind_dict.items():
                results_dict[f'{k}_{ind_name}'] = v
    wandb_info = {}
    for k, v in results_dict.items():
        wandb_info[f'test/{k}'] = v
    if wandb.run:
        wandb.log(wandb_info, commit=True)

    return results_dict, test_auc, val_auc


def train_fl_point(datasets, cur, args, ind_dataset=None, multi_site_class_dict=None):
    """
        train for a single fold
    """
    # number of institutions
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    print('\nInit loss function...', end=' ')
    if args.loss == "focal":
        # alpha is a weight factor for classes, [alpha, 1-alpha] for [0, 1]
        loss_fn = FocalLoss(gamma=2, alpha=None, size_average=True)
    elif args.loss == "focal_con":
        loss_fn = FocalSupConLoss(temperature=0.5, scale_by_temperature=True, gamma=2, alpha=None, size_average=True)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print('Done!')

    print('\nInit train/val/test splits...', end=' ')
    train_splits, val_split, test_split = datasets
    num_insti = len(train_splits)
    print('Done!')
    for idx in range(num_insti):
        print("Worker_{} Training on {} samples".format(idx, len(train_splits[idx])))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit Model...', end=' ')
    model_dict = {'model_type': args.model_type,
                  "dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  'batch_size': args.batch_size,
                  'feature_dim': 3 + args.feat_dim,
                  'base_dim': args.feat_dim,
                  'norm': args.norm,
                  'aux': args.aux, 'ihc': args.ihc, 'mutual_info': args.mutual_info, 'fast_sim': args.fast_sim}

    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})

    model = get_model(**model_dict)
    worker_models = []
    for worker_id in range(num_insti):
        site_model = get_model(**model_dict)
        site_model.institute = args.institutes[worker_id]
        worker_models.append(site_model)

    param_keys = [k for k in model.state_dict().keys()]
    ignore_batch_buffer_keys = [k for k in param_keys if
                                len(re.findall(r'running_mean|running_var|num_batches_tracked', k)) != 0]
    ignore_batch_all_keys = [k for k in param_keys if len(re.findall(r'bn', k)) != 0]
    site_spec_keys = [k for k in param_keys if len(re.findall(r'aux_cls', k)) != 0]
    print('Done!')

    sync_models(model, worker_models)
    device_counts = torch.cuda.device_count()
    if device_counts > 1:
        device_ids = [idx % device_counts for idx in range(num_insti)]
    else:
        device_ids = [0] * num_insti

    model.relocate(device_id=0)
    for idx in range(num_insti):
        worker_models[idx].relocate(device_id=device_ids[idx])

    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    worker_optims = [get_optim(worker_models[i], args) for i in range(num_insti)]
    worker_schedules = [get_cosine_schedule_with_warmup(worker_optims[i],
                                                        args.max_epochs,
                                                        num_warmup_steps=10,
                                                        last_epoch=-1) for i in range(num_insti)]
    print('Done!\n')

    print('Init Loaders...', end=' ')
    train_loaders = []
    for idx in range(num_insti):
        train_loaders.append(get_split_loader(train_splits[idx], training=True, testing=args.testing,
                                              weighted=False, batch_size=args.batch_size, task_type=args.task,
                                              num_workers=args.num_worker, model_type=args.model_type))
    val_loader = get_split_loader(val_split, batch_size=args.batch_size, task_type=args.task, testing=args.testing,
                                  num_workers=args.num_worker, model_type=args.model_type)
    test_loader = get_simple_loader(test_split, batch_size=args.batch_size, num_workers=args.num_worker,
                                    model_type=args.model_type)

    if ind_dataset is not None:
        ind_dataloader = {}
        for k, v in ind_dataset.items():
            ind_dataloader[k] = get_simple_loader(v, batch_size=args.batch_size, num_workers=args.num_worker,
                                                  model_type=args.model_type)
    val_site_labels = val_split.site_labels
    test_site_labels = test_split.site_labels

    print('Done!\n')

    print('Setup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=35, verbose=True)
    else:
        early_stopping = None
    print('Done!\n')

    if not args.eval:
        for epoch in range(args.max_epochs):
            train_info = train_loop_fl_point(epoch, model, worker_models, train_loaders, worker_optims, args.n_classes,
                                             loss_fn, model_type=args.model_type,
                                             worker_schedules=worker_schedules, args=args,
                                             ignore_keys=ignore_batch_buffer_keys,
                                             multi_site_class_dict=multi_site_class_dict)
            weights = train_info['site_w']

            if (epoch + 1) % args.E == 0:
                if args.fl_avg == 'FedBN':
                    model, worker_models = federated_averging(model, worker_models, args.noise_level, weights,
                                                              ignore_keys=site_spec_keys)
                    sync_models(model, worker_models, ignore_keys=ignore_batch_all_keys + site_spec_keys)
                else:
                    model, worker_models = federated_averging(model, worker_models, args.noise_level, weights,
                                                              ignore_keys=site_spec_keys)
                    sync_models(model, worker_models, ignore_keys=site_spec_keys)
            stop, val_info = validate_point(cur, epoch, model, val_loader, args.n_classes,
                                            early_stopping, loss_fn, args.results_dir, worker_models=worker_models,
                                            site_labels=val_site_labels)
            wandb_info = {}
            for k, v in val_info.items():
                wandb_info[f'val/{k}'] = v
            if wandb.run:
                wandb.log(wandb_info, commit=True)
            print('\n')

            if stop:
                break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"), map_location='cpu'))
        for i, site in enumerate(args.institutes):
            worker_models[i].load_state_dict(
                torch.load(os.path.join(args.results_dir, f"s_{cur}_checkpoint_{site}.pt"), map_location='cpu'))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_auc = summary_point(model, val_loader, args.n_classes, site_labels=val_site_labels)
    print('Val auc: {:.4f}'.format(val_auc))

    results_dict, test_auc = summary_point(model, test_loader, args.n_classes, site_labels=test_site_labels, return_res=True)
    np.save(f'{args.results_dir}/s_{cur}_site.npy', test_site_labels)

    if args.eval:
        attr_split = copy.deepcopy(test_split)
        attr_split.status = 'attr'
        attr_loader = get_simple_loader(attr_split, batch_size=2, num_workers=args.num_worker,
                                        model_type=args.model_type)
        attr_score, coord_li = point_attr(model, attr_loader)
        np.save(f'{args.results_dir}/s_{cur}_attr_part.npy', attr_score)
        np.save(f'{args.results_dir}/s_{cur}_coord_part.npy', coord_li)
        np.save(f'{args.results_dir}/s_{cur}_sub_id.npy', attr_split.sub_id_li)

    for k in ['her2', 'prob', 'ihc']:
        res_val = results_dict.pop(k)
        np.save(f'{args.results_dir}/s_{cur}_{k}.npy', res_val)
    if args.aux:
        aux_dict, _ = summary_point_site(worker_models, test_loader, args.n_classes, site_labels=test_site_labels,
                                         institutes=args.institutes)
        results_dict = {**aux_dict, **results_dict}

    print('Test auc: {:.4f}'.format(test_auc))
    if ind_dataset is not None:
        for ind_name, ind_loader in ind_dataloader.items():
            ind_dict, ind_auc = summary_point(model, ind_loader, args.n_classes, site_labels=None)
            for k, v in ind_dict.items():
                results_dict[f'{k}_{ind_name}'] = v

    wandb_info = {}
    for k, v in results_dict.items():
        wandb_info[f'test/{k}'] = v
    if wandb.run:
        wandb.log(wandb_info, commit=True)

    return results_dict, test_auc, val_auc


def train_loop_fl_point(epoch, model, worker_models, worker_loaders, worker_optims, n_classes, loss_fn=None,
                        worker_schedules=None, args=None, ignore_keys=[], multi_site_class_dict=None,
                        model_type='point_transformer'):
    num_insti = len(worker_models)
    model.train()
    worker_w_diff = []
    if args.aux > 0:
        aux_fn = loss_fn
    if args.ihc > 0:
        ihc_fn = loss_fn
    if args.prox_weight:
        fedprox_fn = nn.MSELoss(reduction='sum')
    if args.mutual_info > 0:
        info_fn = UnsupInfoLoss()
    if args.mmd > 0:
        mmd_fn = MMD_Loss()

    train_loss = 0.
    site_num = np.array([len(worker_loaders[i]) for i in range(num_insti)])
    total = np.sum(site_num)
    info = {}
    for idx in range(len(worker_loaders)):
        # pdb.set_trace()
        if worker_models[idx].device is not None:
            model_device = worker_models[idx].device
        else:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        last_weights = copy.deepcopy(worker_models[idx].state_dict())
        total_loss = 0
        num_batches = 0
        total_correct = 0
        count = 0
        info[f'loss_{args.institutes[idx]}'] = 0
        if args.aux:
            info[f'loss_aux_{args.institutes[idx]}'] = 0
        if args.ihc > 0:
            info[f'loss_ihc_{args.institutes[idx]}'] = 0
        if args.mutual_info:
            info[f'loss_info_{args.institutes[idx]}'] = 0
        if args.prox_weight:
            info[f'loss_prox_{args.institutes[idx]}'] = 0
        if args.mmd > 0:
            info[f'loss_mmd_{args.institutes[idx]}'] = 0

        start_time = time.time()

        with tqdm(worker_loaders[idx], ascii=True) as tq:
            target_li, pred_li = [], []
            for batch_idx, (data, label, ihc_lb) in enumerate(tq):
                if 'point' in args.model_type:
                    data = data.data.numpy()
                    data = provider.random_point_dropout(data)
                    data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
                    data[:, :, 0:3] = provider.jitter_point_cloud(data[:, :, 0:3])
                    # data[:, :, 0:3] = provider.rotate_point_cloud_z(data[:, :, 0:3])
                    data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
                    data = torch.tensor(data)

                data, label, ihc_lb = data.to(model_device), label.to(model_device).long(), ihc_lb.to(
                    model_device).long()
                num_examples = label.shape[0]

                if 'clam' in model_type.lower():
                    logits, feat, inst_loss = worker_models[idx](data, label=label, instance_eval=True)
                else:
                    logits, feat, h_feat = worker_models[idx](data)

                if 'supcon' in loss_fn._get_name().lower():
                    loss = loss_fn(logits, feat, label)
                elif args.aux > 0:
                    lb_class_dist = multi_site_class_dict[args.institutes[idx]].to(label.device)
                    iter_ratio = epoch / args.max_epochs
                    if args.dist_code == 'linear':
                        lb_class_dist = lb_class_dist + (epoch / args.max_epochs) * (1 - lb_class_dist)
                    elif args.dist_code == 'exp':
                        lb_class_dist = lb_class_dist + ((math.exp(iter_ratio) - 1) / (math.e - 1)) * (
                                    1 - lb_class_dist)
                    elif args.dist_code == 'sigmoid':
                        lb_class_dist = lb_class_dist + (math_sigmoid(iter_ratio - 0.5)) * (1 - lb_class_dist)
                    elif args.dist_code == 'sin' or args.dist_code == 'sine':
                        lb_class_dist = lb_class_dist + (math.sin(iter_ratio * math.pi / 2)) * (1 - lb_class_dist)

                    if args.dist_code == 'simple':
                        loss = loss_fn(logits, label)
                    else:
                        mask_lb = bernouli_mask(lb_class_dist[label])
                        loss = loss_fn(logits[mask_lb == 1], label[mask_lb == 1])

                    aux_logits = worker_models[idx].aux_cls(feat)
                    aux_loss = aux_fn(aux_logits, label)
                    info[f'loss_aux_{args.institutes[idx]}'] += aux_loss.item()
                    loss = loss + args.aux * aux_loss
                    if args.mmd > 0:
                        mmd_loss = mmd_fn(aux_logits, logits.detach())
                        loss = loss + args.mmd * mmd_loss
                else:
                    loss = loss_fn(logits, label)

                if 'clam' in args.model_type.lower():
                    loss = loss + inst_loss
                elif 'dsmil' in model_type.lower():
                    loss = 0.5 * loss + 0.5 * loss_fn(feat, label)

                if args.ihc > 0:
                    if torch.sum(ihc_lb != -1) > 0:
                        ihc_logits = worker_models[idx].ihc_cls(feat)
                        ihc_loss = ihc_fn(ihc_logits[ihc_lb != -1], ihc_lb[ihc_lb != -1])
                        info[f'loss_ihc_{args.institutes[idx]}'] += ihc_loss.item()
                        loss = loss + args.ihc * ihc_loss

                if args.mutual_info > 0:
                    info_loss = info_fn(h_feat, feat)
                    info[f'loss_info_{args.institutes[idx]}'] += info_loss.item()
                    loss = loss + args.mutual_info * info_loss

                # fedprox
                if args.prox_weight:
                    cur_param = list(worker_models[idx].named_parameters())
                    net_reg = 0.0
                    for cur_named_w in cur_param:
                        cur_k = cur_named_w[0]
                        cur_w = cur_named_w[1]
                        if cur_k in ignore_keys:
                            continue
                        net_reg = net_reg + fedprox_fn(cur_w, last_weights[cur_k].to(model_device))

                    info[f'loss_prox_{args.institutes[idx]}'] += net_reg.item()
                    loss = loss + args.prox_weight * net_reg

                loss_value = loss.item()
                info[f'loss_{args.institutes[idx]}'] += loss_value
                train_loss += loss_value

                _, preds = logits.max(1)
                logits_porb = logits.softmax(dim=1)
                target_li.append(label.cpu().detach().numpy())
                pred_li.append(logits_porb[:, 1].cpu().detach().numpy())

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss.item()
                total_correct += correct

                tq.set_postfix(
                    {
                        "AvgLoss": "%.5f" % (total_loss / num_batches),
                        "AvgAcc": "%.5f" % (total_correct / count),
                    }
                )

                # backward pass
                loss.backward()
                # step
                if args.opt == 'sam':
                    worker_optims[idx].first_step(zero_grad=True)
                    # second forward-backward pass
                    logits, feat, h_feat = worker_models[idx](data)
                    if 'supcon' in loss_fn._get_name().lower():
                        sam_loss = loss_fn(logits, feat, label)
                    elif args.aux:
                        sam_loss = loss_fn(logits[mask_lb == 1], label[mask_lb == 1]) + args.aux * aux_fn(
                            worker_models[idx].aux_cls(feat), label)
                    else:
                        sam_loss = loss_fn(logits, label)  # make sure to do a full forward pass
                    if args.ihc:
                        if torch.sum(ihc_lb != -1) > 0:
                            ihc_logits = worker_models[idx].ihc_cls(feat)
                            sam_loss = sam_loss + args.ihc * ihc_fn(ihc_logits[ihc_lb != -1], ihc_lb[ihc_lb != -1])
                    if args.mutual_info:
                        sam_loss = sam_loss + args.mutual_info * info_fn(h_feat, feat)
                    sam_loss.backward()

                    worker_optims[idx].second_step(zero_grad=True)
                    worker_optims[idx].zero_grad()
                else:
                    worker_optims[idx].step()
                    worker_optims[idx].zero_grad()

        for k, v in info.items():
            if args.institutes[idx] in k:
                info[k] = v / num_batches

        if worker_schedules is not None:
            cur_lr = worker_schedules[idx].get_last_lr()[0]
            worker_schedules[idx].step()

        if args.fl_avg in ['FedMGDA', 'FedMGDAProx']:
            difference = copy.deepcopy(last_weights)

            with torch.no_grad():
                for key in difference.keys():
                    difference[key] = worker_models[idx].state_dict()[key].detach().cpu() - last_weights[
                        key].detach().cpu() + 1e-6
                if args.grad_norm == 1:
                    total_grad = 0.0
                    for key in difference.keys():
                        total_grad += torch.norm(difference[key]) ** 2
                    total_grad = np.sqrt(total.item())
                    for key in difference.keys():
                        difference[key] /= total_grad

            worker_w_diff.append(difference)

    if args.fl_avg == 'FedAvg':
        site_w = site_num / total
    elif args.fl_avg == 'FedProx':
        site_w = site_num / total
    elif args.fl_avg == 'SiteAvg':
        site_w = [1 / num_insti for _ in range(num_insti)]
    elif args.fl_avg in ['FedMGDA', 'FedMGDAProx']:
        if args.cap == 1:
            if args.epsilon <= 1 and args.epsilon != 0:
                lower_b = 1. / num_insti - args.epsilon
                upper_b = 1. / num_insti + args.epsilon
                site_w = solve_centered_w(worker_w_diff, lower_b=lower_b, upper_b=upper_b, ignore_keys=ignore_keys)
        elif 1 > args.cap >= 1. / num_insti:
            site_w = solve_capped_w(worker_w_diff, C=args.cap, ignore_keys=ignore_keys)
    else:
        site_w = site_num / total

    # calculate loss and error for epoch
    train_loss = train_loss / total
    target_li = np.concatenate(target_li)
    pred_li = np.concatenate(pred_li)
    mean_auc = roc_auc_score(target_li, pred_li)
    print(
        "[Train] Epoch: {}, AvgLoss: {:.5}, AvgAcc: {:.5}, AvgAUC: {:.5}, train_loss: {:.5}, Time: {:.5}s".format(
            epoch,
            total_loss / num_batches,
            total_correct / count,
            mean_auc,
            train_loss,
            time.time() - start_time,
        )
    )

    info.update({
        'loss': round(train_loss, 4),
        'acc': round(total_correct / count, 4),
        'auc': round(mean_auc, 4),
        'lr': cur_lr,
        'site_w': site_w,
    })

    wandb_info = {}
    for k, v in info.items():
        if k == 'site_w':
            for site_id in range(len(v)):
                wandb_info[f'site_w_{args.institutes[site_id]}'] = v[site_id]
        else:
            wandb_info['train/' + k] = v

    if wandb.run:
        wandb.log(wandb_info)

    return info


def train_loop_point(epoch, model, loader, optimizer, n_classes, loss_fn=None, worker_schedules=None,
                     model_type='point_transformer'):
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    train_error = 0.

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    start_time = time.time()

    with tqdm(loader, ascii=True) as tq:
        target_li, pred_li = [], []
        for batch_idx, (data, label, ihc_lb) in enumerate(tq):
            data = data.data.numpy()
            data = provider.random_point_dropout(data)
            data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
            data[:, :, 0:3] = provider.jitter_point_cloud(data[:, :, 0:3])
            # data[:, :, 0:3] = provider.rotate_point_cloud_z(data[:, :, 0:3])
            data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
            data = torch.tensor(data)

            data, label, ihc_lb = data.to(model_device), label.to(model_device).long(), ihc_lb.to(model_device).long()
            num_examples = label.shape[0]

            if 'clam' in model_type.lower():
                logits, feat, inst_loss = model(data, label=label, instance_eval=True)
            else:
                logits, feat, h_feat = model(data)

            if 'supcon' in loss_fn._get_name().lower():
                loss = loss_fn(logits, feat, label)
            else:
                loss = loss_fn(logits, label)

            if 'clam' in model_type.lower():
                loss = loss + inst_loss
            elif 'dsmil' in model_type.lower():
                loss = 0.5 * loss + 0.5 * loss_fn(feat, label)

            loss_value = loss.item()
            train_loss += loss_value

            _, preds = logits.max(1)
            logits_porb = logits.softmax(dim=1)
            target_li.append(label.cpu().detach().numpy())
            pred_li.append(logits_porb[:, 1].cpu().detach().numpy())

            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss.item()
            total_correct += correct

            tq.set_postfix(
                {
                    "AvgLoss": "%.5f" % (total_loss / num_batches),
                    "AvgAcc": "%.5f" % (total_correct / count),
                }
            )

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()

    if worker_schedules is not None:
        cur_lr = worker_schedules.get_last_lr()[0]
        worker_schedules.step()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    # print('model updated: ', torch.abs(model_params - model.classifier.weight).sum())
    target_li = np.concatenate(target_li)
    pred_li = np.concatenate(pred_li)
    mean_auc = roc_auc_score(target_li, pred_li)
    print(
        "[Train] Epoch: {}, AvgLoss: {:.5}, AvgAcc: {:.5}, AvgAUC: {:.5}, train_loss: {:.5}, Time: {:.5}s".format(
            epoch,
            total_loss / num_batches,
            total_correct / count,
            mean_auc,
            train_loss,
            time.time() - start_time,
        )
    )
    info = {
        'loss': round(total_loss / num_batches, 4),
        'acc': round(total_correct / count, 4),
        'auc': round(mean_auc, 4),
        'lr': cur_lr,
    }
    wandb_info = {}
    for k, v in info.items():
        wandb_info['train/' + k] = v
    if wandb.run:
        wandb.log(wandb_info)


def validate_point(cur, epoch, model, loader, n_classes, early_stopping=None, loss_fn=None,
                   results_dir=None, site_labels=None, worker_models=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    total_correct = 0
    total_loss = 0
    count = 0
    num_batches = 0
    start_time = time.time()
    with torch.no_grad():
        with tqdm(loader, ascii=True) as tq:
            target_li, pred_li = [], []
            for data, label, ihc_lb in tq:
                num_examples = label.shape[0]
                data, label, ihc_lb = data.to(device), label.to(device).long(), ihc_lb.to(device).long()

                logits, feat, h_feat = model(data)

                if 'supcon' in loss_fn._get_name().lower():
                    loss = loss_fn(logits, feat, label)
                else:
                    loss = loss_fn(logits, label)
                _, preds = logits.max(1)
                logits_porb = logits.softmax(dim=1)
                target_li.append(label.cpu().detach().numpy())
                pred_li.append(logits_porb[:, 1].cpu().detach().numpy())

                correct = (preds == label).sum().item()
                total_loss += loss.item()
                total_correct += correct
                count += num_examples
                num_batches += 1

                tq.set_postfix({"AvgAcc": "%.5f" % (total_correct / count)})
            target_li = np.concatenate(target_li)
            pred_li = np.concatenate(pred_li)
            if np.sum(pred_li) > 0:
                mean_auc = roc_auc_score(target_li, pred_li)
            else:
                mean_auc = 0.0

    print(
        "[Test] AvgLoss: {:.5}, AvgAcc: {:.5}, AvgAUC: {:.5}, Time: {:.5}s".format(
            total_loss / num_batches, total_correct / count, mean_auc, time.time() - start_time
        )
    )

    info = {
        'loss': round(total_loss / num_batches, 4),
        'acc': round(total_correct / count, 4),
        'auc': round(mean_auc, 4)
    }

    if site_labels is not None:
        sites = np.unique(site_labels)
        for site in sites:
            if np.sum(pred_li[site_labels == site]) > 0:
                site_auc = roc_auc_score(target_li[site_labels == site], pred_li[site_labels == site])
            else:
                site_auc = 0.0
            info[f'auc_{site}'] = round(site_auc, 4)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, -mean_auc, model, worker_models=worker_models,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, info

    return False, info


def summary_point(model, loader, n_classes, site_labels=None, return_res=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    test_loss = 0.

    patient_results = {}

    total_correct = 0
    total_loss = 0
    count = 0
    num_batches = 0
    start_time = time.time()
    with torch.no_grad():
        with tqdm(loader, ascii=True) as tq:
            target_li, pred_li, ihc_li = [], [], []
            for data, label, ihc_lb in tq:
                num_examples = label.shape[0]
                data, label, ihc_lb = data.to(device), label.to(device).long(), ihc_lb.to(device).long()
                logits, feat, h_feat = model(data)

                _, preds = logits.max(1)
                logits_porb = logits.softmax(dim=1)
                target_li.append(label.cpu().detach().numpy())
                pred_li.append(logits_porb[:, 1].cpu().detach().numpy())
                ihc_li.append(ihc_lb.cpu().detach().numpy())

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples
                num_batches += 1

                tq.set_postfix({"AvgAcc": "%.5f" % (total_correct / count)})
            target_li = np.concatenate(target_li)
            pred_li = np.concatenate(pred_li)
            ihc_li = np.concatenate(ihc_li)
            mean_auc = roc_auc_score(target_li, pred_li)

    print(
        "[Test] AvgAcc: {:.5}, AvgAUC: {:.5}, Time: {:.5}s".format(
            total_correct / count, mean_auc, time.time() - start_time
        )
    )
    info = {
        'acc': round(total_correct / count, 4),
        'auc': round(mean_auc, 4)
    }
    if site_labels is not None:
        sites = np.unique(site_labels)
        for site in sites:
            site_auc = roc_auc_score(target_li[site_labels == site], pred_li[site_labels == site])
            info[f'auc_{site}'] = round(site_auc, 4)
    if return_res:
        info['her2'] = target_li
        info['prob'] = pred_li
        info['ihc'] = ihc_li
    return info, mean_auc


def summary_point_site(worker_models, loader, n_classes, institutes=None, site_labels=None, ):
    info = {}
    for idx in range(len(worker_models)):
        inst = institutes[idx]
        model = worker_models[idx]
        device = model.device
        model.eval()
        test_loss = 0.

        patient_results = {}

        total_correct = 0
        total_loss = 0
        count = 0
        num_batches = 0
        start_time = time.time()
        with torch.no_grad():
            with tqdm(loader, ascii=True) as tq:
                target_li, pred_li, ihc_li = [], [], []
                for data, label, ihc_lb in tq:
                    num_examples = label.shape[0]
                    data, label, ihc_lb = data.to(device), label.to(device).long(), ihc_lb.to(device).long()
                    _, feat, h_feat = model(data)
                    logits = model.aux_cls(feat)

                    _, preds = logits.max(1)
                    logits_porb = logits.softmax(dim=1)
                    target_li.append(label.cpu().detach().numpy())
                    pred_li.append(logits_porb[:, 1].cpu().detach().numpy())
                    ihc_li.append(ihc_lb.cpu().detach().numpy())

                    correct = (preds == label).sum().item()
                    total_correct += correct
                    count += num_examples
                    num_batches += 1

                    tq.set_postfix({"AvgAcc": "%.5f" % (total_correct / count)})
                target_li = np.concatenate(target_li)
                pred_li = np.concatenate(pred_li)
                ihc_li = np.concatenate(ihc_li)
                mean_auc = roc_auc_score(target_li, pred_li)

        print(
            "[Test] AvgAcc: {:.5}, AvgAUC: {:.5}, Time: {:.5}s".format(
                total_correct / count, mean_auc, time.time() - start_time
            )
        )
        info.update({
            f'acc_{inst}': round(total_correct / count, 4),
            f'auc_{inst}': round(mean_auc, 4)
        })
        if site_labels is not None:
            sites = np.unique(site_labels)
            for site in sites:
                site_auc = roc_auc_score(target_li[site_labels == site], pred_li[site_labels == site])
                info[f'auc_{inst}_{site}'] = round(site_auc, 4)

    return info, mean_auc


def h5_to_npy(datapath, feat_name='feat_nuhtc'):
    import glob, os, h5py
    import numpy as np
    from tqdm import tqdm
    file_path_li = glob.glob(f'{datapath}/{feat_name}/h5_files/*.h5')

    for file_path in tqdm(file_path_li):

        filename = os.path.basename(file_path)[:-3]
        basedir = os.path.dirname(file_path)
        basedir = os.path.dirname(basedir)

        with h5py.File(file_path, mode='r') as f:
            coords = f['coords'][()]
            feats = f['features'][()]

        coords_3d = np.zeros((coords.shape[0], 3))
        coords_3d[:, :2] = coords
        point_set = np.concatenate([coords_3d, feats], axis=1).astype(np.float32)

        os.makedirs(f'{basedir}/npy_files', exist_ok=True)

        if not os.path.exists(f'{basedir}/npy_files/{filename}.npy'):
            np.save(f'{basedir}/npy_files/{filename}.npy', point_set)


def point_attr(model, loader, **kwargs):
    from captum.attr import IntegratedGradients
    attr_li = []
    coord_li = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    model.forward = model.forward_attr

    IG_model = IntegratedGradients(model)

    with tqdm(loader, ascii=True) as tq:
        for data, coord in tq:
            data = data.to(device)
            # logits, feat, h_feat = model(data)
            attr_score = IG_model.attribute(data, target=1)
            attr_li.append(attr_score.cpu().numpy())
            coord_li.append(coord.cpu().numpy())

    attr_li = np.concatenate(attr_li)
    coord_li = np.concatenate(coord_li)

    model.forward = model.forward_cls

    return attr_li, coord_li

def vis_wsi(attr_li, coord_li, testset):
    sys.path.append('CLAM')
    import pandas as pd
    total_df = pd.read_csv('dataset_csv/BRCA_HER2_ALL.csv', index_col=0)
    from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap

    for i in range(len(attr_li)):
        df = total_df.loc[total_df['ID']==testset.sub_id_li[i]]
        filepath = df['FilePath_Pyramid']
        if pd.isna(filepath).values[0] != True:
            slide_path = f'{testset.data_dir}/{filepath}'
        wsi_object = initialize_wsi(slide_path, seg_mask_path=None, seg_params=None, filter_params=None)
        print('Done!')

        heatmap = drawHeatmap(attr_li[i].max(axis=-1), coord_li[i], slide_path, wsi_object=wsi_object, cmap='jet',
                              alpha=0.4, use_holes=True, binarize=False, vis_level=-1,
                              blank_canvas=False,
                              thresh=-1, patch_size=512, convert_to_percentiles=True)


    return attr_li, coord_li

