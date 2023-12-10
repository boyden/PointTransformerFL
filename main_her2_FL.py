import argparse
import pdb
import os
import math
import pandas as pd
import numpy as np
import wandb

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_point import train_fl_point, train_point
from datasets.dataset_point import Generic_Point_Dataset, Independent_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

from timeit import default_timer as timer


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    multi_site_ratio_dict = dataset.multi_site_ratio_dict
    multi_site_class_dict = {}
    for inst in inst_li:
        lb_class_dist = [0 for _ in range(args.n_classes)]
        for i in range(args.n_classes):
            lb_class_dist[i] = multi_site_ratio_dict[(i, inst)]
        multi_site_class_dict[inst] = torch.tensor(lb_class_dist)
        # compute imb ratio
        multi_site_class_dict[inst] = multi_site_class_dict[inst].min()/multi_site_class_dict[inst]

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    folds = np.arange(start, end)

    for i in folds:
        res_pkl_filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if os.path.exists(res_pkl_filename) and not args.no_skip:
            res_pkl = load_pkl(res_pkl_filename)
            all_test_auc.append(res_pkl['test_auc'])
            all_val_auc.append(res_pkl['val_auc'])
            continue

        if args.wandb:
            run = wandb.init(project='HER2', name=f"{args.task}_{args.exp_code}_fold{i}", config=args,
                             reinit=True)

        start_time = timer()

        train_datasets, val_dataset, test_dataset = dataset.return_splits(from_id=True,
                                                                          csv_path=f'{args.results_dir}/splits_{i}.csv',
                                                                          no_fl=args.no_fl)

        if len(train_datasets) > 1:
            for idx in range(len(train_datasets)):
                if args.frac < 1:
                    assert args.frac > 0
                    frac_len = int(len(train_datasets[idx])*args.frac)
                    train_datasets[idx].slide_data = train_datasets[idx].slide_data.iloc[:frac_len]
                print("worker_{} training on {} samples".format(idx, len(train_datasets[idx])))
            print('validation: {}, testing: {}'.format(len(val_dataset), len(test_dataset)))
            datasets = (train_datasets, val_dataset, test_dataset)
            results, test_auc, val_auc = train_fl_point(datasets, i, args, ind_dataset=ind_dataset, multi_site_class_dict=multi_site_class_dict)
        else:
            train_dataset = train_datasets[0]
            if args.frac < 1:
                assert args.frac > 0
                frac_len = int(len(train_datasets[0])*args.frac)
                train_datasets[0].slide_data = train_datasets[0].slide_data.iloc[:frac_len]
            print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset),
                                                                     len(test_dataset)))
            datasets = (train_dataset, val_dataset, test_dataset)
            results, test_auc, val_auc = train_point(datasets, i, args, ind_dataset=ind_dataset)

        results['test_auc'] = test_auc
        results['val_auc'] = val_auc
        print(results)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)

        # write results to pkl
        save_pkl(res_pkl_filename, results)
        if wandb.run:
            run.finish()
        end_time = timer()
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

    final_df = pd.DataFrame({'folds': folds,
                             'test_auc': all_test_auc,
                             'val_auc': all_val_auc, })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--no_skip', action='store_true', default=False, help='skip the fold that has been trained')
parser.add_argument('--eval', action='store_true', default=False, help='evaluation mode')

# Dataset Settings
parser.add_argument('--data_dir', type=str, default='HER2',
                    help='data directory')
parser.add_argument('--csv_path', type=str, default='data_csv/BRCA_HER2_Public.csv',
                    help='data directory')
parser.add_argument('--ind_name', type=str, default=['yale', 'her2c'], nargs='+', help='independent validation name')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_worker", type=int, default=8)
parser.add_argument('--frac', type=float, default=1.0, help='fraction of training dataset')
parser.add_argument('--fps', action='store_true', default=False, help='use fps')
parser.add_argument('--feat_name', type=str, default='feat_nuhtc', help='feature name')
parser.add_argument('--feat_dim', type=int, default=256, help='feature channel numbers')
parser.add_argument('--norm', type=str, default='batch', help='batch, wsconv, group normalization')
parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')

# Model Settings
parser.add_argument('--model_type', type=str, default='point_transformer', help='model name: point_transformer, point_net, sage, slidegraph, patch_gcn, gat')
parser.add_argument('--max_epochs', type=int, default=200,  help='maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0002)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--drop_out', type=float, default=0.5, help='dropout rate (default: 0.5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--aux', type=float, default=0., help='use aux')
parser.add_argument('--ihc', type=float, default=0., help='use ihc aux loss')
parser.add_argument('--mmd', type=float, default=0., help='use mmd loss')
parser.add_argument('--mutual_info', type=float, default=0., help='use mutual information loss')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd', 'adamw', 'sam'], default='adam')
parser.add_argument('--label_smoothing', type=float, default=0, help='(1-epsilon) + epsilon/K')
parser.add_argument('--loss', type=str, default='ce', help='loss function name')
parser.add_argument('--dist_code', type=str, default='exp', help='dynamic distribution method')
parser.add_argument('--load_path', type=str, default=None, help='model load path')
parser.add_argument('--fast_sim', action='store_true', default=False, help='fast simaliraty sampling')

# Federated Learning Settings
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--exp_code', type=str, default='baseline', help='experiment code for saving results')
parser.add_argument('--task', type=str, default='point_cls_fl')
parser.add_argument('--inst_name', type=str, default=None)
parser.add_argument('--fl_avg', type=str, default='FedAvg',
                    help='Update the global model using {FedAvg, FedMGDA, FedProx, FedMGDAProx, SiteAvg, FedBN}')
parser.add_argument('--noise_level', type=float, default=0,
                    help='noise level added on the shared weights in federated learning (default: 0)')
parser.add_argument('--no_fl', action='store_true', default=False, help='train on centralized data')
parser.add_argument('--testing', action='store_true', default=False, help='testing')
parser.add_argument('--E', type=int, default=1, help='communication_freq')

# MGDA Settings
parser.add_argument('--grad_norm', type=int, default=0,
                    help="Default set to no normalization. Set to 1 for normalization")
parser.add_argument('--epsilon', type=float, default=1.,
                    help="Interpolation between FedMGDA and FedAvg. \
                    When set to 0, recovers FedAvg; When set to 1, is FedMGDA without any constraint")
parser.add_argument('--lower_b', type=float, default=0.0,
                    help="lower bound for the site weight")
parser.add_argument('--upper_b', type=float, default=1.0,
                    help="lower bound for the site weight")
parser.add_argument('--cap', type=float, default=1.,
                    help="Capped MGDA parameter, when set to 1, same as default MGDA. \
                        Set to smaller values to restrict individual participation.")
parser.add_argument('--vip', type=int, default=-1,
                    help='the ID of a user that participates in each communication round; {-1 no vip, 0....number of users}')

# Proximal Settings
parser.add_argument('--prox_weight', type=float, default=0.0,
                    help='the weight of proximal regularization term in FedProx and FedMGDA')

# Q-fair Federated Learning
parser.add_argument("--qffl", type=float, default=0.0, help="the q-value in the qffl algorithm. \
                                                                qffl with q=0 reduces to FedAvg")
parser.add_argument('--Lipschitz_constant', type=float, default=1.0)

# Visualization Settings
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for visualization')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### task
print("Experiment Name:", args.exp_code)


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

args.early_stopping = True
args.model_size = 'small'

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'E': args.E,
            'opt': args.opt,
            'loss': args.loss}

if args.inst_name is not None:
    settings.update({'inst_name': args.inst_name})

else:
    settings.update({'noise_level': args.noise_level,
                     'fl_avg': args.fl_avg})

print('\nLoad Dataset')
dataset_setting = dict(
    data_dir=args.data_dir,
    npoint=args.num_point,
    num_splits=args.k,
    shuffle=False,
    seed=args.seed,
    print_info=True,
    label_col='HER2',
    inst=args.inst_name,
    feat_name=args.feat_name,
    feat_dim=args.feat_dim,
    num_classes=args.n_classes,
    site_col='site',
    multi_site=True,
    label_frac=1.0,
    ignore=[])

dataset = Generic_Point_Dataset(csv_path=args.csv_path, **dataset_setting)

inst_li = dataset.institutes
args.institutes = inst_li

csv_path_dict = {
    'yale': 'dataset_csv/Yale_HER2.csv',
    'her2c': 'dataset_csv/HER2_Contest.csv',
}

ind_dataset = {}
for ind_k in args.ind_name:
    ind_dataset[ind_k] = Independent_Dataset(csv_path=csv_path_dict[ind_k], **dataset_setting)

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('./splits', args.task)
else:
    args.split_dir = os.path.join('./splits', args.split_dir)

print("split_dir", args.split_dir)

os.makedirs(args.split_dir, exist_ok=True)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
