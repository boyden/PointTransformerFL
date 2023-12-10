import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from file_utils import load_pkl


def print_stat(res_dir, exp, splits, ignore_keys=[]):
    res_info = {}
    for i in range(splits):
        if not os.path.exists(f'{res_dir}/{exp}/split_{i}_results.pkl'):
            continue
        res = load_pkl(f'{res_dir}/{exp}/split_{i}_results.pkl')

        for k, v in res.items():
            if k in ignore_keys:
                continue
            if k not in res_info.keys():
                res_info[k] = [v]
            else:
                res_info[k].append(v)
    res_info_stat = {'exp_code': exp}
    for k, v in res_info.items():
        res_info_stat[f'{k}_mean'] = round(np.mean(v), 3)
        res_info_stat[f'{k}_std'] = round(np.std(v), 3)
        res_info_stat[f'{k}'] = f"{res_info_stat[f'{k}_mean']}±{res_info_stat[f'{k}_std']}"
    return res_info_stat


parser = argparse.ArgumentParser(description='Summary print')

# Dataset Settings
parser.add_argument('--results_dir', type=str, default='/home/bao/code/HE_Pointnet/point_transformer_FL/results',
                    help='result data directory')
parser.add_argument('--exp', type=str, default=None,
                    help='experiment name')
parser.add_argument('--out', type=str, default='summary',
                    help='experiment name')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')

if __name__ == '__main__':
    args = parser.parse_args()
    res_dir = args.results_dir
    splits = args.k
    ignore_keys = []
    inst_li = ['tcga', 'hx', 'hz', 'hero', 'yale', 'her2c']
    for i in range(len(inst_li)):
        ignore_keys.append(f'acc_{inst_li[i]}')
        for j in range(len(inst_li)):
            if i != j:
                ignore_keys.append(f'auc_{inst_li[i]}_{inst_li[j]}')
                ignore_keys.append(f'acc_{inst_li[i]}_{inst_li[j]}')
    exp_li = os.listdir(res_dir)
    if args.exp is not None:
        exp_li = [exp for exp in exp_li if re.match(args.exp, exp)]

    df_li = []
    for exp in exp_li:
        if 'summary' in exp:
            continue
        if not os.path.exists(f'{res_dir}/{exp}/summary.csv'):
            continue
        res_info = print_stat(res_dir, exp, splits, ignore_keys=ignore_keys)
        print(exp)
        print(res_info)
        print('\n')
        res_info['exp'] = exp
        df_li.append(res_info)
    df = pd.DataFrame(df_li)
    df_col = ['exp_code', 'test_auc_mean', 'test_auc', 'auc_tcga', 'auc_hx', 'auc_hz', 'auc_hero', 'auc_yale', 'auc_her2c', 'auc_mean', 'auc_tcga_mean', 'auc_hx_mean','auc_hz_mean', 'auc_hero_mean', 'auc_yale_mean', 'auc_her2c_mean', 'auc_std', 'auc_tcga_std', 'auc_hx_std','auc_hz_std', 'auc_hero_std', 'auc_yale_std', 'auc_her2c_std', 'test_auc_mean', 'test_auc_std', 'val_auc_mean', 'val_auc_std']
    # df_col = list(set(df_col) - set(df.columns.to_list()))
    df = df[df_col]
    df.to_csv(f'{res_dir}/{args.out}.csv', index=False)

