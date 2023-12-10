import torch
import pdb
import numpy as np
import quadprog
import copy


def sync_models(server_model, worker_models, ignore_keys=[]):
    server_params = server_model.state_dict()
    for key in ignore_keys:
        server_params.pop(key, None)
    worker_models = [worker_model.load_state_dict(server_params, strict=False) for worker_model in worker_models]

def is_pos_def(x):
    eig_vals = np.linalg.eigvals(x)
    return np.all(np.round(eig_vals, 6) > 0)

def federated_averging(model, worker_models, noise_level, weights=None, ignore_keys=[]):
    if model.device is not None:
        central_device = model.device
    else:
        central_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_insti = len(worker_models)
    if weights is None:
        weights = [1 / num_insti for i in range(num_insti)]
    central_params = model.state_dict()
    all_worker_params = [worker_models[idx].state_dict() for idx in range(num_insti)]
    keys = central_params.keys()
    for key in keys:
        if 'labels' in key:
            continue
        elif key in ignore_keys:
            continue
        else:
            temp = torch.zeros_like(central_params[key])
            for idx in range(num_insti):
                if noise_level > 0 and 'bias' not in key:
                    temp_params = all_worker_params[idx][key].reshape(-1).float()
                    if len(temp_params) > 1:
                        noise = noise_level * torch.empty(all_worker_params[idx][key].size()).normal_(mean=0,
                                                                                                      std=temp_params.std())
                    else:
                        noise = noise_level * torch.zeros(all_worker_params[idx][key].size())
                    temp = temp + weights[idx] * all_worker_params[idx][key].to(central_device) + noise.to(
                        central_device)
                else:
                    temp = temp + weights[idx] * all_worker_params[idx][key].to(central_device)

            central_params[key] = temp
    model.load_state_dict(central_params)
    return model, worker_models


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def print_qp(A, b_concat, n):
    x_var = [f'x{i + 1}' for i in range(n)]
    for k in range((len(A.T))):
        s = f'{A[0, k]}*{x_var[0]}'
        for i in range(1, n):
            s = s + f' + {A[i, k]}*{x_var[i]}'
        s = s + f' >= {b_concat[k]}'
        print(s)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def solve_centered_w(U, lower_b=0.0, upper_b=1.0, ignore_keys=[]):
    """
        U is a list of gradients (stored as state_dict()) from n users, need to be modified
    """

    n = len(U)
    K = np.eye(n, dtype=float)
    for i in range(0, n):
        for j in range(0, n):
            K[i, j] = 0
            for key in U[i].keys():
                if key in ignore_keys:
                    continue
                K[i, j] += torch.mul(U[i][key], U[j][key]).sum()
    # in case of super large value in K without influcing on final value
    K = K/K.max()
    Q = 0.5 * (K + K.T)
    if not is_pos_def(Q):
        alpha = 1/n *np.ones(n, dtype=float)
        print(alpha)
        return alpha
    p = np.zeros(n, dtype=float)
    a = np.ones(n, dtype=float).reshape(-1, 1)
    Id = np.eye(n, dtype=float)
    neg_Id = -1. * np.eye(n, dtype=float)
    lower_b_val = lower_b * np.ones(n, dtype=float)
    upper_b_val = -upper_b * np.ones(n, dtype=float)
    A = np.concatenate((a, Id, Id, neg_Id), axis=1)
    b = np.zeros(n + 1)
    b[0] = 1.
    b_concat = np.concatenate((b, lower_b_val, upper_b_val))
    alpha = quadprog.solve_qp(Q, p, A, b_concat, meq=1)[0]
    print(np.round(alpha, 4))
    return alpha


def solve_capped_w(U, C=1, ignore_keys=[]):
    """
        U is a list of gradients (stored as state_dict()) from n users
    """

    n = len(U)
    K = np.eye(n, dtype=float)
    for i in range(0, n):
        for j in range(0, n):
            K[i, j] = 0
            for key in U[i].keys():
                if key in ignore_keys:
                    continue
                K[i, j] += torch.mul(U[i][key], U[j][key]).sum()
    K = K/K.max()
    Q = 0.5 * (K + K.T)
    if not is_pos_def(Q):
        alpha = 1/n *np.ones(n, dtype=float)
        print(alpha)
        return alpha
    p = np.zeros(n, dtype=float)
    a = np.ones(n, dtype=float).reshape(-1, 1)
    Id = np.eye(n, dtype=float)
    neg_Id = -1. * np.eye(n, dtype=float)
    cap_b = (-C) * np.ones(n, dtype=float)
    A = np.concatenate((a, Id, neg_Id), axis=1)
    b = np.zeros(n + 1)
    b[0] = 1.
    b_concat = np.concatenate((b, cap_b))
    alpha = quadprog.solve_qp(Q, p, A, b_concat, meq=1)[0]
    print(alpha)
    return alpha
