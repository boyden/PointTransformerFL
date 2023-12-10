import copy
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from utils.utils import generate_split, nth, generate_split_mutli_site


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Farthest point sampler works as follows:
    1. Initialize the sample set S with a random point
    2. Pick point P not in S, which maximizes the distance d(P, S)
    3. Repeat step 2 until |S| = npoint

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def random_point_sample(point, npoint):
    """
    Farthest point sampler works as follows:
    1. Initialize the sample set S with a random point
    2. Pick point P not in S, which maximizes the distance d(P, S)
    3. Repeat step 2 until |S| = npoint

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    centroids = np.random.choice(N, npoint, replace=True)
    point = point[centroids.astype(np.int32)]
    return point


def df_filter_TCGA(df):
    df = df.drop_duplicates(keep='first')
    select_id = df['HER2'].isin(['+', '-'])
    return df[select_id]


def preprocess_df(df, data_path, feat_name='feat_nuhtc'):
    for idx in df.index:
        x = df.loc[idx]
        if 'graph' in feat_name.lower():
            df.loc[idx, 'FEAT'] = os.path.exists(
                f"{data_path}/WSI_FEAT/{x['Scanner']}/{feat_name}/{os.path.splitext(x['Filename'])[0]}.pt")
        else:
            df.loc[idx, 'FEAT'] = os.path.exists(
                f"{data_path}/WSI_FEAT/{x['Scanner']}/{feat_name}/npy_files/{os.path.splitext(x['Filename'])[0]}.npy")
    df = df[df['FEAT'] == True]
    return df


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()


class Generic_WSI_Dataset(Dataset):
    def __init__(self,
                 csv_path='data_csv/BRCA_HER2_ALL.csv',
                 data_dir=None,
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_col='HER2',
                 train_col=None,
                 site_col='site',
                 multi_site=False,
                 inst='tcga',
                 num_classes=2,
                 num_splits=5,
                 patient_strat=False,
                 patient_voting='max',
                 npoint=1024,
                 feat_name='feat_vit',
                 feat_dim=256,
                 fps=False,
                 normal_channel=True,
                 split='train',
                 cache_size=15000,
                 process_data=False,
                 resample=None,
                 patient_dict=None,
                 val_ratio=0.1,
                 test_ratio=0.3,
                 label_frac=1.0,
                 val_num=(198, 71),
                 test_num=(593, 212),
                 eps=1e-6,
                 filter_dict={},
                 **kwargs,
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.multi_site_ratio_dict = None
        self.custom_test_ids = None
        self.seed = seed
        self.num_splits = num_splits
        self.label_frac = label_frac
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = data_dir
        self.feat_name = feat_name
        self.multi_site = multi_site
        self.kwargs = dict(data_dir=data_dir,
                           label_col=label_col,
                           patient_dict=patient_dict,
                           num_classes=num_classes,
                           feat_name=feat_name,
                           feat_dim=feat_dim,
                           npoint=npoint,
                           split=split,
                           fps=fps,
                           normal_channel=normal_channel,
                           cache_size=cache_size,
                           process_data=process_data,
                           resample=resample,
                           site_col=site_col)
        self.her2_ann2label = {
            '+': 1,
            '-': 0
        }
        self.her2_label2ann = {
            0: '-',
            1: '+'
        }
        self.ihc_label2ann = {
            0: '0',
            1: '1+',
            2: '2+',
            3: '3+'
        }
        self.inst = inst

        slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False)
        slide_data = df_filter_TCGA(slide_data)
        slide_data = preprocess_df(slide_data, data_path=self.data_dir, feat_name=feat_name)

        self.site_col = site_col
        self.institutes = slide_data[site_col].unique()

        if 'case_id' not in slide_data:
            slide_data['case_id'] = slide_data['ID']
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'HER2'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data[[self.site_col, self.label_col, 'ID', 'Filename', 'Scanner', 'HER2_Score']]
        self.sub_id_li = self.slide_data['ID'].values
        self.num_classes = num_classes
        self.slide_data[self.label_col] = self.slide_data[self.label_col].apply(lambda x: self.her2_ann2label[x])
        self.site_labels = self.slide_data[self.site_col].values
        label_dict = {}
        for lb in self.slide_data[self.label_col].unique():
            label_dict[lb] = (self.slide_data[self.label_col] == lb).sum()

        self.label_dict = label_dict
        if multi_site:
            multi_site_label_dict = self.init_multi_site_label_dict(self.slide_data, label_dict)
            self.multi_site_label_dict = multi_site_label_dict

        if self.patient_strat:
            self.patient_data_prep(patient_voting)
        self.slide_data.loc[:, 'slide_id'] = self.slide_data.index.values
        self.cls_ids_prep()

        if train_col is not None:
            test_ids = np.where(self.slide_data[train_col] == 0)[0]
            self.test_ids = test_ids

        if print_info:
            self.summarize()

        if val_ratio is None or test_ratio is None:
            self.create_splits(k=self.num_splits, val_num=val_num, test_num=test_num, label_frac=self.label_frac)
        else:
            test_ids = []
            for site in self.institutes:
                site_idx = self.slide_data[self.site_col] == site
                site_cls_ids = self.slide_data.loc[site_idx, 'slide_id'].values
                site_test_label = self.slide_data.loc[site_idx, self.label_col].values
                _, test_site_id = train_test_split(site_cls_ids, test_size=test_ratio, stratify=site_test_label)
                test_ids.extend(test_site_id)
            self.test_ids = test_ids
            self.create_split_multi_sites(k=self.num_splits, val_num=val_num, test_num=test_num,
                                          label_frac=self.label_frac,
                                          custom_test_ids=self.test_ids,
                                          site_labels=self.slide_data[self.site_col].values)

        # self.set_splits(start_from=0)

    def sample_held_out(self, test_num=(50, 50)):

        test_ids = []
        np.random.seed(self.seed)  # fix seed

        cls_ids = self.slide_cls_ids

        for c in range(len(test_num)):
            test_ids.extend(np.random.choice(cls_ids[c], test_num[c], replace=False))  # validation ids

        return test_ids

    def init_multi_site_label_dict(self, slide_data, label_dict):
        print('initiating multi-source label dictionary')
        sites = np.unique(slide_data['site'].values)
        multi_site_dict = {}
        multi_site_ratio_dict = {}
        num_classes = len(label_dict)
        for key, val in label_dict.items():
            for idx, site in enumerate(sites):
                site_key = (key, site)
                site_num = (slide_data['site'] == site).sum()
                site_val = ((slide_data['site'] == site) & (slide_data[self.label_col] == key)).sum()
                multi_site_dict.update({site_key: site_val})
                multi_site_ratio_dict.update({site_key: site_val / site_num})
                print('{} : {}'.format(site_key, site_val))
        self.multi_site_ratio_dict = multi_site_ratio_dict
        return multi_site_dict

    def cls_ids_prep(self):
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.site_cls_ids = {}
        for site in self.institutes:
            self.site_cls_ids[site] = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_col] == i)[0]
            for site in self.institutes:
                self.site_cls_ids[site][i] = \
                    np.where((self.slide_data[self.label_col] == i) & (self.slide_data[self.site_col] == site))[0]

    def __len__(self):
        return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("patient-level counts:\n", self.slide_data[self.label_col].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Class %i:' % i)
            print('Patient-LVL; Number of samples: %d' % (self.slide_cls_ids[i].shape[0]))
            if self.test_ids is not None:
                print('Number of held-out test samples: {}'.format(
                    len(np.intersect1d(self.test_ids, self.slide_cls_ids[i]))))

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': self.custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def create_split_multi_sites(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, val_ratio=0.1,
                                 test_ratio=0.3, custom_test_ids=None, site_labels=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': self.custom_test_ids,
            'site_labels': site_labels,
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.site_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split_mutli_site(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, split, **kwargs):
        split = split.dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            split_kwargs = {}
            split_kwargs.update(self.kwargs)
            split_kwargs.update(kwargs)
            split = Generic_Split(df_slice, **split_kwargs)
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            split = Generic_Split(df_slice, **self.kwargs)
        else:
            split = None

        return split

    def return_splits(self, from_id=True, csv_path=None, no_fl=False):

        if from_id:
            # set_splites will yeild the train, val, test id for each folds
            if os.path.exists(csv_path):
                all_splits = pd.read_csv(csv_path, index_col=0)
            else:
                self.set_splits(start_from=None)
                all_len = max(len(self.train_ids), len(self.val_ids), len(self.test_ids))
                all_splits = pd.DataFrame(index=range(all_len), columns=['train', 'val', 'test'])
                all_splits.loc[range(len(self.train_ids)), 'train'] = self.train_ids
                all_splits.loc[range(len(self.val_ids)), 'val'] = self.val_ids
                all_splits.loc[range(len(self.test_ids)), 'test'] = self.test_ids
                all_splits.to_csv(csv_path)
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
        # train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits['val'])
        test_split = self.get_split_from_df(all_splits['test'])

        train_splits = []
        if no_fl:
            train_split = all_splits['train']
            train_splits.append(self.get_split_from_df(train_split))
        elif self.inst is not None:
            mask = all_splits['train'].isin(self.slide_data[self.slide_data[self.site_col] == self.inst].index)
            train_split = all_splits.loc[mask, 'train']
            train_splits.append(self.get_split_from_df(train_split))
        else:
            for inst in self.institutes:
                mask = all_splits['train'].isin(self.slide_data[self.slide_data[self.site_col] == inst].index)
                train_split = all_splits.loc[mask, 'train']
                train_splits.append(self.get_split_from_df(train_split))

        return train_splits, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def key_pair2desc(self, key_pair):
        if len(key_pair) == 2:
            label, her2_status = key_pair
            label_desc = (self.bins[label], self.bins[label] + 1, her2_status)
        else:
            institute, label, her2_status = key_pair
            label_desc = (institute, self.bins[label], self.bins[label] + 1, her2_status)
        return label_desc

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in
                     range(self.num_classes)]
            index = [self.key_pair2desc(key_pair) for key_pair in index]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index,
                              columns=columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
        unique = np.append(unique, missing_classes)
        counts = np.append(counts, np.full(len(missing_classes), 0))
        inds = unique.argsort()
        counts = counts[inds]
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
        unique = np.append(unique, missing_classes)
        counts = np.append(counts, np.full(len(missing_classes), 0))
        inds = unique.argsort()
        counts = counts[inds]
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
        unique = np.append(unique, missing_classes)
        counts = np.append(counts, np.full(len(missing_classes), 0))
        inds = unique.argsort()
        counts = counts[inds]
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


class Generic_Point_Dataset(Generic_WSI_Dataset):
    def __init__(
            self,
            data_dir=None,
            npoint=1024,
            split='train',
            fps=False,
            normal_channel=True,
            cache_size=15000,
            process_data=False,
            resample=None,
            num_splits=5,
            feat_name='feat_vit',
            feat_dim=256,
            **kwargs):
        super(Generic_Point_Dataset, self).__init__(data_dir=data_dir,
                                                    feat_name=feat_name,
                                                    feat_dim=feat_dim,
                                                    npoint=npoint,
                                                    num_splits=num_splits,
                                                    fps=fps,
                                                    **kwargs)
        self.data_dir = data_dir
        self.npoints = npoint
        self.fps = fps
        self.normal_channel = normal_channel
        self.process_data = process_data
        self.status = split
        self.feat_name = feat_name
        self.feat_dim = feat_dim
        if isinstance(resample, (float, int)):
            self.resample = [resample, 1 - resample]
        elif isinstance(resample, list):
            self.resample = resample
        self.her2_ann2label = {
            '+': 1,
            '-': 0
        }
        self.her2_label2ann = {
            0: '-',
            1: '+'
        }
        self.ihc_label2ann = {
            0: '0',
            1: '1+',
            2: '2+',
            3: '3+'
        }
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.sub_id_li)

    def __getitem__(self, index):
        # if index in self.cache and self.status != 'train':
        #     point_set, her2_status = self.cache[index]
        # else:
        center = self.slide_data.loc[self.slide_data['ID'] == self.sub_id_li[index], 'Scanner'].values[0]
        filename = self.slide_data.loc[self.slide_data['ID'] == self.sub_id_li[index], 'Filename'].values[0]
        her2_status = self.slide_data.loc[self.slide_data['ID'] == self.sub_id_li[index], 'HER2'].values[0]
        ihc_status = self.slide_data.loc[self.slide_data['ID'] == self.sub_id_li[index], 'HER2_Score'].values[0]
        try:
            ihc_status = int(ihc_status)
        except:
            ihc_status = -1

        filename = os.path.splitext(filename)[0]
        if 'graph' in self.feat_name:
            graph = torch.load(f'{self.data_dir}/WSI_FEAT/{center}/{self.feat_name}/{filename}.pt')
            if 'nuhtc' in self.feat_name:
                mean_arr = torch.tensor([128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2]).repeat_interleave(64).reshape((1, -1))
                graph.ndata['feat'] = graph.ndata['feat'] / mean_arr
            graph.ndata['feat'] = graph.ndata['feat'][:, -self.feat_dim:]
            # return graph, torch.tensor([her2_status], dtype=torch.long), torch.tensor([ihc_status], dtype=torch.long)
            return graph, her2_status, ihc_status

        point_set = np.load(f'{self.data_dir}/WSI_FEAT/{center}/{self.feat_name}/npy_files/{filename}.npy')
        if self.feat_name == 'feat_nuhtc':
            mean_arr = np.array([128 ** 2, 64 ** 2, 32 ** 2, 16 ** 2]).repeat(64)
            mean_arr = np.concatenate(([1, 1, 1], mean_arr)).reshape((1, -1)).astype(np.float32)
            point_set = point_set / mean_arr
        if self.feat_dim + 3 != point_set.shape[1]:
            sel_idx = np.arange(point_set.shape[1])
            sel_idx = np.concatenate((sel_idx[:3], sel_idx[-self.feat_dim:]))
            point_set = point_set[:, sel_idx]

        if self.status == 'attr':
            if point_set.shape[0] < 1024:
                point_set = np.tile(point_set, (1+1024//point_set.shape[0], 1))
            point_set = point_set[:1024, :]
            ori_xyz = copy.deepcopy(point_set[:, 0:3])
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            return torch.from_numpy(point_set), torch.from_numpy(ori_xyz)

        if self.fps:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = random_point_sample(point_set, self.npoints)
            # point_set = point_set[0 : self.npoints, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, her2_status)


        return torch.from_numpy(point_set), her2_status, ihc_status


class Generic_Split(Generic_Point_Dataset):
    def __init__(self,
                 slide_data,
                 data_dir=None,
                 label_col=None,
                 patient_dict=None,
                 num_classes=2,
                 feat_name='feat_vit',
                 feat_dim=256,
                 npoint=1024,
                 split='train',
                 fps=False,
                 normal_channel=True,
                 cache_size=15000,
                 process_data=False,
                 resample=None,
                 site_col='site',
                 ):
        self.use_h5 = False
        self.data_dir = data_dir
        self.slide_data = slide_data
        self.sub_id_li = slide_data['ID'].values
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.site_col = site_col
        self.institutes = slide_data[site_col].unique()
        self.site_labels = self.slide_data[site_col].values

        self.npoints = npoint
        self.fps = fps
        self.normal_channel = normal_channel
        self.process_data = process_data
        self.status = split
        self.feat_name = feat_name
        self.feat_dim = feat_dim
        self.cache_size = cache_size
        self.cache = {}

        self.site_cls_ids = {}
        for site in self.institutes:
            self.site_cls_ids[site] = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_col] == i)[0]
            for site in self.institutes:
                self.site_cls_ids[site][i] = \
                    np.where((self.slide_data[self.label_col] == i) & (self.slide_data[self.site_col] == site))[0]

    def __len__(self):
        return len(self.slide_data)


class Independent_Dataset(Generic_Point_Dataset):
    def __init__(self,
                 csv_path='dataset_csv/BRCA_HER2_PUBLIC.csv',
                 data_dir=None,
                 label_col=None,
                 patient_dict=None,
                 num_classes=2,
                 feat_name='feat_vit',
                 npoint=1024,
                 split='train',
                 fps=False,
                 normal_channel=True,
                 cache_size=15000,
                 process_data=False,
                 feat_dim=256,
                 resample=None,
                 site_col='site',
                 **kwargs
                 ):

        self.her2_ann2label = {
            '+': 1,
            '-': 0
        }
        self.her2_label2ann = {
            0: '-',
            1: '+'
        }
        self.ihc_label2ann = {
            0: '0',
            1: '1+',
            2: '2+',
            3: '3+'
        }
        self.use_h5 = False
        self.data_dir = data_dir
        slide_data = pd.read_csv(csv_path, index_col=None, low_memory=False)
        slide_data = df_filter_TCGA(slide_data)
        slide_data = preprocess_df(slide_data, data_path=self.data_dir, feat_name=feat_name)

        self.slide_data = slide_data
        self.label_col = label_col
        self.slide_data[self.label_col] = self.slide_data[self.label_col].apply(lambda x: self.her2_ann2label[x])
        self.sub_id_li = self.slide_data['ID'].values
        self.num_classes = num_classes

        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.site_col = site_col
        self.institutes = self.slide_data[site_col].unique()
        self.site_labels = self.slide_data[site_col].values

        self.npoints = npoint
        self.fps = fps
        self.normal_channel = normal_channel
        self.process_data = process_data
        self.status = split
        self.feat_name = feat_name
        self.feat_dim = feat_dim
        self.cache_size = cache_size
        self.cache = {}

        self.site_cls_ids = {}
        for site in self.institutes:
            self.site_cls_ids[site] = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_col] == i)[0]
            for site in self.institutes:
                self.site_cls_ids[site][i] = \
                    np.where((self.slide_data[self.label_col] == i) & (self.slide_data[self.site_col] == site))[0]

    def __len__(self):
        return len(self.slide_data)
