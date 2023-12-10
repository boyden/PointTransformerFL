import numpy as np
import os
import torch
import torch.nn.functional as F
from .helper import TransitionUp, index_points, square_distance, PointNetSetAbstraction, PointNetSetAbstractionAtten
from torch import nn

import sys

"""
Part of the code are adapted from
https://github.com/qq456cvb/Point-Transformers
"""


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels, norm='batch'):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True,
                                         norm=norm)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionDownAtten(nn.Module):
    def __init__(self, k, nneighbor, channels, norm='batch'):
        super().__init__()
        self.sa = PointNetSetAbstractionAtten(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True,
                                              norm=norm)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, n_neighbors, transformer_dim=None):
        super(PointTransformerBlock, self).__init__()
        if transformer_dim is None:
            transformer_dim = input_dim
        self.fc1 = nn.Linear(input_dim, transformer_dim)
        self.fc2 = nn.Linear(transformer_dim, input_dim)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.w_qs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_ks = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_vs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.n_neighbors = n_neighbors

    def forward(self, x, pos):
        dists = square_distance(pos, pos)
        knn_idx = dists.argsort()[:, :, : self.n_neighbors]  # b x n x k
        knn_pos = index_points(pos, knn_idx)

        h = self.fc1(x)
        q, k, v = (
            self.w_qs(h),
            index_points(self.w_ks(h), knn_idx),
            index_points(self.w_vs(h), knn_idx),
        )

        pos_enc = self.fc_delta(pos[:, :, None] - knn_pos)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = torch.softmax(
            attn / np.sqrt(k.size(-1)), dim=-2
        )  # b x n x k x f

        res = torch.einsum("bmnf,bmnf->bmf", attn, v + pos_enc)
        res = self.fc2(res) + x
        return res, attn


class PointTransformer(nn.Module):
    def __init__(
            self,
            n_points,
            batch_size,
            feature_dim=3,
            n_blocks=4,
            downsampling_rate=4,
            hidden_dim=32,
            transformer_dim=None,
            n_neighbors=16,
            fast_sim=False,
            norm='batch',
    ):
        super(PointTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ptb = PointTransformerBlock(
            hidden_dim, n_neighbors, transformer_dim
        )
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            block_hidden_dim = hidden_dim * 2 ** (i + 1)
            block_n_points = n_points // (downsampling_rate ** (i + 1))
            # self.transition_downs.append(
            #     TransitionDown(
            #         block_n_points,
            #         batch_size,
            #         [
            #             block_hidden_dim // 2 + 3,
            #             block_hidden_dim,
            #             block_hidden_dim,
            #         ],
            #         n_neighbors=n_neighbors,
            #     )
            # )
            if fast_sim:
                self.transition_downs.append(
                    TransitionDownAtten(
                        block_n_points, n_neighbors, [block_hidden_dim // 2 + 3, block_hidden_dim, block_hidden_dim],
                        norm=norm
                    )
                )
            else:
                self.transition_downs.append(
                    TransitionDown(
                        block_n_points, n_neighbors, [block_hidden_dim // 2 + 3, block_hidden_dim, block_hidden_dim],
                        norm=norm
                    )
                )

            self.transformers.append(
                PointTransformerBlock(
                    block_hidden_dim, n_neighbors, transformer_dim
                )
            )

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
        else:
            pos = x

        feat = x
        h = self.fc(feat)
        h, _ = self.ptb(h, pos)

        hidden_state = [(pos, h)]
        for td, tf in zip(self.transition_downs, self.transformers):
            pos, h = td(pos, h)
            h, _ = tf(h, pos)
            hidden_state.append((pos, h))

        return h, hidden_state


class PointTransformerCLS(nn.Module):
    def __init__(
            self,
            n_classes=2,
            batch_size=16,
            n_points=1024,
            feature_dim=3,
            n_blocks=4,
            downsampling_rate=4,
            hidden_dim=32,
            transformer_dim=None,
            n_neighbors=16,
            aux=0,
            mutual_info=0,
            ihc=0,
            fast_sim=False,
            norm='batch',
            **kwargs
    ):
        super(PointTransformerCLS, self).__init__()
        self.aux = aux
        self.ihc = ihc
        self.mutual_info = mutual_info
        self.backbone = PointTransformer(
            n_points,
            batch_size,
            feature_dim,
            n_blocks,
            downsampling_rate,
            hidden_dim,
            transformer_dim,
            n_neighbors,
            fast_sim=fast_sim,
            norm=norm,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2 ** (n_blocks), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.cls = nn.Linear(64, n_classes)
        if self.aux > 0:
            self.aux_cls = nn.Linear(64, n_classes)
        if self.ihc > 0:
            self.ihc_cls = nn.Linear(64, n_classes)
        if self.mutual_info > 0:
            self.h_out = nn.Sequential(
                nn.Linear(hidden_dim * 2 ** (n_blocks), 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
            )

    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.backbone = self.backbone.to(device)
            self.out = self.out.to(device)
            self.cls = self.cls.to(device)
            if self.aux > 0:
                self.aux_cls = self.aux_cls.to(device)
            if self.ihc > 0:
                self.ihc_cls = self.ihc_cls.to(device)
            if self.mutual_info > 0:
                self.h_out = self.h_out.to(device)
            self.device = device

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.backbone = self.backbone.to(device)
            self.out = self.out.to(device)
            self.cls = self.cls.to(device)
            if self.aux > 0:
                self.aux_cls = self.aux_cls.to(device)
            if self.ihc > 0:
                self.ihc_cls = self.ihc_cls.to(device)
            if self.mutual_info > 0:
                self.h_out = self.h_out.to(device)
            self.device = None

    def forward(self, x):
        out, out_feat, h_feat = self.forward_cls(x)
        return out, out_feat, h_feat

    def forward_cls(self, x):
        h, _ = self.backbone(x)
        out_feat = self.out(torch.mean(h, dim=1))
        if self.mutual_info > 0:
            h_feat = self.h_out(h)
        else:
            h_feat = h
        # out_feat = F.normalize(out_feat, p=2, dim=1)
        out = self.cls(out_feat)
        return out, out_feat, h_feat

    def forward_attr(self, x):
        h, _ = self.backbone(x)
        out_feat = self.out(torch.mean(h, dim=1))
        out = self.cls(out_feat)
        return out


class PointNet2(nn.Module):
    def __init__(self,
                 n_classes=2,
                 batch_size=16,
                 n_points=1024,
                 feature_dim=3,
                 n_blocks=4,
                 downsampling_rate=4,
                 hidden_dim=32,
                 transformer_dim=None,
                 n_neighbors=16,
                 aux=0,
                 mutual_info=0,
                 ihc=0,
                 fast_sim=False,
                 norm='batch',
                 normal_channel=True,
                 **kwargs):
        super(PointNet2, self).__init__()
        if not normal_channel:
            feature_dim = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=feature_dim, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        feat = self.drop2(F.relu(self.bn2(self.fc2(x))))
        out = self.fc3(feat)

        return out, feat, l3_points.view(B, 1024)

    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.to(device)
            self.device = device

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            self.device = None


class PointTransformerSeg(nn.Module):
    def __init__(
            self,
            n_classes,
            batch_size,
            n_points=2048,
            feature_dim=3,
            n_blocks=4,
            downsampling_rate=4,
            hidden_dim=32,
            transformer_dim=None,
            n_neighbors=16,
    ):
        super().__init__()
        self.backbone = PointTransformer(
            n_points,
            batch_size,
            feature_dim,
            n_blocks,
            downsampling_rate,
            hidden_dim,
            transformer_dim,
            n_neighbors,
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 2 ** n_blocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** n_blocks),
        )
        self.ptb = PointTransformerBlock(
            32 * 2 ** n_blocks, n_neighbors, transformer_dim
        )

        self.n_blocks = n_blocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(n_blocks)):
            block_hidden_dim = 32 * 2 ** i
            self.transition_ups.append(
                TransitionUp(
                    block_hidden_dim * 2, block_hidden_dim, block_hidden_dim
                )
            )
            self.transformers.append(
                PointTransformerBlock(
                    block_hidden_dim, n_neighbors, transformer_dim
                )
            )

        self.out = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, cat_vec=None):
        _, hidden_state = self.backbone(x)
        pos, h = hidden_state[-1]
        h, _ = self.ptb(self.fc(h), pos)

        for i in range(self.n_blocks):
            h = self.transition_ups[i](
                pos, h, hidden_state[-i - 2][0], hidden_state[-i - 2][1]
            )
            pos = hidden_state[-i - 2][0]
            h, _ = self.transformers[i](h, pos)
        return self.out(torch.cat([h, cat_vec], dim=-1))

class PartSegLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(PartSegLoss, self).__init__()
        self.eps = eps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, y):
        num_classes = logits.shape[1]
        logits = logits.permute(0, 2, 1).contiguous().view(-1, num_classes)
        loss = self.loss(logits, y)
        return loss
