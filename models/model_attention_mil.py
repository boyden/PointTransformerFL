
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nystrom_attention import NystromAttention

from models.helper import Attn_Net, Attn_Net_Gated
from utils.utils import initialize_weights


class MIL_NN(nn.Module):
    def __init__(self, **kwargs):
        super(MIL_NN, self).__init__(**kwargs)

    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.to(device)
            self.device = device

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            self.device = None


class MIL_CLS(MIL_NN):
    def __init__(self, base_dim=256, n_hidden=64, gate=True, size_arg="small", dropout=0.25, n_classes=2, **kwargs):
        super(MIL_CLS, self).__init__()
        self.size_dict = {"small": [base_dim, 2 * n_hidden, n_hidden], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))

        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h = h[:, :, 3:]  # remove point centroid
        A, h = self.attention_net(h)
        # A: batch * point * 1
        # A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A = F.softmax(A, dim=1)
        out_feat = torch.mul(A, h).sum(dim=1)
        out = self.cls(out_feat)

        return out, out_feat, h


######################################
# Deep Attention MISL Implementation #
######################################
class MIL_Cluster_FC_surv(nn.Module):
    def __init__(self, num_clusters=10, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC_surv, self).__init__()
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.num_clusters = num_clusters

        ### FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        ### WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        cluster_id = kwargs['cluster_id'].detach().cpu().numpy()

        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id == i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1, 512)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        ### Attention MIL
        A, h_path = self.attention_net(h_cluster)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None


class CLAM_SB(MIL_NN):
    def __init__(self, base_dim=256, n_hidden=64, gate=True, size_arg="small", dropout=0.25, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, **kwargs):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [base_dim, 2 * n_hidden, n_hidden], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.cls = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        B, inst_num, D = h.shape
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        else:
            A = A.view(B, inst_num)
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1]
        top_p_ids = top_p_ids.unsqueeze(-1).repeat((1, 1, D))
        top_p = torch.gather(h, dim=1, index=top_p_ids)

        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1]
        top_n_ids = top_n_ids.unsqueeze(-1).repeat((1, 1, D))
        top_n = torch.gather(h, dim=1, index=top_n_ids)
        p_targets = self.create_positive_targets(B * self.k_sample, device)
        n_targets = self.create_negative_targets(B * self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p.reshape(-1, D), top_n.reshape(-1, D)], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, attention_only=False):
        h = h[:, :, 3:]
        A, h = self.attention_net(h)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        total_inst_loss = 0.0

        if instance_eval:

            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[:, i]
                classifier = self.instance_classifiers[i]
                if inst_label.sum() > 0:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[inst_label == 1], h[inst_label == 1], classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    total_inst_loss += instance_loss
                # else:  # out-of-the-class
                #     if self.subtyping:
                #         instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                #         all_preds.extend(preds.cpu().numpy())
                #         all_targets.extend(targets.cpu().numpy())
                #     else:
                #         continue

        logits = torch.mul(A, h).sum(dim=1)
        out = self.cls(logits)

        return out, logits, total_inst_loss


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        batch, inst_num, dim = feats.shape
        V = self.v(feats)  # B x N x V, unsorted
        Q = self.q(feats)  # B x N x Q, unsorted

        # handle two classes without for loop
        _, m_indices = torch.sort(c, 1, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C

        m_feats = torch.gather(feats, dim=1, index=m_indices[:, 0, :].reshape(batch, -1, 1).repeat((1, 1, dim)))# select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.bmm(Q, q_max.transpose(2, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 1)  # normalize attention scores, A in shape N x C,
        B = torch.bmm(A.transpose(2, 1), V)  # compute bag representation, B in shape C x V

        C = self.fcc(B)  # 2 x C x 1
        C = C.view(batch, -1)
        return C, A, B


class DSMIL(MIL_NN):
    def __init__(self, base_dim=256, n_hidden=64, dropout=0.25, n_classes=2, **kwargs):
        super(DSMIL, self).__init__()
        self.i_classifier = FCLayer(base_dim, n_classes)
        self.b_classifier = BClassifier(base_dim, n_classes, dropout_v=dropout)
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = x[:, :, 3:]
        feats, h_feat = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, h_feat)
        out, _ = torch.max(h_feat, 1)
        return out, prediction_bag, h_feat

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class TransMIL(MIL_NN):
    def __init__(self,  base_dim=256, n_hidden=64, dropout=0.25, n_classes=2, **kwargs):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=n_hidden)
        self._fc1 = nn.Sequential(nn.Linear(base_dim, n_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_hidden))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=n_hidden)
        self.layer2 = TransLayer(dim=n_hidden)
        self.norm = nn.LayerNorm(n_hidden)
        self._fc2 = nn.Linear(n_hidden, self.n_classes)
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(n_hidden, self.n_classes)

    def forward(self, x):

        h = x[:, :, 3:] #[B, n, base_dim]

        h = self._fc1(h) #[B, n, n_hidden]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        logits = self.norm(h)[:,0]

        #---->predict
        out = self._fc2(logits) #[B, n_classes]

        return out, logits, h
