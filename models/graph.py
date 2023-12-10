import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, BatchNorm1d, Linear, ModuleList, ReLU, Sequential, LayerNorm

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv, GINConv, EdgeConv, NNConv, Set2Set, GATConv, GlobalAttentionPooling
from dgl.nn import AvgPooling, SumPooling
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from models.helper import Attn_Net_Gated_Score

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class MLP_GENConv(nn.Sequential):
    r"""

    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(LayerNorm(channels[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super(MLP_GENConv, self).__init__(*layers)


class MessageNorm(nn.Module):
    r"""

    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """

    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=learn_scale
        )

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale


class GAT(nn.Module):
    def __init__(
            self,
            in_dim,
            num_hidden,
            num_classes,
            heads=1,
            activation=F.elu,
            dropout=0,
            negative_slope=0.2,
            residual=True,
            num_layers=3,
    ):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads,
                self.dropout,
                self.dropout,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATConv(
                    num_hidden * heads,
                    num_hidden,
                    heads,
                    self.dropout,
                    self.dropout,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        g.ndata['h'] = h
        h = dgl.mean_nodes(g, 'h')
        return h


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 3
        self.dropout = dropout
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(self.dropout)
        self.pool = (
            AvgPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = []
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

    def extract_feat(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = []
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        global_h = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            global_h += self.drop(self.linear_prediction[i](pooled_h))

        local_h = h

        return local_h, global_h


class GENConv(nn.Module):
    r"""

    Description
    -----------
    Generalized Message Aggregator was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    in_dim: int
        Input size.
    out_dim: int
        Output size.
    aggregator: str
        Type of aggregation. Default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            aggregator="softmax",
            beta=1.0,
            learn_beta=False,
            p=1.0,
            learn_p=False,
            msg_norm=False,
            learn_msg_scale=False,
            mlp_layers=1,
            eps=1e-7,
    ):
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP_GENConv(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = (
            nn.Parameter(torch.Tensor([beta]), requires_grad=True)
            if learn_beta and self.aggr == "softmax"
            else beta
        )
        self.p = (
            nn.Parameter(torch.Tensor([p]), requires_grad=True)
            if learn_p
            else p
        )

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata["h"] = node_feats
            g.edata["h"] = edge_feats
            g.apply_edges(fn.u_add_e("h", "h", "m"))

            if self.aggr == "softmax":
                g.edata["m"] = F.relu(g.edata["m"]) + self.eps
                g.edata["a"] = edge_softmax(g, g.edata["m"] * self.beta)
                g.update_all(
                    lambda edge: {"x": edge.data["m"] * edge.data["a"]},
                    fn.sum("x", "m"),
                )

            elif self.aggr == "power":
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata["m"], minv, maxv)
                g.update_all(
                    lambda edge: {"x": torch.pow(edge.data["m"], self.p)},
                    fn.mean("x", "m"),
                )
                torch.clamp_(g.ndata["m"], minv, maxv)
                g.ndata["m"] = torch.pow(g.ndata["m"], self.p)

            else:
                raise NotImplementedError(
                    f"Aggregator {self.aggr} is not supported."
                )

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats + g.ndata["m"]

            return self.mlp(feats)

class GraphNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.to(device)
            self.device = device

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            self.device = None

class SAGE(GraphNN):
    def __init__(self, base_dim=256, n_hidden=64, n_classes=2, dropout=0.5, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(base_dim, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(n_hidden, n_classes)

    def forward(self, sg):
        h = sg.ndata['feat']
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        sg.ndata['h'] = h
        out_feat = dgl.mean_nodes(sg, 'h')
        out = self.cls(out_feat)
        return out, out_feat, h


# Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks
class PatchGCN(GraphNN):
    def __init__(self, base_dim=256, num_layers=3, edge_agg='spatial', multires=False, resample=0,
                 num_features=1024, n_hidden=64, linear_dim=64,
                 use_edges=False, pool=False, dropout=0.25, n_classes=2, **kwargs):
        super(PatchGCN, self).__init__()
        self.use_edges = use_edges
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers
        self.resample = resample
        self.dropout = dropout

        if self.resample > 0:
            self.n_fc = nn.Sequential(
                *[nn.Dropout(self.resample), nn.Linear(base_dim, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.n_fc = nn.Sequential(*[nn.Linear(base_dim, n_hidden), nn.ReLU(), nn.Dropout(0.25)])

        self.e_fc = nn.Sequential(*[nn.Linear(1, n_hidden), nn.ReLU(), nn.Dropout(0.25)])

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(n_hidden, n_hidden, aggregator='softmax',
                           beta=1.0, learn_beta=True, mlp_layers=2, msg_norm=False, )
            norm = LayerNorm(n_hidden, elementwise_affine=True)
            self.gcns.append(conv)
            self.norms.append(norm)

        self.path_phi = nn.Sequential(
            *[nn.Linear(n_hidden * (self.num_layers + 1), n_hidden * (self.num_layers + 1)), nn.ReLU(),
              nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated_Score(L=n_hidden * (self.num_layers + 1),
                                                        D=n_hidden * (self.num_layers + 1), dropout=dropout,
                                                        n_classes=1)
        self.path_rho = nn.Sequential(
            *[nn.Linear(n_hidden * (self.num_layers + 1), n_hidden), nn.ReLU(), nn.Dropout(dropout)])

        # self.pooling = SumPooling()
        self.pooling = GlobalAttentionPooling(gate_nn=self.path_attention_head)
        self.cls = nn.Linear(n_hidden, n_classes)
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(n_hidden, n_classes)

    def forward(self, g):
        edge_feats = torch.ones((g.num_edges(), 1)).to(g.device)

        with g.local_scope():
            hv = self.n_fc(g.ndata['feat'])
            he = self.e_fc(edge_feats)
            x_ = hv
            for layer in range(self.num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv
                x_ = torch.cat([x_, hv], dim=1)

            h_path = self.path_phi(x_)
            h_g = self.pooling(g, h_path)

        # A_path, h_path = self.path_attention_head(h_path)
        # A_path = torch.transpose(A_path, 1, 0)
        # h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        out_feat = self.path_rho(h_g)
        out = self.cls(out_feat)
        return out, out_feat, h_path

# SlideGraph+ : Whole slide image level graphs to predict HER2 status inbreast cancer
class SlideGraph(GraphNN):
    def __init__(self, base_dim=256, n_classes=2, n_hidden=64, layers=[128, 128, 128, 64], pooling='mean', dropout=0.5, conv='EdgeConv',
                 gembed=True, aggr='max', **kwargs):
        """
        Parameters
        ----------
        base_dim : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [128, 128, 64].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'mean'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(SlideGraph, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {"mean": AvgPooling()}[pooling]
        self.cls = nn.Linear(n_hidden, n_classes)
        self.aux = kwargs.get('aux', 0)
        if self.aux > 0:
            self.aux_cls = nn.Linear(n_hidden, n_classes)

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(base_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, n_hidden))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.linears.append(Linear(out_emb_dim, n_hidden))
                if conv == 'GINConv':
                    subnet = Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                    self.nns.append(subnet)
                    self.convs.append(
                        GINConv(apply_func=self.nns[-1], learn_eps=False))  # Eq. 4.2 eps=100, train_eps=False
                elif conv == 'EdgeConv':
                    self.convs.append(
                        EdgeConv(input_emb_dim, out_emb_dim, batch_norm=True))  # DynamicEdgeConv#EdgeConv aggr='mean'
                else:
                    raise NotImplementedError

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, g):

        x = g.ndata['feat']

        out = []
        pooling = self.pooling
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                dout = F.dropout(pooling(g, z), p=self.dropout, training=self.training)
                out.append(dout)
            else:
                x = F.relu(self.convs[layer - 1](g, x))
                z = self.linears[layer](x)
                dout = F.dropout(pooling(g, z), p=self.dropout, training=self.training)
                out.append(dout)
        out_feat = torch.stack(out, dim=0).sum(dim=0)
        out = self.cls(out_feat)

        return out, out_feat, x
