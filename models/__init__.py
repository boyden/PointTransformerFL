from .point_transformer import PointTransformerCLS, PointNet2
from .graph import SAGE, SlideGraph, PatchGCN
from .model_attention_mil import MIL_CLS, CLAM_SB, DSMIL, TransMIL


def get_model(model_type='point_transformer', **kwargs):
    if model_type == 'point_transformer':
        model = PointTransformerCLS(**kwargs)
        return model
    if model_type == 'point_net':
        model = PointNet2(**kwargs)
        return model
    if 'sage' in model_type.lower():
        model = SAGE(**kwargs)
        return model
    if model_type == 'patch_gcn':
        model = PatchGCN(**kwargs)
        return model
    if model_type == 'slidegraph' or model_type == 'slide_graph' or model_type == 'slidegraph+':
        model = SlideGraph(**kwargs)
        return model
    if model_type == 'mil':
        model = MIL_CLS(**kwargs)
        return model
    if model_type == 'clam_sb':
        model = CLAM_SB(**kwargs)
        return model
    if model_type == 'dsmil':
        model = DSMIL(**kwargs)
        return model
    if model_type == 'transmil':
        model = TransMIL(**kwargs)
        return model
