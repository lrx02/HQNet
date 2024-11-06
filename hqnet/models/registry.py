from hqnet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')
POS_ENCODE = Registry('positional_encoding')
BOX_POS_ENCODE = Registry('box_positional_encoding')
TRANSFORMER = Registry('transformer')
FUSION_TRANSFORMER = Registry('fusion_transformer')
CAM_HEADS = Registry('cam_head')
SFM_HEADS = Registry('sfm_head')
NECKS = Registry('necks')
NETS = Registry('nets')
MATCHER = Registry('matcher')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))


def build_cam_head(cfg):
    return build(cfg.cam_head, CAM_HEADS, default_args=dict(cfg=cfg))


def build_sfm_head(cfg):
    return build(cfg.sfm_head, SFM_HEADS, default_args=dict(cfg=cfg))


def build_positional_encoding(cfg):
    return build(cfg.positional_encoding, POS_ENCODE, default_args=dict(cfg=cfg))

def build_box_positional_encoding(cfg):
    return build(cfg.box_positional_encoding, BOX_POS_ENCODE, default_args=dict(cfg=cfg))

def build_transformer(cfg):
    return build(cfg.transformer, TRANSFORMER, default_args=dict(cfg=cfg))

def build_fusion_transformer(cfg):
    return build(cfg.fusion_transformer, FUSION_TRANSFORMER, default_args=dict(cfg=cfg))


def build_matcher(cfg):
    return build(cfg.matcher, MATCHER, default_args=dict(cfg=cfg))

def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))

def build_necks(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))
