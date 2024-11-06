import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from copy import deepcopy
from hqnet.models.registry import MODULES
from ..registry import build_backbones, build_positional_encoding, build_heads, build_necks, build_transformer


@MODULES.register_module
class SFM(nn.Module):
    def __init__(self, cfg):
        super(SFM, self).__init__()
        self.cfg = cfg
        hidden_dim = cfg.positional_encoding['num_pos_feats']
        self.backbone = build_backbones(cfg)
        self.positional_encoding = build_positional_encoding(cfg)
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
        self.transformer = build_transformer(cfg)
        self.input_proj = nn.Conv2d(128, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(cfg.num_queries, hidden_dim)

    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}

        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        fea1 = fea[-1]

        pmasks = F.interpolate(batch['seg'][:, 0, :, :][None].to(torch.float32), size=fea1.shape[-2:]).to(torch.bool)[0]
        pos = self.positional_encoding(fea1, pmasks)

        hs, encoded_feature, weights, query_embed = self.transformer(self.input_proj(fea1), pmasks,
                                                                     self.query_embed.weight, pos)

        # if self.neck:
        #     fea = self.neck(fea)

        if self.training:
            output = self.heads(hs, batch=batch)
        else:
            output = self.heads(hs)

        return output
