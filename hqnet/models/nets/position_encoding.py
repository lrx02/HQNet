# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from hqnet.models.registry import POS_ENCODE, BOX_POS_ENCODE

@POS_ENCODE.register_module
class SinePositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, cfg=None):
        super(SinePositionalEncoding, self).__init__()
        self.num_pos_feats = num_pos_feats // 2  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True
        self.cfg = cfg

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x, mask):
        # x = tensor_list.tensors
        # mask = tensor_list.mask  # the image location which is padded with 0 is set to be 1 at the corresponding mask location
        assert mask is not None
        not_mask = ~mask  # image 0 -> 0

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # print(20 ** (2 * (dim_t // 2) / self.num_pos_feats))
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


@BOX_POS_ENCODE.register_module
class BoxPositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, cfg=None):
        super(BoxPositionalEncoding, self).__init__()
        self.cfg = cfg
        self.num_pos_feats = num_pos_feats // 2  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x, mask):
        # x = tensor_list.tensors
        # mask = tensor_list.mask  # the image location which is padded with 0 is set to be 1 at the corresponding mask location
        # assert mask is not None
        # not_mask = (~mask).reshape(x.shape[:-1])  # image 0 -> 0

        x_embed = (x[:,:,:,0]*self.cfg.img_w)
        y_embed = (x[:,:,:,1]*self.cfg.img_h)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos



