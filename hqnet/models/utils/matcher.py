import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import cv2
import random

from hqnet.models.registry import MATCHER

@MATCHER.register_module
class HungarianMatcher(nn.Module):

    def __init__(self, cost_class=1,
                 curves_weight=1, lower_weight=1, upper_weight=1, cfg=None):
        super(HungarianMatcher, self).__init__()
        self.cfg = cfg
        self.cost_class = cost_class
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold ** 2, 0.)

        self.curves_weight = curves_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

    @torch.no_grad()
    def forward(self, outputs, targets, images, paths):

        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        tgt_ids = torch.cat([tgt[:, 0] for tgt in targets]).long()

        cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_curves"]
        tgt_lowers_y = torch.cat([tgt[:, 1] for tgt in targets])
        tgt_uppers_y = torch.cat([tgt[:, 2] for tgt in targets])
        tgt_lowers_x = torch.cat([tgt[:, 3] for tgt in targets])
        tgt_uppers_x = torch.cat([tgt[:, 4] for tgt in targets])

        cost_lower = torch.cdist(out_bbox[:, :, 4].reshape((-1, 1)), tgt_lowers_y.unsqueeze(-1), p=1)
        cost_upper = torch.cdist(out_bbox[:, :, 5].reshape((-1, 1)), tgt_uppers_y.unsqueeze(-1), p=1)
        cost_lower += torch.cdist(out_bbox[:, :, 6].reshape((-1, 1)), tgt_lowers_x.unsqueeze(-1), p=1)
        cost_upper += torch.cdist(out_bbox[:, :, 7].reshape((-1, 1)), tgt_uppers_x.unsqueeze(-1), p=1)
        cost_lower /= 2
        cost_upper /= 2

        # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 5:] for tgt in targets])  # N x (xs+ys)
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        valid_xs = tgt_xs >= 0

        try:
            if valid_xs.shape[0] == 0 or (torch.sum(valid_xs, dim=1, dtype=torch.float32).any() == 0).item():
                raise AssertionError
        except AssertionError as e:
            print(f"valid_xs: {e}", valid_xs)

        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1,
                                                                        dtype=torch.float32)) ** 0.5  # with more points, have smaller weight
        weights = weights / torch.max(weights)

        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]  # N x len(points)
        tgt_ys = tgt_ys.unsqueeze(-1).repeat(1, 1, bs * num_queries)  # N x len(points) x (bs * num_queries)

        b3 = out_bbox[:, :, 3]
        b2 = out_bbox[:, :, 2]
        b1 = out_bbox[:, :, 1]
        b0 = out_bbox[:, :, 0]  # bs x num_queries

        b0 = b0.reshape(-1)
        b1 = b1.reshape(-1)
        b2 = b2.reshape(-1)
        b3 = b3.reshape(-1)  # (bs * num_queries)

        output_xs = b3 * tgt_ys ** 3 + b2 * tgt_ys ** 2 + b1 * tgt_ys + b0

        tgt_xs = tgt_xs.unsqueeze(0).repeat(bs * num_queries, 1, 1)
        tgt_xs = tgt_xs.permute(1, 2, 0)

        cost_polys = [torch.sum((torch.abs(tgt_x[valid_x] - out_x[valid_x])), dim=0) for tgt_x, out_x, valid_x in
                      zip(tgt_xs, output_xs, valid_xs)]  # N x [bs*num_queries]
        cost_polys = torch.stack(cost_polys, dim=-1)
        cost_polys = cost_polys * weights

        C = self.cost_class * cost_class + self.curves_weight * cost_polys / 10 \
            + self.lower_weight * cost_lower + self.upper_weight * cost_upper
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class,
                  curves_weight, lower_weight, upper_weight):
    return HungarianMatcher(cost_class=set_cost_class,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight)

