import math
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from copy import deepcopy
import json
from torchvision.ops import roi_align

from hqnet.models.losses.focal_loss import FocalLoss
from hqnet.models.losses.accuracy import accuracy
from hqnet.ops import nms

from hqnet.models.utils.roi_gather import ROIGather, LinearModule
from hqnet.models.utils.seg_decoder import SegDecoder
from hqnet.models.utils.mlp import MLP
from hqnet.models.utils.dynamic_assign import assign

from hqnet.models.losses.lineiou_loss import liou_loss
from ..registry import SFM_HEADS, build_box_positional_encoding, build_fusion_transformer

from ..utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment



class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Feature_fusion(nn.Module):
    def __init__(self, in_size, out_size):
        super(Feature_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, 1, padding=0, stride=1, bias=True, dilation=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU6(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Output_mask(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output_mask, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size // 2, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size // 2, in_size // 4, 1, 0, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size // 4, out_size, 1, 0, 1, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                          nn.BatchNorm2d(n_filters),
                                          # nn.ReLU(inplace=True),)
                                          nn.PReLU(), )
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias,
                                      dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size // 2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size // 2, in_size // 4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size // 4, out_size, 1, 0, 1, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@SFM_HEADS.register_module
class SFMHead(nn.Module):
    def __init__(self, cfg=None):
        super(SFMHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.hidden_dim = self.cfg.positional_encoding['num_pos_feats']
        self.num_queries = self.cfg.num_queries
        self.sampling_steps = self.cfg.sampling_steps

        self.conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.up = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 2, stride=2)

        self.position_embedding_box = build_box_positional_encoding(cfg)

        self.query_embed = nn.Embedding(self.sampling_steps, self.hidden_dim)
        self.lane_pts_embed = nn.Embedding(self.num_queries * self.sampling_steps, self.hidden_dim)

        self.input_proj = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.fusion_transformer = build_fusion_transformer(cfg)


        # self.out_z = Output(self.hidden_dim, 1)
        self.out_xy = Output(self.hidden_dim, 2)
        self.out_scope = Output(self.hidden_dim, 1)
        self.out_length = Output(self.hidden_dim, 1)

    def forward(self, encoded_feature, fea2, cam_outputs, imgC_dict, imgC_affine_dict, batch_img, sfm_targets, sfm_paths, indices, cur_epoch, batchsize):

        global_fea = self.up(encoded_feature)
        local_fea = self.conv2(fea2)
        encoded_feature = global_fea + local_fea

        pooled_hs_dict = {}
        for qi in range(self.cfg.queue_length):
            boxes = imgC_dict['imgC_{}'.format(str(qi))].clone()
            boxes[..., 1:] = boxes[..., 1:].clamp(0, self.cfg.img_h-1)

            pooled_hs_dict['pooled_{}'.format(str(qi))] = roi_align(encoded_feature[batchsize * qi:batchsize * (qi + 1)],
                                                                    boxes.reshape((-1, 5)),
                                                                    output_size=1,
                                                                    spatial_scale=1 / 16,
                                                                    sampling_ratio=8)

            imgC_shape = imgC_dict['imgC_{}'.format(str(qi))].shape[0:3]
            pooled_hs_dict['pooled_{}'.format(str(qi))] = pooled_hs_dict['pooled_{}'.format(str(qi))].reshape((imgC_shape[0],
                                                                                                               imgC_shape[1],
                                                                                                               imgC_shape[2],
                                                                                                               pooled_hs_dict['pooled_{}'.format(str(qi))].shape[1])).cuda()

        pos_dict = {}
        pmasks_dict = {}
        for qi in range(self.cfg.queue_length):
            center_x = (imgC_affine_dict['imgC_affine_{}'.format(str(qi))][:,:,:,1] + imgC_affine_dict['imgC_affine_{}'.format(str(qi))][:,:,:,3]) / 2
            center_y = (imgC_affine_dict['imgC_affine_{}'.format(str(qi))][:,:,:,2] + imgC_affine_dict['imgC_affine_{}'.format(str(qi))][:,:,:,4]) / 2
            center_x_ori = (imgC_dict['imgC_{}'.format(str(qi))][:,:,:,1] + imgC_dict['imgC_{}'.format(str(qi))][:,:,:,3]) / 2
            center_y_ori = (imgC_dict['imgC_{}'.format(str(qi))][:,:,:,2] + imgC_dict['imgC_{}'.format(str(qi))][:,:,:,4]) / 2

            valid_anchors_affine = (center_x > 0) & (center_x < self.cfg.img_w) & (center_y >= 0) & (center_y < self.cfg.img_h)
            valid_anchors_ori = ((center_x_ori > 0) & (center_x_ori < self.cfg.img_w) & (center_y_ori >= 0) & (center_y_ori < self.cfg.img_h))

            valid_anchors = valid_anchors_ori & valid_anchors_affine


            pmasks_dict['pmask_{}'.format(str(qi))] = (~valid_anchors.reshape((batchsize, -1)))

            pos_dict['pos_{}'.format(str(qi))] = self.position_embedding_box(torch.stack([center_x/self.cfg.img_w, center_y/self.cfg.img_h], -1),
                                                                             pmasks_dict['pmask_{}'.format(str(qi))])

        pooled_hs_list = []
        pmasks_list = []
        pos_list = []
        for qi in range(self.cfg.queue_length):
            pooled_hs_list.append(pooled_hs_dict['pooled_{}'.format(str(qi))])
            pmasks_list.append(pmasks_dict['pmask_{}'.format(str(qi))])
            pos_list.append(pos_dict['pos_{}'.format(str(qi))])

        pooled_hs = torch.cat(pooled_hs_list, 0)
        pmasks = torch.cat(pmasks_list, 0)
        pos = torch.cat(pos_list, 0)

        fused_feature = pooled_hs.permute((0, 3, 1, 2)).flatten(2, 3)
        fused_query = (pooled_hs[:batchsize].permute(0,3,1,2) + pos[:batchsize]).reshape(batchsize, self.cfg.attn_dim, -1)
        hs, encoder_fmap = self.fusion_transformer(self.input_proj(fused_feature), pmasks, fused_query, pos)

        hs = hs.permute(0, 1, 3, 2)
        hs = hs[-1].reshape(batchsize, self.cfg.attn_dim, self.cfg.num_queries,
                            self.cfg.sampling_steps)

        # out_z = self.out_z(hs)
        out_scope = self.out_scope(hs)
        out_length = self.out_length(hs)
        out_xy = self.out_xy(hs)

        result = [out_scope, out_length, out_xy, imgC_dict]
        return self.loss(result, imgC_dict, cam_outputs, batch_img, sfm_targets, sfm_paths, indices, cur_epoch, batchsize), result

    def get_gt(self, tgt, tgt_y):

        # get gt_lanes_points
        gt_nums_lane = [y.shape[0] for y in tgt]
        tgt = torch.cat([i for i in tgt], dim=0)


        target_points = tgt[:, 5:].cuda()
        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]
        start_end = tgt[:, 1:5]

        assert ((target_ys > -1) != (target_xs > -1)).sum().item() == 0
        points_num = torch.where(target_xs > -100, 1, 0).sum(-1)
        # target_zs = target_zs[target_ys > -1]
        # target_zs = target_zs.split(points_num.tolist())
        target_xs = target_xs[target_xs > -1]
        target_xs = target_xs.split(points_num.tolist())  # points for each lane
        target_ys = target_ys[target_ys > -1]
        target_ys = target_ys.split(points_num.tolist())  # points for each lane

        target_points = torch.ones((len(target_xs), 830, 2)).cuda()
        for index, line_x in enumerate(target_xs):
            line_x = line_x.unsqueeze(0).unsqueeze(0)
            line_x_dense = interpolate(line_x, size=830, mode='linear', align_corners=True)
            line_x_dense = line_x_dense[0, 0, :]
            target_points[index, :, 0] = line_x_dense[:]
        for index, line_y in enumerate(target_ys):
            line_y = line_y.unsqueeze(0).unsqueeze(0)
            line_y_dense = interpolate(line_y, size=830, mode='linear', align_corners=True)
            line_y_dense = line_y_dense[0, 0, :]
            target_points[index, :, 1] = line_y_dense[:]
        target_points_split = target_points.split(gt_nums_lane)


        gt_xs = []
        gt_center = []
        gt_length = []
        gt_angle = []
        for idx, lane in enumerate(target_points):
            tgt_x = torch.ones_like(tgt_y) * (-100)
            lower_y = tgt[idx,1]
            upper_y = tgt[idx,2]

            valid_pts = (tgt_y >= lower_y) & (tgt_y <= upper_y)

            y_values = lane[:, 1]
            dist = torch.abs(tgt_y[:, None] - y_values)
            nearest_distances, nearest_indices = torch.min(dist, dim=1)
            tgt_x[valid_pts] = lane[:,0][nearest_indices][valid_pts]

            tgt_center = torch.stack([torch.ones_like(tgt_y) * (-100)]*2 ,-1)
            tgt_length = torch.ones_like(tgt_y) * (-100)
            tgt_angle = torch.ones_like(tgt_y) * (-100)
            tgt_center[valid_pts, 0] = torch.cat([(lane[:,0][nearest_indices][valid_pts][1:] + lane[:,0][nearest_indices][valid_pts][:-1]) / 2, torch.tensor([-200]).cuda()])
            tgt_center[valid_pts, 1] = torch.cat([(tgt_y[valid_pts][1:] + tgt_y[valid_pts][:-1]) / 2, torch.tensor([-200]).cuda()])
            tgt_length[valid_pts] = torch.cat([torch.sqrt((lane[:,0][nearest_indices][valid_pts][1:] - lane[:,0][nearest_indices][valid_pts][:-1]) ** 2 + (tgt_y[valid_pts][1:] - tgt_y[valid_pts][:-1]) ** 2 + 1e-12), torch.tensor([-200]).cuda()])
            tgt_angle[valid_pts] = torch.cat([torch.atan((tgt_y[valid_pts][1:] - tgt_y[valid_pts][:-1]) / (lane[:,0][nearest_indices][valid_pts][1:] - lane[:,0][nearest_indices][valid_pts][:-1] + 1e-12)), torch.tensor([-200]).cuda()])

            gt_xs.append(tgt_x)
            gt_center.append(tgt_center)
            gt_length.append(tgt_length)
            gt_angle.append(tgt_angle)

        gt_xs_stack = torch.stack(gt_xs, 0)
        gt_xs_split = gt_xs_stack.split(gt_nums_lane)
        gt_start_end_split = start_end.split(gt_nums_lane)

        gt_center_stack = torch.stack(gt_center, 0)
        gt_center_split = gt_center_stack.split(gt_nums_lane)

        gt_length_stack = torch.stack(gt_length, 0)
        gt_length_split = gt_length_stack.split(gt_nums_lane)

        gt_angle_stack = torch.stack(gt_angle, 0)
        gt_angle_split = gt_angle_stack.split(gt_nums_lane)

        return gt_start_end_split, gt_xs_split, gt_center_split, gt_length_split, gt_angle_split

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def xy_wh_r_2_xy_sigma(self, xywhr):
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)
        sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))
        return xy, sigma

    def kld_loss(self,pred, target, alpha=1.0, sqrt=True):
        """Kullback-Leibler Divergence loss.

        Args:
            pred (torch.Tensor): Predicted bboxes.
            target (torch.Tensor): Corresponding gt bboxes.
            alpha (float): Defaults to 1.0.
            sqrt (bool): Whether to sqrt the distance. Defaults to True.

        Returns:
            loss (torch.Tensor)
        """
        xy_p, Sigma_p = pred
        xy_t, Sigma_t = target

        _shape = xy_p.shape

        xy_p = xy_p.reshape(-1, 2)

        xy_t = xy_t.reshape(-1, 2)
        Sigma_p = Sigma_p.reshape(-1, 2, 2)
        Sigma_t = Sigma_t.reshape(-1, 2, 2)

        Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                                   -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                                  dim=-1).reshape(-1, 2, 2)
        Sigma_p_inv = Sigma_p_inv / (1e-25 + Sigma_p.det()).unsqueeze(-1).unsqueeze(-1)

        dxy = (xy_p - xy_t).unsqueeze(-1)
        xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

        whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
            dim1=-2, dim2=-1).sum(dim=-1)

        Sigma_p_det_log = Sigma_p.det().log()
        Sigma_t_det_log = Sigma_t.det().log()
        whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
        whr_distance = whr_distance - 1
        distance = (xy_distance / (alpha * alpha) + whr_distance)
        if sqrt:
            distance = torch.sqrt(distance.clamp(0) + 1e-30)

        distance = distance.reshape(_shape[:-1])

        return distance

    def jd_loss(self, pred, target, alpha=1.0, sqrt=True):
        """Symmetrical Kullback-Leibler Divergence loss.

        Args:
            pred (torch.Tensor): Predicted bboxes.
            target (torch.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.
            alpha (float): Defaults to 1.0.
            sqrt (bool): Whether to sqrt the distance. Defaults to True.

        Returns:
            loss (torch.Tensor)
        """
        jd = self.kld_loss(
            pred,
            target,
            alpha=alpha,
            sqrt=False,
        )
        jd = jd + self.kld_loss(
            target,
            pred,
            alpha=alpha,
            sqrt=False,
        )
        jd = jd * 0.5
        if sqrt:
            jd = torch.sqrt(jd.clamp(0) + 1e-30)
        jd = torch.log1p(jd + 1e-30)
        return 1 - 1 / (1 + jd)

    def loss(self,
             sfm_outputs,
             imgC_dict,
             cam_outputs,
             batch_img,
             sfm_targets,
             sfm_paths,
             CAM_indices,
             cur_epoch,
             bs,
             loss_loc_weight=1.,
             loss_shape_weight=1.):
        if self.cfg.haskey('loss_loc_weight'):
            loss_loc_weight = self.cfg.loss_loc_weight
        if self.cfg.haskey('loss_shape_weight') and cur_epoch >= 150:
            loss_shape_weight = self.cfg.loss_shape_weight
        else:
            loss_shape_weight = 0

        loss_sfm = 0
        pr_theta, pr_length, pr_xy, _ = sfm_outputs
        outputs_without_aux = {}
        outputs_without_aux['pred_logits'] = cam_outputs['pred_logits'][:pr_xy.shape[0]]

        steps = self.cfg.sampling_steps
        tgt_y = torch.linspace(0, 0.998, steps).cuda()
        output_xs = pr_xy[:,0,:,:]
        output_ys = tgt_y.repeat(output_xs.shape[0], output_xs.shape[1], 1)
        outputs_without_aux['pred_curves'] = torch.stack([output_xs, output_ys], 1)
        outputs_without_aux['pred_curves_param'] = cam_outputs['pred_curves'][:pr_xy.shape[0]]

        targets = sfm_targets[:bs]
        gt_start_end, gt_xs, gt_centers, gt_lengths, gt_angles = self.get_gt(targets, tgt_y)
        all_anchors = (imgC_dict['imgC_0'][...,1] + imgC_dict['imgC_0'][...,3]) / 2 / self.cfg.img_h

        indices = CAM_indices[:bs]
        # print("CAM_indices:", CAM_indices)

        idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)

        # loc loss
        pr_xy = pr_xy.permute(0,2,3,1)[idx]
        pr_x = pr_xy[:, :, 0]

        sorted_gt_xs = torch.stack([gt_xs[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        sorted_gt_centers = torch.stack([gt_centers[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        sorted_gt_lengths = torch.stack([gt_lengths[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        sorted_gt_angles = torch.stack([gt_angles[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        # sorted_gt_start_end = torch.stack([gt_start_end[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        valid_x = ((sorted_gt_xs >= 0) & (sorted_gt_xs < 1))
        valid_seg = ((sorted_gt_centers[...,0] >= 0) & (sorted_gt_centers[...,0] < 1))

        curve_anchor = all_anchors[idx]
        weights = (torch.sum(valid_x, dtype=torch.float32) / torch.sum(valid_x, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)

        offset_gt_x = sorted_gt_xs - curve_anchor
        offset_gt_x = offset_gt_x.transpose(1, 0) * weights
        offset_gt_x = offset_gt_x.transpose(1, 0)
        pr_x = pr_x.transpose(1, 0) * weights
        pr_x = pr_x.transpose(1, 0)

        loss_loc1 = (F.l1_loss(offset_gt_x[valid_x], pr_x[valid_x], reduction='none').sum()) / len(sorted_gt_xs) * 0.5
        loss_loc = F.smooth_l1_loss(offset_gt_x[valid_x] * self.cfg.img_w,
                          pr_x[valid_x] * self.cfg.img_w, reduction='none').mean()

        # (F.mse_loss(offset_gt_x[valid_x] * self.cfg.img_w, pr_x[valid_x] * self.cfg.img_w))

        # loss_z = F.l1_loss(gt_z_pre[gt_confidence_pre].cuda()*25, pr_z[gt_confidence_pre]*25)
        # loss += loss_z
        # padded = torch.tensor([-200, -200]).cuda().unsqueeze(0).repeat(pr_x.shape[0], 1, 1)
        padded1 = torch.tensor([-200]).cuda().unsqueeze(0).repeat(pr_x.shape[0], 1)
        pr_center_x = torch.cat([((pr_x+curve_anchor)[:,1:] + (pr_x+curve_anchor)[:,:-1]) / 2, padded1], 1)
        pr_y = tgt_y.cuda().unsqueeze(0).repeat(pr_x.shape[0], 1)
        pr_center_y = torch.cat([(pr_y[:,1:] + pr_y[:,:-1]) / 2, padded1], 1)
        pr_center = torch.stack([pr_center_x, pr_center_y], -1)
        pr_length = torch.cat([torch.sqrt(((pr_x+curve_anchor)[:,1:] - (pr_x+curve_anchor)[:,:-1,])**2 + (pr_y[:,1:] - pr_y[:,:-1])**2 + 1e-12), padded1], 1)
        pr_angle = torch.cat([torch.atan((pr_y[:,1:] - pr_y[:,:-1]) / ((pr_x+curve_anchor)[:,1:] - (pr_x+curve_anchor)[:,:-1] + 1e-12)), padded1], 1)
        valid_seg = torch.cat([valid_seg[:,:-1], padded1>0], 1)
        xywhr_pr = torch.stack((pr_center[valid_seg][:, 0], pr_center[valid_seg][:, 1],
                                pr_length[valid_seg], pr_length[valid_seg] / 4,
                                pr_angle[valid_seg]), -1)
        xywhr_gt = torch.stack((sorted_gt_centers[valid_seg][:, 0], sorted_gt_centers[valid_seg][:, 1],
                                sorted_gt_lengths[valid_seg],
                                sorted_gt_lengths[valid_seg] / 4 ,
                                sorted_gt_angles[valid_seg]), -1)

        GT = self.xy_wh_r_2_xy_sigma(xywhr_gt)
        PR = self.xy_wh_r_2_xy_sigma(xywhr_pr)
        loss_shape = torch.mean(self.jd_loss(PR, GT))

        loss_sfm += loss_loc * loss_loc_weight
        loss_sfm += loss_loc1 * loss_loc_weight
        loss_sfm += loss_shape * loss_shape_weight
        return_value={
            'loss': loss_sfm,
            'loss_stats': {
            'loss_sfm': loss_sfm,
            'loss_loc1': loss_loc1 * loss_loc_weight,
            'loss_loc': loss_loc * loss_loc_weight,
            'loss_shape': loss_shape * loss_shape_weight
        }}

        return return_value

    def get_lanes(self, cam_outputs, sfm_outputs, data, save_img_path, save_json_path, as_lanes=True):
        '''
        Convert model outputs to lanes.
        '''
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        PURPLE = (128, 0, 128)
        WHITE = (255, 255, 255)
        CYAN = (0, 255, 255)

        colors = [RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, WHITE]

        lamda_box = np.array(torch.linspace(0, 0.998, self.cfg.sampling_steps))

        pred_labels = cam_outputs['pred_logits'].detach()[:(self.cfg.batch_size // self.cfg.queue_length)]
        imgs = data['img'][:(self.cfg.batch_size // self.cfg.queue_length)]
        pred_curves = cam_outputs['pred_curves'][:(self.cfg.batch_size // self.cfg.queue_length)]

        prob = F.softmax(pred_labels, -1)
        scores, batch_labels = prob.max(-1)

        out_scope, out_length, out_xy, imgCs = sfm_outputs
        valid_xy = out_xy.permute(0, 2, 3, 1).detach().cpu()
        for bid in range((self.cfg.batch_size // self.cfg.queue_length)):
            imgs_dic = {}
            # imgs_dic["task_name"] = data['meta'].data[0][bid]['full_img_path'].split("/")[-1]
            imgs_dic["task_name"] = data['meta'][bid]['full_img_path'].split("/")[-1]
            imgs_dic["lane_mark"] = []

            valid_xs = valid_xy[..., 0][bid][batch_labels[bid] > 0]
            boxes_list = imgCs['imgC_0'][bid][batch_labels[bid] > 0]

            cur_frame = np.ascontiguousarray(deepcopy(cv2.resize(imgs[bid].detach().cpu().permute(1,2,0).numpy(), (self.cfg.img_h, self.cfg.img_w))))

            for lane_index, (boxes, out_x) in enumerate(
                    zip(boxes_list, valid_xs)):
                lane_points = []


                # lowers_y = pred_curves[bid][lane_index][4].item()
                # uppers_y = pred_curves[bid][lane_index][5].item()
                # lowers_x = pred_curves[bid][lane_index][6].item()
                # uppers_x = pred_curves[bid][lane_index][7].item()
                for index, (box, loc) in enumerate(zip(boxes, out_x)):
                    loc = loc * self.cfg.img_h

                    if (box[1] + box[3]) / 2 <= 0 or (box[1] + box[3]) / 2 >= self.cfg.img_w:
                        continue
                    newpoint = (((box[1] + box[3]) / 2 + loc).item(), lamda_box[index] * self.cfg.img_w)

                    lane_points.append(newpoint)

                    if newpoint[0] <= 0 or newpoint[1] <= 0 or newpoint[0] >= self.cfg.img_h or newpoint[1] >= self.cfg.img_w:
                        continue
                    cv2.circle(cur_frame, (int(newpoint[0]), int(newpoint[1])), 4, colors[lane_index % 7], -1)

                lane = [[x / 640 * 800, y / 640 * 800, 0] for (x, y) in lane_points]

                dic = {}
                dic["node_list"] = lane
                dic["index"] = lane_index
                dic["acce_line_info"] = 'x'
                dic["lane_mark_type"] = 'x'
                dic["lane_mark_color"] = 'x'
                dic["index_uniq"] = 'x'
                imgs_dic["lane_mark"].append(dic)

            save_name = imgs_dic["task_name"]

            cv2.imwrite(os.path.join(save_img_path, save_name),
                        cur_frame * 255)

            save_pr_path = os.path.join(save_json_path, save_name[:-4] + '_pr.json')
            if os.path.exists(save_pr_path):
                os.remove(save_pr_path)
            with open(save_pr_path, 'a') as prfile:
                json.dump(imgs_dic, prfile)
                prfile.write('\n')


        return imgs_dic
