import math
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import json

from hqnet.models.utils.mlp import MLP

from ..registry import CAM_HEADS, build_matcher

from ..utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment

@CAM_HEADS.register_module
class CAMHead(nn.Module):
    def __init__(self, cfg=None):
        super(CAMHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.hidden_dim = self.cfg.positional_encoding['num_pos_feats']
        self.num_cls = 2
        self.aux_loss = True

        self.class_embed = nn.Linear(self.hidden_dim, self.num_cls)
        self.specific_embed = MLP(self.hidden_dim, self.hidden_dim, 8, 3)

        self.matcher = build_matcher(cfg)
        self.losses = cfg.losses


        empty_weight = torch.ones(2)
        empty_weight[0] = 1
        self.register_buffer('empty_weight', empty_weight)


    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self, hs, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            hs: input features (list[Tensor])
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        '''

        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)

        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:  # -2 decoder output
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        # if self.training:
            # up1 = self.upsample1(p50)
            # up2 = self.upsample2(up1)
            # up3 = self.upsample3(up2)
            # out['pred_seg'] = up3
        return self.loss(out, kwargs['batch']), out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def loss(self,
             output,
             batch,
             loss_ce_weight=2.,
             loss_curves_weight=0.5,
             loss_lowers_weight=1.,
             loss_uppers_weight=1.):
        if self.cfg.haskey('loss_ce_weight'):
            loss_ce_weight = self.cfg.loss_ce_weight
        if self.cfg.haskey('loss_curves_weight'):
            loss_curves_weight = self.cfg.loss_curves_weight
        if self.cfg.haskey('loss_lowers_weight'):
            loss_lowers_weight = self.cfg.loss_lowers_weight
        if self.cfg.haskey('loss_uppers_weight'):
            loss_uppers_weight = self.cfg.loss_uppers_weight


        tracks = [track[:-1] for track in batch['track']]
        outputs_without_aux = {k: v for k, v in output.items() if k != 'aux_outputs'}
        targets = [target[target[:,0] > 0] for target in batch['lane_line']]
        images = batch['img']
        paths = [path['full_img_path'] for path in batch['meta']]

        indices = self.matcher(outputs_without_aux, targets, images, paths)
        num_curves = sum(tgt.shape[0] for tgt in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(output.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_curves)
        num_curves = torch.clamp(num_curves / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses: # ['labels', 'curves', 'seg_map' (optional)]
            losses.update(self.get_loss(loss, output, targets, indices, num_curves))


        if 'aux_outputs' in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets, images, paths)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    # if loss == 'seg_map':
                    #     continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_curves, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)



        loss_ce = 0
        loss_lowers = 0
        loss_uppers = 0
        loss_curves = 0

        loss_ce += losses['loss_ce']
        loss_lowers += losses['loss_lowers']
        loss_uppers += losses['loss_uppers']
        loss_curves += losses['loss_curves']


        for layer_item in range(len(output['aux_outputs'])):
            loss_ce += losses['loss_ce_{}'.format(str(layer_item))]
            loss_lowers += losses['loss_lowers_{}'.format(str(layer_item))]
            loss_uppers += losses['loss_uppers_{}'.format(str(layer_item))]
            loss_curves += losses['loss_curves_{}'.format(str(layer_item))]

        loss_ce /= (len(output['aux_outputs'])+1)
        loss_lowers /= (len(output['aux_outputs'])+1)
        loss_uppers /= (len(output['aux_outputs'])+1)
        loss_curves /= (len(output['aux_outputs'])+1)

        loss = loss_ce * loss_ce_weight + loss_lowers * loss_lowers_weight \
            + loss_uppers * loss_uppers_weight + loss_curves * loss_curves_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss_cam': loss,
                'loss_ce': loss_ce * loss_ce_weight,
                'loss_lowers': loss_lowers * loss_lowers_weight,
                'loss_uppers': loss_uppers * loss_uppers_weight,
                'loss_curves': loss_curves * loss_curves_weight
            }
        }

        return return_value, indices

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)

    def loss_labels(self, outputs, targets, indices, num_curves, log=False):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction = 'mean')
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def loss_curves(self, outputs, targets, indices, num_curves):

        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        out_bbox = outputs['pred_curves']

        src_lowers_y = out_bbox[:, :, 4][idx]
        src_uppers_y = out_bbox[:, :, 5][idx]
        src_lowers_x = out_bbox[:, :, 6][idx]
        src_uppers_x = out_bbox[:, :, 7][idx]

        target_lowers_y = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers_y = torch.cat([tgt[:, 2][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_lowers_x = torch.cat([tgt[:, 3][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers_x = torch.cat([tgt[:, 4][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_points = torch.cat([tgt[:, 5:][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        target_xs = target_points[:, :target_points.shape[1] // 2]
        ys = target_points[:, target_points.shape[1] // 2:].transpose(1, 0)
        valid_xs = target_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)

        # Calculate the predicted xs
        b3 = out_bbox[:, :, 3][idx]
        b2 = out_bbox[:, :, 2][idx]
        b1 = out_bbox[:, :, 1][idx]
        b0 = out_bbox[:, :, 0][idx]  # N

        output_xs = (b3 * ys ** 3+ b2 * ys ** 2 + b1 * ys + b0) * weights # N x len(points)
        output_xs = output_xs.transpose(1, 0)
        tgt_xs = target_xs.transpose(1, 0) * weights
        tgt_xs = tgt_xs.transpose(1, 0)

        loss_polys = F.l1_loss(tgt_xs[valid_xs], output_xs[valid_xs], reduction='none')

        losses = {}

        loss_lowers = F.l1_loss(target_lowers_y, src_lowers_y, reduction='none')
        loss_uppers =  F.l1_loss(target_uppers_y, src_uppers_y, reduction='none')
        loss_lowers += F.l1_loss(target_lowers_x, src_lowers_x, reduction='none')
        loss_uppers +=  F.l1_loss(target_uppers_x, src_uppers_x, reduction='none')
        loss_lowers /= 2
        loss_uppers /= 2

        losses['loss_lowers'] = loss_lowers.sum() / num_curves
        losses['loss_uppers'] = loss_uppers.sum() / num_curves
        losses['loss_curves'] = loss_polys.sum() / num_curves

        return losses

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def clamp(self, min, max, para):
        if para >= min and para <= max:
            return para
        elif para > max:
            return max
        else:
            return min


    def get_lanes(self, outputs, data, save_img_path, save_json_path, as_lanes=True):
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

        pred_labels = outputs['pred_logits'].detach()
        imgs = data['img']

        prob = F.softmax(pred_labels, -1)
        scores, batch_labels = prob.max(-1)

        pred_curves = outputs['pred_curves'].detach()
        batch_curves = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)

        for bid in range(len(pred_labels)):
            imgs_dic = {}
            if isinstance(data['meta'], list):
                imgs_dic["task_name"] = data['meta'][bid]['full_img_path'].split("/")[-1]
            else:
                imgs_dic["task_name"] = data['meta'].data[0][bid]['full_img_path'].split("/")[-1]
            imgs_dic["lane_mark"] = []
            cur_frame = np.ascontiguousarray(cv2.resize(deepcopy(imgs[bid].detach().cpu().permute(1,2,0).numpy()), (800, 800)))
            batch_curve = batch_curves[bid]
            pred_valid = batch_curve.cpu().numpy()[batch_labels[bid].cpu().numpy() > 0]
            if len(pred_valid) == 0:

                dic = {}

                dic["index"] = 0
                dic["node_list"] = []
                dic["acce_line_info"] = 'x'
                dic["lane_mark_type"] = 'x'
                dic["lane_mark_color"] = 'x'
                dic["index_uniq"] = 0
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
                continue
            for n, lane in enumerate(pred_valid):
                lane = lane[1:]

                b3 = lane[3]
                b2 = lane[2]
                b1 = lane[1]
                b0 = lane[0]

                # inspect start/end points
                lowers_y = lane[4]
                uppers_y = lane[5]
                lowers_x = lane[6]
                uppers_x = lane[7]
                cv2.circle(cur_frame, (int(self.clamp(0,1,lowers_x) * cur_frame.shape[1]), int(self.clamp(0,1,lowers_y) * cur_frame.shape[0])), 5, colors[n % len(colors)], -1)
                cv2.circle(cur_frame, (int(self.clamp(0,1,uppers_x) * cur_frame.shape[1]), int(self.clamp(0,1,uppers_y) * cur_frame.shape[0])), 5, colors[n % len(colors)], -1)

                lamda_box = np.linspace(0, 800, 81)
                points = np.zeros((len(lamda_box), 2))
                points[:, 1] = lamda_box / 800
                points[:, 0] = (b3 * points[:, 1] ** 3 + b2 * points[:, 1] ** 2 + b1 * points[:, 1] + b0)
                points *= cur_frame.shape[0]
                lane_points = []
                for pt in points:
                    if (pt[0] > 0 and pt[0] < 800 and pt[1] > lowers_y * 800 and pt[1] < uppers_y * 800):
                        cv2.circle(cur_frame, (int(pt[0]), int(pt[1])), 3, colors[n % len(colors)], -1)
                        lane_points.append((pt[0].item(), pt[1].item()))

                lane = [[x, y, 0] for (x, y) in lane_points]

                dic = {}
                dic["index"] = n
                dic["node_list"] = lane
                dic["acce_line_info"] = 'x'
                dic["lane_mark_type"] = 'x'
                dic["lane_mark_color"] = 'x'
                dic["index_uniq"] = 0
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
            print()


        return imgs_dic