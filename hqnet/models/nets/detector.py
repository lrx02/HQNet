import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from copy import deepcopy
from hqnet.models.registry import NETS
from ..registry import build_backbones, build_positional_encoding, build_cam_head, build_sfm_head, build_necks, build_transformer


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        hidden_dim = cfg.positional_encoding['num_pos_feats']
        self.backbone = build_backbones(cfg)
        self.positional_encoding = build_positional_encoding(cfg)
        self.transformer = build_transformer(cfg)
        self.input_proj = nn.Conv2d(128, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(cfg.num_queries, hidden_dim)
        # self.heads = build_heads(cfg)
        self.cam_head = build_cam_head(cfg)
        self.sfm_head = build_sfm_head(cfg)

    def get_lanes(self):
        return self.heads.get_lanes(output)

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_latest_input_bbox(self, outputs, targets, images, indices, tracks, paths):
        idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)

        # anchor sampling settings
        queue_length, batch_size, num_queries, img_res = self.cfg.queue_length, self.cfg.batch_size // self.cfg.queue_length, self.cfg.num_queries, self.cfg.img_res
        h, w, ori_h, ori_w = self.cfg.img_h,  self.cfg.img_w, self.cfg.ori_img_h, self.cfg.ori_img_w

        steps = self.cfg.sampling_steps
        r = self.cfg.roi_r
        tgt_y = torch.linspace(0, 0.998, steps).cuda()

        if (img_res * (ori_h-1) / (h-1)) == (img_res * (ori_w-1) / (w-1)):
            resized_img_res = (img_res * (ori_h-1) / (h-1))
        else:
            print("Warning! Image resolution is not equal!!!")

        inds = {}
        for i in range(queue_length):
            inds['inds_{}'.format(str(i))] = (idx[0][torch.logical_and((idx[0] >= batch_size*i), (idx[0] < batch_size*(i+1)))],
                                              idx[1][torch.logical_and((idx[0] >= batch_size*i), (idx[0] < batch_size*(i+1)))])

        # track
        track_dict = {}
        tracks_change_dict = {}

        for qi in range(queue_length):
            track_dict['track_{}'.format(str(qi))] = tracks[batch_size*qi:batch_size*(qi+1)]
            tracks_change_dict['track_{}'.format(str(qi))] = [(track_dict['track_{}'.format(str(qi))][i] - tracks[:batch_size][i]) for i in range(batch_size)]

        b3_dict, b2_dict, b1_dict, b0_dict = {}, {}, {}, {}
        out_bbox = outputs['pred_curves']
        for qi in range(queue_length):

            b3_dict['b3_{}'.format(str(qi))] = out_bbox[:, :, 3][batch_size*qi:batch_size*(qi+1)].reshape(-1)
            b2_dict['b2_{}'.format(str(qi))] = out_bbox[:, :, 2][batch_size*qi:batch_size*(qi+1)].reshape(-1)
            b1_dict['b1_{}'.format(str(qi))] = out_bbox[:, :, 1][batch_size*qi:batch_size*(qi+1)].reshape(-1)
            b0_dict['b0_{}'.format(str(qi))] = out_bbox[:, :, 0][batch_size*qi:batch_size*(qi+1)].reshape(-1)

        imgC_dict, imgC_affine_dict = {}, {}
        valid_dict = {}

        for qi in range(queue_length):
            # target frame
            if qi == 0:
                # curve semi-parameters
                batch_b3 = b3_dict['b3_{}'.format(str(qi))].unsqueeze(-1).repeat(1, tgt_y.shape[-1])
                batch_b2 = b2_dict['b2_{}'.format(str(qi))].unsqueeze(-1).repeat(1, tgt_y.shape[-1])
                batch_b1 = b1_dict['b1_{}'.format(str(qi))].unsqueeze(-1).repeat(1, tgt_y.shape[-1])
                batch_b0 = b0_dict['b0_{}'.format(str(qi))].unsqueeze(-1).repeat(1, tgt_y.shape[-1])

                output_xs = batch_b3 * tgt_y ** 3 + batch_b2 * tgt_y ** 2 + batch_b1 * tgt_y + batch_b0
                output_ys = tgt_y.repeat(output_xs.shape[0], 1)

                # disturb_xs = (torch.rand(output_ys.shape[0],
                #                          output_ys.shape[1]).cuda() * 2.2 - 1.1) / 2 * r / w
                # disturb_ys = (torch.rand(output_ys.shape[0],
                #                          output_ys.shape[1]).cuda() * 2 - 1) * 4 * r / h
                # output_xs += disturb_xs
                # output_ys += disturb_ys


                imgC_dict['imgC_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1], 5), -1.0)
                imgC_affine_dict['imgC_affine_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1], 5), -1.0)
                valid_dict['valid_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1]), False)

                ys = tgt_y * h

                xs_p = output_xs * w
                ys_p = output_ys * h
                xy_cord = torch.stack((xs_p, ys_p), -1)

                # all (x, y)
                imgC_dict['imgC_{}'.format(str(qi))][:, :, 0] = (torch.arange(batch_size).repeat(num_queries,1).T).reshape(batch_size*num_queries).unsqueeze(-1).repeat(1, steps)
                imgC_dict['imgC_{}'.format(str(qi))][:, :, 1] = xy_cord[:, :, 0] - r
                imgC_dict['imgC_{}'.format(str(qi))][:, :, 2] = xy_cord[:, :, 1] - r
                imgC_dict['imgC_{}'.format(str(qi))][:, :, 3] = xy_cord[:, :, 0] + r
                imgC_dict['imgC_{}'.format(str(qi))][:, :, 4] = xy_cord[:, :, 1] + r

                imgC_affine_dict['imgC_affine_{}'.format(str(qi))] = imgC_dict['imgC_{}'.format(str(qi))]

            # auxiliary frames
            else:
                imgC_dict['imgC_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1], 5), -1.0)
                imgC_affine_dict['imgC_affine_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1], 5), -1.0)
                valid_dict['valid_{}'.format(str(qi))] = torch.full((output_xs.shape[0], output_xs.shape[1]), False)

                for b_id in range(batch_size):
                    b3l = b3_dict['b3_{}'.format(str(qi))].reshape(batch_size, num_queries, -1)[b_id][:,0]
                    b2l = b2_dict['b2_{}'.format(str(qi))].reshape(batch_size, num_queries, -1)[b_id][:,0]
                    b1l = b1_dict['b1_{}'.format(str(qi))].reshape(batch_size, num_queries, -1)[b_id][:,0]
                    b0l = b0_dict['b0_{}'.format(str(qi))].reshape(batch_size, num_queries, -1)[b_id][:,0]

                    bl_stack = torch.stack((b3l, b2l, b1l, b0l), -1)
                    xs_l = ((((tgt_y.repeat(bl_stack.shape[0], 1)).T ** 3 * b3l + (
                        tgt_y.repeat(bl_stack.shape[0], 1)).T ** 2 * b2l + (
                                 tgt_y.repeat(bl_stack.shape[0], 1)).T * b1l + b0l).T)) * h

                    if tracks[b_id][-1].item() == 0.0:
                        yyy = torch.stack([ys] * xs_l.shape[0], 0)
                        ys_temp = (yyy + tracks_change_dict['track_{}'.format(str(qi))][b_id][1] / resized_img_res) / h
                        xs = ((ys_temp.T ** 3 * b3l + ys_temp.T ** 2 * b2l + ys_temp.T * b1l + b0l).T) * w
                        xxx = xs + tracks_change_dict['track_{}'.format(str(qi))][b_id][0] / resized_img_res
                        ys1 = (yyy + tracks_change_dict['track_{}'.format(str(qi))][b_id][1] / resized_img_res)
                    else:
                        yyy = torch.stack([ys] * xs_l.shape[0], 0)
                        ys_temp = (yyy + tracks_change_dict['track_{}'.format(str(qi))][b_id][0] / resized_img_res) / h  # * 640
                        xs = ((ys_temp.T ** 3 * b3l + ys_temp.T ** 2 * b2l + ys_temp.T * b1l + b0l).T) * w
                        xxx = xs - tracks_change_dict['track_{}'.format(str(qi))][b_id][1] / resized_img_res
                        ys1 = (yyy + tracks_change_dict['track_{}'.format(str(qi))][b_id][0] / resized_img_res)

                    xy_aux = torch.stack((xs, ys1), -1)
                    xy_cord_affine = torch.stack((xxx, yyy), -1)

                    overlap_valid = (xs > 0) & (xs < w) & (ys1 >= 0) & (ys1 < h) & (xxx >= 0) & (xxx < w) & (yyy >= 0) & (yyy < h)

                    # original positions used for roi_align
                    imgC_dict['imgC_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 0] = b_id
                    imgC_dict['imgC_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 1] = xy_aux[:, :, 0] - r
                    imgC_dict['imgC_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 2] = xy_aux[:, :, 1] - r
                    imgC_dict['imgC_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 3] = xy_aux[:, :, 0] + r
                    imgC_dict['imgC_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 4] = xy_aux[:, :, 1] + r
                    # affine positions used for unified positional encoding
                    imgC_affine_dict['imgC_affine_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 0] = b_id
                    imgC_affine_dict['imgC_affine_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 1] = xy_cord_affine[:, :, 0] - r
                    imgC_affine_dict['imgC_affine_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 2] = xy_cord_affine[:, :, 1] - r
                    imgC_affine_dict['imgC_affine_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 3] = xy_cord_affine[:, :, 0] + r
                    imgC_affine_dict['imgC_affine_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :, 4] = xy_cord_affine[:, :, 1] + r

                    valid_dict['valid_{}'.format(str(qi))][b_id*num_queries:(b_id+1)*num_queries, :] = overlap_valid

        target_imgC = imgC_dict['imgC_{}'.format(str(0))]

        centerx = (target_imgC[:, :, 1] + target_imgC[:, :, 3]) / 2
        centery = (target_imgC[:, :, 2] + target_imgC[:, :, 4]) / 2
        valid = (centerx > 0) & (centerx < w) & (centery >= 0) & (centery < h)
        valid_dict['valid_{}'.format(str(0))] = valid

        for qi in range(queue_length):
            imgC_dict['imgC_{}'.format(str(qi))] = imgC_dict['imgC_{}'.format(str(qi))].reshape(batch_size, num_queries, steps, -1).cuda().detach()
            imgC_affine_dict['imgC_affine_{}'.format(str(qi))] = imgC_affine_dict['imgC_affine_{}'.format(str(qi))].reshape(batch_size, num_queries, steps, -1).cuda().detach()
            valid_dict['valid_{}'.format(str(qi))] = valid_dict['valid_{}'.format(str(qi))].reshape(batch_size, num_queries, steps).cuda().detach()
        return imgC_dict, valid_dict, imgC_affine_dict, gt_idx

    def forward(self, batch, cur_epoch):
        output = {}
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        fea1 = fea[-1]
        fea2 = fea[-2]

        pmasks = F.interpolate(batch['seg'][:, 0, :, :][None].to(torch.float32), size=fea1.shape[-2:]).to(torch.bool)[0]
        pos = self.positional_encoding(fea1, pmasks)

        hs, encoded_feature = self.transformer(self.input_proj(fea1), pmasks,
                                                                     self.query_embed.weight, pos)

        (output, indices), cam_outputs = self.cam_head(hs, batch=batch)

        if cur_epoch < self.cfg.cam_epoch:
            return output, {'loss': torch.zeros_like(output['loss']), 'loss_stats': {'loss_sfm': torch.zeros_like(output['loss'])}}, cam_outputs, {}

        targets = [batch['lane_line'][i][batch['lane_line'][i][:,0]>0] for i in range(len(batch['lane_line']))]
        paths = [batch['meta'][i]['full_img_path'] for i in range(len(batch['meta']))]

        imgC_dict, valid_dict, imgC_affine_dict, gt_idx = self.get_latest_input_bbox(
            cam_outputs, targets, batch['img'], indices, batch['track'][:,:-1], paths)

        sfm_loss, sfm_outputs = self.sfm_head(encoded_feature, fea2, cam_outputs, imgC_dict, imgC_affine_dict, batch['img'], targets, paths, indices, cur_epoch, batchsize = (self.cfg.batch_size // self.cfg.queue_length))


        return output, sfm_loss, cam_outputs, sfm_outputs
