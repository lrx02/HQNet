import os.path as osp
import numpy as np
import cv2
import os
import pickle
import json
import torchvision
from .base_dataset import BaseDataset
from .registry import DATASETS
import logging
import random
from collections import Counter
from tqdm import tqdm
from copy import deepcopy

@DATASETS.register_module
class MultiBEV4CAM(BaseDataset):
    def __init__(self, split, processes=None, cfg=None):
        super().__init__(split, processes, cfg)

        # dataset setting
        self.manual_forced_max_lanes = 9
        self.max_points = 0
        self.img_h, self.img_w = cfg.ori_img_h, cfg.ori_img_w
        self._split = split
        self._image_file = []
        self._image_ids = []


        self.queue_length = cfg.queue_length
        self.detection_pattern = cfg.detection_pattern  # det3d | det2d
        self._cache_file = os.path.join(cfg.cache_dir, "{}_{}.pkl".format(cfg.dataset_type, self._split))
        self.sample_list_root = os.path.join(
            '/data/lrx/TITS/MF_HAN_new_versions/MF_HAN_data_processing/data/{}_index.txt'.format(split))
        self._load_data()
        self.load_annotations()

    def _load_data(self):
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            self._transform_annotations()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._annotations,
                             self._image_ids,
                             self._image_file,
                             self.max_lanes,
                             self.max_points], f)
        else:
            print("Loading from cache file: {}...\nMake sure your data is not changed!".format(self._cache_file))
            with open(self._cache_file, "rb") as f:
                (self._annotations,
                 self._image_ids,
                 self._image_file,
                 self.max_lanes,
                 self.max_points) = pickle.load(f)


    def readTxt(self, file_path):
        img_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                item = lines.strip().split()
                img_list.append(item)
        file_to_read.close()
        return img_list

    def _extract_data(self):
        max_lanes = 0
        image_id  = 0
        self.img_list = self.readTxt(self.sample_list_root)
        self.img_list.sort()
        self._old_annotations = {}

        C = Counter()
        for i in tqdm(range(len(self.img_list)), ncols=67, desc="Reading raw data..."):
            sample_name = self.img_list[i][-2]
            anno_name = self.img_list[i][-1]
            sample_weight = 0
            anno_path = anno_name
            # if not os.path.exists(anno_path):
            #     continue
            with open(anno_path, 'r') as data_file:
                anno_data = data_file.readlines()
                anno_data = [line.strip("\n") for line in anno_data]
                lanes = [line.split("\b") for line in anno_data]
                cur_track = [list(map(float, line[-4:])) + [sample_weight] for line in lanes]
                lanes = [line[:-4] for line in lanes]
                lanes = [list(map(float, lane)) for lane in lanes if len(lane)>10*3]
                if self.detection_pattern == 'det2d':
                    lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 5)] for lane in lanes]
                elif self.detection_pattern == 'det3d':
                    lanes = [[(lane[i], lane[i + 1], lane[i + 2]) for i in range(0, len(lane), 5)] for lane in lanes]
                else:
                    raise Exception('Format of lane detection error, choose det2d or det3d.')
                lanes = [lane for lane in lanes if len(lane) >= 2]
                C[len(lanes)] += 1
                flag = False
                if not len(lanes):
                    continue
                for lane in range(len(lanes)):
                    lanes[lane] = sorted(lanes[lane], key=lambda x: x[1], reverse=False)
                    if abs(lanes[lane][0][1] - lanes[lane][-1][1]) < 10:
                        flag = True
                        break
                if flag or len(lanes) > self.manual_forced_max_lanes:
                    continue
                max_lanes = max(max_lanes, len(lanes))
                self.max_lanes = max_lanes
                self.max_lanes = self.manual_forced_max_lanes
                if lanes:
                    self.max_points = max(self.max_points, max([len(l) for l in lanes]))

                if self.detection_pattern == 'det2d':
                    image_path = sample_name
                elif self.detection_pattern == 'det3d':
                    image_path = sample_name.replace('.jpg', '.npy')

                self._image_file.append(image_path)
                self._image_ids.append(image_id)
                self._old_annotations[image_id] = {
                    'path': image_path,
                    'raw_lanes': lanes,
                    'track': cur_track,
                    'categories': [1] * len(lanes)
                }
                image_id += 1
        print(sorted(dict(C).items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

    def _transform_annotations(self):
        print('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in tqdm(self._old_annotations.items()):
            self._annotations[image_id] = self._transform_annotation(old_anno)


    def _transform_annotation(self, anno, img_wh=None):
        old_lanes   = anno['raw_lanes'] # num_lane * [(num_point, 3)]
        categories  = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        if img_wh is None:
            track = anno['track'][0]
            img_h, img_w = self.img_h, self.img_w
        else:
            track = anno['track']
            img_w, img_h = img_wh
        z_range = (-2, 1)

        old_lanes   = zip(old_lanes, categories)
        old_lanes   = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes       = np.ones((self.max_lanes, 1 + 4 + 2 * self.max_points), dtype=np.float32) * -1e5 # (max_lanes, cat1+updown4+(max)xs+(max)ys)
        lanes_zs       = np.ones((self.max_lanes, self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes   = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lane = np.array(lane)
            lane_x = lane[:, 0]
            lane_y = lane[:, 1]
            lane_z = lane[:, -1]
            ind = np.argsort(lane_y, axis=0)
            lane_x = np.take_along_axis(lane_x, ind, axis=0)
            lane_y = np.take_along_axis(lane_y, ind, axis=0)
            lane_z = np.take_along_axis(lane_z, ind, axis=0)
            lane = np.stack((lane_x,lane_y), -1)
            lower_y, upper_y       = lane[0][1], lane[-1][1]
            lower_x, upper_x       = lane[0][0], lane[-1][0]
            xs                 = np.array([p[0] for p in lane]) / img_w
            ys                 = np.array([p[1] for p in lane]) / img_h
            zs                 = (lane_z - track[2] - z_range[0]) / (z_range[1] - z_range[0])

            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower_y / img_h
            lanes[lane_pos, 2] = upper_y / img_h
            lanes[lane_pos, 3] = lower_x / img_w
            lanes[lane_pos, 4] = upper_x / img_w
            lanes[lane_pos, 5:5 + len(xs)] = xs
            lanes[lane_pos, (5 + self.max_points):(5 + self.max_points + len(ys))] = ys
            lanes_zs[lane_pos, :len(zs)] = zs
        if self.detection_pattern == 'det2d':
            new_anno = {
                'label': lanes,
                'track': track,
                'old_anno': anno,
            }
        elif self.detection_pattern == 'det3d':
            new_anno = {
                'label': lanes,
                'zs': lanes_zs,
                'track': track,
                'old_anno': anno,
            }

        return new_anno

    def load_annotations(self):
        self.logger.info('Loading MultiBEV4CAM annotations...')
        self.data_infos = []
        # max_lanes = 0
        for anno_idx in self._annotations:
            anno_file = self._annotations[anno_idx]
            self.data_infos.append({
                'img_path':
                anno_file['old_anno']['path'],
                'label':
                anno_file['label'],
                'track':
                anno_file['track'],
                'old_anno':
                anno_file['old_anno']
            })

        if self.training:
            random.shuffle(self.data_infos)
        # self.max_lanes = max_lanes


