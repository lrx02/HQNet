import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.training = 'train' in split
        self.processes = Process(processes, cfg)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample
