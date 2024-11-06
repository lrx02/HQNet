import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from hqnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from hqnet.datasets import build_dataloader
from hqnet.utils.recorder import build_recorder
from hqnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()

        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader, cam_epoch):
        self.net.train()
        if epoch < cam_epoch:
            self.net.module.sfm_head.eval()
        else:
            self.net.module.sfm_head.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            if self.net.module.sfm_head.training:
                combined_dict = {key: [] for key in data[0].keys()}

                for data_item in data:
                    for key, value in data_item.items():
                        if isinstance(value, torch.Tensor):
                            combined_dict[key].append(value)
                        elif isinstance(value, DC):
                            combined_dict[key].extend(value._data[0])
                        else:
                            raise TypeError(f'Unsupported data type for key {key}')

                for key, value in combined_dict.items():
                    if isinstance(value[0], torch.Tensor):
                        combined_dict[key] = torch.cat(value, dim=0)
                    elif isinstance(value, dict):
                        combined_dict[key] = DC(value)
                data = combined_dict


            data = self.to_cuda(data)
            output, sfm_out, _, _ = self.net(data, epoch)
            self.optimizer.zero_grad()
            loss = output['loss'].sum() + sfm_out['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            if epoch == self.cfg.cam_epoch:
                self.recorder.reset()
            self.recorder.update_loss_stats({'total_loss': loss})
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.update_loss_stats(sfm_out['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')
        return loss

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        min_loss = 9999
        cam_epoch = self.cfg.cam_epoch
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            if epoch >= cam_epoch:
                train_loader = build_dataloader(self.cfg.dataset.train_sfm,
                                                self.cfg,
                                                is_train=True)
            loss = self.train_epoch(epoch, train_loader, cam_epoch)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()

            if loss.item() < min_loss:
                min_loss = loss.item()
                self.save_ckpt(is_best=True)
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        cam_save_img_path = os.path.join(self.recorder.work_dir, 'cam_test_images')
        sfm_save_img_path = os.path.join(self.recorder.work_dir, 'sfm_test_images')
        cam_save_json_path = os.path.join(self.recorder.work_dir, 'cam_test_compare')
        sfm_save_json_path = os.path.join(self.recorder.work_dir, 'sfm_test_compare')
        if not os.path.exists(cam_save_img_path):
            os.mkdir(cam_save_img_path)
        if not os.path.exists(sfm_save_img_path):
            os.mkdir(sfm_save_img_path)
        if not os.path.exists(cam_save_json_path):
            os.mkdir(cam_save_json_path)
        if not os.path.exists(sfm_save_json_path):
            os.mkdir(sfm_save_json_path)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):

            combined_dict = {key: [] for key in data[0].keys()}

            for data_item in data:
                for key, value in data_item.items():
                    if isinstance(value, torch.Tensor):
                        combined_dict[key].append(value)
                    elif isinstance(value, DC):
                        combined_dict[key].extend(value._data[0])
                    else:
                        raise TypeError(f'Unsupported data type for key {key}')

            for key, value in combined_dict.items():
                if isinstance(value[0], torch.Tensor):
                    combined_dict[key] = torch.cat(value, dim=0)
                elif isinstance(value, dict):
                    combined_dict[key] = DC(value)
            data = combined_dict
            data = self.to_cuda(data)
            with torch.no_grad():

                if self.cfg.resume_from is not None:
                    test_epoch = int(self.cfg.resume_from.split("/")[-1].split(".pth")[0])
                if self.cfg.load_from is not None:
                    if self.cfg.load_from.split("/")[-1].split(".pth")[0] == 'best':
                        test_epoch = self.cfg.cam_epoch
                    else:
                        test_epoch = int(self.cfg.load_from.split("/")[-1].split(".pth")[0])
                _, _, cam_outputs, sfm_outputs = self.net(data, test_epoch)
                output_cam = self.net.module.cam_head.get_lanes(cam_outputs, data, cam_save_img_path, cam_save_json_path)
                if test_epoch <= self.cfg.cam_epoch:
                    continue
                output_sfm = self.net.module.sfm_head.get_lanes(cam_outputs, sfm_outputs, data, sfm_save_img_path, sfm_save_json_path)
                predictions.extend(output_sfm)


    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
